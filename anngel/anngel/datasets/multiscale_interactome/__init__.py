from .. import Dataset
from ..utils import download_url, tarextract
from .msi import MSI

from pathlib import Path


# TODO: implement this
class MultiscaleInteractomeDataset(Dataset):
    DATASET_NAME = "MultiscaleInteractome"
    DATASET_URL = "http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz"

    def __init__(self, data_path="."):
        self.path = Path(data_path)
        self.dataset_tgz_path = self.path / f"data.tar.gz"
        self.dataset_path = self.path / self.DATASET_NAME

        self.nodes_file = self.dataset_path / "nodes.tsv"
        self.edges_file = self.dataset_path / "edges.tsv"

        if (not self.nodes_file.exists()) or (not self.edges_file.exists()):
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            self.download()
            self.unzip()
            self.prepare()
            self.cleanup()

    def get_edges(self):
        import pandas as pd

        edges_df = pd.read_csv(self.edges_file, sep="\t")

        return edges_df

    def get_nodes(self):
        import pandas as pd

        nodes_df = pd.read_csv(self.nodes_file, sep="\t")

        return nodes_df

    def to_pytorch_geometric_heterodata(self):
        try:
            import torch
        except ImportError:
            msg = "Missing dependency torch. Please install torch."
            raise ImportError(msg)

        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            msg = "Missing dependency torch_geometric. Please install torch_geometric."
            raise ImportError(msg)

        nodes_df = self.get_nodes()
        edges_df = self.get_edges()

        nodes_df["_id"] = nodes_df.groupby("node_type").cumcount()

        edges_df = (
            edges_df.merge(
                right=nodes_df,
                how="left",
                left_on="source_node_id",
                right_on="node_id",
            )
            .merge(
                right=nodes_df,
                how="left",
                left_on="target_node_id",
                right_on="node_id",
                suffixes=["__source", "__target"],
            )
            .drop(columns=["node_id__source", "node_id__target"])
        )

        data = HeteroData()

        nodes_df["node_index"] = torch.arange(0, nodes_df.shape[0])
        for node_type, df_n in nodes_df.groupby("node_type"):
            data[node_type].node_indices = torch.tensor(df_n["node_index"].values)
            data[node_type].x = torch.empty((df_n.shape[0], 0))

        edges_df["edge_index"] = torch.arange(0, edges_df.shape[0])
        for edge_type, df_e in edges_df.groupby("edge_type"):
            for node_type_source, df_s in df_e.groupby("node_type__source"):
                for node_type_target, df_t in df_s.groupby("node_type__target"):
                    if df_t.shape[0] > 0:
                        data[
                            (node_type_source, edge_type, node_type_target)
                        ].edge_index = torch.tensor(
                            df_t[["_id__source", "_id__target"]].values.T
                        )
                        data[
                            (node_type_source, edge_type, node_type_target)
                        ].edge_indices = torch.tensor(df_t["edge_index"].values)

        return data

    def to_pytorch_geometric_data(self):
        return self.to_pytorch_geometric_heterodata().to_homogeneous(
            add_node_type=True, add_edge_type=True
        )

    def download(self):
        download_url(url=self.DATASET_URL, output_path=self.dataset_tgz_path)

    def unzip(self):
        tarextract(path=self.dataset_tgz_path, output_path=self.dataset_path)

    def cleanup(self):
        if self.dataset_tgz_path.exists():
            self.dataset_tgz_path.unlink()

    def prepare(self):
        msi = MSI(
            drug2protein_file_path=self.dataset_path / "data/1_drug_to_protein.tsv",
            indication2protein_file_path=self.dataset_path
            / "data/2_indication_to_protein.tsv",
            protein2protein_file_path=self.dataset_path
            / "data/3_protein_to_protein.tsv",
            protein2biological_function_file_path=self.dataset_path
            / "data/4_protein_to_biological_function.tsv",
            biological_function2biological_function_file_path=self.dataset_path
            / "data/5_biological_function_to_biological_function.tsv",
        )
        msi.load()

        import pandas as pd

        NODES_FILE_COLUMNS = ["node_id", "node_type"]
        nodes_df = pd.DataFrame.from_records(
            [(k, v["type"]) for k, v in msi.graph.nodes.data()],
            columns=NODES_FILE_COLUMNS,
            # dtype={c: str for c in range(len(NODES_FILE_COLUMNS))},
        )
        nodes_df.to_csv(self.nodes_file, sep="\t", index=False)

        EDGES_FILE_COLUMNS = ["source_node_id", "edge_type", "target_node_id"]
        edges_df = pd.DataFrame.from_records(
            [(s, "TO", t) for s, t, d in msi.graph.edges.data()],
            columns=EDGES_FILE_COLUMNS,
            # dtype={c: str for c in range(len(EDGES_FILE_COLUMNS))},
        )
        edges_df.to_csv(self.edges_file, sep="\t", index=False)
