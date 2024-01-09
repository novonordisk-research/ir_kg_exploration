from .. import Dataset

from pathlib import Path


class CustomDataset(Dataset):
    EDGES_FILE_COLUMNS = (
        "source_node_id",
        "edge_type",
        "target_node_id",
        "quality",
        "data_source",
    )
    NODES_FILE_COLUMNS = ("node_id", "node_type")

    def __init__(self, data_path=".", dataset_name="custom"):
        self.path = Path(data_path)
        self.dataset_path = self.path / dataset_name

        self.nodes_file = self.dataset_path / "nodes.tsv"
        self.edges_file = self.dataset_path / "edges.tsv"

    def get_edges(self):
        import pandas as pd

        edges_df = pd.read_csv(self.edges_file, sep="\t")
        edges_df.columns = self.EDGES_FILE_COLUMNS

        return edges_df

    def get_nodes(self):
        import pandas as pd

        nodes_df = pd.read_csv(self.nodes_file, sep="\t")
        nodes_df.columns = self.NODES_FILE_COLUMNS

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
