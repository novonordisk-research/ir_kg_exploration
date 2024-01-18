from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData, Data
    from pandas import DataFrame
    from networkx import Graph


class Dataset:
    def to_pytorch_geometric_heterodata(self) -> "HeteroData":
        raise NotImplementedError()

    def to_pytorch_geometric_data(self) -> "Data":
        raise NotImplementedError()

    def to_networkx(self) -> "Graph":
        from torch_geometric.utils.convert import to_networkx

        return to_networkx(
            self.to_pytorch_geometric_data(), node_attrs=["node_indices"]
        )

    def get_nodes(self) -> "DataFrame":
        raise NotImplementedError()

    def get_edges(self) -> "DataFrame":
        raise NotImplementedError()
