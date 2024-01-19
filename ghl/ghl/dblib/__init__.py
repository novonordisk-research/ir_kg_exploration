from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pathlib


class Database:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def ensure_connectivity(self, *args, **kwargs):
        raise NotImplementedError()

    def export_nodes(self, output_file: Union[str, "pathlib.Path"]):
        raise NotImplementedError()

    def export_edges(self, output_file: Union[str, "pathlib.Path"]):
        raise NotImplementedError()

    def export_node_features(self, output_directory: Union[str, "pathlib.Path"]):
        raise NotImplementedError()

    def run_query(self, query: str, output_file: Union[str, "pathlib.Path"]):
        raise NotImplementedError()
