from .. import Database
from .writers import write_records_to_csv, write_features_to_csv

from pathlib import Path
import re
from typing import Union, Iterable, Dict, Callable, TYPE_CHECKING, List, Any


if TYPE_CHECKING:
    import pathlib
    from neo4j import Result

from neo4j import GraphDatabase
from contextlib import contextmanager


class Neo4JDatabase(Database):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._args = args
        self._kwargs = kwargs

    @contextmanager
    def _driver(self):
        driver = GraphDatabase.driver(*self._args, **self._kwargs)
        driver.verify_connectivity()
        try:
            yield driver
        finally:
            driver.close()

    def ensure_connectivity(self, *args, **kwargs):
        with self._driver():
            pass

    def export_nodes(
        self,
        output_file: Union[str, "pathlib.Path"],
        node_labels: Union[None, Iterable[str]] = None,
        properties: Union[None, Dict[str, str]] = None,
        write_function: Callable[[str, "Result"], None] = write_records_to_csv,
    ):
        with self._driver() as driver:
            with driver.session() as session:
                where_str = (
                    ""
                    if node_labels is None
                    else "WHERE " + " OR ".join([f"n:{l}" for l in node_labels]) + "\n"
                )
                return_str = "RETURN id(n) as id"
                if properties is not None:
                    return_str += ", " + ", ".join(
                        [f"{e} AS {k}" for k, e in properties.items()]
                    )
                query = f"""
                    MATCH (n)
                    {where_str}
                    {return_str}
                """
                result = session.run(query=query)
                out = write_function(output_file, result)

        return out

    def export_edges(
        self,
        output_file: Union[str, "pathlib.Path"],
        edge_types: Union[None, Iterable[str]] = None,
        properties: Union[None, Dict[str, str]] = None,
        write_function: Callable[[str, "Result"], None] = write_records_to_csv,
    ):
        with self._driver() as driver:
            with driver.session() as session:
                where_str = (
                    ""
                    if edge_types is None
                    else "WHERE " + " OR ".join([f"r:{l}" for l in edge_types]) + "\n"
                )

                return_str = "RETURN id(s) as source, id(t) as target"
                if properties is not None:
                    return_str += ", " + ", ".join(
                        [f"{e} AS {k}" for k, e in properties.items()]
                    )

                result = session.run(
                    f"""
                    MATCH (s)-[r]->(t)
                    {where_str}
                    {return_str}
                """
                )
                out = write_function(output_file, result)
        return out

    def _export_node_features(
        self,
        output_file: Union[str, "pathlib.Path"],
        node_label: str,
        write_function: Callable[[str, "Result", List[str]], None],
    ):
        with self._driver() as driver:
            with driver.session() as session:
                property_result = session.run(
                    f"MATCH (n:{node_label}) UNWIND keys(n) as properties RETURN DISTINCT properties"
                )
                properties = ["id"] + [p.value() for p in property_result]
                nodes_result = session.run(f"MATCH (g:{node_label}) RETURN g")

                out = write_function(
                    output_file=str(output_file),
                    result=nodes_result,
                    properties=properties,
                )
        return out

    def _export_edge_features(
        self,
        output_file: Union[str, "pathlib.Path"],
        edge_type: str,
        write_function: Callable[[str, "Result", List[str]], None],
    ):
        with self._driver() as driver:
            with driver.session() as session:
                property_result = session.run(
                    f"MATCH ()-[r:{edge_type}]->() UNWIND keys(r) as properties RETURN DISTINCT properties"
                )
                properties = ["id"] + [p.value() for p in property_result]
                edges_result = session.run(f"MATCH ()-[r:{edge_type}]->() RETURN r")

                out = write_function(
                    output_file=str(output_file),
                    result=edges_result,
                    properties=properties,
                )
        return out

    def export_node_features(
        self,
        output_directory: Union[str, "pathlib.Path"],
        exclude_regex: Union[str, None] = "_.*",
        node_labels: Union[Iterable[str], None] = None,
        write_function: Callable[
            [str, "Result", List[str]], None
        ] = write_features_to_csv,
    ):
        from tqdm import tqdm

        output_directory = Path(output_directory)

        if node_labels is None:
            node_labels = self.get_node_labels()
            if exclude_regex is not None:
                node_labels = [l for l in node_labels if not re.match(exclude_regex, l)]
        outs = []
        for label in tqdm(node_labels):
            out = self._export_node_features(
                output_file=output_directory / f"{label}",
                node_label=label,
                write_function=write_function,
            )
            outs.append(out)

        return outs

    def export_edge_features(
        self,
        output_directory: Union[str, "pathlib.Path"],
        exclude_regex: Union[str, None] = None,
        edge_types: Union[Iterable[str], None] = None,
        write_function: Callable[
            [str, "Result", List[str]], None
        ] = write_features_to_csv,
    ):
        from tqdm import tqdm

        output_directory = Path(output_directory)

        if edge_types is None:
            edge_types = self.get_edge_types()
            if exclude_regex is not None:
                edge_types = [t for t in edge_types if not re.match(exclude_regex, t)]
        outs = []
        for et in tqdm(edge_types):
            out = self._export_edge_features(
                output_file=output_directory / f"{et}",
                edge_type=et,
                write_function=write_function,
            )
            outs.append(out)

        return outs

    def get_node_labels(self) -> List[str]:
        with self._driver() as driver:
            with driver.session() as session:
                result = session.run(
                    "MATCH (n) UNWIND labels(n) as l RETURN DISTINCT l as labels"
                )
                labels = [p.value() for p in result]
        return labels

    def get_edge_types(self) -> List[str]:
        with self._driver() as driver:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN DISTINCT type(r)
                    """
                )
                types = [p.value() for p in result]

        return types

    def run_query(
        self,
        query: str,
        output_file: Union[str, "pathlib.Path"],
        write_function: Callable[[str, "Result"], Any] = write_records_to_csv,
    ):
        with self._driver() as driver:
            with driver.session() as session:
                result = session.run(query=query)
                return write_function(output_file, result)
