"""
!!! ATTENTION !!!

Running the below tests requires a .env file in this files directory, i.e. "./.env" relative to this file.
The .env file must define the following variables:

- DB_URI
- DB_NAME
- USER_NAME
- USER_PW
"""

import pytest
import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import dotenv_values

from ghl.dblib import neo4jlib

if TYPE_CHECKING:
    from neo4j import Result


@pytest.fixture(scope="module")
def config():
    d = dotenv_values(Path(__file__).parent / ".env")
    d["AUTH"] = (d["USER_NAME"], d["USER_PW"])

    return d


@pytest.fixture(scope="module")
def _temporary_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def temporary_directory(_temporary_directory):
    tmp_dir = Path(_temporary_directory) / str(uuid.uuid4())
    tmp_dir.mkdir()
    return tmp_dir


@pytest.fixture()
def temporary_file(_temporary_directory):
    return Path(_temporary_directory) / str(uuid.uuid4())


@pytest.fixture()
def db(config):
    return neo4jlib.Neo4JDatabase(
        uri=config["DB_URI"],
        database=config["DB_NAME"],
        auth=config["AUTH"],
    )


def test_general(config):
    db = neo4jlib.Neo4JDatabase(
        uri=config["DB_URI"],
        database=config["DB_NAME"],
        auth=config["AUTH"],
    )
    db.ensure_connectivity()


def test_export_nodes(db: neo4jlib.Neo4JDatabase, temporary_file):
    db.export_nodes(
        output_file=temporary_file,
        node_labels=["GeneProtein"],
        properties={"labels": "labels(n)"},
    )
    df = pd.read_csv(temporary_file)
    assert "id" in df.columns
    assert "labels" in df.columns

    db.export_nodes(
        output_file=temporary_file,
        node_labels=["GeneProtein", "Disease"],
        properties={
            "labels": "labels(n)",
            "gene_label": "n.gene_label",
            "disease_label": "n.label",
        },
    )
    df = pd.read_csv(temporary_file, encoding="latin-1")


def test_export_edges(db: neo4jlib.Neo4JDatabase, temporary_file):
    db.export_edges(
        output_file=temporary_file,
        edge_types=["ENCODES"],
    )
    df = pd.read_csv(temporary_file)
    assert "source" in df.columns
    assert "target" in df.columns


def test_get_node_labels(db: neo4jlib.Neo4JDatabase):
    node_labels = db.get_node_labels()
    print(node_labels)


def test_get_edge_types(db: neo4jlib.Neo4JDatabase):
    edge_types = db.get_edge_types()
    print(edge_types)


def test_export_node_features(db: neo4jlib.Neo4JDatabase, temporary_directory):
    db.export_node_features(
        output_directory=temporary_directory, node_labels=["GeneProtein", "Disease"]
    )


def test_run_query(db: neo4jlib.Neo4JDatabase, temporary_file):
    db.run_query(
        query="MATCH (n) RETURN id(n) as id, labels(n) as labels LIMIT 10",
        output_file=temporary_file,
    )
    with open(temporary_file) as f:
        print(f.read())


def test_custom_write_function(db: neo4jlib.Neo4JDatabase):
    def pandas_writer(output_file: str, result: "Result"):
        import pandas as pd

        record = next(result)
        columns = record.keys()
        records = [record.values()]

        for record in result:
            records.append(record.values())

        return pd.DataFrame.from_records(data=records, columns=columns)

    df = db.run_query(
        query="MATCH (n) RETURN id(n) as id, labels(n) as labels LIMIT 10",
        output_file="",
        write_function=pandas_writer,
    )
    assert df.shape == (10, 2)
    assert tuple(df.columns) == ("id", "labels")
