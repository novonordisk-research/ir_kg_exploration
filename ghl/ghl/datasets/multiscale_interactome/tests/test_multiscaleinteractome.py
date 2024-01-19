import pytest
import tempfile
import uuid
from pathlib import Path

from ghl.datasets.multiscale_interactome import MultiscaleInteractomeDataset


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


def test_general(temporary_directory):
    ds = MultiscaleInteractomeDataset(data_path=temporary_directory)
    edges_df = ds.get_edges()
    nodes_df = ds.get_nodes()

    hetero_data = ds.to_pytorch_geometric_heterodata()
    nodes_df.iloc[hetero_data["biological_function"].node_indices]
    edges_df.iloc[hetero_data[("biological_function", "TO", "protein")].edge_indices]

    data = ds.to_pytorch_geometric_data()
    nodes_df.iloc[data.node_indices[:10]]
    edges_df.iloc[data.edge_indices[:10]]
