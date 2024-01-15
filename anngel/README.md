# Accessible Novo Nordisk Graph Embedding Library (ANNGEL)

## Table of Contents

- [Accessible Novo Nordisk Graph Embedding Library (ANNGEL)](#accessible-novo-nordisk-graph-embedding-library-anngel)
  - [Table of Contents](#table-of-contents)
  - [Requirements:](#requirements)
  - [Installation](#installation)
  - [Examples](#examples)
    - [Export nodes and edges from Neo4j database](#export-nodes-and-edges-from-neo4j-database)
    - [Connect to Amazon S3 bucket](#connect-to-amazon-s3-bucket)
    - [Use an included Dataset](#use-an-included-dataset)

## Requirements:

- Python >= 3.10
- [Poetry](https://python-poetry.org/)

## Installation

```bash
poetry install
```

## Examples

### Export nodes and edges from Neo4j database

```Python
from anngel.dblib.neo4jlib import Neo4JDatabase

DB_URI="neo4j://ENTER IP ADDRESS:7687"
DB_NAME="ENTER DB NAME"
USER_NAME="ENTER USER NAME"

db = Neo4JDatabase(
    uri=DB_URI,
    database=DB_NAME,
    auth=(USER_NAME, getpass("Neo4J password:")),
)

db.export_nodes(
    output_file="nodes.csv",
    properties={"labels": "labels(n)"},
)
db.export_edges(
    output_file="nodes.csv",
    properties={"type": "type(r)"},
)
```

### Connect to Amazon S3 bucket

```Python
from anngel.datalib.s3datastore import S3DataStore

BUCKET_NAME = "ENTER S3 BUCKET NAME"
PREFIX = "ENTER S3 PREFIX"

# AWS credential will be collected from ~/.credentials
# if not passed to S3DataStore
ds = S3DataStore(
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    bucket_name=BUCKET_NAME,
    prefix=PREFIX,
    local_directory="./data",
)

new_file = ds.get_file("test.txt", new_if_not_exist=True)
with open(new_file.local_path, "w") as f:
    f.write("Hello World!\n")
new_file.upload()
new_file.delete(delete_remote=False)

existing_file = ds.get_file("test.txt") # downloads the file automatically
with open(existing_file.local_path, "r") as f:
    print(f.read())
# cleanup
existing_file.delete(delete_remote=True)
```

### Use an included Dataset

```Python
from anngel.datasets.openbiolink import OpenBioLinkDataset

# download the dataset to ./data
ds = OpenBioLinkDataset(data_path='./data')

# get nodes and edges as pandas.DataFrame
edges = ds.get_edges()
nodes = ds.get_nodes()

# get dataset as torch_geometric Data or HeteroData object
# note that ".x" might be empty, if the dataset does not provide
# node or edge features like it is the case for the OpenBioLink dataset
hetero_data = ds.to_pytorch_geometric_heterodata()
data = ds.to_pytorch_geometric_data()

# Access original node and edge data based on a pytorch geometric
# Data or HeteroData object.
# NOTE: .node_indices and .edge_indices is a custom property
nodes_df.iloc[hetero_data["DIS"].node_indices]
edges_df.iloc[hetero_data[("GO", "PART_OF", "GO")].edge_indices]

nodes_df.iloc[data.node_indices[:10]]
edges_df.iloc[data.edge_indices[:10]]
```
