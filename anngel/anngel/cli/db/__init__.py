from cloup import (
    help_option,
    group,
)

from .export_nodes import export_nodes
from .export_edges import export_edges
from .export_node_features import export_node_features
from .export_edge_features import export_edge_features


@group(
    name="db",
    help="Database subcommand. Mainly used to exporting data.",
    no_args_is_help=True,
)
@help_option("-h", "--help")
def db():
    pass


db.add_command(export_nodes)
db.add_command(export_edges)
db.add_command(export_node_features)
db.add_command(export_edge_features)
