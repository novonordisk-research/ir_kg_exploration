from cloup import (
    help_option,
    group,
)


@group(
    name="algorithms",
    aliases=["algo"],
    help="Algorithms subcommand for calculating node and graph features.",
    no_args_is_help=True,
)
@help_option("-h", "--help")
def algorithms():
    pass


from .node_features import calculate_node_feature

algorithms.add_command(calculate_node_feature)

from .pairwise_features import calculate_pairwise_feature

algorithms.add_command(calculate_pairwise_feature)
