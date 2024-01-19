from typing import Union

from pathlib import Path

from cloup import command, option, option_group, help_option, Path as PathType, Choice

from ghl.algorithms.topology import (
    n_common_neighbors,
    shortest_path_length,
    adamic_adar_index,
    preferential_attachment,
    jaccard_coefficient,
    resource_allocation_index,
    leicht_holme_newman_index,
    personalized_pagerank,
)

ALGORITHMS_MAP = {
    "n_common_neighbors": n_common_neighbors,
    "shortest_path_length": shortest_path_length,
    "adamic_adar_index": adamic_adar_index,
    "preferential_attachment": preferential_attachment,
    "jaccard_coefficient": jaccard_coefficient,
    "resource_allocation_index": resource_allocation_index,
    "leicht_holme_newman_index": leicht_holme_newman_index,
    "personalized_pagerank": personalized_pagerank,
}


@command(
    name="calculate_pairwise_feature",
    help="""
        Calculate pairwise features for set of nodes pairs
        of an underlying graph.
    """,
    no_args_is_help=True,
)
@option_group(
    "Input options",
    option(
        "-g",
        "--graph_file",
        type=PathType(file_okay=True, dir_okay=False, writable=False, path_type=Path),
        required=True,
        help="Pickle file of a networkx.Graph.",
    ),
    option(
        "-p",
        "--pairs_file",
        type=PathType(file_okay=True, dir_okay=False, writable=False, path_type=Path),
        required=True,
        help="Pickle file of pairs.",
    ),
)
@option_group(
    "Output options",
    option(
        "-o",
        "--output_file",
        type=PathType(file_okay=True, dir_okay=False, writable=True, path_type=Path),
        required=True,
        help="CSV file to write the outputs to.",
    ),
)
@option_group(
    "Algorithm options",
    option(
        "-a",
        "--algorithm",
        type=Choice(list(ALGORITHMS_MAP.keys())),
        required=True,
        help="Algorithm to use for feature calculation.",
    ),
    option(
        "--algorithm_kwargs",
        type=str,
        default=None,
        required=False,
        help="""
                Dictionary describing additional keyword arguments for the algorithm.
                
                Example: '{"alpha": 0.85, "max_iter": 100}'

                Note: strings must be escaped on Windows, i.e., '{\\"alpha\\": 0.85}'
            """,
    ),
    option(
        "-n",
        "--n_processes",
        type=int,
        default=1,
        required=True,
        help="Number of parallel processes to run.",
    ),
)
@help_option("-h", "--help")
def calculate_pairwise_feature(
    graph_file: Path,
    pairs_file: Path,
    output_file: Path,
    algorithm: str,
    algorithm_kwargs: str,
    n_processes: int,
):
    import time
    import json
    import pickle
    import pandas as pd

    if algorithm_kwargs is not None:
        import json

        kwargs = json.loads(algorithm_kwargs)
    else:
        kwargs = {}

    fun = ALGORITHMS_MAP[algorithm]

    with open(graph_file, "rb") as f:
        graph = pickle.load(f)

    with open(pairs_file, "rb") as f:
        pairs = pickle.load(f)

    if algorithm == "personalized_pagerank":
        r = fun(graph=graph, input_pairs=pairs, n_cores=n_processes, **kwargs)

        import csv

        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(r)

    else:
        df = pd.DataFrame(
            {algorithm: fun(graph=graph, pairs=pairs, n_cores=n_processes, **kwargs)}
        )
        df.to_csv(output_file, index=False)
