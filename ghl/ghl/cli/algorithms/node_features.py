from typing import Union

from pathlib import Path

from cloup import command, option, option_group, help_option, Path as PathType, Choice

from ghl.algorithms.topology import (
    degree_centrality,
    in_degree_centrality,
    out_degree_centrality,
    eigenvector_centrality,
    pagerank,
    closeness_centrality,
    betweenness_centrality,
    katz_centrality,
    load_centrality,
    triangles,
    average_neighbor_degree,
    clustering,
)

ALGORITHMS_MAP = {
    "degree_centrality": degree_centrality,
    "in_degree_centrality": in_degree_centrality,
    "out_degree_centrality": out_degree_centrality,
    "eigenvector_centrality": eigenvector_centrality,
    "pagerank": pagerank,
    "closeness_centrality": closeness_centrality,
    "betweenness_centrality": betweenness_centrality,
    "katz_centrality": katz_centrality,
    "load_centrality": load_centrality,
    "triangles": triangles,
    "average_neighbor_degree": average_neighbor_degree,
    "clustering": clustering,
}


@command(
    name="calculate_node_feature",
    help="""
        Calculate node features for all nodes
        of a graph.
    """,
    no_args_is_help=True,
)
@option_group(
    "Input options",
    option(
        "-i",
        "--input_file",
        type=PathType(file_okay=True, dir_okay=False, writable=False, path_type=Path),
        required=True,
        help="Pickle file of a networkx.Graph.",
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
)
@help_option("-h", "--help")
def calculate_node_feature(
    input_file: Path,
    output_file: Path,
    algorithm: str,
    algorithm_kwargs: str,
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

    with open(input_file, "rb") as f:
        graph = pickle.load(f)

    t0 = time.time()
    df = pd.DataFrame({algorithm: fun(graph=graph, **kwargs).values()})
    df.to_csv(output_file, index=False)
    t1 = time.time()

    print(f'Finished running "{algorithm}" after {t1 - t0:1f} seconds.')
