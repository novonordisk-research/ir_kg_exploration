import pytest
import networkx

from anngel.algorithms.topology import (
    # pairwise
    n_common_neighbors,
    shortest_path_length,
    adamic_adar_index,
    preferential_attachment,
    jaccard_coefficient,
    resource_allocation_index,
    leicht_holme_newman_index,
    personalized_pagerank,
    # node-wise
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


@pytest.fixture(scope="module")
def graph():
    g = networkx.Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 5)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)

    return g


@pytest.fixture(scope="module")
def pairs(graph):
    import itertools

    return list(itertools.combinations(list(graph.nodes()), 2))


@pytest.fixture(scope="module")
def input_pairs(graph):
    import itertools

    return [
        (idcs[0], idcs[1:])
        for idcs in itertools.combinations(list(graph.nodes())[:10], 3)
    ]


def test_degree_centrality(graph):
    degree_centrality(graph=graph)


def test_in_degree_centrality(graph):
    in_degree_centrality(graph=graph.to_directed())


def test_out_degree_centrality(graph):
    out_degree_centrality(graph=graph.to_directed())


def test_eigenvector_centrality(graph):
    eigenvector_centrality(graph=graph)


def test_pagerank(graph):
    pagerank(graph=graph)


def test_closeness_centrality(graph):
    closeness_centrality(graph=graph)


def test_betweenness_centrality(graph):
    betweenness_centrality(graph=graph)


def test_katz_centrality(graph):
    katz_centrality(graph=graph)


def test_load_centrality(graph):
    load_centrality(graph=graph)


def test_triangles(graph):
    triangles(graph=graph)


def test_average_neighbor_degree(graph):
    average_neighbor_degree(graph=graph)


def test_clustering(graph):
    clustering(graph=graph)


def test_n_common_neighbors(graph, pairs):
    n_common_neighbors(graph=graph, pairs=pairs, n_cores=1)
    n_common_neighbors(graph=graph, pairs=pairs, n_cores=1)


def test_shortest_path_length(graph, pairs):
    shortest_path_length(graph=graph, pairs=pairs, n_cores=1)
    shortest_path_length(graph=graph, pairs=pairs, n_cores=2)


def test_adamic_adar_index(graph, pairs):
    adamic_adar_index(graph=graph, pairs=pairs, n_cores=1)
    adamic_adar_index(graph=graph, pairs=pairs, n_cores=2)


def test_preferential_attachment(graph, pairs):
    preferential_attachment(graph=graph, pairs=pairs, n_cores=1)
    preferential_attachment(graph=graph, pairs=pairs, n_cores=2)


def test_jaccard_coefficient(graph, pairs):
    jaccard_coefficient(graph=graph, pairs=pairs, n_cores=1)
    jaccard_coefficient(graph=graph, pairs=pairs, n_cores=2)


def test_resource_allocation_index(graph, pairs):
    resource_allocation_index(graph=graph, pairs=pairs, n_cores=1)
    resource_allocation_index(graph=graph, pairs=pairs, n_cores=2)


def test_leicht_holme_newman_index(graph, pairs):
    leicht_holme_newman_index(graph=graph, pairs=pairs, n_cores=1)
    leicht_holme_newman_index(graph=graph, pairs=pairs, n_cores=2)


def test_personalized_pagerank(graph, input_pairs):
    personalized_pagerank(graph=graph, input_pairs=input_pairs, n_cores=1)
    personalized_pagerank(graph=graph, input_pairs=input_pairs, n_cores=2)
