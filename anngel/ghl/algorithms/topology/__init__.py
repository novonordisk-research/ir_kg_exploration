import networkx
import networkx.algorithms

from .networkx_utils import (
    n_common_neighbors,
    shortest_path_length,
    adamic_adar_index,
    preferential_attachment,
    jaccard_coefficient,
    resource_allocation_index,
    leicht_holme_newman_index,
    personalized_pagerank,
)


def degree_centrality(graph: networkx.Graph) -> float:
    return networkx.algorithms.degree_centrality(graph)


def in_degree_centrality(graph: networkx.Graph) -> float:
    return networkx.algorithms.in_degree_centrality(graph.to_directed())


def out_degree_centrality(graph: networkx.Graph) -> float:
    return networkx.algorithms.out_degree_centrality(graph.to_directed())


def eigenvector_centrality(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.eigenvector_centrality(graph, **kwargs)


def pagerank(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.pagerank(graph, **kwargs)


def closeness_centrality(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.closeness_centrality(graph, **kwargs)


def betweenness_centrality(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.betweenness_centrality(graph, **kwargs)


def katz_centrality(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.katz_centrality(graph, **kwargs)


def load_centrality(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.load_centrality(graph, **kwargs)


def triangles(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.triangles(graph, **kwargs)


def average_neighbor_degree(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.average_neighbor_degree(graph, **kwargs)


def clustering(graph: networkx.Graph, **kwargs) -> float:
    return networkx.algorithms.clustering(graph, **kwargs)
