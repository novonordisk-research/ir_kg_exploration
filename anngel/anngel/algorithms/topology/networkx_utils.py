import multiprocessing
import networkx
import numpy as np
from tqdm import tqdm
from typing import Union

_graph: Union[networkx.Graph, None] = None


def _apply_to_pairs(pairs, graph, fun, n_cores=1):
    def set_global_graph(graph):
        global _graph
        _graph = graph

    with multiprocessing.Pool(
        n_cores, initializer=set_global_graph, initargs=(graph,)
    ) as p:
        return list(tqdm(p.imap(fun, pairs), total=len(pairs)))


def n_common_neighbors(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_n_common_neighbors, n_cores=n_cores
    )


def _n_common_neighbors(pair):
    (g_idx, p_idx) = pair
    return len(
        list(
            networkx.common_neighbors(
                _graph,
                g_idx,
                p_idx,
            )
        )
    )


def shortest_path_length(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_shortest_path_length, n_cores=n_cores
    )


def _shortest_path_length(pair):
    (g_idx, p_idx) = pair
    try:
        return networkx.shortest_path_length(
            _graph,
            g_idx,
            p_idx,
        )
    except networkx.NetworkXNoPath:
        return -1


def adamic_adar_index(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_adamic_adar_index, n_cores=n_cores
    )


def _adamic_adar_index(pair):
    (g_idx, p_idx) = pair
    return np.sum(
        1 / np.log(_graph.degree(w))
        for w in networkx.common_neighbors(
            _graph,
            g_idx,
            p_idx,
        )
    )


def preferential_attachment(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_preferential_attachment, n_cores=n_cores
    )


def _preferential_attachment(pair):
    (g_idx, p_idx) = pair
    return _graph.degree(g_idx) * _graph.degree(p_idx)


def jaccard_coefficient(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_jaccard_coefficient, n_cores=n_cores
    )


def _jaccard_coefficient(pair):
    (g_idx, p_idx) = pair
    union_size = len(set(_graph[g_idx]) | set(_graph[p_idx]))
    if union_size == 0:
        return 0
    return len(list(networkx.common_neighbors(_graph, g_idx, p_idx))) / union_size


def resource_allocation_index(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_resource_allocation_index, n_cores=n_cores
    )


def _resource_allocation_index(pair):
    (g_idx, p_idx) = pair
    return sum(
        1 / _graph.degree(w) for w in networkx.common_neighbors(_graph, g_idx, p_idx)
    )


def leicht_holme_newman_index(graph, pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=pairs, graph=graph, fun=_leicht_holme_newman_index, n_cores=n_cores
    )


def _leicht_holme_newman_index(pair):
    ncn = _n_common_neighbors(pair)
    pa = _preferential_attachment(pair)

    if ncn == 0 and pa == 0:
        return 0

    return float(ncn) / pa


def personalized_pagerank(graph, input_pairs, n_cores=1):
    return _apply_to_pairs(
        pairs=input_pairs, graph=graph, fun=_personalized_pagerank, n_cores=n_cores
    )


def _personalized_pagerank(input_pair):
    global _graph
    (g_idx, p_idcs) = input_pair
    personalization = {k: 0 for k in range(len(_graph.nodes))}
    personalization[g_idx] = 1
    r = networkx.pagerank(_graph, personalization=personalization)
    return [r[p_idx] for p_idx in p_idcs]
