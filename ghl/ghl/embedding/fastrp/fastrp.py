from typing import List, Union

import pandas as pd
import numpy as np

from sklearn import random_projection
from sklearn.preprocessing import normalize, scale
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags


# projection method: choose from Gaussian and Sparse
# input matrix: choose from adjacency and transition matrix
# alpha adjusts the weighting of nodes according to their degree
def fastrp_projection(
    A, q=3, dim=128, projection_method="gaussian", input_matrix="adj", alpha=None
):
    assert input_matrix == "adj" or input_matrix == "trans"
    assert projection_method == "gaussian" or projection_method == "sparse"

    if input_matrix == "adj":
        M = A
    else:
        N = A.shape[0]
        normalizer = spdiags(np.squeeze(1.0 / csc_matrix.sum(A, axis=1)), 0, N, N)
        M = normalizer @ A
    # Gaussian projection matrix
    if projection_method == "gaussian":
        transformer = random_projection.GaussianRandomProjection(
            n_components=dim, random_state=42
        )
    # Sparse projection matrix
    else:
        transformer = random_projection.SparseRandomProjection(
            n_components=dim, random_state=42
        )
    Y = transformer.fit(M)
    # Random projection for A
    if alpha is not None:
        Y.components_ = Y.components_ @ spdiags(
            np.squeeze(np.power(csc_matrix.sum(A, axis=1), alpha)), 0, N, N
        )
    cur_U = transformer.transform(M)
    U_list = [cur_U]

    for i in range(2, q + 1):
        cur_U = M @ cur_U
        U_list.append(cur_U)
    return U_list


# When weights is None, concatenate instead of linearly combines the embeddings from different powers of A
def fastrp_merge(U_list, weights, normalization=False):
    dense_U_list = (
        [_U.todense() for _U in U_list] if type(U_list[0]) == csc_matrix else U_list
    )
    _U_list = (
        [normalize(_U, norm="l2", axis=1) for _U in dense_U_list]
        if normalization
        else dense_U_list
    )

    if weights is None:
        return np.concatenate(_U_list, axis=1)
    U = np.zeros_like(_U_list[0])
    for cur_U, weight in zip(_U_list, weights):
        U += cur_U * weight
    # U = scale(U.todense())
    # U = normalize(U.todense(), norm='l2', axis=1)
    return scale(U.todense()) if type(U) == csr_matrix else scale(U)


def fastrp_wrapper(
    a: coo_matrix,
    weights: List[float],
    embedding_dimension: int = 3,
    alpha: Union[None, float] = None,
    normalization: bool = True,
    matrix_type: str = "adj",
    projection_method: str = "gaussian",
):
    u_list = fastrp_projection(
        A=a,
        q=len(weights),
        dim=embedding_dimension,
        projection_method=projection_method,
        input_matrix=matrix_type,
        alpha=alpha,
    )
    u = fastrp_merge(U_list=u_list, weights=weights, normalization=normalization)
    return u
