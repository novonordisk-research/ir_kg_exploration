from functools import partial

import numpy as np
import pandas as pd
from pykeen.metrics.ranking import (
    InverseHarmonicMeanRank,
    AdjustedArithmeticMeanRankIndex,
    ArithmeticMeanRank,
)


# NOTE: this function does bootstrap by default but
#       can also be used to do random subsampling
def bootstrap(df, fun, n_rep=10, replace=True, size=None, **kwargs):
    n_rows = df.shape[0]
    idcs = np.arange(n_rows - 1)

    if size is None:
        size = n_rows

    r = []
    for _ in range(n_rep):
        idcs_samp = np.random.choice(idcs, size=size, replace=True)
        df_samp = df.iloc[idcs_samp,]
        r.append(fun(df=df_samp, **kwargs))

    return np.array(r)


def calc_order(x, ascending=False):
    order = np.argsort(x)
    if not ascending:
        order = order[::-1]

    return order


def rank_metric_on_proba_df(
    fun, df, colname_scores, colname_y, ascending=False, hit_value=True, **kwargs
):
    scores = df[colname_scores].values
    y = df[colname_y].values

    order = calc_order(scores, ascending=ascending)
    y = y[order]

    ranks = np.argwhere(y == hit_value).ravel() + 1
    num_candidates = len(y)

    return fun(ranks=ranks, num_candidates=num_candidates, **kwargs)


amri = AdjustedArithmeticMeanRankIndex()
mrr = InverseHarmonicMeanRank()
mr = ArithmeticMeanRank()

amri_on_proba_df = partial(rank_metric_on_proba_df, amri)
mrr_on_proba_df = partial(rank_metric_on_proba_df, mrr)
mr_on_proba_df = partial(rank_metric_on_proba_df, mr)
