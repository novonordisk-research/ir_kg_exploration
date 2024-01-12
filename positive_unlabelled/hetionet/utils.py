from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from anngel.metrics import calc_order, mr, mrr, amri
from anngel.metrics import amri_on_proba_df, mrr_on_proba_df, mr_on_proba_df
from sklearn.model_selection import GridSearchCV


def predict_proba(m, X):
    try:
        proba = m.predict_proba(X)
    except:
        try:
            proba = m.decision_function(X)
        except:
            proba = m.best_estimator_.decision_function(X)

    return proba


def evaluate(m, X, y, n=100, scale=True):
    y_pred = m.predict(X) > 0
    y_prob_pred = predict_proba(m, X)

    order = np.argsort(y_prob_pred)[::-1]
    if scale:
        return y[order][:n].sum() / y.sum()
    else:
        return y[order][:n].sum()


def train(input_dir, output_dir, n_jobs, cv, model_names, pul_cfg, rerun=False):
    ref_df = pd.read_csv(input_dir / "ref_df.csv", index_col=0)
    ref_df.head()
    for model_name in tqdm(model_names):
        out_dir = output_dir / model_name
        out_dir.mkdir(exist_ok=True, parents=True)
        for i in tqdm(range(10)):
            m_file = out_dir / f"{model_name}_{i}.pkl"
            if m_file.exists():
                try:
                    # try to load the file
                    with open(m_file, "rb") as f:
                        pickle.load(f)
                    if not rerun:
                        print(m_file, "exists. Skipping...")
                        continue
                except:
                    pass

            X_all = np.load(input_dir / f"{model_name}_X_{i}.npy")

            ds_names = ("train", "test", "val")
            Xs = {ds: X_all[ref_df.query(ds)["id"]] for ds in ds_names}
            ys = {ds: ref_df.query(ds)["y"].values for ds in ds_names}

            ds = "train"
            X = Xs[ds]
            y = ys[ds]

            pipe = pul_cfg.build_pipeline()
            param_grid = pul_cfg.build_param_grid()

            grid_search = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=cv,
                refit=True,
                n_jobs=n_jobs,
                scoring=evaluate,
            )
            grid_search.fit(X=X, y=y)

            with open(m_file, "wb") as f:
                pickle.dump(grid_search, f)


def confusion_matrix(m, X, y, n=None, scale=None):
    y_pred = m.predict(X) > 0
    cmat = sklearn_confusion_matrix(y_true=y, y_pred=y_pred)

    return cmat


def evaluate2(m, X, y, n=None, scale=None):
    y_prob_pred = predict_proba(m, X)

    order = calc_order(y_prob_pred)

    y = y[order]
    ranks = np.argwhere(y).ravel() + 1
    num_candidates = len(y)

    return dict(
        mr=mr(ranks=ranks, num_candidates=num_candidates),
        mrr=mrr(ranks=ranks, num_candidates=num_candidates),
        amri=amri(ranks=ranks, num_candidates=num_candidates),
    )


def load_grid_searches(output_dir, model_names):
    grid_searches = {}
    for model_name in model_names:
        grid_searches[model_name] = {}
        for j in range(10):
            m_name = f"{model_name}_{j}"
            m_file = output_dir / model_name / f"{m_name}.pkl"
            with open(m_file, "rb") as f:
                grid_search = pickle.load(f)
            grid_searches[model_name][m_name] = grid_search

    return grid_searches


def score_df_from_grid_searches(grid_searches, input_dir, ns=(10, 100)):
    ref_df = pd.read_csv(input_dir / "ref_df.csv", index_col=0)
    ref_df.head()

    score_dict = {
        "model_name": [],
        "m_name": [],
        "score10_val": [],
        "score100_val": [],
        "score10_test": [],
        "score100_test": [],
        "score10_all": [],
        "score100_all": [],
        "mr_all": [],
        "mrr_all": [],
        "amri_all": [],
        "mr_test": [],
        "mrr_test": [],
        "amri_test": [],
        "tp_test": [],
        "fp_test": [],
        "fn_test": [],
        "tn_test": [],
        "tp_all": [],
        "fp_all": [],
        "fn_all": [],
        "tn_all": [],
    }
    scale = False
    for model_name in grid_searches.keys():
        for i, m_name in enumerate(grid_searches[model_name].keys()):
            score_dict["model_name"].append(model_name)
            score_dict["m_name"].append(m_name)
            X_all = np.load(input_dir / f"{model_name}_X_{i}.npy")
            X = X_all[ref_df["id"]]
            y = ref_df["y"].values

            ds_names = ("train", "test", "val")
            Xs = {ds: X_all[ref_df.query(ds)["id"]] for ds in ds_names}
            ys = {ds: ref_df.query(ds)["y"].values for ds in ds_names}

            grid_search = grid_searches[model_name][m_name]
            for n in ns:
                val_score = evaluate(
                    m=grid_search, X=Xs["val"], y=ys["val"], n=n, scale=scale
                )
                score_dict[f"score{n}_val"].append(val_score)

                test_score = evaluate(
                    m=grid_search, X=Xs["test"], y=ys["test"], n=n, scale=scale
                )
                score_dict[f"score{n}_test"].append(test_score)

                all_score = evaluate(m=grid_search, X=X, y=y, n=n, scale=scale)
                score_dict[f"score{n}_all"].append(all_score)

            other_all = evaluate2(
                m=grid_search,
                X=X,
                y=y,
            )
            score_dict["mr_all"].append(other_all["mr"])
            score_dict["mrr_all"].append(other_all["mrr"])
            score_dict["amri_all"].append(other_all["amri"])

            other_test = evaluate2(
                m=grid_search,
                X=Xs["test"],
                y=ys["test"],
            )
            score_dict["mr_test"].append(other_test["mr"])
            score_dict["mrr_test"].append(other_test["mrr"])
            score_dict["amri_test"].append(other_test["amri"])

            cmat_test = confusion_matrix(m=grid_search, X=Xs["test"], y=ys["test"])
            score_dict[f"tp_test"].append(cmat_test[0, 0])
            score_dict[f"fp_test"].append(cmat_test[0, 1])
            score_dict[f"fn_test"].append(cmat_test[1, 0])
            score_dict[f"tn_test"].append(cmat_test[1, 1])

            cmat_all = confusion_matrix(m=grid_search, X=X, y=y)
            score_dict[f"tp_all"].append(cmat_all[0, 0])
            score_dict[f"fp_all"].append(cmat_all[0, 1])
            score_dict[f"fn_all"].append(cmat_all[1, 0])
            score_dict[f"tn_all"].append(cmat_all[1, 1])

    score_df = pd.DataFrame(score_dict)
    return score_df


def calc_predictions(grid_searches, input_dir):
    ref_df = pd.read_csv(input_dir / "ref_df.csv", index_col=0)
    ref_df.head()

    ids = ref_df["id"].values
    y = ref_df["y"].values
    train = ref_df["train"].values
    val = ref_df["val"].values
    test = ref_df["test"].values

    y_preds = {}
    for model_name in grid_searches.keys():
        y_preds[model_name] = {}
        for i, m_name in enumerate(grid_searches[model_name].keys()):
            X_all = np.load(input_dir / f"{model_name}_X_{i}.npy")
            X = X_all[ids]

            grid_search = grid_searches[model_name][m_name]
            y_preds[model_name][m_name] = predict_proba(grid_search, X)

    return y_preds


def build_proba_df(
    grid_searches,
    input_dir,
    gene_df_file="./data/gene_df.csv",
    mns=["RotatE", "TransE", "CompGCN"],
):
    ref_df = pd.read_csv(input_dir / "ref_df.csv", index_col=0)
    id_to_entity = {i: e for i, e in zip(ref_df["id"], ref_df["entity"])}

    y_preds = calc_predictions(grid_searches=grid_searches, input_dir=input_dir)

    ids = ref_df["id"].values
    y = ref_df["y"].values
    train = ref_df["train"].values
    val = ref_df["val"].values
    test = ref_df["test"].values

    gene_mapping_df = pd.read_csv(gene_df_file)
    gene_mapping_df["node_id"] = [f"Gene::{i}" for i in gene_mapping_df["gene_label"]]
    gene_mapping_df = gene_mapping_df[["node_id", "gene_name"]]
    gene_map = {
        node_id: gene_name for _, (node_id, gene_name) in gene_mapping_df.iterrows()
    }
    gene_names = [gene_map[node_id] for node_id in [id_to_entity[i] for i in ids]]

    d = dict(
        node_id=[id_to_entity[i] for i in ids],
        gene_name=gene_names,
        irr=y,
        train=train,
        val=val,
        test=test,
    )
    for mn in mns:
        d = dict(**d, **y_preds[mn])

    proba_df = pd.DataFrame(d, index=ids)

    return proba_df


def build_ranked_lists(proba_df, prefix):
    gene_lists = {}
    for i in range(10):
        col_nm = f"{prefix}_{i}"
        genes_100 = (
            proba_df[["gene_name", col_nm]]
            .sort_values(col_nm, ascending=False)
            .iloc[:100]["gene_name"]
            .values
        )
        gene_lists[col_nm] = genes_100

    return pd.DataFrame(gene_lists)


def calc_scores(proba_df, prefix="Topology", th=0.5):
    columns = {
        "model_name": [],
        "hits@10": [],
        "hits@100": [],
        "mr": [],
        "mrr": [],
        "amri": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": [],
    }
    for i in range(10):
        col_nm = f"{prefix}_{i}"
        columns["model_name"].append(col_nm)

        tmp_df = proba_df[["irr", col_nm]].sort_values(col_nm, ascending=False)
        columns["hits@10"].append(tmp_df.iloc[:10]["irr"].sum())
        columns["hits@100"].append(tmp_df.iloc[:100]["irr"].sum())
        columns["mr"].append(
            mr_on_proba_df(proba_df, colname_scores=col_nm, colname_y="irr")
        )
        columns["mrr"].append(
            mrr_on_proba_df(proba_df, colname_scores=col_nm, colname_y="irr")
        )
        columns["amri"].append(
            amri_on_proba_df(proba_df, colname_scores=col_nm, colname_y="irr")
        )

        cmat = sklearn_confusion_matrix(
            y_true=tmp_df["irr"], y_pred=tmp_df[col_nm] > th
        )
        columns["tp"].append(cmat[0, 0])
        columns["fp"].append(cmat[0, 1])
        columns["fn"].append(cmat[1, 0])
        columns["tn"].append(cmat[1, 1])

    score_df = pd.DataFrame(columns)
    score_df = score_df.set_index("model_name").T
    score_df["mean"] = score_df.mean(axis=1)

    return score_df
