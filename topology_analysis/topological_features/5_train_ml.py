from click import command, option, Path as PathType, help_option

from pathlib import Path

def calc_metrics(y_pred, y_true):
    from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
    return {
        "f1": f1_score(y_pred=y_pred, y_true=y_true),
        "recall": recall_score(y_pred=y_pred, y_true=y_true),
        "precision": precision_score(y_pred=y_pred, y_true=y_true, zero_division=0),
        "confusion_matrix": confusion_matrix(y_pred=y_pred, y_true=y_true),
    }

def custom_scorer(clf, X, y):
    y_pred = clf.predict(X)
    metrics = calc_metrics(y_pred=y_pred, y_true=y)
    cm = metrics["confusion_matrix"]
    metrics["TP"] = cm[0, 0]
    metrics["FP"] = cm[0, 1]
    metrics["FN"] = cm[1, 0]
    metrics["TN"] = cm[1, 1]
    del metrics["confusion_matrix"]
    return metrics


def evaluate(m, X, y, n=100, scale=True):
    import numpy as np
    y_pred = m.predict(X) > 0
    y_prob_pred = m.predict_proba(X)

    order = np.argsort(y_prob_pred)[::-1]
    # print(y_pred[order][:n])
    if scale:
        return y[order][:n].sum() / y.sum()
    else:
        return y[order][:n].sum()

@command('train_ml')
@option('-i', '--input_dir', type=PathType(dir_okay=True, file_okay=False, writable=True, path_type=Path), required=True)
@option('-o', '--output_dir', type=PathType(dir_okay=True, file_okay=False, writable=True, path_type=Path), required=True)
@option('-n', '--n_jobs', type=int, required=True)
@option('-r', '--rerun', is_flag=True)
@help_option('-h', '--help')
def main(input_dir, output_dir, n_jobs, rerun):
    import pickle
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from tqdm import tqdm

    
    output_dir.mkdir(exist_ok=True, parents=True)

    xy_dir = input_dir / 'XY'
    train_dir = xy_dir / 'train'
    val_dir = xy_dir / 'val'
    test_dir = xy_dir / 'test'

    y_train_file = train_dir / 'y.csv'
    y_val_file = val_dir / 'y.csv'
    y_test_file = test_dir / 'y.csv'

    X_train_files = [f for f in train_dir.glob('*.csv') if not f.name == 'y.csv']
    X_val_files = [val_dir / f.name for f in X_train_files]
    X_test_files = [test_dir / f.name for f in X_train_files]

    def fit_model(
        X_train,
        y_train,
        model,
        parameters,
        scoring=evaluate,
        cv=5,
        **kwargs
    ):    
        clf = GridSearchCV(model, parameters, scoring=scoring, cv=cv, **kwargs)
        clf.fit(X=X_train, y=y_train)
        return clf

    CV = 5
    def train_predictor(
        X_train_file, y_train_file,
        out_file,
        cv=CV,
        n_jobs=n_jobs,
        scoring=evaluate,
    ):
        m = RandomForestClassifier()
        parameters = dict(
            n_estimators=[100, 250, 500],
            max_depth=[None, 2, 5, 10],
            min_samples_leaf=[1, 32, 128, 256],
            max_features=['sqrt', 1, 2, 4, 8,]
        )

        X_train = pd.read_csv(X_train_file, index_col=0).values
        y_train = pd.read_csv(y_train_file, index_col=0).iloc[:,0].values

        clf = fit_model(
            X_train=X_train,
            y_train=y_train,
            model=m,
            parameters=parameters,
            cv=cv,
            n_jobs=n_jobs,
        )

        with open(out_file, 'wb') as f:
            pickle.dump(clf, f)
        return clf


    classifiers_dir = output_dir / 'classifiers_ml'
    classifiers_dir.mkdir(exist_ok=True, parents=True)

    rerun=False
    for X_train_file in tqdm(X_train_files):
        nm = X_train_file.name.split('.')[0]
        out_file = classifiers_dir / f'{nm}.pkl'
        if out_file.exists():
            try:
                # try to load the file
                with open(out_file, 'rb') as f:
                    pickle.load(f)
                if not rerun:
                    print(out_file, 'exists. Skipping...')
                    continue
            except:
                pass

        train_predictor(
            X_train_file=X_train_file,
            y_train_file=y_train_file,
            out_file=out_file,
            cv=CV,
            n_jobs=n_jobs,
        )
        
    classifier_paths = list(classifiers_dir.glob('*.pkl'))

    probas = {}
    val_results = {}
    indices = {}
    for clf_file in tqdm(classifier_paths):
        with open(clf_file, 'rb') as f:
            clf = pickle.load(f)

        nm = clf_file.name.split('.')[0]
        for ds in ('train', 'val', 'test'):
            if probas.get(ds) is None:
                probas[ds] = {}
            X_file = xy_dir / ds / f'{nm}.csv'
            y_file = xy_dir / ds / 'y.csv'

            X_df = pd.read_csv(X_file, index_col=0)
            indices[ds] = X_df.index
            X = X_df.values
            y = pd.read_csv(y_file, index_col=0).iloc[:,0].values

            probas[ds][nm] = clf.predict_proba(X)
            if ds == 'val':
                y_pred = clf.predict(X)
                val_results[nm] = calc_metrics(y_true=y, y_pred=y_pred)


    # In[14]:


    train_probas_df = pd.DataFrame({nm: v[:,1] for nm, v in probas['train'].items()}, index=indices['train'])
    val_probas_df = pd.DataFrame({nm: v[:,1] for nm, v in probas['val'].items()}, index=indices['val'])
    test_probas_df = pd.DataFrame({nm: v[:,1] for nm, v in probas['test'].items()}, index=indices['test'])

    train_probas_df.to_csv(output_dir / 'train_probabilities.csv')
    val_probas_df.to_csv(output_dir / 'val_probabilities.csv')
    test_probas_df.to_csv(output_dir / 'test_probabilities.csv')


    # In[15]:


    proba_dfs = {}
    Xs = {}
    y_dfs = {}
    ys = {}
    indices={}
    for ds in ('train', 'val', 'test'):
        proba_df = pd.read_csv(output_dir / f'{ds}_probabilities.csv', index_col=0)
        proba_dfs[ds] = proba_df
        y_df = pd.read_csv(xy_dir / f'{ds}/y.csv', index_col=0)
        y_dfs[ds] = y_df

        X = proba_df.values
        Xs[ds] = X
        y = y_df.iloc[:, 0].values
        ys[ds] = y

        indices[ds] = proba_df.index

    # lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=(1), max_iter=10000, C=1.0)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xs['train'], ys['train'])
    lr.coef_
    
    lr_file = output_dir / 'lr.pkl'
    with open(lr_file, 'wb') as f:
        pickle.dump(lr, f)
        
if __name__ == '__main__':
    main()