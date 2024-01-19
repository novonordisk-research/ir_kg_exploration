import json
import pandas as pd


def ohe_labels(df: pd.DataFrame, labels_column: str = "labels") -> pd.DataFrame:
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(sparse_output=True)
    df[labels_column] = [json.loads(l.replace("'", '"')) for l in df[labels_column]]
    df = df.join(
        pd.DataFrame(
            mlb.fit_transform(df.pop(labels_column)).todense(),
            index=df.index,
            columns=mlb.classes_,
        )
    )

    return df
