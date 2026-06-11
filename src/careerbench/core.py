import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dataset(samples=3000, seed=7):
    x, y = make_multilabel_classification(
        n_samples=samples,
        n_features=24,
        n_classes=6,
        n_labels=2,
        allow_unlabeled=False,
        random_state=seed,
    )
    group = (x[:, 0] > np.median(x[:, 0])).astype(int)
    return pd.DataFrame(x), y, group


def evaluate(samples=3000):
    x, y, group = dataset(samples)
    split = int(samples * 0.7)
    models = {
        "logistic": MultiOutputClassifier(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        ),
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=150, random_state=7, n_jobs=-1)
        ),
    }
    records = []
    for name, model in models.items():
        model.fit(x.iloc[:split], y[:split])
        prediction = model.predict(x.iloc[split:])
        record = {
            "model": name,
            "micro_f1": f1_score(y[split:], prediction, average="micro"),
            "macro_f1": f1_score(y[split:], prediction, average="macro"),
            "hamming_loss": hamming_loss(y[split:], prediction),
        }
        for value in [0, 1]:
            mask = group[split:] == value
            record[f"group_{value}_micro_f1"] = f1_score(
                y[split:][mask], prediction[mask], average="micro"
            )
        record["subgroup_gap"] = abs(record["group_0_micro_f1"] - record["group_1_micro_f1"])
        records.append(record)
    return records
