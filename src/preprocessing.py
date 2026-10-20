"""Training preprocessing for the URL-only model."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_extraction import features_to_frame


def prepare_splits(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    features = features_to_frame(frame["URL"].tolist())
    labels = frame["label"].astype("int8")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        scaler,
        list(features.columns),
        frame.loc[x_train.index, "URL"].tolist(),
        frame.loc[x_test.index, "URL"].tolist(),
    )
