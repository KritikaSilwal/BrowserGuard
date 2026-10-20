"""Prediction reliability and training-distribution unfamiliarity helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def threshold_metrics(y_true, phishing_probabilities, threshold: float) -> dict[str, float]:
    actual = (np.asarray(y_true) == 0).astype(int)
    predicted = (np.asarray(phishing_probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[1, 0])
    tp, fn, fp, tn = matrix.ravel()
    return {
        "threshold": float(threshold),
        "phishing_predictions": int(predicted.sum()),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }


def unfamiliarity(scaled_features, reference_mean, reference_std, unusual_z: float = 3.0) -> dict:
    values = np.asarray(scaled_features).reshape(-1)
    mean = np.asarray(reference_mean)
    std = np.maximum(np.asarray(reference_std), 1e-9)
    z = np.abs((values - mean) / std)
    unusual = np.flatnonzero(z >= unusual_z)
    level = "familiar to training distribution"
    if len(unusual) >= 3:
        level = "substantially different from training examples"
    elif len(unusual):
        level = "some unusual feature values"
    return {
        "level": level,
        "unusual_feature_count": int(len(unusual)),
        "max_absolute_z": float(z.max(initial=0)),
        "threshold": unusual_z,
        "unusual_indices": unusual.tolist(),
    }
