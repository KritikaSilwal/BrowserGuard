"""Domain-held-out evaluation using only raw URL and label columns."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data_loader import find_dataset, load_url_dataset
from .feature_extraction import features_to_frame


def registrable_domain(raw_url: str) -> str | None:
    """Extract a registrable domain using tldextract when available.

    IP hosts are returned unchanged. Without tldextract, a conservative
    hostname fallback is used and marked in the experiment metadata.
    """
    value = str(raw_url).strip()
    candidate = value if "://" in value else f"http://{value}"
    hostname = urlsplit(candidate).hostname
    if not hostname:
        return None
    try:
        import ipaddress

        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    try:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=())
        extracted = extractor(hostname)
        if not extracted.domain or not extracted.suffix:
            return None
        return f"{extracted.domain}.{extracted.suffix}".lower()
    except ImportError:
        labels = hostname.lower().rstrip(".").split(".")
        return ".".join(labels[-2:]) if len(labels) >= 2 else hostname.lower()


def _metrics(y_true, predictions, probabilities) -> dict:
    phishing = (np.asarray(y_true) == 0).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(phishing, probabilities)) if len(np.unique(phishing)) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "false_positive_rate": float(matrix[1, 0] / max(matrix[1].sum(), 1)),
        "false_negative_rate": float(matrix[0, 1] / max(matrix[0].sum(), 1)),
    }


def run_domain_held_out(
    dataset_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    path = Path(dataset_path) if dataset_path else find_dataset()
    frame, summary = load_url_dataset(path)
    frame = frame.copy()
    frame["domain"] = frame["URL"].map(registrable_domain)
    frame = frame.dropna(subset=["domain"]).reset_index(drop=True)
    groups = frame["domain"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, frame["label"], groups))
    train = frame.iloc[train_index]
    test = frame.iloc[test_index]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features_to_frame(train["URL"].tolist()))
    x_test = scaler.transform(features_to_frame(test["URL"].tolist()))
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(0)]
        results[name] = _metrics(y_test, model.predict(x_test), probabilities)
    train_domains, test_domains = set(train["domain"]), set(test["domain"])
    return {
        "dataset": summary,
        "method": "GroupShuffleSplit by registrable domain",
        "random_state": random_state,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_domains": len(train_domains),
        "test_domains": len(test_domains),
        "overlap_domains": len(train_domains & test_domains),
        "test_domains_absent_from_training_pct": 100.0 * len(test_domains - train_domains) / max(len(test_domains), 1),
        "train_class_counts": train["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test["label"].value_counts().sort_index().to_dict(),
        "models": results,
        "registrable_domain_method": "tldextract Public Suffix List when installed; conservative fallback otherwise",
    }
