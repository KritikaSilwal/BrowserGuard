"""Train and evaluate the URL-only AI BrowserGuard model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from urllib.parse import urlsplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, f1_score)

from src.data_loader import find_dataset, load_url_dataset
from src.preprocessing import prepare_splits


def train(dataset_path: str | Path | None = None, model_path: str | Path = "models/url_model.joblib"):
    path = Path(dataset_path) if dataset_path else find_dataset()
    frame, summary = load_url_dataset(path)
    (
        x_train, x_test, y_train, y_test, scaler, feature_names, train_urls, test_urls
    ) = prepare_splits(frame)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=None)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(0)]
    forest = RandomForestClassifier(
        n_estimators=120, class_weight="balanced", random_state=42, n_jobs=-1
    )
    forest.fit(x_train, y_train)
    forest_predictions = forest.predict(x_test)
    forest_probabilities = forest.predict_proba(x_test)[:, list(forest.classes_).index(0)]
    permutation = permutation_importance(
        model, x_test, y_test, n_repeats=5, random_state=42, scoring="f1", n_jobs=-1
    )
    calibrated = CalibratedClassifierCV(
        estimator=LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(x_train, y_train)
    calibrated_probabilities = calibrated.predict_proba(x_test)[:, list(calibrated.classes_).index(0)]
    calibrated_forest = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=120, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        method="sigmoid",
        cv=5,
    )
    calibrated_forest.fit(x_train, y_train)
    calibrated_forest_probabilities = calibrated_forest.predict_proba(x_test)[:, list(calibrated_forest.classes_).index(0)]
    calibration_target = (y_test == 0).astype(int)
    calibration_fraction, calibration_mean = calibration_curve(
        calibration_target, calibrated_probabilities, n_bins=10, strategy="quantile"
    )
    train_domains = {urlsplit(url if "://" in url else f"http://{url}").hostname for url in train_urls}
    test_domains = {urlsplit(url if "://" in url else f"http://{url}").hostname for url in test_urls}
    unseen_test_domains = test_domains.difference(train_domains)
    metrics = {
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_auc": float(roc_auc_score((y_test == 0).astype(int), probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=0)),
        "roc_curve": {
            "fpr": roc_curve((y_test == 0).astype(int), probabilities)[0].tolist(),
            "tpr": roc_curve((y_test == 0).astype(int), probabilities)[1].tolist(),
        },
        "precision_recall_curve": {
            "precision": precision_recall_curve((y_test == 0).astype(int), probabilities)[0].tolist(),
            "recall": precision_recall_curve((y_test == 0).astype(int), probabilities)[1].tolist(),
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, forest_predictions)),
            "precision": float(precision_score(y_test, forest_predictions, pos_label=0)),
            "recall": float(recall_score(y_test, forest_predictions, pos_label=0)),
            "f1": float(f1_score(y_test, forest_predictions, pos_label=0)),
            "roc_auc": float(roc_auc_score((y_test == 0).astype(int), forest_probabilities)),
            "confusion_matrix": confusion_matrix(y_test, forest_predictions).tolist(),
        },
        "coefficient_importance": (-model.coef_[0]).tolist(),
        "permutation_importance": permutation.importances_mean.tolist(),
        "calibration": {
            "method": "sigmoid calibration fitted with 5-fold CV on training partition only",
            "brier_score": float(brier_score_loss(calibration_target, calibrated_probabilities)),
            "mean_predicted_probability": calibration_mean.tolist(),
            "fraction_phishing": calibration_fraction.tolist(),
            "random_forest_brier_score": float(brier_score_loss(calibration_target, calibrated_forest_probabilities)),
        },
        "domain_split": {
            "method": "random stratified row split; domain overlap audited, not a domain-held-out test",
            "train_unique_domains": len(train_domains),
            "test_unique_domains": len(test_domains),
            "unseen_test_domains": len(unseen_test_domains),
            "unseen_domain_evaluation": "not performed because the primary split is row-stratified",
        },
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "summary": summary, "metrics": metrics, "x_test": x_test, "y_test": y_test.to_numpy(),
        "test_probabilities": probabilities, "random_forest": forest,
        "calibrated_model": calibrated, "calibrated_random_forest": calibrated_forest,
        "test_urls": test_urls, "train_urls": train_urls,
        "train_feature_reference": {
            "mean": np.asarray(x_train).mean(axis=0).tolist(),
            "std": np.asarray(x_train).std(axis=0).tolist(),
        },
    }, destination)
    destination.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": summary, "metrics": metrics, "model": str(destination)}, indent=2))
    return metrics


if __name__ == "__main__":
    train()
