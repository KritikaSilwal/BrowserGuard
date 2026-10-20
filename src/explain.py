"""Human-readable and SHAP-compatible explanations."""

from __future__ import annotations

import numpy as np


def _phishing_linear_terms(model, scaled_row: np.ndarray) -> np.ndarray:
    """Return per-feature terms in the model's phishing log-odds space."""
    # The trained positive class is label 1 (legitimate), so phishing is the
    # opposite logit and has the negated coefficient vector.
    return -np.asarray(model.coef_[0]) * np.asarray(scaled_row).reshape(-1)


def _linear_contributions(model, scaled_row: np.ndarray, feature_names: list[str], top_n: int):
    contributions = _phishing_linear_terms(model, scaled_row)
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {"feature": feature_names[index], "contribution": float(contributions[index])}
        for index in order
    ]


def explain_linear_prediction(model, scaled_row: np.ndarray, feature_names: list[str], top_n: int = 8):
    """Return coefficient contributions for phishing class (label 0)."""
    return _linear_contributions(model, scaled_row, feature_names, top_n)


def explain_with_shap_and_method(
    model, scaled_row: np.ndarray, feature_names: list[str], top_n: int = 8
):
    """Return local contributions and the method actually used."""
    try:
        import shap

        explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_names))))
        # SHAP's binary LogisticRegression values are for class 1, which is
        # legitimate in this project. Negate them into phishing log-odds.
        values = -np.asarray(explainer.shap_values(scaled_row))
        if values.ndim == 3:
            values = values[0, :, 0]
        elif values.ndim == 2:
            values = values[0]
        order = np.argsort(np.abs(values))[::-1][:top_n]
        return [
            {"feature": feature_names[index], "contribution": float(values[index])}
            for index in order
        ], "SHAP LinearExplainer (phishing log-odds)"
    except (ImportError, AttributeError, ValueError):
        return _linear_contributions(model, scaled_row, feature_names, top_n), "exact linear contribution fallback (phishing log-odds)"


def shap_waterfall_data(model, scaled_row: np.ndarray, feature_names: list[str], top_n: int = 12):
    """Return genuine SHAP values for a local waterfall when SHAP supports the model."""
    import shap

    explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_names))))
    values = -np.asarray(explainer.shap_values(scaled_row))
    if values.ndim == 3:
        values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    order = np.argsort(np.abs(values))[::-1][:top_n]
    return [
        {"feature": feature_names[index], "value": float(values[index])}
        for index in order
    ], "SHAP LinearExplainer (phishing log-odds)"


def explain_with_shap(model, scaled_row: np.ndarray, feature_names: list[str], top_n: int = 8):
    """Use SHAP when installed, with a clear fallback to linear contributions."""
    return explain_with_shap_and_method(model, scaled_row, feature_names, top_n)[0]


def global_shap_importance(model, scaled_features: np.ndarray, feature_names: list[str]):
    """Return mean absolute SHAP values, using the exact linear fallback if needed."""
    try:
        import shap

        explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_names))))
        values = -np.asarray(explainer.shap_values(scaled_features))
        if values.ndim == 3:
            values = values[:, :, 0]
        return np.abs(values).mean(axis=0), "SHAP LinearExplainer (phishing log-odds)"
    except (ImportError, AttributeError, ValueError):
        values = -np.asarray(scaled_features) * np.asarray(model.coef_[0])
        return np.abs(values).mean(axis=0), "Exact linear SHAP equivalent"
