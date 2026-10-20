"""Shared model loading and prediction helpers for the Streamlit pages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from .feature_extraction import extract_url_features, features_to_frame, parse_url

HF_REPOSITORY = "kritika2005/browserguard-model"
HF_MODEL_FILENAME = "url_model.joblib"


def _configured_hf_token() -> str | None:
    """Read the private Hub token without exposing it in errors or logs."""
    try:
        token = st.secrets["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        token = os.environ.get("HF_TOKEN")
    if token is None:
        return None
    token = str(token).strip()
    return token or None


def _download_model_from_huggingface() -> Path:
    token = _configured_hf_token()
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPOSITORY,
            filename=HF_MODEL_FILENAME,
            token=token,
        )
    except HfHubHTTPError as exc:
        if token is None:
            raise RuntimeError(
                "The local model is unavailable and the Hugging Face repository is private. "
                "Configure st.secrets['HF_TOKEN'] in Streamlit Cloud or set the HF_TOKEN "
                "environment variable for local use."
            ) from exc
        raise RuntimeError(
            "The BrowserGuard model could not be downloaded from Hugging Face. "
            "Check that HF_TOKEN has read access to the model repository."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "The BrowserGuard model could not be downloaded from Hugging Face. "
            "Check network access and the HF_TOKEN configuration."
        ) from exc
    return Path(downloaded_path)


def load_bundle(path: str | Path = "models/url_model.joblib") -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        model_path = _download_model_from_huggingface()
    return joblib.load(model_path)


def predict_url(raw_url: str, bundle: dict[str, Any]) -> dict[str, Any]:
    raw_features = extract_url_features(raw_url)
    parsed, explicit_scheme = parse_url(raw_url)
    row = features_to_frame([raw_url])
    scaled = bundle["scaler"].transform(row)
    model = bundle["model"]
    classes = list(model.classes_)
    phishing_probability = float(model.predict_proba(scaled)[0, classes.index(0)])
    prediction = int(model.predict(scaled)[0])
    return {
        "url": raw_url.strip(),
        "explicit_scheme": explicit_scheme,
        "parsed_scheme": parsed.scheme,
        "raw_features": raw_features,
        "scaled_features": scaled,
        "prediction": prediction,
        "phishing_probability": phishing_probability,
        "classification": "Potentially phishing" if prediction == 0 else "Likely legitimate",
        "risk": "Higher risk" if phishing_probability >= 0.5 else "Lower risk",
    }


def predict_batch(urls: list[str], bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row_number, value in enumerate(urls, start=1):
        try:
            result = predict_url(str(value), bundle)
            results.append({
                "URL": result["url"],
                "prediction": result["classification"],
                "phishing_probability": result["phishing_probability"],
                "risk": result["risk"],
            })
        except (ValueError, TypeError) as exc:
            errors.append({"row": str(row_number), "URL": str(value), "error": str(exc)})
    return results, errors


def predict_with_model(raw_url: str, model: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    """Predict with a saved comparison model using the shared URL pipeline."""
    row = features_to_frame([raw_url])
    scaled = bundle["scaler"].transform(row)
    classes = list(model.classes_)
    probability = float(model.predict_proba(scaled)[0, classes.index(0)])
    prediction = int(model.predict(scaled)[0])
    return {
        "prediction": prediction,
        "phishing_probability": probability,
        "classification": "Potentially phishing" if prediction == 0 else "Likely legitimate",
    }
