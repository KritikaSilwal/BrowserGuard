"""Shared model loading and prediction helpers for the Streamlit pages."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from streamlit.errors import StreamlitSecretNotFoundError

from .feature_extraction import extract_url_features, features_to_frame, parse_url

HF_REPOSITORY = "kritika2005/browserguard-model"
HF_MODEL_FILENAME = "url_model.joblib"
LOGGER = logging.getLogger(__name__)


def _configured_hf_token() -> str | None:
    """Read the private Hub token without exposing it in errors or logs."""
    secret_key_exists = False
    try:
        token = st.secrets["HF_TOKEN"]
        secret_key_exists = True
    except (KeyError, FileNotFoundError, StreamlitSecretNotFoundError):
        token = os.environ.get("HF_TOKEN")
    if token is None:
        LOGGER.info(
            "Hugging Face token configuration: secrets_key_exists=%s "
            "non_empty_token=%s",
            secret_key_exists,
            False,
        )
        return None
    token = str(token).strip()
    resolved_token = token or None
    LOGGER.info(
        "Hugging Face token configuration: secrets_key_exists=%s "
        "non_empty_token=%s",
        secret_key_exists,
        resolved_token is not None,
    )
    return resolved_token


def _http_status_code(exception: BaseException) -> int | None:
    response = getattr(exception, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _download_model_from_huggingface() -> Path:
    token = _configured_hf_token()
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPOSITORY,
            filename=HF_MODEL_FILENAME,
            token=token,
        )
    except HfHubHTTPError as exc:
        status_code = _http_status_code(exc)
        LOGGER.warning(
            "Hugging Face model download failed: exception_class=%s "
            "http_status=%s non_empty_token=%s",
            type(exc).__name__,
            status_code if status_code is not None else "unavailable",
            token is not None,
        )
        if status_code == 401:
            raise RuntimeError(
                "Hugging Face returned HTTP 401 for the model download. "
                "If the repository is intended to be public, change its visibility in "
                "Hugging Face settings; otherwise configure a read-enabled HF_TOKEN."
            ) from exc
        if status_code == 403:
            raise RuntimeError(
                "Hugging Face denied access to the model (HTTP 403). "
                "Grant the token read access to the private repository."
            ) from exc
        if status_code == 404:
            raise RuntimeError(
                "The Hugging Face model repository or file was not found (HTTP 404). "
                f"Verify {HF_REPOSITORY}/{HF_MODEL_FILENAME}."
            ) from exc
        raise RuntimeError(
            "The BrowserGuard model could not be downloaded from Hugging Face "
            f"(HTTP {status_code if status_code is not None else 'unknown'}). "
            "Check repository visibility, network access, and optional HF_TOKEN permissions."
        ) from exc
    except OSError as exc:
        LOGGER.warning(
            "Hugging Face model download failed: exception_class=%s "
            "http_status=%s non_empty_token=%s",
            type(exc).__name__,
            "unavailable",
            token is not None,
        )
        raise RuntimeError(
            "The BrowserGuard model could not be downloaded from Hugging Face. "
            "Check network access and the optional HF_TOKEN configuration."
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
