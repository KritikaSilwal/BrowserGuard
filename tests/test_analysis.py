import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
from sklearn.linear_model import LogisticRegression

from src.explain import explain_linear_prediction
from src.feature_extraction import FEATURE_NAMES, extract_url_features, features_to_frame
from src.prediction import load_bundle, predict_url, predict_with_model
from src.security_analysis import generate_counterfactual_urls, parse_url_structure
from src.generalization import registrable_domain
from src.reliability import threshold_metrics, unfamiliarity
from src.display import stringify_value_column
import src.prediction as prediction_module


def test_explanation_uses_feature_order_and_phishing_direction():
    model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    values = explain_linear_prediction(model, np.array([[1.0, 2.0]]), ["first", "second"])
    assert [item["feature"] for item in values] == ["second", "first"]
    assert all(np.isfinite(item["contribution"]) for item in values)


def test_feature_order_matches_extractor():
    frame = features_to_frame(["https://example.com"])
    assert list(frame.columns) == FEATURE_NAMES
    assert set(frame.columns) == set(extract_url_features("https://example.com"))


def test_structure_and_counterfactuals_are_url_only_and_valid():
    structure = parse_url_structure("http://192.0.2.1:8080/login?q=1#top")
    assert structure["scheme"] == "http"
    assert structure["is_ip_hostname"] is True
    assert structure["has_explicit_port"] is True
    variants = generate_counterfactual_urls("http://example.com/login?q=1")
    assert variants
    assert all("://" in url for _, url in variants)


def test_registrable_domain_handles_public_suffix_and_ip():
    assert registrable_domain("https://login.example.co.uk/path") == "example.co.uk"
    assert registrable_domain("http://192.0.2.1/path") == "192.0.2.1"


def test_reliability_helpers_are_explicit_and_bounded():
    metrics = threshold_metrics([0, 1, 0, 1], [0.9, 0.1, 0.2, 0.8], 0.5)
    assert metrics["phishing_predictions"] == 2
    result = unfamiliarity([0, 0, 4], [0, 0, 0], [1, 1, 1])
    assert result["level"] == "some unusual feature values"
    assert result["unusual_feature_count"] == 1


@pytest.mark.parametrize(
    "values",
    [
        [95, "95.5%"],
        [0.0, "0.0%"],
        [True, 12, 0.25],
        [],
    ],
)
def test_display_value_column_is_consistently_stringified(values):
    frame = pd.DataFrame({"metric": range(len(values)), "value": values})
    display_frame = stringify_value_column(frame)
    assert display_frame["value"].astype(object).map(type).drop_duplicates().tolist() in (
        [],
        [str],
    )
    assert frame["value"].tolist() == values


def test_saved_bundle_maps_phishing_class_probability():
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        pytest.skip("trained artifact is not available")
    result = predict_url("https://example.com", bundle)
    assert result["phishing_probability"] == pytest.approx(
        bundle["model"].predict_proba(
            bundle["scaler"].transform(features_to_frame(["https://example.com"]))
        )[0, list(bundle["model"].classes_).index(0)]
    )


def test_saved_logistic_and_forest_use_class_index_zero_for_phishing():
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        pytest.skip("trained artifact is not available")
    url = "www.facebook.com"
    row_result = predict_url(url, bundle)
    forest_result = predict_with_model(url, bundle["random_forest"], bundle)
    assert list(bundle["model"].classes_) == [0, 1]
    assert list(bundle["random_forest"].classes_) == [0, 1]
    assert row_result["phishing_probability"] == pytest.approx(
        bundle["model"].predict_proba(row_result["scaled_features"])[0, 0]
    )
    assert forest_result["phishing_probability"] == pytest.approx(
        bundle["random_forest"].predict_proba(row_result["scaled_features"])[0, 0]
    )


def test_load_bundle_prefers_existing_local_model(tmp_path, monkeypatch):
    local_model = tmp_path / "url_model.joblib"
    local_model.write_bytes(b"local")
    loader = Mock(side_effect=AssertionError("Hub download should not run"))
    monkeypatch.setattr(prediction_module, "_download_model_from_huggingface", loader)
    monkeypatch.setattr(prediction_module.joblib, "load", lambda path: {"source": str(path)})

    result = prediction_module.load_bundle(local_model)

    assert result["source"] == str(local_model)
    loader.assert_not_called()


def test_load_bundle_downloads_missing_model_without_downloading_in_tests(tmp_path, monkeypatch):
    downloaded_model = tmp_path / "cached-url-model.joblib"
    downloaded_model.write_bytes(b"cached")
    downloader = Mock(return_value=downloaded_model)
    monkeypatch.setattr(prediction_module, "_download_model_from_huggingface", downloader)
    monkeypatch.setattr(prediction_module.joblib, "load", lambda path: {"source": str(path)})

    result = prediction_module.load_bundle(tmp_path / "missing-url-model.joblib")

    assert result["source"] == str(downloaded_model)
    downloader.assert_called_once_with()


def test_huggingface_token_uses_streamlit_secret_before_environment(monkeypatch):
    monkeypatch.setattr(prediction_module.st, "secrets", {"HF_TOKEN": "secret-token"})
    monkeypatch.setenv("HF_TOKEN", "environment-token")

    assert prediction_module._configured_hf_token() == "secret-token"


def test_huggingface_token_falls_back_to_environment(monkeypatch):
    monkeypatch.setattr(prediction_module.st, "secrets", {})
    monkeypatch.setenv("HF_TOKEN", "environment-token")

    assert prediction_module._configured_hf_token() == "environment-token"
