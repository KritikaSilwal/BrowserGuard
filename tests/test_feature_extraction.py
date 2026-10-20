import pytest

from src.feature_extraction import FEATURE_NAMES, extract_url_features, features_to_frame, parse_url


def test_extractor_is_deterministic_and_ordered():
    first = extract_url_features("https://example.com/a?id=4")
    second = extract_url_features("https://example.com/a?id=4")
    assert first == second
    assert list(first) == FEATURE_NAMES
    assert first["https"] == 1
    assert first["has_query"] == 1


def test_missing_scheme_is_supported_without_network_access():
    values = extract_url_features("example.com/login")
    assert values["http"] == 1
    assert values["https"] == 0
    assert values["hostname_length"] == len("example.com")


def test_explicit_scheme_is_distinguished_from_inferred_scheme():
    parsed, explicit = parse_url("example.com")
    assert parsed.scheme == "http"
    assert explicit is False
    parsed, explicit = parse_url("https://example.com")
    assert parsed.scheme == "https"
    assert explicit is True


def test_representative_scheme_features_are_mutually_exclusive():
    for url, expected_http, expected_https in [
        ("example.com", 1, 0),
        ("http://example.com", 1, 0),
        ("https://example.com", 0, 1),
        ("https://example.com/login", 0, 1),
        ("https://sub.example.com/a?x=1", 0, 1),
    ]:
        values = extract_url_features(url)
        assert values["http"] == expected_http
        assert values["https"] == expected_https
        assert values["http"] + values["https"] == 1


def test_invalid_url_is_rejected():
    with pytest.raises(ValueError):
        extract_url_features("not a URL")


def test_frame_has_numeric_columns():
    frame = features_to_frame(["https://example.com", "http://192.0.2.1/a"])
    assert list(frame.columns) == FEATURE_NAMES
    assert frame.shape == (2, len(FEATURE_NAMES))
