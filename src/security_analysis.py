"""URL-only security analysis helpers with no network access."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .feature_extraction import extract_url_features, parse_url
from .prediction import predict_url


def parse_url_structure(raw_url: str) -> dict[str, str | int | float]:
    value = raw_url.strip()
    parsed, explicit_scheme = parse_url(value)
    features = extract_url_features(value)
    return {
        "scheme": parsed.scheme,
        "scheme_was_explicit": explicit_scheme,
        "hostname": parsed.hostname or "",
        "path": parsed.path or "/",
        "query_string": parsed.query or "(none)",
        "fragment": parsed.fragment or "(none)",
        "hostname_label_count": int(features["hostname_label_count"]),
        "url_length": int(features["url_length"]),
        "digit_count": int(features["digit_count"]),
        "special_character_count": int(features["special_count"]),
        "is_ip_hostname": bool(features["is_ipv4"]),
        "has_punycode": bool(features["has_punycode"]),
        "has_explicit_port": bool(features["has_port"]),
        "has_encoded_characters": bool(features["has_percent_encoding"]),
        "url_entropy": features["url_entropy"],
        "hostname_entropy": features["hostname_entropy"],
    }


def structural_observations(structure: dict[str, str | int | float]) -> list[str]:
    observations: list[str] = []
    if structure["is_ip_hostname"]:
        observations.append("The hostname is an IPv4 address rather than a named domain.")
    if structure["has_punycode"]:
        observations.append("The hostname contains punycode; internationalized names need careful review.")
    if structure["has_explicit_port"]:
        observations.append("The URL specifies a non-default port.")
    if structure["has_encoded_characters"]:
        observations.append("The URL contains percent-encoded characters.")
    if int(structure["hostname_label_count"]) >= 4:
        observations.append("The hostname has several labels/subdomains.")
    if not observations:
        observations.append("No notable structural flags were found by this URL-only inspector.")
    return observations


def generate_counterfactual_urls(raw_url: str) -> list[tuple[str, str]]:
    """Create valid, transparent hypothetical URL transformations."""
    value = raw_url.strip()
    candidate = value if "://" in value else f"http://{value}"
    parsed = urlsplit(candidate)
    variants: list[tuple[str, str]] = []
    if parsed.scheme == "http":
        variants.append(("Switch scheme to HTTPS", urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))))
    elif parsed.scheme == "https":
        variants.append(("Switch scheme to HTTP (hypothetical only)", urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))))
    if parsed.query:
        variants.append(("Remove query string", urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment))))
    if parsed.fragment:
        variants.append(("Remove fragment", urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))))
    return variants


def compare_counterfactuals(raw_url: str, bundle: dict) -> list[dict[str, object]]:
    original = predict_url(raw_url, bundle)
    comparisons: list[dict[str, object]] = []
    for change, variant in generate_counterfactual_urls(raw_url):
        hypothetical = predict_url(variant, bundle)
        comparisons.append({
            "change": change,
            "hypothetical_url": variant,
            "original_probability": original["phishing_probability"],
            "hypothetical_probability": hypothetical["phishing_probability"],
            "probability_delta": hypothetical["phishing_probability"] - original["phishing_probability"],
            "hypothetical_classification": hypothetical["classification"],
        })
    return comparisons
