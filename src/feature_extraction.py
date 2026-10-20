"""Deterministic URL-only feature extraction for AI BrowserGuard."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import SplitResult, urlsplit

import pandas as pd

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "fragment_length", "dot_count", "hyphen_count", "underscore_count",
    "slash_count", "question_count", "equals_count", "ampersand_count",
    "at_count", "percent_count", "digit_count", "letter_count",
    "special_count", "digit_ratio", "special_ratio", "hostname_label_count",
    "subdomain_count", "is_ipv4", "has_punycode", "hostname_has_hyphen",
    "hostname_has_digit", "has_suspicious_label", "has_port", "https",
    "http", "has_at_symbol", "has_percent_encoding", "has_query",
    "has_fragment", "url_entropy", "hostname_entropy",
]

_SUSPICIOUS_LABEL = re.compile(r"(?:xn--|login|secure|account|verify|update|signin|banking|password)", re.I)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def parse_url(raw_url: str) -> tuple[SplitResult, bool]:
    """Parse a URL and report whether its scheme was explicitly supplied."""
    value = str(raw_url).strip()
    if not value:
        raise ValueError("URL cannot be empty.")
    explicit_scheme = "://" in value
    candidate = value if explicit_scheme else f"http://{value}"
    parsed = urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any(char.isspace() for char in parsed.hostname)
    ):
        raise ValueError("URL must contain a valid HTTP or HTTPS hostname.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port.") from exc
    return parsed, explicit_scheme


def _parse_url(raw_url: str) -> SplitResult:
    return parse_url(raw_url)[0]


def extract_url_features(raw_url: str) -> dict[str, float]:
    """Extract numerical features without resolving or visiting the URL."""
    parsed = _parse_url(raw_url)
    value = str(raw_url).strip()
    hostname = parsed.hostname or ""
    labels = hostname.split(".") if hostname else []
    try:
        is_ipv4 = int(isinstance(ipaddress.ip_address(hostname), ipaddress.IPv4Address))
    except ValueError:
        is_ipv4 = 0
    special_count = sum(not char.isalnum() for char in value)
    length = max(len(value), 1)
    return {
        "url_length": float(len(value)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "fragment_length": float(len(parsed.fragment)),
        "dot_count": float(value.count(".")),
        "hyphen_count": float(value.count("-")),
        "underscore_count": float(value.count("_")),
        "slash_count": float(value.count("/")),
        "question_count": float(value.count("?")),
        "equals_count": float(value.count("=")),
        "ampersand_count": float(value.count("&")),
        "at_count": float(value.count("@")),
        "percent_count": float(value.count("%")),
        "digit_count": float(sum(char.isdigit() for char in value)),
        "letter_count": float(sum(char.isalpha() for char in value)),
        "special_count": float(special_count),
        "digit_ratio": float(sum(char.isdigit() for char in value) / length),
        "special_ratio": float(special_count / length),
        "hostname_label_count": float(len(labels)),
        "subdomain_count": float(max(len(labels) - 2, 0)),
        "is_ipv4": float(is_ipv4),
        "has_punycode": float(int(any(label.lower().startswith("xn--") for label in labels))),
        "hostname_has_hyphen": float(int("-" in hostname)),
        "hostname_has_digit": float(int(any(char.isdigit() for char in hostname))),
        "has_suspicious_label": float(int(any(_SUSPICIOUS_LABEL.search(label) for label in labels))),
        "has_port": float(int(parsed.port is not None)),
        "https": float(int(parsed.scheme == "https")),
        "http": float(int(parsed.scheme == "http")),
        "has_at_symbol": float(int("@" in value)),
        "has_percent_encoding": float(int(bool(re.search(r"%[0-9A-Fa-f]{2}", value)))),
        "has_query": float(int(bool(parsed.query))),
        "has_fragment": float(int(bool(parsed.fragment))),
        "url_entropy": _entropy(value),
        "hostname_entropy": _entropy(hostname),
    }


def features_to_frame(urls: list[str] | pd.Series) -> pd.DataFrame:
    """Extract an ordered feature frame, raising a useful error for bad input."""
    rows: list[dict[str, Any]] = []
    for url in urls:
        rows.append(extract_url_features(str(url)))
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype="float64")
