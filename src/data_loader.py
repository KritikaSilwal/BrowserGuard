"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"URL", "label"}


def find_dataset(data_dir: str | Path = "data/raw") -> Path:
    directory = Path(data_dir)
    expected = directory / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if expected.exists():
        return expected
    candidates = sorted(directory.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV dataset found in {directory.resolve()}.")
    return candidates[0]


def load_url_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only URL and label, returning a truthful processing summary."""
    source = Path(path)
    frame = pd.read_csv(source, usecols=lambda column: column in REQUIRED_COLUMNS)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")
    original_rows = len(frame)
    frame["URL"] = frame["URL"].astype("string").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["URL", "label"])
    frame = frame[frame["URL"].str.len() > 0]
    frame = frame[frame["label"].isin([0, 1])]
    invalid_urls = int((~frame["URL"].str.contains(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://)?[^/\s]+\.[^/\s]+", regex=True, na=False)).sum())
    duplicate_urls = int(frame.duplicated("URL").sum())
    frame = frame.drop_duplicates("URL").reset_index(drop=True)
    frame["label"] = frame["label"].astype("int8")
    summary = {
        "source": str(source),
        "original_rows": original_rows,
        "usable_rows": len(frame),
        "duplicate_urls_removed": duplicate_urls,
        "malformed_url_candidates": invalid_urls,
        "class_counts": frame["label"].value_counts().sort_index().to_dict(),
    }
    return frame[["URL", "label"]], summary
