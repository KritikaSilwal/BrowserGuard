"""Helpers for preparing UI-only tables without changing model data."""

from __future__ import annotations

import pandas as pd


def stringify_value_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy whose mixed-type ``value`` column is Arrow-safe."""
    display_frame = frame.copy()
    if "value" in display_frame.columns:
        display_frame["value"] = display_frame["value"].map(str)
    return display_frame
