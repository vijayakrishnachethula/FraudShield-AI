"""Tests for EDA dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.constants import PAYSIM_REQUIRED_COLUMNS
from ml.eda.loader import load_paysim_dataset


def test_load_paysim_dataset_returns_dataframe_and_report(tmp_path: Path) -> None:
    """The loader should validate and return a dataframe plus validation metadata."""
    dataset_path = tmp_path / "paysim.csv"
    row = {column: 0 for column in PAYSIM_REQUIRED_COLUMNS}
    row["type"] = "CASH_OUT"
    row["nameOrig"] = "C100"
    row["nameDest"] = "M100"
    pd.DataFrame([row]).to_csv(dataset_path, index=False)

    dataframe, report = load_paysim_dataset(dataset_path)

    assert dataframe.shape == (1, len(PAYSIM_REQUIRED_COLUMNS))
    assert report.row_count == 1
    assert report.column_count == len(PAYSIM_REQUIRED_COLUMNS)
