"""Tests for EDA validation behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.constants import PAYSIM_REQUIRED_COLUMNS
from ml.eda.validator import DataValidationError, validate_dataset


def _valid_row() -> dict[str, object]:
    """Return a minimally valid PaySim-like row."""
    row = {column: 0 for column in PAYSIM_REQUIRED_COLUMNS}
    row["type"] = "TRANSFER"
    row["nameOrig"] = "C123"
    row["nameDest"] = "M123"
    row["amount"] = 100.5
    return row


def test_validate_dataset_detects_duplicates(tmp_path: Path) -> None:
    """Validation should report duplicate row counts."""
    dataset_path = tmp_path / "paysim.csv"
    row = _valid_row()
    pd.DataFrame([row, row]).to_csv(dataset_path, index=False)

    report = validate_dataset(dataset_path)

    assert report.duplicate_rows == 1


def test_validate_dataset_reports_missing_values(tmp_path: Path) -> None:
    """Validation should surface missing counts and percentages."""
    dataset_path = tmp_path / "paysim.csv"
    row = _valid_row()
    row["nameDest"] = None
    pd.DataFrame([row]).to_csv(dataset_path, index=False)

    report = validate_dataset(dataset_path)

    assert report.missing_values_by_column["nameDest"] == 1
    assert report.missing_percentage_by_column["nameDest"] == 100.0


def test_validate_dataset_rejects_invalid_dtypes(tmp_path: Path) -> None:
    """Validation should fail when a numeric column has an invalid dtype."""
    dataset_path = tmp_path / "paysim.csv"
    row = _valid_row()
    row["amount"] = "invalid"
    pd.DataFrame([row]).to_csv(dataset_path, index=False)

    with pytest.raises(DataValidationError, match="Invalid data types"):
        validate_dataset(dataset_path)
