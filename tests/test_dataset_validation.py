"""Tests for dataset validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.constants import PAYSIM_REQUIRED_COLUMNS
from utils.dataset_validation import DatasetValidationError, validate_paysim_dataset


def test_validate_paysim_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    """Validation should fail when the dataset file does not exist."""
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(DatasetValidationError, match="Dataset file not found"):
        validate_paysim_dataset(missing_file)


def test_validate_paysim_dataset_raises_for_missing_columns(tmp_path: Path) -> None:
    """Validation should fail when required columns are absent."""
    dataset_path = tmp_path / "paysim.csv"
    pd.DataFrame({"step": [1], "amount": [100.0]}).to_csv(dataset_path, index=False)

    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        validate_paysim_dataset(dataset_path)


def test_validate_paysim_dataset_succeeds_for_valid_schema(tmp_path: Path) -> None:
    """Validation should succeed for a non-empty dataset with required columns."""
    dataset_path = tmp_path / "paysim.csv"
    row = {column: 0 for column in PAYSIM_REQUIRED_COLUMNS}
    row["type"] = "PAYMENT"
    row["nameOrig"] = "C123"
    row["nameDest"] = "M456"
    pd.DataFrame([row]).to_csv(dataset_path, index=False)

    result = validate_paysim_dataset(dataset_path)

    assert result.row_count == 1
    assert result.column_count == len(PAYSIM_REQUIRED_COLUMNS)
