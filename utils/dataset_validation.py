"""Dataset validation utilities for the PaySim dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.constants import PAYSIM_REQUIRED_COLUMNS


class DatasetValidationError(Exception):
    """Raised when the input dataset fails validation."""


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    """Represents the outcome of dataset validation."""

    dataset_path: Path
    row_count: int
    column_count: int


def validate_paysim_dataset(dataset_path: str | Path) -> DatasetValidationResult:
    """Validate that the PaySim dataset file exists and matches the expected schema.

    Args:
        dataset_path: Path to `data/raw/paysim.csv`.

    Returns:
        Validation metadata for the dataset.

    Raises:
        DatasetValidationError: If the file is missing, empty, or invalid.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise DatasetValidationError(
            f"Dataset file not found at '{path}'. Place PaySim at data/raw/paysim.csv."
        )

    if path.stat().st_size == 0:
        raise DatasetValidationError(
            f"Dataset file at '{path}' is empty. Provide a valid non-empty CSV file."
        )

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error detail varies
        raise DatasetValidationError(
            f"Unable to read dataset file at '{path}'. Ensure it is a valid CSV."
        ) from exc

    if dataframe.empty:
        raise DatasetValidationError(
            f"Dataset at '{path}' contains no rows. Provide a populated CSV file."
        )

    missing_columns = [
        column for column in PAYSIM_REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise DatasetValidationError(
            "Dataset schema validation failed. Missing required columns: "
            + ", ".join(missing_columns)
        )

    return DatasetValidationResult(
        dataset_path=path.resolve(),
        row_count=len(dataframe.index),
        column_count=len(dataframe.columns),
    )
