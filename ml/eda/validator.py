"""Dataset validation utilities for exploratory data analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from config.constants import PAYSIM_REQUIRED_COLUMNS


EXPECTED_COLUMN_TYPES: dict[str, str] = {
    "step": "numeric",
    "type": "string",
    "amount": "numeric",
    "nameOrig": "string",
    "oldbalanceOrg": "numeric",
    "newbalanceOrig": "numeric",
    "nameDest": "string",
    "oldbalanceDest": "numeric",
    "newbalanceDest": "numeric",
    "isFraud": "numeric",
    "isFlaggedFraud": "numeric",
}

CSV_DTYPE_MAP: dict[str, str] = {
    "step": "int32",
    "type": "string",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


class DataValidationError(Exception):
    """Raised when dataset validation fails."""


@dataclass(frozen=True, slots=True)
class DataValidationReport:
    """Structured dataset validation output."""

    dataset_path: Path
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values_by_column: dict[str, int]
    missing_percentage_by_column: dict[str, float]
    invalid_dtypes: dict[str, str]
    required_columns_present: bool

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passed the required validation checks."""
        return self.required_columns_present and not self.invalid_dtypes


def _find_invalid_dtypes(dataframe: pd.DataFrame) -> dict[str, str]:
    """Return columns whose dtypes do not match the expected contract."""
    invalid_dtypes: dict[str, str] = {}
    for column, expected_type in EXPECTED_COLUMN_TYPES.items():
        if column not in dataframe.columns:
            continue

        series = dataframe[column]
        if expected_type == "numeric" and not is_numeric_dtype(series):
            invalid_dtypes[column] = str(series.dtype)
        if expected_type == "string" and not is_string_dtype(series):
            invalid_dtypes[column] = str(series.dtype)
    return invalid_dtypes


def read_dataset_csv(dataset_path: str | Path) -> pd.DataFrame:
    """Read the PaySim CSV using an explicit memory-conscious dtype contract."""
    return pd.read_csv(
        dataset_path,
        dtype=CSV_DTYPE_MAP,
        low_memory=False,
    )


def validate_dataset(dataset_path: str | Path) -> DataValidationReport:
    """Validate the PaySim dataset for EDA readiness.

    Args:
        dataset_path: Path to the raw PaySim CSV file.

    Returns:
        A structured validation report.

    Raises:
        DataValidationError: If the dataset cannot be read or violates required checks.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise DataValidationError(
            f"Dataset file not found at '{path}'. Expected data/raw/paysim.csv."
        )
    if path.stat().st_size == 0:
        raise DataValidationError(f"Dataset file at '{path}' is empty.")

    try:
        dataframe = read_dataset_csv(path)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "Invalid data types detected while parsing the dataset. "
            "Ensure required numeric columns contain numeric values and text "
            "columns contain string values."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on parser error wording
        raise DataValidationError(
            f"Dataset file at '{path}' is not readable as CSV."
        ) from exc

    if dataframe.empty:
        raise DataValidationError(f"Dataset file at '{path}' contains no rows.")

    missing_required_columns = [
        column for column in PAYSIM_REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_required_columns:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(missing_required_columns)
        )

    invalid_dtypes = _find_invalid_dtypes(dataframe)
    if invalid_dtypes:
        invalid_details = ", ".join(
            f"{column}={dtype}" for column, dtype in invalid_dtypes.items()
        )
        raise DataValidationError(
            "Invalid data types detected for required columns: " + invalid_details
        )

    missing_counts = dataframe.isna().sum()
    missing_percentages = (
        (missing_counts / len(dataframe.index)) * 100 if len(dataframe.index) else 0
    )

    return DataValidationReport(
        dataset_path=path.resolve(),
        row_count=len(dataframe.index),
        column_count=len(dataframe.columns),
        duplicate_rows=int(dataframe.duplicated().sum()),
        missing_values_by_column={
            column: int(value) for column, value in missing_counts.items()
        },
        missing_percentage_by_column={
            column: round(float(value), 6)
            for column, value in missing_percentages.items()
        },
        invalid_dtypes=invalid_dtypes,
        required_columns_present=True,
    )
