"""Validation helpers for preprocessing datasets and feature roles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.constants import PAYSIM_REQUIRED_COLUMNS
from ml.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class FeatureRoleSummary:
    """Describes the role of columns inside preprocessing."""

    usable_features: list[str]
    excluded_features: dict[str, str]
    target_column: str


def validate_preprocessing_columns(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
    require_target: bool = True,
) -> None:
    """Validate that the dataframe contains required preprocessing columns.

    Args:
        dataframe: Raw validated dataset.
        config: Preprocessing configuration.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = set(PAYSIM_REQUIRED_COLUMNS)
    if not require_target:
        required_columns.discard(config.target_column)
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        if missing_columns == [config.target_column]:
            raise ValueError(
                f"Target column '{config.target_column}' is missing from the dataset."
            )
        raise ValueError(
            "Dataset is missing required preprocessing columns: "
            + ", ".join(missing_columns)
        )

    target_missing = require_target and config.target_column not in dataframe.columns
    if target_missing:
        raise ValueError(
            f"Target column '{config.target_column}' is missing from the dataset."
        )


def get_feature_role_summary(config: PreprocessingConfig) -> FeatureRoleSummary:
    """Return usable, excluded, and target column roles."""
    return FeatureRoleSummary(
        usable_features=list(config.base_feature_columns),
        excluded_features=dict(config.excluded_columns),
        target_column=config.target_column,
    )


def detect_leakage_prone_columns(config: PreprocessingConfig) -> dict[str, str]:
    """Return the columns excluded due to leakage or identifier risk."""
    return {
        column: reason
        for column, reason in config.excluded_columns.items()
        if column in {"nameOrig", "nameDest", "isFlaggedFraud"}
    }
