"""Dataset loader for PaySim EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.eda.validator import DataValidationReport, read_dataset_csv, validate_dataset


def load_paysim_dataset(
    dataset_path: str | Path,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Validate and load the PaySim dataset.

    Args:
        dataset_path: Path to the raw PaySim CSV file.

    Returns:
        The loaded dataframe and its validation report.
    """
    validation_report = validate_dataset(dataset_path)
    dataframe = read_dataset_csv(dataset_path)
    return dataframe, validation_report
