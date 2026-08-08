"""Dataset loading utilities for preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.eda.validator import DataValidationReport, read_dataset_csv, validate_dataset
from ml.preprocessing.config import PreprocessingConfig
from ml.preprocessing.validators import validate_preprocessing_columns


LOGGER = logging.getLogger(__name__)


def load_preprocessing_dataset(
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Load the raw PaySim dataset for preprocessing after validation.

    Args:
        config: Preprocessing configuration.

    Returns:
        The raw dataframe and the validation report from Phase 1 utilities.
    """
    dataset_path = Path(config.dataset_path)
    LOGGER.info("Validating preprocessing dataset at %s", dataset_path)
    validation_report = validate_dataset(dataset_path)
    dataframe = read_dataset_csv(dataset_path)
    validate_preprocessing_columns(dataframe, config, require_target=True)
    LOGGER.info(
        "Loaded preprocessing dataset with %s rows and %s columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )
    return dataframe, validation_report
