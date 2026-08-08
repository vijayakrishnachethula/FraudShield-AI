"""Reusable stratified dataset splitting for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Container for train, validation, and test splits."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def stratified_train_valid_test_split(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> DatasetSplit:
    """Split the dataset into stratified train, validation, and test partitions."""
    X = dataframe.drop(columns=[config.target_column])
    y = dataframe[config.target_column]

    total_rows = len(dataframe)
    test_rows = int(round(total_rows * config.test_size))
    validation_rows = int(round(total_rows * config.validation_size))
    temp_rows = test_rows + validation_rows

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_rows,
        random_state=config.random_state,
        stratify=y,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_rows,
        random_state=config.random_state,
        stratify=y_temp,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
