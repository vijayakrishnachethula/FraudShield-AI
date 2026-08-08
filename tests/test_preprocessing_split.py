"""Tests for stratified preprocessing splits."""

from __future__ import annotations

import pandas as pd

from ml.preprocessing.config import get_preprocessing_config
from ml.preprocessing.split import stratified_train_valid_test_split


def test_stratified_split_respects_target_proportions() -> None:
    """The split helper should create 70/15/15 partitions with stratification."""
    rows = []
    for index in range(100):
        rows.append(
            {
                "step": index,
                "type": "TRANSFER" if index % 2 == 0 else "CASH_OUT",
                "amount": float(index + 1),
                "nameOrig": f"C{index}",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": f"M{index}",
                "oldbalanceDest": 500.0,
                "newbalanceDest": 600.0,
                "isFraud": 1 if index < 10 else 0,
                "isFlaggedFraud": 0,
            }
        )
    dataframe = pd.DataFrame(rows)

    split_data = stratified_train_valid_test_split(
        dataframe,
        get_preprocessing_config(),
    )

    assert len(split_data.X_train) == 70
    assert len(split_data.X_valid) == 15
    assert len(split_data.X_test) == 15
    assert abs(split_data.y_train.mean() - 0.10) < 0.02
