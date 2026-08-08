"""Tests for the reusable preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from ml.preprocessing.config import get_preprocessing_config
from ml.preprocessing.preprocessing_pipeline import build_preprocessing_pipeline


def test_preprocessing_pipeline_fits_and_exposes_feature_names() -> None:
    """The preprocessing pipeline should fit and emit output feature names."""
    dataframe = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 400.0,
                "nameDest": "M1",
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 1100.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            },
            {
                "step": 2,
                "type": "CASH_OUT",
                "amount": 50.0,
                "nameOrig": "C2",
                "oldbalanceOrg": 250.0,
                "newbalanceOrig": 200.0,
                "nameDest": "M2",
                "oldbalanceDest": 120.0,
                "newbalanceDest": 170.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
        ]
    )
    X = dataframe.drop(columns=["isFraud"])
    y = dataframe["isFraud"]

    pipeline = build_preprocessing_pipeline(get_preprocessing_config())
    pipeline.fit(X, y)
    transformed = pipeline.transform(X)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    assert transformed.shape[0] == 2
    assert "transaction_type_frequency" in feature_names
    assert any(name.startswith("type_") for name in feature_names)
