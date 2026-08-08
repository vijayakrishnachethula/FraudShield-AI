"""Tests for preprocessing transformers and feature engineering."""

from __future__ import annotations

import pandas as pd

from ml.preprocessing.config import get_preprocessing_config
from ml.preprocessing.transformers import FraudFeatureEngineer, IQRClipper


def _sample_dataframe() -> pd.DataFrame:
    """Return a small valid dataframe for preprocessing tests."""
    return pd.DataFrame(
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
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "nameDest": "M2",
                "oldbalanceDest": 200.0,
                "newbalanceDest": 250.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
        ]
    )


def test_feature_engineer_creates_expected_columns() -> None:
    """Feature engineering should add the configured derived features."""
    config = get_preprocessing_config()
    dataframe = _sample_dataframe()
    transformer = FraudFeatureEngineer(config).fit(dataframe)

    transformed = transformer.transform(dataframe)

    assert "origin_balance_delta" in transformed.columns
    assert "transaction_amount_log" in transformed.columns
    assert "transaction_type_frequency" in transformed.columns
    assert transformed["zero_origin_balance"].tolist() == [0, 1]


def test_iqr_clipper_preserves_shape() -> None:
    """Outlier clipping should keep numeric shape unchanged."""
    dataframe = pd.DataFrame({"a": [1.0, 2.0, 1000.0], "b": [5.0, 6.0, 7.0]})

    clipper = IQRClipper().fit(dataframe)
    clipped = clipper.transform(dataframe)

    assert clipped.shape == dataframe.shape
    assert clipped.iloc[2, 0] <= clipper.upper_bounds_["a"]
