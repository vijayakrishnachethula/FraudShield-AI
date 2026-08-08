"""Tests for preprocessing artifact persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.artifacts import save_preprocessing_artifacts
from ml.preprocessing.config import get_preprocessing_config
from ml.preprocessing.feature_metadata import build_feature_metadata
from ml.preprocessing.preprocessing_pipeline import build_preprocessing_pipeline


def test_save_preprocessing_artifacts_writes_expected_files(tmp_path: Path) -> None:
    """Artifact saving should persist pipeline objects and metadata files."""
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

    metadata = build_feature_metadata(
        config=get_preprocessing_config(),
        output_feature_names=[
            str(name)
            for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
        ],
        numerical_columns=[
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "origin_balance_delta",
            "destination_balance_delta",
            "amount_to_origin_balance_ratio",
            "amount_to_destination_balance_ratio",
            "origin_balance_change",
            "destination_balance_change",
            "transaction_amount_log",
            "is_high_value_transaction",
            "zero_origin_balance",
            "zero_destination_balance",
            "balance_consistency_flag",
            "transaction_type_frequency",
            "account_drain_ratio",
        ],
        categorical_columns=["type"],
    )

    artifact_paths = save_preprocessing_artifacts(pipeline, metadata, tmp_path)

    assert Path(artifact_paths["preprocessing_pipeline"]).exists()
    assert Path(artifact_paths["feature_metadata"]).exists()
    assert Path(artifact_paths["target_column"]).exists()
