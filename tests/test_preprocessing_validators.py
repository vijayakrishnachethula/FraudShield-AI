"""Tests for preprocessing validation and leakage controls."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.preprocessing.config import get_preprocessing_config
from ml.preprocessing.validators import (
    detect_leakage_prone_columns,
    get_feature_role_summary,
    validate_preprocessing_columns,
)


def test_leakage_detection_lists_expected_columns() -> None:
    """Leakage detection should identify excluded identifier and flagged columns."""
    leakage_columns = detect_leakage_prone_columns(get_preprocessing_config())

    assert set(leakage_columns) == {"nameOrig", "nameDest", "isFlaggedFraud"}


def test_feature_role_summary_excludes_identifiers() -> None:
    """Feature roles should separate usable features from excluded ones."""
    summary = get_feature_role_summary(get_preprocessing_config())

    assert "nameOrig" not in summary.usable_features
    assert "nameDest" in summary.excluded_features


def test_validate_preprocessing_columns_raises_for_missing_target() -> None:
    """Schema validation should fail if the target column is absent."""
    dataframe = pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [100.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [400.0],
            "nameDest": ["M1"],
            "oldbalanceDest": [1000.0],
            "newbalanceDest": [1100.0],
            "isFlaggedFraud": [0],
        }
    )

    with pytest.raises(ValueError, match="Target column"):
        validate_preprocessing_columns(dataframe, get_preprocessing_config())
