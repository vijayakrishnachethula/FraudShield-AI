"""Reusable sklearn-compatible transformers for preprocessing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.preprocessing.config import PreprocessingConfig
from ml.preprocessing.feature_engineering import get_engineered_feature_definitions
from ml.preprocessing.validators import validate_preprocessing_columns


LOGGER = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series while avoiding infinities and NaNs."""
    denominator_replaced = denominator.replace(0, np.nan)
    result = numerator / denominator_replaced
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class DataFrameColumnValidator(BaseEstimator, TransformerMixin):
    """Validate the raw dataframe schema before feature engineering."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the validator transformer."""
        self.config = config

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameColumnValidator":
        """Validate the input dataframe during fit."""
        validate_preprocessing_columns(X, self.config, require_target=False)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        LOGGER.info("Validated raw dataframe schema for preprocessing")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a copy of the input dataframe."""
        validate_preprocessing_columns(X, self.config, require_target=False)
        return X.copy()

    def get_feature_names_out(
        self,
        input_features: np.ndarray | list[str] | None = None,
    ) -> np.ndarray:
        """Return output feature names."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generate reusable fraud-specific engineered features."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the transformer with preprocessing configuration."""
        self.config = config

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FraudFeatureEngineer":
        """Learn reusable statistics from the training data."""
        validate_preprocessing_columns(X, self.config, require_target=False)
        type_frequency = X["type"].value_counts(normalize=True, dropna=False)
        self.transaction_type_frequency_map_ = type_frequency.to_dict()
        self.high_value_threshold_ = float(
            X["amount"].quantile(self.config.high_value_quantile)
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.engineered_feature_names_ = np.asarray(
            list(get_engineered_feature_definitions().keys()),
            dtype=object,
        )
        LOGGER.info(
            "Fitted feature engineering transformer with high value threshold %.4f",
            self.high_value_threshold_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the base usable features plus engineered features."""
        validate_preprocessing_columns(X, self.config, require_target=False)
        dataframe = X.loc[:, list(self.config.base_feature_columns)].copy()

        origin_balance_delta = X["oldbalanceOrg"] - X["newbalanceOrig"]
        destination_balance_delta = X["newbalanceDest"] - X["oldbalanceDest"]

        dataframe["origin_balance_delta"] = origin_balance_delta
        dataframe["destination_balance_delta"] = destination_balance_delta
        dataframe["amount_to_origin_balance_ratio"] = _safe_divide(
            X["amount"],
            X["oldbalanceOrg"],
        )
        dataframe["amount_to_destination_balance_ratio"] = _safe_divide(
            X["amount"],
            X["oldbalanceDest"],
        )
        dataframe["origin_balance_change"] = X["amount"] - origin_balance_delta
        dataframe["destination_balance_change"] = (
            X["amount"] - destination_balance_delta
        )
        dataframe["transaction_amount_log"] = np.log1p(X["amount"].clip(lower=0))
        dataframe["is_high_value_transaction"] = (
            X["amount"] >= self.high_value_threshold_
        ).astype(np.int8)
        dataframe["zero_origin_balance"] = (X["oldbalanceOrg"] == 0).astype(np.int8)
        dataframe["zero_destination_balance"] = (
            X["oldbalanceDest"] == 0
        ).astype(np.int8)
        dataframe["balance_consistency_flag"] = (
            np.isclose(origin_balance_delta, X["amount"], atol=1e-6)
            & np.isclose(destination_balance_delta, X["amount"], atol=1e-6)
        ).astype(np.int8)
        dataframe["transaction_type_frequency"] = (
            X["type"].map(self.transaction_type_frequency_map_).fillna(0.0)
        )
        dataframe["account_drain_ratio"] = _safe_divide(
            X["amount"],
            X["oldbalanceOrg"],
        )

        LOGGER.info("Generated engineered preprocessing features for %s rows", len(X))
        return dataframe

    def get_feature_names_out(
        self,
        input_features: np.ndarray | list[str] | None = None,
    ) -> np.ndarray:
        """Return output feature names for the engineered dataframe."""
        return np.asarray(
            list(self.config.base_feature_columns)
            + list(self.engineered_feature_names_),
            dtype=object,
        )


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to the interquartile range bounds learned at fit time."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "IQRClipper":
        """Learn lower and upper clipping bounds."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            values = X.to_numpy(dtype=np.float32, copy=False)
            columns = list(X.columns)
        else:
            values = np.asarray(X, dtype=np.float32)
            self.feature_names_in_ = np.asarray(
                [f"feature_{index}" for index in range(values.shape[1])],
                dtype=object,
            )
            columns = list(self.feature_names_in_)

        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = pd.Series(
            (q1 - 1.5 * iqr).astype(np.float32, copy=False),
            index=columns,
        )
        self.upper_bounds_ = pd.Series(
            (q3 + 1.5 * iqr).astype(np.float32, copy=False),
            index=columns,
        )
        LOGGER.info("Fitted IQR clipper for %s numeric features", values.shape[1])
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Clip numeric values to learned bounds."""
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        columns = X.columns if is_dataframe else None
        values = np.asarray(X, dtype=np.float32).copy()
        np.clip(
            values,
            self.lower_bounds_.to_numpy(dtype=np.float32, copy=False),
            self.upper_bounds_.to_numpy(dtype=np.float32, copy=False),
            out=values,
        )
        if is_dataframe:
            return pd.DataFrame(values, index=index, columns=columns)
        return values

    def get_feature_names_out(
        self,
        input_features: np.ndarray | list[str] | None = None,
    ) -> np.ndarray:
        """Return the unmodified feature names."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_
