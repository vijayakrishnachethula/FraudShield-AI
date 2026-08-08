"""Metadata helpers for preprocessing features and artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ml.preprocessing.config import PreprocessingConfig
from ml.preprocessing.feature_engineering import (
    get_engineered_feature_definitions,
    get_feature_exclusion_notes,
)


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Structured metadata for the fitted preprocessing pipeline."""

    target_column: str
    categorical_columns: list[str]
    numerical_columns: list[str]
    engineered_features: dict[str, str]
    excluded_columns: dict[str, str]
    output_feature_names: list[str]
    base_feature_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary representation."""
        return asdict(self)


def build_feature_metadata(
    config: PreprocessingConfig,
    output_feature_names: list[str],
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> FeatureMetadata:
    """Build metadata for the fitted preprocessing pipeline."""
    excluded_columns = dict(config.excluded_columns)
    excluded_columns.update(get_feature_exclusion_notes())
    return FeatureMetadata(
        target_column=config.target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        engineered_features=get_engineered_feature_definitions(),
        excluded_columns=excluded_columns,
        output_feature_names=output_feature_names,
        base_feature_columns=list(config.base_feature_columns),
    )
