"""Production preprocessing pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from config.constants import DEFAULT_ENCODING
from config.logging_config import configure_logging
from ml.preprocessing.artifacts import save_preprocessing_artifacts
from ml.preprocessing.config import PreprocessingConfig, get_preprocessing_config
from ml.preprocessing.feature_metadata import build_feature_metadata
from ml.preprocessing.loader import load_preprocessing_dataset
from ml.preprocessing.split import stratified_train_valid_test_split
from ml.preprocessing.transformers import (
    DataFrameColumnValidator,
    FraudFeatureEngineer,
    IQRClipper,
)
from ml.preprocessing.validators import (
    detect_leakage_prone_columns,
    get_feature_role_summary,
)


LOGGER = logging.getLogger(__name__)


def build_preprocessing_pipeline(config: PreprocessingConfig) -> Pipeline:
    """Build the shared reusable preprocessing pipeline."""
    engineered_numerical_columns = [
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
    ]
    categorical_columns = ["type"]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("outlier_clipper", IQRClipper()),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, engineered_numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("column_validator", DataFrameColumnValidator(config)),
            ("feature_engineering", FraudFeatureEngineer(config)),
            ("preprocessor", preprocessor),
        ]
    )


def _resolve_output_feature_names(pipeline: Pipeline) -> list[str]:
    """Resolve the fitted output feature names from the preprocessing pipeline."""
    raw_feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    return [str(name) for name in raw_feature_names]


def _write_markdown(path: Path, content: str) -> None:
    """Persist markdown content to disk."""
    path.write_text(content, encoding=DEFAULT_ENCODING)


def _build_feature_dictionary_markdown(config: PreprocessingConfig) -> str:
    """Build the feature dictionary report."""
    rows = [
        ("step", "Raw numeric", "Time step index from the PaySim simulation."),
        ("type", "Raw categorical", "Transaction type category."),
        ("amount", "Raw numeric", "Transaction amount."),
        (
            "oldbalanceOrg",
            "Raw numeric",
            "Origin account balance before the transaction.",
        ),
        (
            "newbalanceOrig",
            "Raw numeric",
            "Origin account balance after the transaction.",
        ),
        (
            "oldbalanceDest",
            "Raw numeric",
            "Destination account balance before the transaction.",
        ),
        (
            "newbalanceDest",
            "Raw numeric",
            "Destination account balance after the transaction.",
        ),
        ("isFraud", "Target", "Binary fraud label."),
        (
            "nameOrig",
            "Excluded",
            config.excluded_columns["nameOrig"],
        ),
        (
            "nameDest",
            "Excluded",
            config.excluded_columns["nameDest"],
        ),
        (
            "isFlaggedFraud",
            "Excluded",
            config.excluded_columns["isFlaggedFraud"],
        ),
    ]
    header = "| feature | role | description |\n| --- | --- | --- |"
    body = "\n".join(f"| {name} | {role} | {description} |" for name, role, description in rows)
    return f"# Feature Dictionary\n\n{header}\n{body}\n"


def _build_engineered_features_markdown() -> str:
    """Build documentation for engineered features."""
    from ml.preprocessing.feature_engineering import (
        get_engineered_feature_definitions,
        get_feature_exclusion_notes,
    )

    header = "| feature | definition |\n| --- | --- |"
    body = "\n".join(
        f"| {name} | {description} |"
        for name, description in get_engineered_feature_definitions().items()
    )
    excluded_body = "\n".join(
        f"- `{name}`: {reason}"
        for name, reason in get_feature_exclusion_notes().items()
    )
    return (
        "# Engineered Features\n\n"
        f"{header}\n{body}\n\n"
        "## Requested Items Not Materialized Separately\n\n"
        f"{excluded_body}\n"
    )


def _build_pipeline_summary_markdown(
    pipeline: Pipeline,
    config: PreprocessingConfig,
    split_statistics: pd.DataFrame,
) -> str:
    """Build the preprocessing pipeline summary report."""
    leakage_columns = detect_leakage_prone_columns(config)
    leakage_bullets = "\n".join(
        f"- `{column}`: {reason}" for column, reason in leakage_columns.items()
    )
    return f"""# Pipeline Summary

## Shared Preprocessing Design

- The preprocessing code lives in a single sklearn `Pipeline`.
- Feature engineering happens before the `ColumnTransformer`.
- Numerical features use mean imputation, IQR-based clipping, and `RobustScaler`.
- Categorical features use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`.
- Unknown categories are ignored at inference time to preserve FastAPI and batch prediction consistency.

## Split Strategy

{split_statistics.to_csv(index=False)}

## Leakage Prevention

{leakage_bullets}

## Artifact Reuse

- The fitted pipeline can be reused by training, explainability, API inference, LangGraph, Streamlit, and batch scoring without duplicating preprocessing logic.
- Output feature names are saved after fit so downstream systems can align model inputs and explanations consistently.
"""


def _build_artifact_summary_markdown(artifact_paths: dict[str, str]) -> str:
    """Build the artifact summary report."""
    lines = [f"- `{name}`: `{path}`" for name, path in artifact_paths.items()]
    return "# Artifact Summary\n\n" + "\n".join(lines) + "\n"


def _build_split_statistics(split_data: object) -> pd.DataFrame:
    """Create a compact split statistics table."""
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(split_data.X_train),
                "fraud_rate": float(split_data.y_train.mean()),
            },
            {
                "split": "validation",
                "rows": len(split_data.X_valid),
                "fraud_rate": float(split_data.y_valid.mean()),
            },
            {
                "split": "test",
                "rows": len(split_data.X_test),
                "fraud_rate": float(split_data.y_test.mean()),
            },
        ]
    )


def run_preprocessing_phase(config: PreprocessingConfig | None = None) -> dict[str, str]:
    """Run the full Phase 2A preprocessing build and save artifacts.

    Args:
        config: Optional preprocessing configuration override.

    Returns:
        Saved artifact paths.
    """
    configure_logging("preprocessing.log")
    active_config = config or get_preprocessing_config()
    LOGGER.info("Starting preprocessing phase using dataset %s", active_config.dataset_path)

    dataframe, _ = load_preprocessing_dataset(active_config)
    split_data = stratified_train_valid_test_split(dataframe, active_config)
    split_statistics = _build_split_statistics(split_data)

    pipeline = build_preprocessing_pipeline(active_config)
    LOGGER.info("Fitting shared preprocessing pipeline on training split")
    pipeline.fit(split_data.X_train, split_data.y_train)

    feature_engineering_step = pipeline.named_steps["feature_engineering"]
    output_feature_names = _resolve_output_feature_names(pipeline)
    feature_metadata = build_feature_metadata(
        config=active_config,
        output_feature_names=output_feature_names,
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
    artifact_paths = save_preprocessing_artifacts(
        pipeline=pipeline,
        metadata=feature_metadata,
        output_dir=active_config.artifacts_dir,
    )

    reports_dir = Path(active_config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(
        reports_dir / "feature_dictionary.md",
        _build_feature_dictionary_markdown(active_config),
    )
    _write_markdown(
        reports_dir / "engineered_features.md",
        _build_engineered_features_markdown(),
    )
    _write_markdown(
        reports_dir / "pipeline_summary.md",
        _build_pipeline_summary_markdown(pipeline, active_config, split_statistics),
    )
    _write_markdown(
        reports_dir / "artifact_summary.md",
        _build_artifact_summary_markdown(artifact_paths),
    )
    (reports_dir / "preprocessing_statistics.csv").write_text(
        split_statistics.to_csv(index=False),
        encoding=DEFAULT_ENCODING,
    )

    role_summary = get_feature_role_summary(active_config)
    (reports_dir / "feature_roles.json").write_text(
        json.dumps(
            {
                "usable_features": role_summary.usable_features,
                "excluded_features": role_summary.excluded_features,
                "target_column": role_summary.target_column,
                "engineered_output_columns": list(
                    feature_engineering_step.get_feature_names_out()
                ),
            },
            indent=2,
        ),
        encoding=DEFAULT_ENCODING,
    )

    LOGGER.info("Completed preprocessing phase and saved outputs")
    return artifact_paths


if __name__ == "__main__":
    run_preprocessing_phase()
