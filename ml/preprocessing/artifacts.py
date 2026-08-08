"""Artifact persistence for preprocessing assets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from config.constants import DEFAULT_ENCODING
from ml.preprocessing.feature_metadata import FeatureMetadata


LOGGER = logging.getLogger(__name__)


def save_preprocessing_artifacts(
    pipeline: Pipeline,
    metadata: FeatureMetadata,
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist fitted preprocessing artifacts for later reuse.

    Args:
        pipeline: Fitted preprocessing pipeline.
        metadata: Serialized feature metadata.
        output_dir: Directory where artifacts will be stored.

    Returns:
        A mapping of artifact names to file paths.
    """
    artifacts_dir = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    numerical_pipeline = preprocessor.named_transformers_["numerical"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]

    artifact_paths = {
        "preprocessing_pipeline": str(
            artifacts_dir / "preprocessing_pipeline.joblib"
        ),
        "encoders": str(artifacts_dir / "encoders.joblib"),
        "scalers": str(artifacts_dir / "scalers.joblib"),
        "feature_metadata": str(artifacts_dir / "feature_metadata.json"),
        "feature_names": str(artifacts_dir / "feature_names.json"),
        "categorical_columns": str(artifacts_dir / "categorical_columns.json"),
        "numerical_columns": str(artifacts_dir / "numerical_columns.json"),
        "excluded_columns": str(artifacts_dir / "excluded_columns.json"),
        "target_column": str(artifacts_dir / "target_column.json"),
    }

    joblib.dump(pipeline, artifact_paths["preprocessing_pipeline"])
    joblib.dump(categorical_pipeline, artifact_paths["encoders"])
    joblib.dump(numerical_pipeline, artifact_paths["scalers"])

    (artifacts_dir / "feature_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding=DEFAULT_ENCODING,
    )
    (artifacts_dir / "feature_names.json").write_text(
        json.dumps(metadata.output_feature_names, indent=2),
        encoding=DEFAULT_ENCODING,
    )
    (artifacts_dir / "categorical_columns.json").write_text(
        json.dumps(metadata.categorical_columns, indent=2),
        encoding=DEFAULT_ENCODING,
    )
    (artifacts_dir / "numerical_columns.json").write_text(
        json.dumps(metadata.numerical_columns, indent=2),
        encoding=DEFAULT_ENCODING,
    )
    (artifacts_dir / "excluded_columns.json").write_text(
        json.dumps(metadata.excluded_columns, indent=2),
        encoding=DEFAULT_ENCODING,
    )
    (artifacts_dir / "target_column.json").write_text(
        json.dumps(metadata.target_column, indent=2),
        encoding=DEFAULT_ENCODING,
    )

    LOGGER.info("Saved preprocessing artifacts to %s", artifacts_dir)
    return artifact_paths
