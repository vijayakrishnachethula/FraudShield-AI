"""Configuration for the production preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.paths import ARTIFACTS_DIR, PROJECT_ROOT
from config.settings import settings


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration values shared by preprocessing components."""

    dataset_path: str = str(settings.dataset_path)
    reports_dir: str = str(PROJECT_ROOT / "reports" / "preprocessing")
    artifacts_dir: str = str(ARTIFACTS_DIR)
    target_column: str = "isFraud"
    categorical_columns: tuple[str, ...] = ("type",)
    numerical_columns: tuple[str, ...] = (
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    )
    excluded_columns: dict[str, str] = field(
        default_factory=lambda: {
            "nameOrig": (
                "Excluded from model inputs because it is a high-cardinality origin "
                "account identifier and would encourage memorization rather than "
                "generalizable behavior."
            ),
            "nameDest": (
                "Excluded from model inputs because it is a high-cardinality "
                "destination account identifier and may leak entity identity."
            ),
            "isFlaggedFraud": (
                "Excluded from model inputs because EDA showed it is strongly "
                "leakage-prone and nearly reveals the target directly."
            ),
            "transaction_type_encoded": (
                "Not materialized as a standalone engineered feature because "
                "categorical encoding is handled inside the shared ColumnTransformer. "
                "Adding a separate encoded column would duplicate logic and can "
                "introduce an artificial ordinal relationship."
            ),
        }
    )
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    high_value_quantile: float = 0.95

    @property
    def base_feature_columns(self) -> tuple[str, ...]:
        """Return the raw columns available for preprocessing."""
        return self.numerical_columns + self.categorical_columns


def get_preprocessing_config() -> PreprocessingConfig:
    """Return the default preprocessing configuration."""
    return PreprocessingConfig()
