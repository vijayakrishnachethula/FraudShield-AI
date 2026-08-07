"""Project-wide constants for FraudShield AI."""

from __future__ import annotations

from typing import Final


PROJECT_NAME: Final[str] = "FraudShield AI"
PROJECT_SLUG: Final[str] = "fraudshield-ai"
DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
DEFAULT_LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

PAYSIM_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)
