"""Logging configuration helpers for FraudShield AI."""

from __future__ import annotations

import logging
from pathlib import Path

from config.constants import DEFAULT_LOG_DATE_FORMAT, DEFAULT_LOG_FORMAT
from config.settings import settings


def configure_logging(log_file_name: str = "fraudshield.log") -> None:
    """Configure application logging for console and file output.

    Args:
        log_file_name: Name of the log file inside the logs directory.
    """
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(settings.logs_dir) / log_file_name

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
