"""Environment variable loading utilities for FraudShield AI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment(env_file: str = ".env") -> Path:
    """Load environment variables from a dotenv file if it exists.

    Args:
        env_file: Relative or absolute path to the dotenv file.

    Returns:
        The resolved path that was checked for environment loading.
    """
    env_path = Path(env_file).resolve()
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    return env_path


def get_env(name: str, default: str | None = None) -> str | None:
    """Fetch an environment variable value.

    Args:
        name: Name of the environment variable.
        default: Value returned when the variable is missing.

    Returns:
        The environment value or the provided default.
    """
    return os.getenv(name, default)


def get_bool_env(name: str, default: bool = False) -> bool:
    """Return a boolean environment variable.

    Args:
        name: Name of the environment variable.
        default: Default boolean value when missing.

    Returns:
        Parsed boolean value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
