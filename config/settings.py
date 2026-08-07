"""Centralized application settings for FraudShield AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.constants import PROJECT_NAME
from config.environment import get_bool_env, get_env, load_environment
from config.paths import LOGS_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR


load_environment()


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    project_name: str = get_env("PROJECT_NAME", PROJECT_NAME) or PROJECT_NAME
    project_version: str = get_env("PROJECT_VERSION", "0.1.0") or "0.1.0"
    app_env: str = get_env("APP_ENV", "development") or "development"
    debug: bool = get_bool_env("DEBUG", default=True)
    log_level: str = get_env("LOG_LEVEL", "INFO") or "INFO"
    gemini_api_key: str = get_env("GEMINI_API_KEY", "") or ""
    llm_provider: str = get_env("LLM_PROVIDER", "gemini") or "gemini"
    dataset_path: Path = Path(
        get_env("DATASET_PATH", str(PROJECT_ROOT / "data" / "raw" / "paysim.csv"))
        or PROJECT_ROOT / "data" / "raw" / "paysim.csv"
    )
    chroma_persist_dir: Path = Path(
        get_env(
            "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "processed" / "chroma")
        )
        or PROJECT_ROOT / "data" / "processed" / "chroma"
    )
    sqlite_db_path: Path = Path(
        get_env("SQLITE_DB_PATH", str(PROJECT_ROOT / "database" / "fraudshield.db"))
        or PROJECT_ROOT / "database" / "fraudshield.db"
    )
    model_dir: Path = Path(get_env("MODEL_DIR", str(MODELS_DIR)) or MODELS_DIR)
    reports_dir: Path = Path(get_env("REPORTS_DIR", str(REPORTS_DIR)) or REPORTS_DIR)
    logs_dir: Path = Path(get_env("LOGS_DIR", str(LOGS_DIR)) or LOGS_DIR)


settings = Settings()
