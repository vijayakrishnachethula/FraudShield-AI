"""Executable entry point for the FraudShield AI preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.preprocessing_pipeline import run_preprocessing_phase


def main() -> None:
    """Run the shared preprocessing phase."""
    run_preprocessing_phase()


if __name__ == "__main__":
    main()
