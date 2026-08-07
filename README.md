# FraudShield AI

FraudShield AI is a production-style portfolio project for fraud detection, explainability, retrieval-assisted decision support, and analyst workflows. This repository currently contains only the Phase 0 engineering foundation: architecture, configuration, documentation, project standards, and dataset validation utilities.

## Project Overview

The goal of FraudShield AI is to provide an end-to-end fraud analysis platform built entirely with free tools and deployment options. The planned system combines classical machine learning, SHAP-based explainability, retrieval-augmented policy search, a LangGraph-driven multi-agent workflow, a FastAPI backend, and a Streamlit frontend.

The current repository state includes the engineering foundation plus an EDA-only analysis phase for understanding the PaySim dataset before preprocessing or modeling. No training, feature engineering, API endpoints, dashboard code, agents, or RAG workflows are implemented yet.

## Architecture

The architecture is organized into dedicated domains:

- `ml/` for dataset handling, features, training, evaluation, packaging, and inference contracts
- `backend/` for FastAPI application layers and API contracts
- `frontend/` for Streamlit pages and UI components
- `agents/` for LangGraph orchestration and agent responsibilities
- `rag/` for document ingestion, retrieval, and citation workflows
- `database/` for SQLite schema, data access, and persistence
- `config/` for environment loading, settings, paths, constants, and logging
- `utils/` for cross-cutting reusable helpers
- `docs/` for architecture and delivery documentation

Detailed architecture references live under [docs](docs).

## Tech Stack

- Python 3.11+
- Scikit-Learn
- XGBoost
- CatBoost
- LightGBM (optional when supported)
- SHAP
- LangGraph
- LangChain
- Gemini 2.5 Flash
- FastAPI
- Streamlit
- SQLite
- ChromaDB
- Plotly
- Matplotlib
- Pytest
- Ruff
- Black

## Planned Features

- Fraud detection model benchmarking and selection
- SHAP-powered global and local explanations
- FastAPI prediction and reporting services
- SQLite-backed transaction, prediction, and feedback persistence
- Retrieval over policies, guidelines, and historical reports
- LangGraph multi-agent fraud decision workflow
- Streamlit analyst dashboard
- Monitoring, feedback loop, and reporting exports

## Current Phase Deliverables

- Project architecture and configuration foundation
- PaySim dataset validation for analysis readiness
- Exploratory data analysis package under `ml/eda/`
- Generated EDA reports and plots under `reports/eda/`
- Unit tests for loading and validation behaviors

## Folder Structure

```text
FraudShield-AI/
|-- agents/
|-- backend/
|-- config/
|-- data/
|   |-- processed/
|   `-- raw/
|-- database/
|-- docs/
|-- frontend/
|-- logs/
|-- ml/
|-- models/
|-- notebooks/
|-- rag/
|-- reports/
|-- scripts/
|-- tests/
`-- utils/
```

## Installation

1. Create a Python 3.11+ virtual environment.
2. Activate the environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in required values such as `GEMINI_API_KEY`.
3. Keep `.env` local only. It is ignored by Git.

Configuration modules live in [config](config) and are designed around `python-dotenv`.

## Dataset Setup

The PaySim dataset must be downloaded manually by the user.

- Expected location: `data/raw/paysim.csv`
- Dataset instructions: [data/README.md](data/README.md)
- Validation utility: [utils/dataset_validation.py](utils/dataset_validation.py)
- EDA package: [ml/eda](ml/eda)
- EDA report generator: [ml/eda/report_generator.py](ml/eda/report_generator.py)

Validation checks include:

- File existence
- Non-empty file
- Required schema columns
- CSV readability
- Duplicate row detection
- Invalid dtype detection
- Missing value summary

## EDA Workflow

Run the exploratory analysis pipeline with:

```bash
python -m ml.eda.report_generator
```

This generates:

- `reports/eda/eda_summary.md`
- `reports/eda/validation_report.md`
- `reports/eda/business_insights.md`
- `reports/eda/feature_statistics.csv`
- `reports/eda/missing_values.csv`
- `reports/eda/correlation.csv`
- PNG plots in `reports/eda/plots/`

## Deployment Strategy

The planned free deployment path is:

- Frontend: Streamlit Community Cloud
- Backend: Render Free Tier
- Database: SQLite
- Vector Store: ChromaDB

Detailed deployment notes are documented in [docs/09_Deployment_Strategy.md](docs/09_Deployment_Strategy.md).

## Future Roadmap

- Phase 1: Project setup assets beyond architecture
- Phase 2: Dataset and feature engineering
- Phase 3: Model benchmarking
- Phase 4: Explainability
- Phase 5+: Packaging, backend, RAG, agents, dashboard, monitoring, reports, deployment

See [docs/10_Project_Roadmap.md](docs/10_Project_Roadmap.md) for the full sequence.

## Screenshots Placeholder

Screenshots and dashboard captures will be added in later phases once the frontend is implemented. EDA visualizations are currently exported as PNG files under `reports/eda/plots/`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
