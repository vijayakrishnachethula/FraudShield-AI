# 03 Tech Stack

## Core Language

- Python 3.11+

## Machine Learning And Explainability

- Scikit-Learn for baseline models, pipelines, metrics, and utilities
- XGBoost for high-performance gradient boosting
- CatBoost for categorical-friendly boosting experiments
- LightGBM as an optional benchmark when supported in the environment
- SHAP for local and global interpretability

## Backend And Interfaces

- FastAPI for REST API implementation
- Streamlit for analyst-facing dashboard delivery
- Pydantic for input and output contracts

## Agent And Retrieval Layer

- LangGraph for multi-agent orchestration
- LangChain for retrieval and integration patterns
- ChromaDB for local vector storage
- Gemini 2.5 Flash for free-tier LLM support

## Persistence And Visualization

- SQLite for lightweight relational persistence
- Plotly for interactive dashboard visuals
- Matplotlib for static analytical plots

## Engineering Tooling

- Pytest for testing
- Ruff for linting
- Black for formatting
- python-dotenv for environment management

## Free-Only Constraint

Every selected component supports local development and a free deployment path without introducing paid dependencies.
