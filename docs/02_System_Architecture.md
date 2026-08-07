# 02 System Architecture

## High-Level Design

FraudShield AI is organized as a modular Python application with explicit separation between platform concerns and domain concerns. The system is divided into backend, frontend, machine learning, retrieval, agents, and persistence layers, all supported by shared configuration and utilities.

## Architectural Modules

- `ml/`: future home for training, evaluation, model packaging, inference contracts, and explainability adapters
- `backend/`: future FastAPI application entry points, routers, schemas, and service orchestration
- `frontend/`: future Streamlit pages, UI state management, and visualization components
- `database/`: future SQLite schema, ORM models, migrations strategy, and CRUD access
- `rag/`: future ingestion, embedding, vector indexing, retrieval, and citation handling
- `agents/`: future LangGraph state definitions, agent nodes, orchestration, and decision policies
- `config/`: environment loading, path management, constants, logging, and application settings
- `utils/`: reusable shared utilities that do not belong to a single domain

## Design Principles

- Single responsibility per file and module
- Configuration separate from business logic
- Explicit contracts between layers
- Replaceable infrastructure components where practical
- Clear portability between local development and free-tier deployment

## Planned Runtime Relationships

1. The frontend will call backend APIs for prediction, explanation, reporting, and history.
2. The backend will coordinate model inference, database persistence, retrieval, and agent orchestration.
3. The machine learning layer will expose packaged prediction and explainability interfaces.
4. The RAG layer will provide policy and document context to the multi-agent workflow.
5. The database layer will store transactions, predictions, feedback, notes, and reports.

## LLM Provider Strategy

The architecture is designed to use Gemini 2.5 Flash initially while keeping provider access abstract enough for a later Ollama adapter without changing higher-level orchestration boundaries.
