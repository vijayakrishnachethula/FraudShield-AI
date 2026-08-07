# 09 Deployment Strategy

## Deployment Targets

- Frontend: Streamlit Community Cloud
- Backend: Render Free Tier
- Database: SQLite
- Vector Store: ChromaDB local persistence

## Strategy

The system should support local development first, then package cleanly for free-tier deployment with environment-variable based configuration. The backend and frontend should remain deployable as separate services, while shared configuration stays consistent across environments.

## Key Requirements

- No paid infrastructure
- No secret values committed to source control
- Reproducible dependency installation
- Clear separation between data, models, reports, and logs
- A documented environment setup path

## Phase 0 Constraint

Deployment configuration files are not introduced yet because this phase is limited to architecture and engineering foundation. This document preserves the deployment direction for later phases.
