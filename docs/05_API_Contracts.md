# 05 API Contracts

## Purpose

This document defines the architectural intent for future API boundaries without implementing any endpoints in Phase 0.

## Planned Service Areas

- Prediction services
- Explainability services
- Report access services
- Transaction history services
- Analyst feedback services
- Health and operational services

## Contract Design Principles

- Use explicit Pydantic request and response schemas
- Version public endpoints from the start
- Validate all external inputs at the boundary
- Return structured error responses
- Separate transport models from internal domain models

## Planned Contract Domains

- Transaction scoring request
- Prediction response with fraud probability and decision metadata
- Explainability response with top positive and negative contributors
- Report summary and report detail responses
- Feedback submission and audit response models

## Phase 0 Constraint

No FastAPI application, routers, or schemas are implemented in this phase. This document exists to preserve the architecture and guide later delivery.
