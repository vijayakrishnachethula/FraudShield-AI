# 08 RAG Architecture

## Goal

The RAG subsystem will provide document-grounded context for fraud review decisions, especially policies, guidelines, and historical references.

## Planned Inputs

- Fraud SOP documents
- Regulatory guidance
- Merchant policy references
- Analyst playbooks
- Historical report knowledge

## Planned Components

- Document loaders
- Chunking strategy
- Embedding generation
- ChromaDB persistence
- Retriever configuration
- Citation-aware response assembly

## Design Constraints

- Use free-compatible tooling only
- Keep ingestion separate from retrieval
- Preserve document provenance for citations
- Allow local storage and reproducible rebuilds

## Phase 0 Constraint

No ingestion, embeddings, or retrieval execution is implemented in this phase.
