# 07 Agent Architecture

## Multi-Agent Objective

The future LangGraph workflow will coordinate specialized agents that each handle a well-bounded responsibility in the fraud review lifecycle.

## Planned Agent Roles

- Detection Agent: consumes the packaged fraud model for inference
- Explainability Agent: interprets SHAP outputs for local reasoning
- Policy Agent: retrieves relevant SOPs, guidelines, and policy excerpts
- Decision Agent: combines score, explanation, context, and case evidence
- Notification Agent: prepares structured alerts for downstream consumption
- Report Agent: assembles investigation-ready summaries

## Orchestration Principles

- Shared graph state should be explicit and typed
- Agent outputs should be deterministic where possible
- LLM-backed reasoning should be constrained by structured context
- Retrieval evidence should be citable
- Final decisions should preserve traceability

## Phase 0 Constraint

No LangGraph state, nodes, or execution graph is implemented in this phase.
