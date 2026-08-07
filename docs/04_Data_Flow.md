# 04 Data Flow

## Source To Insight Flow

The planned system data flow is intentionally staged and modular.

1. The user manually places the PaySim dataset at `data/raw/paysim.csv`.
2. Future preprocessing components in `ml/` will validate and transform the raw dataset.
3. Future training and benchmarking components will produce candidate models and evaluation outputs.
4. The selected model package will expose prediction-ready inference contracts.
5. The backend will receive transaction data and route it through prediction, explanation, retrieval, and persistence workflows.
6. The RAG layer will retrieve policy and reference context relevant to the transaction or case.
7. The agent layer will combine prediction, explanation, and retrieval outputs to support a decision.
8. The database layer will store predictions, feedback, notes, and reports.
9. The frontend will render outputs for analysts and capture review feedback.

## Phase 0 Boundary

In Phase 0, only the data entry assumption and validation contract are formalized. No transformation, model, retrieval, or serving logic is implemented.
