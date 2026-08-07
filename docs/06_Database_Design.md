# 06 Database Design

## Storage Strategy

FraudShield AI will use SQLite as the default relational persistence layer for local development and free deployment alignment.

## Planned Core Tables

- `transactions`
- `predictions`
- `feedback`
- `analyst_notes`
- `reports`
- `users`

## Planned Relationships

- A transaction can have one or more prediction records over time
- Feedback attaches to analyst review of a prediction or case outcome
- Analyst notes provide additional contextual observations
- Reports summarize decision trails and supporting evidence

## Design Principles

- Store auditable timestamps on all mutable entities
- Keep prediction inputs and outputs traceable
- Separate reference entities from event entities
- Maintain schema naming consistency for future migrations

## Phase 0 Constraint

No database models, migrations, or CRUD code are created yet. This phase captures storage direction only.
