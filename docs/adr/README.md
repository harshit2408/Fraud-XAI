# Architecture Decision Records

Records of architectural decisions that constrain later implementation work.
Each ADR is immutable once **Accepted** — a decision that turns out wrong is
superseded by a new ADR, not edited in place.

Format: Status · Context · Decision Drivers · Options Considered · Decision ·
Consequences · Open Questions. An ADR states what was decided and why the
rejected options were rejected; it is not a design doc and does not contain
implementation code.

| ADR | Title | Status | Gates |
|-----|-------|--------|-------|
| [ADR-001](ADR-001-inference-orchestration.md) | Inference orchestration: `ModelRegistry` + `InferenceService` | Accepted (2026-08-26) | PRD Phase 5 (FastAPI serving), Phase 4 (SHAP in the request path) |
| [ADR-002](ADR-002-realtime-feature-state.md) | Real-time card aggregates and target encodings | Accepted (2026-08-26) | PRD Phase 6 (Kafka streaming), and the servability of the batch feature set |
| [ADR-003](ADR-003-trailing-window-velocity-state.md) | Serving the trailing-window velocity features (extends ADR-002) | Accepted (2026-09-08) | PRD Phase 9 step 9.4, task E4 (train/serve equivalence) |
| [ADR-004](ADR-004-operating-point-selection.md) | Operating-point selection under an infeasible target band — two-stage cascade rejected (extends ADR-001) | Accepted (2026-09-09) | PRD Phase 9 step 9.5 / phase close-out |
| [ADR-005](ADR-005-gnn-architecture-evaluation.md) | GNN-GraphSAGE architecture evaluation — **rejected on evidence** (test PR-AUC 0.4410 vs. 0.5502 baseline); infrastructure kept | Accepted (2026-09-10) | PRD Phase 12 close-out; 12.2.4 not triggered |

Both correspond to tasks **E1** and **E2** in
[`docs/IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) Phase E. The plan's
standing instruction — *"do not begin Phase E implementation before ADR-002 is
written: the real-time feature question determines whether the current batch
feature set is servable at all"* — is discharged by ADR-002's feature-by-feature
servability audit.
