"""
src/serving/

Transport-free inference layer (docs/adr/ADR-001-inference-orchestration.md §4.1).

This package imports NOTHING from `src/api/`: the FastAPI routes and the
Phase 6 Kafka consumer are both adapters over the same `InferenceService`, so
an HTTP request and a streamed message produce identical decisions for
identical input.
"""
