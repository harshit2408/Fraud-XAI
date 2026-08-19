<!-- Generated: 2026-08-05 | Files scanned: 30 | Token estimate: ~650 -->

# Architecture — Explainable Fraud Detection System

Single-repo Python ML system (portfolio project). No monorepo/packages split.
Entry points: `src/api/main.py` (FastAPI app), `src/streaming/producer.py` (CLI),
training scripts under `src/training/*.py` (CLI, run via `make train`).

## System Diagram

```
Kaggle CSV ──▶ download_data.py ──▶ data/raw/*.csv
                                        │
                                        ▼
                              preprocess.py (orchestrator)
                    ┌───────────────┬───────────────┬──────────────┐
                    ▼               ▼               ▼              ▼
              data_loader   feature_engineering  imbalance_handler  data_splitter
                    └───────────────┴───────────────┴──────────────┘
                                        │
                                        ▼
                     data/processed/{train,val,test}_{features,labels}.parquet
                                        │
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
            train_xgb.py         train_lgbm.py         train_tft.py
          (+ tune_xgb.py)       (uses losses.py,      (+ tune_tft.py,
                                  sequence_builder)     sequence_builder)
                    │                   │                    │
                    ▼                   ▼                    ▼
           models/xgb_model.pkl  models/lgbm_model.pkl  models/tft_model.pt
                    └───────────────────┴────────────────────┘
                                        │
                                        ▼
                          models/ensemble.py (ModelEnsemble)
                                        │
                                        ▼
                            evaluator.py (metrics, plots)
                                        │
                                        ▼
                    src/api/main.py (FastAPI) ── /health (Phase 5: /predict, /metrics)
                                        │
                          consumes Kafka topic "transactions"
                          produces Kafka topic "fraud_alerts"
                                        ▲
                                        │
                          src/streaming/producer.py (one-shot CLI, `make stream`)
```

## Service Boundaries (Docker Compose — strictly 4 containers, see `.ai/rules.md`)

1. `fraud-api` — FastAPI + async Kafka consumer (single process, no separate consumer container)
2. `kafka` — KRaft mode (no Zookeeper), topics: `transactions` → `fraud_alerts`
3. `prometheus` — scrapes `fraud-api` `/metrics` every 15s
4. `grafana` — dashboards over Prometheus, auto-provisioned from `monitoring/grafana/`

MLflow (`mlruns/`) is explicitly NOT containerized — run locally via `make mlflow`.

## Build Phases (see `.ai/prd.md` for full detail)

Phase 0 Foundation → 1 Data/Feature Eng → 2 XGBoost baseline → 3 TFT sequential model
→ 4 Explainability (SHAP, `src/explainability/` — stub, not yet implemented)
→ 5 FastAPI serving (stub: `/health` only) → 6 Kafka real-time simulation
→ 7 Monitoring/drift (`src/monitoring/` — stub, `drift_reporter.py` referenced by
Makefile but not yet created) → 8 Business impact → 9 Testing → 10 Packaging.

## Status of Stub Modules

- `src/explainability/` — package exists, empty (`__init__.py` only). Planned: SHAP.
- `src/monitoring/` — package exists, empty. Makefile's `make monitor` calls
  `src/monitoring/drift_reporter.py`, which does not exist yet.
- `src/api/routes/`, `src/api/schemas/`, `src/api/middleware/` — empty packages,
  scaffolded for Phase 5 (`/predict` route, Pydantic request/response schemas).
