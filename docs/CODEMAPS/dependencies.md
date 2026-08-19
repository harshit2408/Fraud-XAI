<!-- Generated: 2026-08-05 | Files scanned: 6 | Token estimate: ~450 -->

# Dependencies

## External Services (Docker Compose — exactly 4 containers, no more, per `.ai/rules.md`)

| Service | Image | Purpose |
|---------|-------|---------|
| `fraud-api` | built from `Dockerfile` | FastAPI inference + async Kafka consumer |
| `kafka` | `confluentinc/cp-kafka:7.5.0` | KRaft mode (no Zookeeper), topics `transactions`/`fraud_alerts` |
| `prometheus` | `prom/prometheus:latest` | Scrapes `fraud-api:8000/metrics` every 15s |
| `grafana` | `grafana/grafana:latest` | Dashboards over Prometheus (auto-provisioned) |

Explicitly NOT containerized: MLflow (local CLI only), Kafka producer (one-shot script).

## Third-Party Libraries (`requirements.txt`)

- **ML/Training**: xgboost 2.0.3, lightgbm 4.3.0, torch 2.2.0,
  pytorch-forecasting 1.0.0, pytorch-lightning 2.2.0, scikit-learn 1.4.0,
  imbalanced-learn 0.12.0 (SMOTE), optuna 3.6.1 (hyperparam tuning)
- **Explainability**: shap 0.44.0
- **Experiment Tracking**: mlflow 2.11.0 (local file store `./mlruns`)
- **Serving**: fastapi 0.110.0, uvicorn[standard] 0.27.0,
  prometheus-fastapi-instrumentator 6.4.0, pydantic 2.6.0
- **Streaming**: kafka-python 2.0.2
- **Monitoring/Drift**: evidently 0.4.22
- **Data**: pandas 2.2.0, numpy 1.26.0, pyarrow 15.0.0 (parquet I/O)
- **Viz**: matplotlib 3.8.0, seaborn 0.13.2, plotly 5.19.0
- **Dev/Quality**: pytest 8.0.0, black 24.2.0, isort 5.13.0, flake8 7.0.0,
  pre-commit 3.6.0
- **Data acquisition**: kaggle 1.6.6 (IEEE-CIS Fraud Detection competition)

## Shared/Internal Modules (cross-cutting)

- `config/config.yaml` — single source of truth for all thresholds, hyperparams,
  paths; loaded via each module's local `load_config()` helper (never hardcode
  values, per `.ai/rules.md`)
- `src/data/feature_engineering.FeatureEngineer` transformers — persisted to
  `data/processed/transformers/`, shared between training and (future) serving
  to ensure train/serve feature parity

## Environment / Secrets

`.env.example` documents required env vars (Kaggle API creds for
`download_data.py`, `KAFKA_BOOTSTRAP_SERVERS` override). No secrets committed.

## Dataset Dependency

IEEE-CIS Fraud Detection (Kaggle competition) — external, ~1.3GB raw CSVs,
downloaded on demand via `make data`, not committed to git.
