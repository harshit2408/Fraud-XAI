<!-- Generated: 2026-08-05 | Files scanned: 5 | Token estimate: ~150 -->

# Frontend

No custom frontend/SPA exists in this project. UI surface is limited to
operator-facing dashboards, not end-user pages.

## Visualization Layer (Grafana, provisioned not hand-coded)

```
monitoring/grafana/datasources/prometheus.yaml  → auto-registers Prometheus datasource
monitoring/grafana/dashboards/dashboards.yaml    → provider config (folder scan)
monitoring/grafana/dashboards/fraud_detection.json → dashboard panel definitions
```

Served at `localhost:3000` (container `grafana`, see `docker-compose.yml`),
admin credentials via `GF_SECURITY_ADMIN_PASSWORD` env var.

## MLflow UI

Local-only (not containerized, per `.ai/rules.md`): `make mlflow` →
`mlflow ui --backend-store-uri ./mlruns --port 5000`.

## Notebooks (exploratory, not shipped UI)

`notebooks/01_eda.ipynb`, `02_imbalance_ablation.ipynb`, `03_model_comparison.ipynb`.

## Future Scope

Phase 5 API responses (`POST /predict`) are consumed programmatically —
no dashboard/frontend planned for prediction results beyond Grafana metrics.
