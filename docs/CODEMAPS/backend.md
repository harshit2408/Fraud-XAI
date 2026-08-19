<!-- Generated: 2026-08-05 | Files scanned: 12 | Token estimate: ~700 -->

# Backend — FastAPI Service (`src/api/`)

## Routes

```
GET /health → main.health_check → returns {status, model_loaded, uptime_seconds}
```

Phase 5 (planned, not yet implemented): `POST /predict`, `GET /metrics`
(via `prometheus-fastapi-instrumentator`). Route handlers will live in
`src/api/routes/` (currently empty `__init__.py`), schemas in `src/api/schemas/`,
cross-cutting concerns (auth/logging/rate-limit) in `src/api/middleware/`.

## App Factory

`src/api/main.py` (74 lines)
- `app = FastAPI(lifespan=lifespan)` — modern lifespan context manager, NOT
  deprecated `@app.on_event` (mandated by `.ai/rules.md`)
- Startup TODO (Phase 5): load XGBoost/TFT models + SHAP explainer into a
  ModelRegistry
- Startup TODO (Phase 6): start Kafka consumer as asyncio background task
- Shutdown TODO (Phase 6): gracefully stop Kafka consumer

## Training Pipeline (`src/training/`)

| File | Contents |
|------|----------|
| `train_xgb.py` (126 lines) | `XGBTrainer`: build/train/predict_proba/save/load XGBoost classifier |
| `train_lgbm.py` (134 lines) | `LGBMTrainer`: same interface as XGBTrainer, for LightGBM |
| `train_tft.py` (589 lines) | `TFTTrainer` + `FraudSequenceDataset`: full TFT train loop, checkpointing |
| `tune_xgb.py` (63 lines) | Optuna hyperparam search for XGBoost, `objective()` per trial |
| `tune_tft.py` (97 lines) | Optuna search for TFT with card-ID subsampling + pruning |
| `losses.py` (102 lines) | `FocalLoss`, `WeightedBCELoss` — for imbalanced-class NN training |

Model → Repository style: each `*Trainer` class encapsulates `build_model` →
`train` → `predict_proba` → `save`/`load(cls, path)` (classmethod loader).
All read hyperparams from `config/config.yaml` via `load_config()`.

## Data Pipeline (`src/data/`)

```
download_data.py → data_loader.py → feature_engineering.py → imbalance_handler.py
                                              │
                                              ▼
                                     sequence_builder.py (TFT only)
                                              │
                                              ▼
                                       data_splitter.py
```

Orchestrated end-to-end by `preprocess.py::run_pipeline(config)`.

## Models (`src/models/`)

- `tft_model.py` (531 lines) — custom TFT implementation: `GatedLinearUnit`,
  `GatedResidualNetwork`, `VariableSelectionNetwork`,
  `InterpretableMultiHeadAttention`, `TemporalFusionTransformer`
- `ensemble.py` (114 lines) — `ModelEnsemble`: weighted blend of XGB + TFT
  probabilities, `find_optimal_weights()` grid search over PR-AUC

## Evaluation (`src/evaluation/evaluator.py`, 183 lines)

`ModelEvaluator`: PR-AUC, ROC-AUC, `find_optimal_threshold` (cost-based, uses
`thresholds.cost_fn/cost_fp/revenue_tp` from config), confusion matrix +
PR/ROC curve plotting, business-value-vs-threshold plot, classification report.

## Streaming (`src/streaming/producer.py`, 45 lines)

One-shot CLI (`make stream`): reads test transactions, publishes to Kafka
topic `transactions` at a configurable rate (`--rate`, `--limit`). Not a
long-running service/container.

## Middleware Chain

None implemented yet — `src/api/middleware/` is a scaffolded empty package.
