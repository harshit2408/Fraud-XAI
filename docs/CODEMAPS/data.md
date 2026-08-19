<!-- Generated: 2026-08-05 | Files scanned: 8 | Token estimate: ~550 -->

# Data — IEEE-CIS Fraud Detection Dataset

No SQL database. Data lives as flat files (CSV → Parquet), config-driven via
`config/config.yaml`. "Migration history" = the preprocessing pipeline stages below.

## Raw Files (`data/raw/`, from Kaggle via `download_data.py`)

| File | Role |
|------|------|
| `train_transaction.csv` (~683MB) | Transaction-level features, target `isFraud` |
| `train_identity.csv` | Identity/device features, joined to transactions |
| `test_transaction.csv` / `test_identity.csv` | Held-out (no label used in training) |
| `sample_submission.csv` | Kaggle submission template (unused in pipeline) |

`download_data.py`: checks Kaggle API creds → downloads → extracts zip →
computes checksums for integrity verification.

## Processed Files (`data/processed/`, from `preprocess.py`)

```
train_features.parquet (~149MB) / train_labels.parquet
val_features.parquet (~24MB)    / val_labels.parquet
test_features.parquet (~47MB)   / test_labels.parquet
processed/transformers/  → fitted encoders (target encoding, hashing) via
                            FeatureEngineer.save_transformers/load_transformers
```

## Pipeline Stages (relationships / lineage)

```
raw CSVs
  → data_loader.DataLoader.load_raw()      join transaction+identity, validate_schema
  → data_loader.sort_temporal()             sort by TransactionDT (no shuffling — see rules.md)
  → feature_engineering.FeatureEngineer     20+ transform methods (temporal, amount,
                                             card aggregates, email, D/C columns, card
                                             hashing, velocity, null counts, address,
                                             device, target encoding, categorical encode,
                                             V-feature PCA reduction, missing-value handling)
  → data_splitter.time_based_split_3way()   70/10/20 time-ordered split (train_split_ratio,
                                             val_split_ratio in config) — NO random shuffle
  → imbalance_handler.ImbalanceHandler       SMOTE / class_weight / focal_loss strategy
                                             (config: imbalance.strategy)
  → [TFT only] sequence_builder.SequenceBuilder.build_sequences()
                                             flat rows → per-card sequences,
                                             sequence_length=10 (config)
  → data/processed/*.parquet
```

## Key Config-Driven Parameters (`config/config.yaml`)

- `data.temporal_col: TransactionDT` — strict time-based ordering, prevents leakage
- `data.sequence_length: 10` — TFT lookback window per card
- `features.v_features_pca_components: 30` — PCA compression of V1-V339
- `imbalance.strategy: smote` (alt: class_weight, focal_loss, none)
- `thresholds.cost_fn/cost_fp/revenue_tp` — business cost model for threshold tuning

## Models Directory (`models/`, binary artifacts — not source)

`xgb_model.pkl` (~14MB), `lgbm_model.pkl` (~41KB), `tft_model.pt` (~224KB).

## Staleness Note

`reports/RESULTS.md` and `reports/phase2_results.json` hold point-in-time
eval results — verify against latest `mlruns/` before citing numbers.
