# TDD Evidence — Phase B1–B3

**Source plan:** [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (Phase B — Fix TFT Training Correctness)

**Date:** 2026-08-07  
**Env:** `conda run -n fraudx python -m pytest …`

## User journeys

1. As an ML engineer, I want imbalance sampling and loss chosen by explicit config keys that fail fast on typos, so focal loss is reachable and misconfiguration cannot silently fall through to WeightedBCE.
2. As an ML engineer, I want exactly one imbalance-correction mechanism active at a time, so effective positive weighting stays ~27× (not ~729×).
3. As an ML engineer, I want every training entry point to call `set_seed(config.project.random_seed)` and log that seed to MLflow, so two consecutive runs can reproduce val metrics.

## Task report

| Task | Summary | Validation | RED → GREEN |
|------|---------|------------|-------------|
| B1 | Split `imbalance.strategy` → `sampling_strategy` + `loss_function`; `resolve_imbalance_config` fails fast | `pytest tests/unit/test_imbalance_config.py tests/unit/test_config.py` | Config/tests already present; all PASS |
| B2 | When sampler is on, `weighted_bce` forces `pos_weight=1.0` | `TestNoDoubleCorrection` | Already implemented; PASS (`pos_weight ≈ 1.0`, not ~729) |
| B3 | `set_seed()` utility + wire 5 entry points + MLflow `random_seed` | `pytest tests/unit/test_seed.py tests/unit/test_training_entry_point_seeding.py` | Entry-point suite was RED (17 fail); after wiring → 25 PASS |

### Commands actually run

```text
# RED (entry points only, before wiring)
conda run -n fraudx python -m pytest tests/unit/test_imbalance_config.py \
  tests/unit/test_seed.py tests/unit/test_training_entry_point_seeding.py -v --tb=short
→ 17 failed, 29 passed

# GREEN (after wiring set_seed + MLflow seed log)
conda run -n fraudx python -m pytest tests/unit/test_imbalance_config.py \
  tests/unit/test_seed.py tests/unit/test_training_entry_point_seeding.py \
  tests/unit/test_config.py -v --tb=short
→ 60 passed

# Broader regression
conda run -n fraudx python -m pytest tests/unit/ -q --tb=line
→ 132 passed
```

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|--------------------|------|------|--------|
| 1 | Unknown `sampling_strategy` / `loss_function` raise `ValueError` | `test_imbalance_config.py::TestFailFastOnUnknownValues` | unit | PASS |
| 2 | Missing imbalance keys fail loudly (no silent default) | `test_missing_keys_raise_rather_than_silently_defaulting` | unit | PASS |
| 3 | `loss_function=focal_loss` returns `FocalLoss` | `TestFocalLossIsReachable` | unit | PASS |
| 4 | `oversample` + `weighted_bce` → `pos_weight=1.0` (no double correction) | `test_oversample_with_weighted_bce_does_not_double_correct` | unit | PASS |
| 5 | Shipped config has no `imbalance.strategy`; valid split keys; not oversample+weighted_bce | `test_config.py` Phase B1 tests | unit | PASS |
| 6 | `set_seed` makes random/numpy/torch (and TFT weight init) reproducible | `test_seed.py` | unit | PASS |
| 7 | All 5 training entry points import/call `set_seed` early from `project.random_seed` | `test_training_entry_point_seeding.py` | unit | PASS |
| 8 | Each entry point logs `random_seed` to MLflow | `test_main_logs_seed_to_mlflow` | unit | PASS |

## Coverage and known gaps

- Full end-to-end “two consecutive TFT runs → identical val PR-AUC” is **not** executed here (needs parquet + GPU/CPU training time). Unit proxy: identical TFT weight init under the same seed (`test_model_weight_initialization_is_reproducible`).
- Optuna samplers remain unseeded (Phase **D2**); B3 only seeds process RNGs and logs the seed.
- `seed_worker` exists for DataLoader multi-worker use; TFT currently uses `num_workers=0` (Windows), so worker init is unused in production path today.
- `ImbalanceHandler` / tree-model `scale_pos_weight` path unchanged (Phase C/E follow-ups).

## Acceptance vs plan

| Done when (plan) | Status |
|------------------|--------|
| B1: Focal loss reachable and covered by a test | Met |
| B2: Effective positive weighting ~27×, not ~729× | Met (sampler XOR count-based weight) |
| B3: `set_seed` at every training entry point; seed logged to MLflow | Met (5/5 entry points) |

## Files touched (this session)

- `src/training/train_xgb.py`, `train_lgbm.py`, `train_tft.py`, `tune_xgb.py`, `tune_tft.py` — import/`set_seed` early in `main`, log seed to MLflow
- `tests/unit/test_training_entry_point_seeding.py` — added `test_main_logs_seed_to_mlflow`
- Pre-existing (already green before this session’s GREEN fix): `config/config.yaml`, `src/utils/seed.py`, `resolve_imbalance_config`, `test_imbalance_config.py`, `test_seed.py`, `test_config.py` B1 assertions
