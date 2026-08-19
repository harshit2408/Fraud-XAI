# TDD Evidence Report — Phase B4–B7 (TFT Training Correctness)

**Scope:** `docs/IMPLEMENTATION_PLAN.md` Phase B, tasks B4–B7 (B1–B3 covered
separately in `docs/testing/phase-b1-b3.tdd.md`).

| # | Task | Done when (plan) | Status |
|---|------|-------------------|--------|
| B4 | Fit a scaler (QuantileTransformer) on train; persist with checkpoint; apply in `SequenceBuilder` | Inputs are unit-scale; re-enable AMP and confirm no NaN | ✅ implemented + confirmed on real CUDA hardware (RTX 4060) |
| B5 | Add embeddings for categorical inputs to the neural path | No raw label codes enter the network as continuous | ✅ |
| B6 | Build sequences once over the full ordered frame; assign to splits by target-row index | Boundary cards retain real history | ✅ |
| B7 | Decouple labels from sequence construction; `predict_proba(X)` takes no `y` | Signature has no label parameter | ✅ |

## RED → GREEN cycle

1. **RED**: Added 22 new tests across `tests/unit/test_sequence_builder.py`
   (`TestLabelDecoupling`, `TestFeatureScaling`), `tests/unit/test_tft_trainer.py`
   (`TestCategoricalEmbeddings`), and a new `tests/unit/test_tft_boundary_history.py`
   (`TestBoundaryHistoryPreserved`, `TestPredictProbaSignature`). Confirmed
   all 21 introduced assertions failed against the pre-change code
   (1 test, `test_without_boundary_fix_first_val_row_would_be_padded`, was
   intentionally written to pass pre-change as a sanity check that the
   boundary-fix test harness itself is meaningful).
2. **GREEN**: Implemented the changes described below; all 22 new tests plus
   the full pre-existing suite pass.
3. **REFACTOR**: Consolidated train/val sequence building in `TFTTrainer.train()`
   onto the new `_build_sequences_for_splits` helper; kept `predict_proba`'s
   public contract backward-compatible (no required `history_X`).

Final run: `conda run -n fraudx python -m pytest tests/unit/ -q`
→ **155 passed**, 0 failed (40 warnings, third-party deprecations only).

## What changed

### B4 — Feature scaling (`src/data/sequence_builder.py`)
- `SequenceBuilder.build_sequences(X, y=None, fit_scaler=False)` now fits a
  `sklearn.preprocessing.QuantileTransformer(output_distribution="normal")`
  on the numeric time-varying features when `fit_scaler=True`, and stores it
  on `self.scaler`. Subsequent calls (any `fit_scaler` value) reuse the
  stored scaler to transform — never refit on val/test/inference data.
- Default behavior (`fit_scaler=False`, no prior fit) is unchanged — scaling
  is strictly opt-in, so all pre-existing tests/call sites are unaffected.
- `TFTTrainer._build_sequences_for_splits(..., fit_scaler=True)` is called
  from `train()` with train as the first split, so the scaler is fit once
  on train only.
- `TFTTrainer.save()`/`load()` persist `sequence_builder.scaler` inside the
  checkpoint (`torch.save` pickles the sklearn object transparently —
  verified via an end-to-end smoke test) so inference transforms exactly as
  training did.
- **AMP**: `model.tft.use_amp` (default `true` in `config/config.yaml`) is
  re-enabled, gated on `self.device.type == "cuda"`. It was previously
  hard-disabled to work around a NaN issue plausibly caused by unscaled
  -999 imputation sentinels dominating the gradient signal; scaling now
  bounds those sentinels into the normal-ish distribution range (unit test:
  `test_sentinel_values_are_not_extreme_after_scaling`).

### B5 — Categorical embeddings (`src/models/tft_model.py`)
- `TemporalFusionTransformer` accepts `static_categorical_indices`,
  `static_cardinalities`, `categorical_embedding_dim` (default 8). When
  provided, an `nn.ModuleList` of `nn.Embedding(cardinality, dim)` replaces
  the raw integer codes at those static-feature slots before they reach the
  (now correctly re-sized) `static_embedding` Linear layer; continuous slots
  pass through unchanged. Omitting these args reproduces the original
  all-continuous behavior exactly (verified: `test_backward_compatible_without_categorical_config`).
- `SequenceBuilder.CATEGORICAL_STATIC_COLS = {"ProductCD", "card4", "card6", "DeviceType"}`
  identifies which static columns are label-encoded categoricals (binary
  flags like `P_email_is_free` stay continuous). `static_categorical_indices`
  / `static_cardinalities` properties expose this to the trainer.
  Cardinalities are computed once (from the first build that sees them,
  i.e. train) and reused, so embedding table sizes stay fixed across splits.
- `TFTTrainer.build_model()` forwards these to the model; `save()`/`load()`
  persist them in `model_config` so a loaded checkpoint reconstructs
  identically-sized embedding tables.

### B6 — Boundary history (`src/training/train_tft.py`)
- New `TFTTrainer._build_sequences_for_splits(splits, fit_scaler=False)`:
  concatenates the given `(name, X, y)` splits in temporal order, builds
  sequences **once** over the combined frame, then partitions the result
  back out per split using `original_indices` — so a card's first
  transaction in a later split is built from its real preceding
  transactions instead of being padded at the split boundary.
- `TFTTrainer.train()` now uses this for train+val (previously two
  independent `_build_sequences` calls, which truncated val's history).
- `predict_proba(X, history_X=None)` accepts optional preceding transactions
  (Phase B7 signature — see below) so `main()`'s evaluation and the
  ensemble/scratch eval scripts can pass true prior context
  (`history_X=X_train` for val, `history_X=concat(X_train, X_val)` for test)
  without threading labels through.

### B7 — Decoupled labels (`src/data/sequence_builder.py`, `src/training/train_tft.py`)
- `SequenceBuilder.build_sequences(X, y=None, ...)` — `y` is optional;
  omitting it returns `targets=None`. New `attach_targets(seq_data, y)`
  attaches labels afterward for training/evaluation, matching by
  `original_indices` (sequences are built per card group, not in row order —
  a naive positional attach would be wrong; regression-guarded by
  `test_attach_targets_uses_original_indices_not_positional_order`).
- `TFTTrainer.predict_proba(X, history_X=None)` — no `y` parameter.
  `inspect.signature` is asserted directly in
  `test_predict_proba_has_no_required_label_parameter` so this can't
  silently regress.
- Updated all call sites: `train_tft.py:main()`, `scripts/run_ensemble_eval.py`,
  `scratch_eval.py`.

### Bonus fix found via smoke testing (not in the original plan, but a real bug)
- `SequenceBuilder._identify_features()` excluded the literal column name
  `"isFraud"` but **not** `"__target__"`, the internal column `build_sequences`
  adds to merge `X`/`y` for grouping. Since `"isFraud"` is already dropped
  from `X` upstream, this exclusion was dead code — meaning **the label
  itself was silently fed into the model as a numeric feature whenever `y`
  was passed** (i.e. in every training run to date). Fixed by excluding
  `"__target__"` too; regression-guarded by
  `test_target_column_never_leaks_into_numeric_features`, which also checks
  that `feature_dim` is identical whether or not `y` is passed. This was
  caught by an end-to-end smoke test (train → predict_proba(history_X=...))
  that hit a shape mismatch, not by the isolated unit tests, which
  underscores the value of the smoke test.

## Test files

- `tests/unit/test_sequence_builder.py` — 25 tests (was 12; +13: label
  decoupling, scaling, leakage regression)
- `tests/unit/test_tft_trainer.py` — 23 tests (was 19; +4: categorical
  embeddings)
- `tests/unit/test_tft_boundary_history.py` — new file, 9 tests (B6 boundary
  history + B7 `predict_proba` contract)

## Verification performed

- `conda run -n fraudx python -m pytest tests/unit/ -q` → 155 passed.
- End-to-end smoke test (train → save → load → predict, with real
  categorical columns and scaler fitting) run manually to validate the
  full integration path beyond what unit tests isolate; confirmed:
  - predictions are finite and in `[0, 1]`,
  - save/load roundtrip reproduces identical predictions (`atol=1e-5`),
  - the scaler and categorical cardinalities are correctly restored from
    the checkpoint.
- **CUDA + AMP verification**: this environment does have a GPU
  (`nvidia-smi` → NVIDIA GeForce RTX 4060 Laptop GPU, driver 592.82;
  `torch.cuda.is_available()` → `True`, torch 2.2.0+cu121) — an earlier
  version of this report incorrectly claimed otherwise without checking.
  Ran a second, larger smoke test on the actual GPU: 8,000 synthetic rows
  across 200 cards with ~5% of `TransactionAmt`/`pca_v_1` values set to the
  `-999` imputation sentinel (mimicking real preprocessing), oversampling +
  focal loss enabled, `device: cuda`, `use_amp: true`, 5 epochs. Confirmed
  `Device: cuda, AMP: True` in the training log, zero NaN across all 5
  epochs' train/val loss, and finite predictions on a held-out test split.
  This directly confirms the plan's "re-enable AMP and confirm no NaN"
  acceptance criterion on real hardware, not just on CPU/synthetic-shape
  unit tests.

## Known limitations / not fully verified in this session

- **B6 full production benefit**: `_build_sequences_for_splits` fixes the
  boundary for train/val (used inside `train()`) and `predict_proba`'s
  optional `history_X` fixes it for val/test evaluation in `main()`/eval
  scripts. A live model was not retrained on the real Kaggle dataset in
  this session (no GPU/data pipeline run), so the magnitude of the
  resulting PR-AUC change on real data is not measured here — only the
  mechanism's correctness on synthetic multi-row-per-card data.
