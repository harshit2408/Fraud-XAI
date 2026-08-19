# Implementation Plan — Code Quality, ML Correctness & Architecture Remediation

<!-- Generated: 2026-08-05 | Reviewers: code-reviewer, mle-reviewer, architect | Files reviewed: 30 src + 3 scripts + 3 notebooks + 12 tests -->

**Scope:** Full quality and correctness audit of the ML pipeline (data loading, EDA,
feature engineering, model training, hyperparameters, evaluation) combined with the
architecture assessment, converted into a prioritized, actionable plan.

**Baseline verification run:** `python -m pytest tests/ -q` → **50 passed in 10.87s**
(run in the `fraudx` conda env; the default `python` on PATH is 3.14 and cannot
install this project's pinned dependencies).

**Status as of 2026-08-07 (post-Phase A):** the CRITICAL finding below is resolved —
`reports/RESULTS.md` and the Phase A section have the details and the re-baselined
number. Current test count is **77 passed** (A6 and A7 added regression coverage).
The Executive Summary, findings, and Review Summary immediately below are left as
originally written to preserve the audit record; treat the Phase A/B/C/D/E/F sections
further down as the live status.

---

## Executive Summary

| Area | Verdict | Headline |
|------|---------|----------|
| Data loading | Good | Memory-aware PyArrow loader, explicit left-join semantics, schema validation |
| Time-based splitting | Good | Genuinely leak-free; hard-fails on missing temporal column |
| Feature engineering *logic* | Good | Card aggregates and target encoding are correctly point-in-time (`shift`/`cumsum` exclusion) |
| Feature engineering *sequencing* | **BLOCK** | Every transformer is fitted on train+val+test **before** the split |
| EDA | **Weak** | Notebooks 01 and 02 have never been executed — zero outputs; conclusions asserted without evidence |
| Hyperparameters | **Warn** | Config diverges from the documented tuning result; tuning is unseeded and its pruner is a no-op |
| XGBoost training | Warn | Sound overall, but the decision threshold is selected on the test set |
| TFT training | **BLOCK** | Double imbalance correction, focal loss silently disabled, no feature scaling, no seeds |
| Evaluation | Warn | Threshold search hits its own grid boundary; two conflicting business objectives; no calibration |
| Serving readiness | Warn | 6 model features cannot be reproduced at inference; `predict_proba` requires labels |

**Overall decision (mle-reviewer): BLOCK**
**Primary risks:** data leakage (transformer fitting), irreproducible training (no seeds,
config/report divergence), weak eval (threshold selected on test), unsafe serving
(feature parity gap).

Two important pieces of context before the findings:

1. Most of these are **pre-production issues in a project that is honestly at Phase 2–3**.
   Nothing here is a live incident. The point is to fix them before Phases 4–7 build on top.
2. The leakage found is mostly *unsupervised* (PCA, frequency maps, imputation constants)
   rather than label leakage. Expect the corrected test PR-AUC to move modestly, not
   collapse. The reason to fix it is methodological defensibility — this project's stated
   selling point is regulatory-grade rigor.

---

## Part 1 — Findings

### CRITICAL

```
[CRITICAL] All feature transformers are fitted on the full dataset before the split
File: src/data/preprocess.py:75, 108, 117, 126
Issue: The pipeline fits every stateful transformer on the complete DataFrame and only
       splits afterwards at line 145. Specifically:
         L75  fe.reduce_v_features(df, fit=True)      → IncrementalPCA fitted on test rows
         L108 fe.create_card_hash_features(df, fit=True) → value_counts over test rows
         L117 fe.create_target_encoding(df)            → global prior = df["isFraud"].mean()
                                                          over test LABELS (feature_engineering.py:483)
         L126 fe.encode_categoricals(df, fit=True)     → frequency maps over test rows
       Failure mode: test-set distribution (and, for the target-encoding prior, test
       labels) informs the feature representation the model trains on. Reported test
       PR-AUC of 0.5678 is therefore not a clean held-out estimate.
       Existing guards do not catch this: each method correctly supports fit=False, and
       the docstrings state "Never fit on test data" — the orchestrator simply never
       uses the fit=False path.
Fix: Split first, then fit on train only and transform val/test with fit=False.
     Requires restructuring run_pipeline() so the split happens immediately after
     sort_temporal(), with per-split transform calls afterwards.
     Note: the PCA-before-sort ordering exists for a real memory reason (764MB → 220MB
     sort buffer). Preserve that by fitting IncrementalPCA on the train slice only and
     transforming the rest in batches, not by reverting the memory optimization.
```

### HIGH

```
[HIGH] TFT applies two independent imbalance corrections simultaneously
File: src/training/train_tft.py:254 and 259-271
Issue: _create_dataloader(..., oversample=True) installs a WeightedRandomSampler that
       rebalances the training stream to ~50/50. Independently, the loss branch computes
       pos_weight = neg/pos ≈ 27 from the PRE-sampling distribution and applies
       WeightedBCELoss. The minority class is therefore up-weighted twice (~27x on top
       of ~27x effective resampling).
       Trigger: any run with the shipped config. Outcome: the model is pushed to predict
       fraud far too aggressively, and output probabilities are badly miscalibrated —
       which then feeds the cost-based threshold logic that assumes calibrated scores.
Fix: Choose one mechanism. Recommended: keep the sampler, set pos_weight=1.0; or drop
     oversample and keep pos_weight. Make it explicit in config, not implicit.
```

```
[HIGH] Focal loss is silently disabled by a config key collision
File: src/training/train_tft.py:259
Issue: loss_strategy = imb_cfg.get("strategy", "focal_loss") reads config key
       imbalance.strategy, whose shipped value is "smote" (config/config.yaml:99).
       "smote" != "focal_loss", so control always falls to the else branch.
       Consequences: (a) FocalLoss in src/training/losses.py is dead code and has never
       run in the shipped configuration; (b) focal_loss_gamma/alpha in config are inert;
       (c) the PRD's "Focal Loss" capability claim is not actually exercised.
       The failure is silent — the log line prints "Using WeightedBCELoss", which reads
       as an intentional choice rather than a fallback.
Fix: Separate the two concerns into distinct config keys, e.g.
     imbalance.sampling_strategy (smote | oversample | none) and
     imbalance.loss_function (focal | weighted_bce | bce). Fail fast on unknown values
     rather than falling through to a default.
```

```
[HIGH] Decision threshold is selected on the test set
File: src/training/train_xgb.py:207, 212-221
      scripts/run_ensemble_eval.py:97
      scripts/run_phase2_eval.py:150
      scripts/run_ablation.py:34
      notebooks/02_imbalance_ablation.ipynb (cells 4-6)
Issue: find_optimal_threshold(y_test, y_prob_test, ...) picks the operating point using
       test labels, and the headline precision/recall/F1 are then reported at that same
       threshold on the same data. The reported "81.30% recall @ 16.13% precision"
       (reports/RESULTS.md:67) is therefore an optimistically biased estimate, not an
       out-of-sample one. The validation split exists and is unused for this purpose.
Fix: Select the threshold on X_val/y_val, freeze it, then report test metrics at that
     frozen threshold. Persist the chosen threshold with the model artifact so serving
     uses the identical value.
```

```
[HIGH] No feature scaling anywhere — raw values and -999 sentinels fed into the TFT
File: src/data/feature_engineering.py:667 (sentinel), src/training/train_tft.py (no scaler)
Issue: A repo-wide search for StandardScaler/MinMaxScaler/normalize returns nothing.
       The TFT therefore consumes, in the same input vector: TransactionAmt in dollars,
       tx_sum_per_card (cumulative, can reach 10^4-10^5), PCA components (~unit scale),
       binary int8 flags, and -999 imputation sentinels.
       Trigger: every TFT training run. Outcome: the -999 sentinels alone dominate the
       gradient signal through the GatedResidualNetwork/LayerNorm stack; this is a
       plausible root cause of the NaN issue that AMP was disabled to work around
       (train_tft.py:290-292).
       Tree models are scale-invariant so XGBoost is unaffected — which is exactly why
       this went unnoticed.
Fix: Fit a StandardScaler (or QuantileTransformer, which is more robust to the -999
     sentinels) on train only, persist it alongside the TFT checkpoint, and apply it in
     SequenceBuilder. Consider a separate is_missing indicator channel instead of -999
     for the neural path.
```

```
[HIGH] TFT training is not reproducible — no seeds are set
File: src/training/train_tft.py (entire module)
Issue: No torch.manual_seed, torch.cuda.manual_seed_all, or np.random.seed anywhere in
       the repo (verified by search). Unseeded sources: model weight init
       (_init_weights, tft_model.py:405), dropout masks, WeightedRandomSampler draws,
       and DataLoader shuffling. config.project.random_seed = 42 is honoured only by
       XGBoost/LightGBM random_state.
       Outcome: two runs of `make train` produce different TFT models and different
       reported metrics; a regression cannot be distinguished from run-to-run variance.
Fix: Add a set_seed(seed) utility called at the top of every training entry point,
     covering random, numpy, torch, torch.cuda, and DataLoader worker seeding. Log the
     seed to MLflow. Document that full CUDA determinism additionally requires
     torch.use_deterministic_algorithms(True).
```

```
[HIGH] Six model features cannot be reproduced at inference time
File: src/data/feature_engineering.py:469-506 (create_target_encoding),
      src/data/feature_engineering.py:683-705 (save_transformers)
Issue: create_target_encoding produces card1/card2/addr1/P_emaildomain/R_emaildomain/
       device_brand _target_enc — six features the trained model depends on. Neither the
       per-category encoding maps nor _global_target_mean are written by
       save_transformers(), which persists only label encoders, freq encoders, PCA,
       imputer values, and card-hash frequencies.
       Trigger: Phase 5 serving. Outcome: the inference path cannot construct these
       columns at all; XGBTrainer.predict_proba would raise "Missing columns in input"
       (train_xgb.py:78) — the loud failure is fortunate, but the feature set is simply
       unavailable.
       Additionally, create_target_encoding has no fit/transform separation at all
       (unlike its sibling methods), so there is no mechanism to apply it in transform-only
       mode even if the state were saved.
Fix: Give create_target_encoding a fit flag; persist the final per-category
     (cum_sum, cum_count) state and the global prior; add them to save/load_transformers.
     For streaming, these become running per-entity counters — see Phase E / ADR-002.
```

```
[HIGH] TFTTrainer.predict_proba requires ground-truth labels
File: src/training/train_tft.py:469 and src/data/sequence_builder.py:149
Issue: The signature is predict_proba(self, X, y) and y is passed straight into
       build_sequences, which does df["__target__"] = y.values. Labels are not available
       at prediction time in production — this is the textbook "feature generation uses
       fields unavailable at prediction time" smell. The labels are only used to build
       the returned targets array (not as model input), so the leakage is latent rather
       than active, but the contract is wrong and invites a real leak the moment someone
       wires this into serving.
Fix: Split sequence construction from label attachment. build_sequences(X) should return
     sequences/static/mask/indices; a separate attach_targets(y) is used only in training
     and evaluation. predict_proba should take X alone.
```

```
[HIGH] Config hyperparameters do not match the documented tuning result
File: config/config.yaml:56-63 vs reports/RESULTS.md:45-50
Issue: RESULTS.md documents the Optuna best trial as n_estimators=897, lr=0.0893,
       subsample=0.8655, colsample_bytree=0.911, min_child_weight=7. The shipped config
       has 887 / 0.09696 / 0.8382 / 0.6186 / 4. Every value differs; colsample_bytree
       differs by 32%. There is no run ID, git SHA, or dataset hash linking either set to
       an actual experiment.
       Root cause: tune_xgb.py:112 ends by telling a human to "Update config/config.yaml
       with these parameters" — a manual promotion step with no provenance record.
       Outcome: the model in models/xgb_model.pkl cannot be traced to a tuning run, and
       the results report describes a model that was never shipped.
Fix: Have tune_xgb.py write the winning params to a versioned artifact
     (e.g. config/tuned/xgb_<run_id>.yaml) and record the MLflow run ID; make the
     training script read the tuned file by reference. Regenerate RESULTS.md from the
     actual promoted run.
```

```
[HIGH] catboost is imported but is not a declared dependency
File: scripts/run_phase2_eval.py:30
Issue: `import catboost as cb` with run_cb_experiment() at line 117, but catboost is
       absent from requirements.txt. The stray catboost_info/ directory at the repo root
       confirms it has been run locally.
       Trigger: a clean `pip install -r requirements.txt` followed by
       `python scripts/run_phase2_eval.py` → ImportError. The Docker image would fail the
       same way.
Fix: Either add a pinned catboost to requirements.txt, or remove the CatBoost arm from
     the script if it was exploratory only. Add a CI smoke step that imports every
     top-level module against a clean env.
```

### MEDIUM

```
[MEDIUM] Optuna study is unseeded and its pruner never fires
File: src/training/tune_xgb.py:89-93
Issue: Two separate defects. (a) create_study() uses the default TPESampler with no seed,
       so the search path is not reproducible. (b) MedianPruner(n_warmup_steps=10) is
       configured, but objective() never calls trial.report() or trial.should_prune() —
       pruning is therefore impossible. Contrast with tune_tft.py, which does report
       correctly via train_tft.py:376-380. The configured pruner gives a false impression
       that compute is being saved.
Fix: sampler=optuna.samplers.TPESampler(seed=config["project"]["random_seed"]); add
     per-boosting-round reporting via an XGBoost callback, or drop the pruner argument.
```

```
[MEDIUM] Cost-optimal thresholding is applied to uncalibrated probabilities
File: src/evaluation/evaluator.py:28-51, src/training/train_xgb.py:207
Issue: The cost model (500 FN / 5 FP) is only valid if p(fraud|x) is calibrated —
       expected cost is computed as a probability-weighted sum. With scale_pos_weight=29
       (XGBoost) and double correction (TFT), the outputs are systematically distorted.
       No calibration step (CalibratedClassifierCV, isotonic, Platt) exists anywhere, and
       no calibration metric (Brier score, reliability curve) is computed.
       This is the mechanism behind the next finding.
Fix: Fit isotonic or sigmoid calibration on the validation split after training; report
     Brier score and a reliability curve alongside PR-AUC; derive the threshold from the
     calibrated probabilities.
```

```
[MEDIUM] Threshold search returns a boundary value, and two different business
         objectives are in use
File: src/evaluation/evaluator.py:37 vs :137-157
Issue: (a) find_optimal_threshold scans np.linspace(0.01, 0.99, 99). The reported optimum
       in RESULTS.md:67 is exactly 0.0100 — the first grid point. When an optimizer
       returns its own boundary, the true optimum is outside the search space; the search
       is clipped, not converged.
       (b) find_optimal_threshold minimizes (FN*cost_fn + FP*cost_fp) and ignores
       revenue_tp, while plot_threshold_vs_business_value maximizes
       (TP*revenue_tp - FP*cost_fp - FN*cost_fn). The two can select different thresholds,
       so the vertical "optimal threshold" line in the generated chart need not match the
       threshold the model actually uses.
Fix: Extend the grid downward (log-spaced from ~1e-4) or derive the threshold
     analytically from the cost ratio on calibrated probabilities; unify both call sites
     on a single business-value function.
```

```
[MEDIUM] Validation and test sequences are built without cross-split card history
File: src/training/train_tft.py:249-250
Issue: _build_sequences is called independently per split, so a card's first transactions
       in val/test get zero-padded sequences even though genuine history exists at the end
       of the train split. This affects a meaningful share of the 10-step windows at each
       boundary and creates a train/serve mismatch: in production, history is always
       available.
Fix: Build sequences over the temporally ordered full frame once, then assign each
     sequence to a split by the index of its target row. History may cross the boundary
     backwards (past → present) without leakage; only the target row's split membership
     matters.
```

```
[MEDIUM] Label-encoded categorical integers are fed to the neural network as continuous
File: src/data/sequence_builder.py:89-109
Issue: static_candidates includes ProductCD, card4, card6, DeviceType, which
       encode_categoricals has already converted to arbitrary integer codes. These are
       consumed as float32 (line 169), imposing a false ordinal relationship
       (e.g. visa=3 > discover=1) on the network. Trees split on these fine; an MLP/GRN
       treats the code as a magnitude.
Fix: Add nn.Embedding layers for categorical inputs in TemporalFusionTransformer, or
     one-hot encode the low-cardinality set for the neural path specifically.
```

```
[MEDIUM] ImbalanceHandler is dead code and config.imbalance.strategy is misleading
File: src/data/imbalance_handler.py (whole module), src/training/train_xgb.py:140-142, 157
Issue: train_xgb.py computes scale_pos_weight inline and hardcodes
       mlflow.log_param("imbalance_strategy", "scale_pos_weight") at line 157, while
       config declares strategy: "smote". No training script instantiates
       ImbalanceHandler. So the config value is inert for XGBoost, means "not focal loss"
       for the TFT, and the module itself is only exercised by its unit tests.
       Also: SMOTE(random_state=42) at imbalance_handler.py:76 hardcodes the seed instead
       of reading config.project.random_seed, contrary to the project rule that nothing
       is hardcoded.
Fix: Either route both trainers through ImbalanceHandler and honour the config value, or
     delete the module and remove the config block. Do not leave a config key that
     describes behaviour the code does not implement.
```

```
[MEDIUM] V-feature missingness signal is destroyed before null-count features are computed
File: src/data/preprocess.py:75 then :84
Issue: reduce_v_features runs first and converts V1-V339 NaNs to 0.0
       (feature_engineering.py:601, na_value=0.0) before dropping the columns.
       create_null_count_features then counts NaNs over the remaining ~125 columns, so
       null_count_total excludes all V-column missingness — which the EDA notebook itself
       identifies as a strong fraud signal. The memory optimization silently degraded a
       documented feature.
       Secondary: null_ratio divides by df.shape[1] - 1 (line 388), a value that changes
       with pipeline ordering, so the feature's meaning is position-dependent.
Fix: Compute the raw null-count features immediately after load_raw(), before any
     transformation. Use a fixed denominator constant for null_ratio.
```

```
[MEDIUM] build_sequences materializes ~3 GB, and the "efficient" variant is not
File: src/data/sequence_builder.py:207-234 and :287-291
Issue: For 590k rows × seq_len 10 × ~125 features × 4 bytes the output array is ~2.9 GB —
       and build_sequences first accumulates that in a Python list of per-row arrays
       before np.array() copies it, briefly doubling to ~6 GB.
       build_sequences_efficient pre-allocates the same full (N, 10, F) array at line 287,
       so it removes the doubling but is not meaningfully "memory-efficient"; its name
       overpromises. The per-row Python loop is also the dominant runtime cost.
Fix: Replace materialization with lazy windowing in the Dataset (__getitem__ slices from
     the flat array using precomputed per-group index lists), or use stride tricks. Rename
     the second method to reflect what it actually does.
```

```
[MEDIUM] Unsafe artifact deserialization on the serving path
File: src/training/train_tft.py:545, src/training/train_xgb.py:103,
      src/data/feature_engineering.py:717-738
Issue: torch.load(..., weights_only=False) explicitly opts out of the safe loader, and
       both the XGB model and all five transformer files are raw pickle. These artifacts
       are loaded inside the fraud-api container. Pickle/torch.load with weights_only=False
       execute arbitrary code on load; there is no checksum or signature on any artifact.
       Severity is MEDIUM rather than HIGH because the artifacts are currently
       self-generated and gitignored, not fetched from a registry — but Phase 5 puts them
       on the container's startup path.
Fix: Persist XGBoost via model.save_model() (JSON/UBJ, no pickle) and transformers via
     joblib with a recorded SHA-256; keep torch.load's weights_only default and save only
     the state_dict. Verify the checksum at load time.
```

```
[MEDIUM] Unseen categorical values crash at inference
File: src/data/feature_engineering.py:552-555
Issue: Unknown labels are mapped to the literal "MISSING" and then passed to
       le.transform(). If a given column had zero NaNs in the training data, "MISSING" is
       not in le.classes_ and transform raises
       ValueError: y contains previously unseen labels: 'MISSING'.
       Trigger: any inference request with a novel category on such a column. The guard at
       line 553-554 looks like it handles this case but does not.
Fix: Reserve "MISSING" in the encoder vocabulary at fit time
     (fit on df[col].tolist() + ["MISSING"]), or replace LabelEncoder with an explicit
     dict mapping that has a defined default index.
```

```
[MEDIUM] load_config duplicated verbatim across six modules, with no validation
File: src/data/preprocess.py:45, src/training/train_xgb.py:113, train_lgbm.py:121,
      train_tft.py:577, tune_xgb.py:28, tune_tft.py:35
Issue: The identical three-line yaml.safe_load helper is copy-pasted six times. There is
       no src/config.py and no schema validation, despite pydantic 2.6 already being a
       dependency. A missing or mistyped config key surfaces as a KeyError deep inside a
       long-running training job rather than at startup.
Fix: Extract src/config.py exposing a validated pydantic Settings model with a cached
     loader; update the six call sites.
```

```
[MEDIUM] device: "cuda" is hardcoded in the tuner and in shipped config
File: src/training/tune_xgb.py:71, config/config.yaml:65
Issue: tune_xgb.py sets "device": "cuda" as a literal rather than reading config,
       violating the project's no-hardcoding rule; and config.model.xgboost.device is
       "cuda" unconditionally. XGBoost 2.0.3 raises on a CPU-only host, and the fraud-api
       container has no GPU. The saved pickle also carries the device parameter into
       serving. Contrast with the TFT, which does this correctly via
       _get_device()'s "auto" mode (train_tft.py:111-127).
Fix: Add an "auto" device resolution helper shared by all trainers; default config to
     "auto"; force CPU at inference.
```

```
[MEDIUM] EDA and ablation notebooks have never been executed
File: notebooks/01_eda.ipynb (0/15 code cells with output),
      notebooks/02_imbalance_ablation.ipynb (0/7 code cells with output)
Issue: Notebook 01's header literally reads "Key Findings (filled after running)" and is
       then followed by findings, and its section 10 presents an eight-row conclusions
       table as established fact — with no computed output anywhere in the file. Two of
       those claims are also mutually inconsistent with other documents: the notebook says
       PCA "retains >85% variance" while reports/RESULTS.md:20 says "~100%".
       Notebook 02 is the sole cited evidence for the SMOTE-vs-scale_pos_weight decision
       (RESULTS.md:33-37) and its conclusion cell asserts "significantly higher PR-AUC"
       with no numbers behind it.
       By contrast notebook 03 is fully executed (11/11 cells, 203 KB of outputs) — so the
       project knows how to do this; 01 and 02 were simply never run or were stripped.
Fix: Execute both notebooks end-to-end and commit with outputs, or convert their content
     to scripts that emit artifacts into reports/. Replace asserted numbers with computed
     ones. Reconcile the PCA variance figure.
```

```
[MEDIUM] The imbalance ablation selected its winner on the test set
File: notebooks/02_imbalance_ablation.ipynb (cells 3-6), scripts/run_ablation.py:43-83
Issue: All three arms are trained on X_train and scored on X_test, and the winning
       strategy is chosen from those test scores — then that same test set is used for the
       final reported model metrics. The validation split is never loaded. The strategy
       choice is therefore contaminated.
       Secondary issues in the same experiment: the SMOTE arms pass X_test.values while
       the baseline passes a DataFrame (inconsistent feature-name handling), and
       "SMOTE + scale_pos_weight=2" uses an unexplained magic constant.
       Also note the notebook and scripts/run_ablation.py are near-duplicate
       implementations of the same experiment, which will drift.
Fix: Re-run the ablation scoring on validation; keep test untouched. Delete one of the
     two copies.
```

### LOW

```
[LOW] enable_categorical=True is a no-op
File: src/training/train_xgb.py:44
Issue: All categoricals are already numerically encoded upstream, so no column has pandas
       'category' dtype. The flag misleads a reader into thinking native categorical
       handling is active.
```

```
[LOW] Row-wise Python apply on 590k rows in two hot paths
File: src/data/feature_engineering.py:131 (_decimal_len via .apply)
      src/data/feature_engineering.py:322-325 (.agg("_".join, axis=1) then md5 .apply)
Issue: Both are per-row Python calls in an otherwise vectorized module; the md5 path also
       does a row-wise string join, the slowest common pandas anti-pattern. Measurable
       preprocessing slowdown, no correctness impact.
Fix: Vectorize the string concat with str.cat; use pd.util.hash_pandas_object instead of
     per-row md5; compute decimal length arithmetically.
```

```
[LOW] Deprecated ReduceLROnPlateau(verbose=True)
File: src/training/train_tft.py:286
Issue: verbose is deprecated in PyTorch 2.2 and emits a warning each run.
```

```
[LOW] build_sequences_efficient silently returns empty group_ids
File: src/data/sequence_builder.py:337
Issue: "group_ids": np.array([]) with a comment "Not tracked in efficient mode". A caller
       that switches methods gets an empty array rather than an error.
Fix: Raise on access or populate it; do not return a silently wrong shape.
```

```
[LOW] Unresolved NaN bug parked behind a disabled feature
File: src/training/train_tft.py:290-291
Issue: "Mixed precision (temporarily disabled to debug NaN issue)" — the root cause was
       never fixed, and the workaround costs training throughput. See the feature-scaling
       finding above for the most likely cause.
```

```
[LOW] Repo-root clutter is not gitignored
File: .gitignore
Issue: scratch_eval.py, scratch_test.py, analyze_optuna.py, optuna_analysis.txt,
       test_tuning.db, test_tuning_2.db, tft_tuning.db, and catboost_info/ all sit at the
       repo root and are not ignored. Two of the .db files are named "test_*" and will be
       collected by some pytest configurations.
Fix: Add *.db, catboost_info/, and scratch_* to .gitignore; move analyze_optuna.py into
     scripts/.
```

### Test Coverage Gaps

The 50 passing tests are real tests, not smoke tests — `test_no_data_leakage_in_split`
in particular asserts the right property. But coverage is concentrated on the
low-risk parts of the system:

| Untested | Why it matters |
|----------|----------------|
| `create_card_aggregates` current-row exclusion | The single most leakage-prone computation in the codebase has no test asserting that row *i* excludes its own amount |
| `create_target_encoding` shift correctness | Same — plus it has no fit/transform separation to test |
| `reduce_v_features` fit/transform separation | Would have caught the CRITICAL finding |
| `encode_categoricals(fit=False)` with unseen labels | Would have caught the inference crash |
| `src/models/ensemble.py` | No test file at all |
| `src/api/main.py` | No test file at all |
| `tests/integration/`, `tests/performance/` | Directories exist with only `__init__.py` — zero tests |

---

## Part 2 — Architecture Recommendations (carried forward)

From the prior architecture assessment, unchanged and still open:

1. **Real-time card-aggregate computation is undesigned.** `create_card_aggregates` uses
   batch `groupby().cumsum()` semantics that have no single-transaction equivalent. The
   Kafka consumer in Phase 6 cannot reproduce these features without a running per-card
   state store. This is the highest-risk unresolved design decision in the system and
   should be an ADR before any Phase 6 code is written.
2. **No inference orchestration layer.** `ModelEnsemble.predict` takes pre-computed
   probability arrays, not features — nothing owns "given one transaction, produce a
   scored, explained decision". Without a designed `InferenceService`, that logic will be
   improvised inside `main.py`'s lifespan.
3. **LightGBM is orphaned.** `train_lgbm.py` + `models/lgbm_model.pkl` + a full config
   block remain, but LightGBM lost the bake-off (0.32 vs 0.57 PR-AUC), is absent from
   `make train`, and is not in the ensemble.
4. **`FeatureEngineer` is approaching god-object size** — 741 lines, 20 public transform
   methods, one shared state bag. Roughly two feature families from the 800-line ceiling.
5. **No model versioning or rollback path.** Artifacts are flat overwritten files with no
   manifest linking them to config, dataset, git SHA, or metrics.
6. **`make monitor` is broken** — it invokes `src/monitoring/drift_reporter.py`, which
   does not exist.

---

## Part 3 — Phased Implementation Plan

Ordering rationale: correctness before capability. Phases A and B invalidate every metric
currently in `reports/RESULTS.md`, so they must land before any new modeling work, and the
report must be regenerated afterwards. Phase C makes results trustworthy; Phase D makes
them reproducible; Phase E unblocks Phases 4–7 of the PRD.

### Phase A — Eliminate Leakage (P0, blocking)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| A1 | Write the failing tests first: assert PCA/freq-encoder/label-encoder state is identical whether or not test rows are present | `tests/unit/test_feature_engineering.py` | Tests fail against current code |
| A2 | Restructure `run_pipeline` to split immediately after `sort_temporal`, then `fit=True` on train and `fit=False` on val/test | `src/data/preprocess.py:63-150` | A1 passes |
| A3 | Fit `IncrementalPCA` on the train slice only, preserving batched transform for memory | `src/data/feature_engineering.py:564-637`, `preprocess.py:75` | Peak RSS stays within current envelope |
| A4 | Add `fit` parameter to `create_target_encoding`; derive the global prior from train labels only | `src/data/feature_engineering.py:469-506` | Prior is provably train-only |
| A5 | Move null-count features to run before PCA so V-column missingness is captured | `src/data/preprocess.py:75,84` | `null_count_total` range reflects ~400 columns |
| A6 | Add regression tests for current-row exclusion in card aggregates and target encoding | `tests/unit/test_feature_engineering.py` | **DONE** — both properties asserted against naive oracles plus perturbation-invariance checks; verified non-vacuous by mutation. Uncovered and fixed a live NaN-entity-key defect (see below) |
| A7 | Re-run preprocessing and retrain; regenerate `reports/RESULTS.md` from the clean run | pipeline + `reports/` | **DONE** — report cites MLflow run `74b8460d68ae400889fb5b128db51751` |

**Acceptance:** no transformer observes val/test rows during fit; test PR-AUC re-baselined
and documented as the honest number, whatever it turns out to be.

**Status: Phase A CLOSED.** A1–A6 were verified previously; A7 re-ran preprocessing and
retrained XGBoost on the leakage-free splits (413,378 / 59,053 / 118,109 rows, 171
features). Re-baselined **test PR-AUC = 0.5602** (val 0.6922, train 1.0000), down from the
pre-fix 0.5678 — a 0.0076 absolute / 1.3% relative drop. That modest move is the predicted
magnitude: the leakage was overwhelmingly unsupervised (PCA basis, frequency maps, fill
values) rather than label-bearing. `reports/RESULTS.md` is regenerated from this run.

Note that A7 re-baselines PR-AUC only. The reported operating point is still selected on
the test set (C1) at a grid-boundary threshold (C3) using uncalibrated probabilities (C2),
and `reports/RESULTS.md` marks it as provisional. The LightGBM row and the regularisation
experiment were not retrained and are withheld rather than restated.

Second defect found and fixed while running A7: the target-encoding carried state keyed
missing entity groups under a literal `NaN`. Because `nan != nan` and each frame's
`groupby` emits a fresh NaN object, every `update_state=True` call wrote a *new*
missing-key entry instead of accumulating onto the previous one. Two splits still read the
first entry back and looked correct, so the A6 train→val test passed; the third call — the
`train → val → test` order `run_pipeline` uses — hit a duplicated key and raised
`InvalidIndexError`, killing preprocessing on the test split. Silent half of the same bug:
val's missing-key history replaced train's rather than adding to it, which is material
because target encoding runs before imputation and `addr1` / `card2` / `R_emaildomain` are
missing on a large share of IEEE-CIS rows. Fixed by canonicalising missing keys to a
single `MISSING_ENTITY_KEY` sentinel in `_carried_totals` and `_accumulate_entity_totals`,
covered by `test_missing_key_history_accumulates_over_three_splits` (77 tests pass).

Defect found and fixed while writing the A6 tests: `create_target_encoding` grouped with
pandas' default `dropna=True`, so rows with a missing entity key got a NaN `cumcount`.
That made the `cum_count == 0` prior fallback unreachable and emitted a **NaN feature**
instead, while `_accumulate_entity_totals` (`dropna=False`) still recorded that group —
leaving carried state written but never read. Silent because the tree models accept NaN,
and material because target encoding runs before imputation
(`preprocess.py` `_apply_stateful_transforms`) and `addr1` / `card2` / `R_emaildomain` are
missing on a large share of IEEE-CIS rows. Fixed by grouping with `dropna=False`.

### Phase B — Fix TFT Training Correctness (P0, blocking)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| B1 | Split `imbalance.strategy` into `sampling_strategy` + `loss_function`; fail fast on unknown values | `config/config.yaml:98-102`, `src/training/train_tft.py:259` | Focal loss is reachable and covered by a test |
| B2 | Remove the double correction — one mechanism only, chosen by config | `src/training/train_tft.py:254,267-271` | Effective positive weighting is ~27x, not ~729x |
| B3 | Add `set_seed()` utility; call from every training entry point; log seed to MLflow | new `src/utils/seed.py` + 5 entry points | Two consecutive runs produce identical val PR-AUC |
| B4 | Fit a scaler (QuantileTransformer recommended, robust to -999) on train; persist with the TFT checkpoint; apply in `SequenceBuilder` | `src/data/sequence_builder.py`, `src/training/train_tft.py:513-540` | Inputs are unit-scale; re-enable AMP and confirm no NaN |
| B5 | Add embeddings (or one-hot) for categorical inputs to the neural path | `src/models/tft_model.py:315-404`, `sequence_builder.py:89-109` | No raw label codes enter the network as continuous |
| B6 | Build sequences once over the full ordered frame; assign to splits by target-row index | `src/training/train_tft.py:249-250` | Boundary cards retain real history |
| B7 | Decouple labels from sequence construction; `predict_proba(X)` takes no `y` | `src/data/sequence_builder.py:118-149`, `train_tft.py:469` | Signature has no label parameter |

**Acceptance:** TFT trains deterministically, uses exactly one imbalance mechanism, and
its inference signature is production-shaped.

### Phase C — Evaluation Integrity (P1)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| C1 | Select thresholds on validation; report test metrics at the frozen threshold | `train_xgb.py:207`, `run_ensemble_eval.py:97`, `run_phase2_eval.py:150` | No `y_test` appears in any selection call |
| C2 | Add probability calibration (isotonic on val) + Brier score + reliability curve | `src/evaluation/evaluator.py`, both trainers | Calibration metrics logged to MLflow |
| C3 | Unify the business objective; extend the threshold grid log-spaced from 1e-4 | `src/evaluation/evaluator.py:28-51,137-157` | Chosen threshold is interior to the grid |
| C4 | Persist the frozen threshold in the model artifact | `train_xgb.py:83-98`, `train_tft.py:513` | Serving reads the threshold, never recomputes it |
| C5 | Add slice metrics (by ProductCD, hour bucket, card tenure) per the mle-reviewer checklist | `src/evaluation/evaluator.py` | Slice table in `reports/` |
| C6 | Re-run the imbalance ablation on validation; delete the duplicate implementation | `notebooks/02`, `scripts/run_ablation.py` | One executed artifact with real numbers |

### Phase D — Reproducibility & Provenance (P1)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| D1 | Extract validated `src/config.py` (pydantic); replace 6 duplicated loaders | 6 modules | Bad config fails at startup with a clear message |
| D2 | Seed the Optuna sampler; fix or remove the no-op pruner | `src/training/tune_xgb.py:89-93` | Repeat study yields identical trials |
| D3 | Emit tuned params to a versioned file with the MLflow run ID; remove the manual copy step | `src/training/tune_xgb.py:103-112` | Config traces to a run |
| D4 | Add a model manifest (config hash, git SHA, dataset hash, metrics, timestamp) next to every artifact | both trainers | Rollback target is identifiable without retraining |
| D5 | Add `catboost` to requirements (or remove its usage); pin the Python version; add a clean-env import smoke test | `requirements.txt`, `scripts/run_phase2_eval.py:30` | **DONE** — CatBoost arm was exploratory-only (confirmed with the user); removed `run_cb_experiment` and its call site rather than adding the dependency. `.python-version` pins 3.10 (the validated `fraudx` env). `tests/unit/test_import_smoke.py` imports every `src/` module and every `scripts/*.py` file; `.github/workflows/ci.yml` runs it on a bare GitHub-hosted runner, the only genuinely clean env available |
| D6 | Replace pickle/`weights_only=False` with `save_model()` + joblib + SHA-256 verification | `train_xgb.py:96,103`, `train_tft.py:539,545`, `feature_engineering.py:683-738` | **DONE** — `src/utils/checksums.py` (new) centralizes the sha256 checksum-manifest format (also adopted by `src/training/manifest.py`, replacing its own copy). `XGBTrainer` now saves the booster via `save_model()` (JSON/UBJ) plus a joblib metadata sidecar; `TFTTrainer` saves only `model.state_dict()` via `torch.save` (loaded with `weights_only=True` explicitly — torch 2.2.0's actual default is `False`, so this is the real control, not reliance on a default) plus a joblib metadata sidecar for config/scaler/threshold/calibrator; `FeatureEngineer.save_transformers`/`load_transformers` moved from raw `pickle` to `joblib`. Every loader calls `verify_checksums()` before deserializing and raises (never falls back) on a mismatch or missing manifest. `scripts/run_slice_metrics.py`, which bypassed `XGBTrainer`/`FeatureEngineer` with its own raw `pickle.load()` on `models/xgb_model.pkl` and `label_encoders.pkl`, was switched to the safe loaders — it would otherwise have both broken (old `.pkl` no longer written) and remained the one unpatched arbitrary-code-on-load path. 28 new/updated tests across `test_checksums.py`, `test_xgb_trainer.py`, `test_tft_artifact_persistence.py`, `test_feature_transformer_persistence.py` |
| D7 | Resolve `device: cuda` — shared auto-resolution, CPU at inference | `tune_xgb.py:71`, `config/config.yaml:65` | CPU-only host trains and serves |
| D8 | Execute notebooks 01 and 02 with outputs; reconcile the PCA variance claim | `notebooks/` | No asserted-but-uncomputed numbers remain |

### Phase E — Serving Readiness (P2, unblocks PRD Phases 4-7)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| E1 | **ADR-001**: inference orchestration — a `ModelRegistry` + `InferenceService` owning load → transform → sequence → ensemble → threshold → SHAP | new `docs/adr/` | Accepted before Phase 5 code |
| E2 | **ADR-002**: real-time card aggregates and target encodings — running per-entity state store vs micro-batch vs cold-start degradation | new `docs/adr/` | Accepted before Phase 6 code |
| E3 | Persist and load target-encoding state; wire `load_transformers()` into API startup | `feature_engineering.py:683-738`, `src/api/main.py:42` | `load_transformers` has a production call site |
| E4 | Train/serve equivalence test: same transaction through batch and serving paths yields identical features | `tests/integration/` | Byte-identical feature vectors |
| E5 | Pydantic request/response schemas with range and staleness validation; include model version in the response | `src/api/schemas/`, `src/api/routes/` | Invalid input rejected at the boundary |
| E6 | Create `src/monitoring/drift_reporter.py` so `make monitor` works | `src/monitoring/` | Target runs |

### Phase F — Hygiene (P2, opportunistic)

| # | Task |
|---|------|
| F1 | Resolve LightGBM: delete, or document as a rejected baseline |
| F2 | Route both trainers through `ImbalanceHandler`, or delete it and its config block |
| F3 | Split `FeatureEngineer` into composable per-family transformers before it crosses 800 lines |
| F4 | Vectorize the md5 hash and decimal-length hot paths |
| F5 | Lazy sequence windowing to remove the ~3 GB materialization; rename the misleading "efficient" method |
| F6 | Update `.gitignore` (`*.db`, `catboost_info/`, `scratch_*`); move `analyze_optuna.py` into `scripts/` |
| F7 | Add tests for `ensemble.py` and `api/main.py`; populate `tests/integration/` |
| F8 | Fix `ReduceLROnPlateau(verbose=True)` deprecation |
| F9 | Add `dropna=False` to the six `groupby("card1")` calls in `create_card_aggregates` (`feature_engineering.py:214,217,227,231`) and `create_velocity_features` (`:427`) — same NaN-key defect fixed in `create_target_encoding` under A6. Latent, not live: `card1` is non-null throughout IEEE-CIS, so this is a consistency fix. Pair it with a NaN-`card1` regression test rather than changing the numerics blind |

---

## Recommended Next Action

Start with **A1** — write the leakage tests before touching the pipeline. They will fail
against the current code, which both proves the finding and gives Phase A an objective
exit criterion. Phases A and B together are the minimum to make any published metric from
this project defensible.

Do not begin Phase E implementation before ADR-002 is written: the real-time feature
question determines whether the current batch feature set is even servable, and it may
force changes back into Phase A's feature engineering.

**Update 2026-08-07:** Phase A is closed (see status block under Phase A above). Next
action is **B1** — split `imbalance.strategy` into `sampling_strategy` / `loss_function`
and fail fast on unknown values — since B2's double-correction fix and B3's seeding both
build on that config split existing first.

---

## Review Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 1     | block  |
| HIGH     | 9     | warn   |
| MEDIUM   | 14    | info   |
| LOW      | 6     | note   |

**code-reviewer verdict:** BLOCK — 1 CRITICAL (train/test contamination in the
preprocessing orchestrator) must be resolved before any results are published.

**mle-reviewer decision:** BLOCK
**Primary risks:** data leakage; irreproducible training; weak eval; unsafe serving
**Tests run:** `python -m pytest tests/ -q` → 50 passed, 29 warnings, 10.87s

**Update 2026-08-07:** the CRITICAL (data leakage) is resolved — Phase A closed, test
PR-AUC re-baselined to 0.5602 at MLflow run `74b8460d68ae400889fb5b128db51751`. The 9 HIGH
findings remain open except the one covered by Phase A (config/report hyperparameter
mismatch is still open — see A7's provenance-gap note in `reports/RESULTS.md` §5). TFT's
double imbalance correction and disabled focal loss (Phase B) are unaffected by this
update and remain BLOCK. Test count is now 77 passed.
