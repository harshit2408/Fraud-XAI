# Implementation Plan — Code Quality, ML Correctness & Architecture Remediation

<!-- Generated: 2026-08-05 | Reviewers: code-reviewer, mle-reviewer, architect | Files reviewed: 30 src + 3 scripts + 3 notebooks + 12 tests -->

**Scope:** Full quality and correctness audit of the ML pipeline (data loading, EDA,
feature engineering, model training, hyperparameters, evaluation) combined with the
architecture assessment, converted into a prioritized, actionable plan.

**Baseline verification run:** `python -m pytest tests/ -q` → **50 passed in 10.87s**
(run in the `fraudx` conda env; the default `python` on PATH is 3.14 and cannot
install this project's pinned dependencies).

**Status as of 2026-08-20 (post-Phase D + metrics audit + F6 exercised + 3-way
ensemble):** Phases **A, B, C and D are closed**; the CRITICAL finding and all 9 HIGH
findings below are resolved. A second, narrower **metrics audit** (2026-08-19/20,
tracked as its own phase below — "Phase F-Audit") closed three more findings (F1, F2,
F6) and fixed a LightGBM training bug. Audit-F6 (target-encoding label lag) was fixed
in code on 2026-08-19 evening and **actually exercised** by a full pipeline re-run on
2026-08-20 afternoon — `preprocess.py` was re-run and all three models retrained on the
resulting data, dropping every model's test PR-AUC by the expected amount (leakage-
adjacent optimism removed, not signal). LightGBM was then reviewed (`mle-reviewer`
agent) and added as a 3rd ensemble input behind a minimum-lift gate; the gate passed and
LightGBM is now part of the deployed blend. Current test count is **356 passed**. Latest
run IDs: XGBoost `77ec7e28d2c4433b80ef8fa31508271b`, TFT `fddd4413666a4cd6a7dd2a0d2cab7de2`,
LightGBM `6434e48efbd1499db82f4e30101661d3`, Ensemble `0fb89a9beb3f4dbd8883dce82819979f`;
`reports/RESULTS.md` is regenerated from all four. The Executive Summary, findings, and
Review Summary immediately below are left as originally written to preserve the audit
record; treat the Phase A/B/C/D/F-Audit/E/F sections further down as the live status.

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
| B1 | Split `imbalance.strategy` into `sampling_strategy` + `loss_function`; fail fast on unknown values | `config/config.yaml:98-102`, `src/training/train_tft.py:259` | **DONE** — `imbalance.strategy` is split into `sampling_strategy` (`none|oversample|smote`) and `loss_function` (`focal_loss|weighted_bce|bce`); `resolve_imbalance_config()` validates both independently and raises on an unknown value instead of silently falling through. Focal loss is no longer dead code — the live training log reads `Using FocalLoss (gamma=2.0, alpha=0.25)`. Covered by `tests/unit/test_imbalance_config.py` |
| B2 | Remove the double correction — one mechanism only, chosen by config | `src/training/train_tft.py:254,267-271` | **DONE** — `resolve_imbalance_config()` guarantees exactly one active mechanism: when the `WeightedRandomSampler` is installed, `weighted_bce`'s count-based `pos_weight` collapses to 1.0, so the ~27x resampling is never multiplied by a ~27x loss weight. FocalLoss's `alpha` is a fixed hyperparameter rather than an imbalance-ratio derivative, so it cannot compound either |
| B3 | Add `set_seed()` utility; call from every training entry point; log seed to MLflow | new `src/utils/seed.py` + 5 entry points | **DONE** — `src/utils/seed.py:set_seed()` (random/numpy/torch/cuda/`PYTHONHASHSEED`, with an opt-in `deterministic_cuda`) plus `seed_worker` for DataLoader workers; called from every training entry point and logged to MLflow. Verified across the pipeline: the 2026-08-19 XGBoost re-run reproduced val PR-AUC 0.6922 / test 0.5602 to four decimals against the independent 2026-08-07 run. Covered by `test_seed.py` and `test_training_entry_point_seeding.py` |
| B4 | Fit a scaler (QuantileTransformer recommended, robust to -999) on train; persist with the TFT checkpoint; apply in `SequenceBuilder` | `src/data/sequence_builder.py`, `src/training/train_tft.py:513-540` | **DONE** — `SequenceBuilder` fits a `QuantileTransformer` on the train split only (`fit_scaler=True`), reuses it for val/test/inference, and persists it in the TFT metadata sidecar. AMP is re-enabled (`use_amp: true`) and the training run proceeds with finite losses (epoch 1 train 0.0473 / val 0.0393), so the NaN issue AMP was disabled to work around is resolved by scaling — as the finding predicted |
| B5 | Add embeddings (or one-hot) for categorical inputs to the neural path | `src/models/tft_model.py:315-404`, `sequence_builder.py:89-109` | **DONE** — `TemporalFusionTransformer` takes `static_categorical_indices` + `static_cardinalities` and routes those slots through `nn.Embedding` via `_embed_static_categoricals()`, concatenating embeddings with the continuous slots. The run log confirms `Static categorical embeddings: 4 columns, cardinalities=[6, 6, 6, 3]` |
| B6 | Build sequences once over the full ordered frame; assign to splits by target-row index | `src/training/train_tft.py:249-250` | **DONE** — `_build_sequences_for_splits()` concatenates the splits in temporal order, builds sequences once over the combined frame, then partitions by each row's original split membership (rebasing `original_indices`). The run log shows a single combined build: `Built 472,431 sequences from 12,730 card groups` for train+val. Covered by `tests/unit/test_tft_boundary_history.py` |
| B7 | Decouple labels from sequence construction; `predict_proba(X)` takes no `y` | `src/data/sequence_builder.py:118-149`, `train_tft.py:469` | **DONE** — `TFTTrainer.predict_proba(self, X, history_X=None)` takes no `y`; label attachment moved to `SequenceBuilder.attach_targets()`, used only in training/eval. `history_X` supplies prior context at inference without labels, preserving the B6 property in serving |

**Acceptance:** TFT trains deterministically, uses exactly one imbalance mechanism, and
its inference signature is production-shaped.

**Status: Phase B CLOSED.** Verified by a full TFT training run on 2026-08-19
(MLflow `3a4f64b83d1a40bdb0a427cb99e4c7ee`, early-stopped at epoch 18 of 100 after
50.7 min). The run log carries direct evidence for each item: `Using FocalLoss
(gamma=2.0, alpha=0.25)` (B1 — focal loss had never executed in the shipped config),
`Static categorical embeddings: 4 columns, cardinalities=[6, 6, 6, 3]` (B5), one combined
sequence build of `472,431 sequences from 12,730 card groups` for train+val (B6),
`predict: 590,540 sequences (no labels)` (B7), and finite train/val losses for all 18
epochs with `AMP: True` (B4 — the NaN that AMP was disabled to work around does not recur
once inputs are QuantileTransformer-scaled, exactly as the finding predicted).

Results: train PR-AUC 0.7007 / val 0.5867 / test 0.4906, ROC-AUC 0.8876. The overfit gap
(0.2100) is roughly half XGBoost's (0.4398). Most striking is calibration: the TFT's raw
Brier score is 0.0784 and isotonic calibration cuts it to 0.0234 — a 70% reduction, which
is the sharpest available demonstration of why C2 was needed, since the pre-Phase-C
cost-based threshold was being applied to exactly these distorted probabilities.

The TFT is weaker than XGBoost standalone but contributes real decorrelated signal: the
validation-optimized ensemble weight is 0.2213 and the blend lifts test PR-AUC to 0.5709
(vs 0.5455 for calibrated XGBoost alone). See `reports/RESULTS.md` §6.

### Phase C — Evaluation Integrity (P1)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| C1 | Select thresholds on validation; report test metrics at the frozen threshold | `train_xgb.py:207`, `run_ensemble_eval.py:97`, `run_phase2_eval.py:150` | **DONE** — every `find_optimal_threshold` call site now passes `y_val`/`y_prob_val` (`train_xgb.py:435`, `train_lgbm.py:298`, `train_tft.py:1133`, `run_ensemble_eval.py:137`); the F1-optimal threshold is likewise picked on validation via `precision_recall_curve(y_val, ...)`. Test metrics are reported at the frozen value. Verified end-to-end in run `9a149f6267f443538f4ed68ff7d0829c` |
| C2 | Add probability calibration (isotonic on val) + Brier score + reliability curve | `src/evaluation/evaluator.py`, both trainers | **DONE** — `Evaluator.fit_calibrator` (isotonic, fitted on validation) + `compute_brier_score`; both trainers log before/after Brier for val and test. Measured in run `9a149f6267f443538f4ed68ff7d0829c`: val Brier 0.0176 → 0.0169, test 0.0219 → 0.0213. The calibrator is persisted in the artifact alongside the threshold |
| C3 | Unify the business objective; extend the threshold grid log-spaced from 1e-4 | `src/evaluation/evaluator.py:28-51,137-157` | **DONE** — `_select_threshold_grid` is log-spaced over `[1e-4, 0.99]` (200 points), and `find_optimal_threshold` / `plot_threshold_vs_business_value` now share one `_business_value` objective so they cannot disagree. Run `9a149f6267f443538f4ed68ff7d0829c` selects **0.0004**, interior to the grid (floor 0.0001). Independently corroborated: that raw threshold maps through the isotonic calibrator to a calibrated p of **0.005043**, versus the analytic cost-optimal `cost_fp/(revenue_tp+cost_fn+cost_fp)` = **0.005076** — a 0.7% agreement, so the grid optimum is the true optimum, not a boundary artifact |
| C4 | Persist the frozen threshold in the model artifact | `train_xgb.py:83-98`, `train_tft.py:513` | **DONE** — `XGBTrainer.set_threshold()`/`save()` persist `threshold` and `calibrator` into the joblib metadata sidecar; `load()` restores them and `predict()` raises rather than inventing a threshold when absent. Confirmed by reloading the shipped artifact (threshold 0.0004002737 round-trips) and by `scripts/run_slice_metrics.py` reporting `Using frozen threshold from model artifact: 0.0004` |
| C5 | Add slice metrics (by ProductCD, hour bucket, card tenure) per the mle-reviewer checklist | `src/evaluation/evaluator.py` | **DONE** — `Evaluator.compute_slice_metrics` + `scripts/run_slice_metrics.py` emit `reports/slice_metrics.csv` and `reports/slice_metrics.md` (ProductCD, hour bucket, card tenure), scored on validation at the single frozen threshold, with a `reliable` flag for slices under 30 rows |
| C6 | Re-run the imbalance ablation on validation; delete the duplicate implementation | `notebooks/02`, `scripts/run_ablation.py` | **DONE** — `notebooks/02_imbalance_ablation.ipynb` deleted; `scripts/run_ablation.py` is the sole implementation and writes `reports/imbalance_ablation_results.json`, which records `"split_used_for_selection": "validation"` and never loads test. Winner `scale_pos_weight` (val PR-AUC 0.6038) over SMOTE (0.5799) |

### Phase D — Reproducibility & Provenance (P1)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| D1 | Extract validated `src/config.py` (pydantic); replace 6 duplicated loaders | 6 modules | **DONE** — `src/config.py` exposes a strict pydantic `Settings` model, a cached `load_settings()`, and `config_hash()`; all six duplicated `yaml.safe_load` helpers now call it, so an unknown or mistyped key raises `ValidationError` naming the field at startup. Covered by `tests/unit/test_config.py` and `test_settings.py` |
| D2 | Seed the Optuna sampler; fix or remove the no-op pruner | `src/training/tune_xgb.py:89-93` | **DONE** — `build_sampler(seed)` returns `TPESampler(seed=config.project.random_seed)`, and the no-op pruner is now real: `XGBPruningCallback` (an `xgb.callback.TrainingCallback`) calls `trial.report()` / `trial.should_prune()` once per boosting round, passed via the `XGBClassifier(callbacks=[...])` constructor kwarg. Covered by `tests/unit/test_tune_xgb.py` |
| D3 | Emit tuned params to a versioned file with the MLflow run ID; remove the manual copy step | `src/training/tune_xgb.py:103-112` | **DONE** — `write_tuned_params()` emits `config/tuned/xgb_<mlflow_run_id>.yaml` and logs it back to the run, so file and run cross-reference each other; `train_xgb.py --tuned-params <path>` reads it by reference instead of the old manual copy-into-config step. Covered by `tests/unit/test_tuned_params_artifact.py`. **Note:** no tuning study has been run against the leakage-free features yet, so `config/tuned/` is still empty and §5 of RESULTS.md carries the hyperparameters forward from the pre-Phase-A study |
| D4 | Add a model manifest (config hash, git SHA, dataset hash, metrics, timestamp) next to every artifact | both trainers | **DONE** — `src/training/manifest.py` builds and writes a `ModelManifest` sidecar (`models/xgb_model.manifest.json`) carrying schema version, model type, MLflow run ID, config hash, git SHA, dataset hash + file list, seed, metrics and UTC timestamp. Run `9a149f6267f443538f4ed68ff7d0829c` records git SHA `f73b8af1cb0d1a77a7f9228b91c51bd26055b92d` — the "not a git repository" gap in the old report is closed. `resolve_git_sha()` degrades to `null` with a warning rather than blocking a save |
| D5 | Add `catboost` to requirements (or remove its usage); pin the Python version; add a clean-env import smoke test | `requirements.txt`, `scripts/run_phase2_eval.py:30` | **DONE** — CatBoost arm was exploratory-only (confirmed with the user); removed `run_cb_experiment` and its call site rather than adding the dependency. `.python-version` pins 3.10 (the validated `fraudx` env). `tests/unit/test_import_smoke.py` imports every `src/` module and every `scripts/*.py` file; `.github/workflows/ci.yml` runs it on a bare GitHub-hosted runner, the only genuinely clean env available |
| D6 | Replace pickle/`weights_only=False` with `save_model()` + joblib + SHA-256 verification | `train_xgb.py:96,103`, `train_tft.py:539,545`, `feature_engineering.py:683-738` | **DONE** — `src/utils/checksums.py` (new) centralizes the sha256 checksum-manifest format (also adopted by `src/training/manifest.py`, replacing its own copy). `XGBTrainer` now saves the booster via `save_model()` (JSON/UBJ) plus a joblib metadata sidecar; `TFTTrainer` saves only `model.state_dict()` via `torch.save` (loaded with `weights_only=True` explicitly — torch 2.2.0's actual default is `False`, so this is the real control, not reliance on a default) plus a joblib metadata sidecar for config/scaler/threshold/calibrator; `FeatureEngineer.save_transformers`/`load_transformers` moved from raw `pickle` to `joblib`. Every loader calls `verify_checksums()` before deserializing and raises (never falls back) on a mismatch or missing manifest. `scripts/run_slice_metrics.py`, which bypassed `XGBTrainer`/`FeatureEngineer` with its own raw `pickle.load()` on `models/xgb_model.pkl` and `label_encoders.pkl`, was switched to the safe loaders — it would otherwise have both broken (old `.pkl` no longer written) and remained the one unpatched arbitrary-code-on-load path. 28 new/updated tests across `test_checksums.py`, `test_xgb_trainer.py`, `test_tft_artifact_persistence.py`, `test_feature_transformer_persistence.py` |
| D7 | Resolve `device: cuda` — shared auto-resolution, CPU at inference | `tune_xgb.py:71`, `config/config.yaml:65` | **DONE** — `src/device.py:resolve_device()` is the shared helper; `config.model.xgboost.device` defaults to `"auto"`, `tune_xgb.py` reads it instead of hardcoding `"cuda"`, and `XGBTrainer.load()` calls `resolve_device("auto", force_cpu=True)` so a GPU-trained artifact always serves on CPU |
| D8 | Execute notebooks 01 and 02 with outputs; reconcile the PCA variance claim | `notebooks/` | **DONE** — `notebooks/02_imbalance_ablation.ipynb` no longer exists: it was deleted under task C6 and replaced by `scripts/run_ablation.py`, which produces a real executed artifact (`reports/imbalance_ablation_results.json`) re-run on validation rather than test. `notebooks/01_eda.ipynb` has now been executed end-to-end in the `fraudx` env (16/16 code cells carry real output, zero errors). A new "§5b PCA Variance Retained (Computed, Train-Only Fit)" cell reuses the actual production code (`FeatureEngineer.reduce_v_features` + `time_based_split_3way`, the same train-only-fit methodology `src/data/preprocess.py` uses post-Phase-A) and independently measures **100.0%** variance retained — reconciling the notebook's old, uncomputed ">85%" claim with `reports/RESULTS.md`'s measured figure: RESULTS.md was correct, the notebook's assertion was not. Every other numeric claim in the notebook's header and §10 summary table was replaced with the value its own executed cells computed. Verified by `mle-reviewer` (PASS on both "no asserted-but-uncomputed numbers remain" and "PCA claim reconciled and correct post-Phase-A"); `reports/RESULTS.md` §2/§8 updated to drop the now-stale "notebook never executed" note |

### Phase F-Audit — 2026-08-19/20 Metrics Audit (P1, closed)

A narrower follow-up audit, conducted after Phase D closed and initially recorded only as
scattered code comments ("Finding F1/F2/F6, 2026-08-19 metrics audit") rather than in this
document — the exact "stale tracking table" failure this plan's own Phase D verification
pass warned about, recurring. Written up here properly so it stops drifting. Numbered
independently of the *original* Phase F hygiene list below — that list's F1 ("resolve
LightGBM: delete, or document as rejected baseline") is answered by this phase's LightGBM
fix.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| Audit-F1 | `scripts/run_ensemble_eval.py` produced no provenance artifact for the ensemble weight sweep | `src/training/run_logging.py` (new), `scripts/run_ensemble_eval.py` | **DONE** — new `RunLogger` gives every trainer/eval script shared TensorBoard + checkpoint infra; `reports/ensemble_results.json` + `.manifest.json` now exist |
| Audit-F2 | LightGBM training wrote a raw pickle and no manifest/checksums — the one D4/D6 gap left after Phase D, since D4/D6 were scoped to "both trainers" (XGBoost, TFT) at the time | `src/training/train_lgbm.py` | **DONE** — mirrors `XGBTrainer`/`TFTTrainer`: native `Booster.save_model` (`models/lgbm_model.txt`), joblib metadata sidecar, sha256 checksum manifest, `ModelManifest`. No raw pickle remains anywhere in `models/` |
| Audit-F6 | Target encoding treated same-window fraud labels as instantly known, which a live system never has | `config/config.yaml`, `src/config.py`, `src/data/feature_engineering.py`, `src/data/preprocess.py`, `tests/unit/test_feature_engineering.py` | **DONE and exercised, 2026-08-20 afternoon.** `preprocess.py` was re-run (`dataset_hash` `4c1059dc…`) and all three models retrained on the result. Measured effect: test PR-AUC dropped for every model — XGBoost 0.5602→0.5289, TFT 0.4906→0.4137, LightGBM 0.5349→0.5165 (`reports/RESULTS.md` §6 "Effect of Audit-F6") — the expected direction, confirming the fix removes leakage-adjacent optimism rather than signal |
| LightGBM undertrained (original plan's F1 decision) | `train_lgbm.py` stopped at iteration 4 of 1200 (`stopping_rounds=100`) — a training failure, not a converged model, that blocked deciding whether to keep or delete LightGBM | `config/config.yaml` (`model.lightgbm.early_stopping_rounds: 0`), `train_lgbm.py` | **DONE 2026-08-20 morning; decision revised 2026-08-20 afternoon.** Root cause of the iteration-4 stop was not isolated (suspected `min_data_in_leaf=50` + `reg_lambda=1.0` interacting with `scale_pos_weight=29`, not yet confirmed); disabling early stopping and training the full 1200 trees fixed the training failure. **Decision: kept, and now part of the deployed 3-way ensemble** (see the addendum below) rather than merely a documented standalone comparator as first decided that morning. Re-enabling early stopping with a working configuration is deliberately left open, not blocking |

**Acceptance:** the two provenance gaps (Audit-F1, Audit-F2) are closed and verified by
reading the code, not just the diffstat. `pytest tests/ -q` → **343 passed** as of the
morning LightGBM-only retrain (up from 328); **356 passed** as of the afternoon ensemble
work below. `reports/RESULTS.md` §6/§8 is regenerated from the full 2026-08-20 afternoon
pipeline run (all four MLflow run IDs current).

**Status: Phase F-Audit CLOSED.** The carry-over noted in an earlier revision of this
section (Audit-F6 unmeasured pending a `preprocess.py` re-run) is resolved — see the
addendum immediately below.

#### Addendum (2026-08-20 afternoon): LightGBM added to the ensemble, mle-reviewer reviewed

User request: add LightGBM as a 3rd ensemble input alongside XGBoost+TFT. Reviewed by the
`mle-reviewer` ECC agent **before** implementation, per instruction — verdict
**"proceed with modifications,"** not a flat yes. Reviewer's core concern: XGBoost and
LightGBM share the identical feature set and imbalance strategy (unlike TFT, which is
architecturally decorrelated), so any lift from adding LightGBM was not assumed
beneficial and had to be gated, not just measured.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| Ens-1 | Pre-registered diversity check before trusting any blend PR-AUC | `src/models/ensemble.py:pairwise_diagnostics` | **DONE** — confirmed the reviewer's prediction: XGB↔LightGBM correlation (0.9253) exceeds XGB↔TFT (0.8247) |
| Ens-2 | Grid search instead of gradient-based (SLSQP) optimization — PR-AUC vs. blend weight is piecewise, not smooth | `src/models/ensemble.py:grid_search_simplex_weights` | **DONE** — coarse-to-fine simplex grid search, generalizes the prior 2-model `scipy.optimize.minimize_scalar` call to N models |
| Ens-3 | Bootstrap weight-stability diagnostic — a point-optimized 2-DOF search on a validation set already reused for calibration/thresholding can chase noise | `src/models/ensemble.py:bootstrap_weight_ci` | **DONE** — 50 resamples; none of the three weights' 90% CIs straddle zero |
| Ens-4 | Minimum-lift gate — keep LightGBM only if the 3-way ensemble beats the 2-way baseline's test PR-AUC by a pre-registered amount, checked once | `config/config.yaml` (`ensemble.min_lightgbm_lift: 0.005`), `scripts/run_ensemble_eval.py` | **DONE** — measured lift +0.0074 ≥ 0.0050 → **gate passes, LightGBM kept** |
| Ens-5 | Resolve `src/models/ensemble.py`'s pre-existing dead, untested 2-model `ModelEnsemble` class | `src/models/ensemble.py`, `tests/unit/test_model_ensemble.py` (new) | **DONE** — rewritten as the single N-model implementation; 13 new tests (none existed before) |

**Result:** final deployed ensemble (XGB 0.692 / TFT 0.134 / LightGBM 0.174) — test PR-AUC
**0.5375**, up from the 2-way baseline's 0.5301 (both post-Audit-F6). Not a TFT-sized
decorrelated-signal lift — the correlation diagnostic said not to expect one — but a real,
gate-passing result. One open item, not fixed in this pass: the final ensemble's own
isotonic calibration slightly worsens test Brier (0.02173 → 0.02204); `run_ensemble_eval.py`
does not yet carry the same "warn if calibration doesn't improve" check the three
per-model trainers do. Full diagnostics table in `reports/RESULTS.md` §6.

**Status: Addendum CLOSED.** `pytest tests/ -q` → **356 passed**.

---

### Phase E — Serving Readiness (P2, unblocks PRD Phases 4-7)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| E1 | **ADR-001**: inference orchestration — a `ModelRegistry` + `InferenceService` owning load → transform → sequence → ensemble → threshold → SHAP | `docs/adr/ADR-001-inference-orchestration.md` | **DONE (2026-08-26)** — Accepted. Key decisions: transport-free `src/serving/` package so HTTP and the Kafka consumer share one scoring path; registry validates `dataset_hash`/`config_hash` agreement across all four artifacts and fails startup rather than serving a mixed-vintage ensemble; the blend is over **calibrated** per-model probabilities (the space the frozen 0.006123 threshold was selected in); ensemble weights+threshold are promoted from `reports/` into a checksummed `models/ensemble.json`; degradation modes must carry their own pre-registered threshold (no weight renormalization); SHAP is TreeSHAP on the XGB component (0.692 of the blend), labelled as such |
| E2 | **ADR-002**: real-time card aggregates and target encodings — running per-entity state store vs micro-batch vs cold-start degradation | `docs/adr/ADR-002-realtime-feature-state.md` | **DONE (2026-08-26)** — Accepted. **The gating question is answered: the batch feature set IS servable and Phase A feature engineering does not change.** Feature-by-feature audit: 8 groups are row-wise (`D_mean`/`C_sum` etc. are `axis=1`, not cross-transaction), 6 are frozen fitted state already persisted, and the card-history group reduces to **five scalars per `card1`** — `(n, Σx, Σx², max_x, last_dt)` — which reproduce the batch aggregates *exactly*, not approximately. Chosen: online per-entity store (read-then-write, mirroring the batch's exclusive-cumsum semantics) for the label-free aggregates and TFT sequence windows; a separate label-feed-driven, 30-day-lag-respecting path for the six target encodings; micro-batch rejected (destroys velocity features), cold-start retained only as a declared fallback. One code change forced: `save_transformers` must also persist the per-card amount accumulators |
| E3 | Persist and load target-encoding state; wire `load_transformers()` into API startup | `feature_engineering.py`, `src/api/main.py`, `src/serving/registry.py` | **DONE (2026-09-01)** — target-encoding state was already persisted; this pass added the piece ADR-002 §6 forces: `CardAggregateState` (frozen, five scalars per `card1`), carried-state support in `create_card_aggregates`/`create_velocity_features`, `update_card_aggregate_state()` called once at the end of `_derive_causal_features`, and `card_agg_state` in `save/load_transformers`. Production call site is `ModelRegistry.load()` from the FastAPI lifespan. Artifacts predating ADR-002 load with a loud warning and cold-start every card rather than raising |
| E4 | Train/serve equivalence test: same transaction through batch and serving paths yields identical features | `tests/integration/test_train_serve_equivalence.py`, `src/serving/transform.py` | **DONE (2026-09-01)** — 5 tests, compared against the real `_derive_causal_features`, not a reimplementation. Engineered features match bit-for-bit; two narrow float-associativity carve-outs are documented in the test (`pca_v_*` float32 BLAS blocking ~2.4e-07; carried-sum features one float64 ulp ~2e-16, intrinsic to carrying a subtotal at all — the alternative is the recompute-per-request Option D that ADR-002 §4 rejected). **The test earned its keep during development**: it caught a real same-window label-leak divergence when the fixture omitted the configured 30-day `target_encoding_label_lag_days`, which is exactly the F6 class of bug |
| E5 | Pydantic request/response schemas with range and staleness validation; include model version in the response | `src/api/schemas/prediction.py`, `src/api/routes/predict.py` | **DONE (2026-09-01)** — 27 tests. Range validation rejects non-positive/NaN/infinite amounts (NaN passes `gt` silently, so bare `Field(gt=0)` was not sufficient) and blank ids; staleness lives in `src/serving/staleness.py`, not the schema, because it needs the card's last-seen `last_dt` and the Kafka consumer needs the same rule without importing `src/api/`. `model_version` = `{dataset[:12]}-{config[:12]}-{git[:7]}` per ADR-001 §4.4. Tests assert rejection happens *at the boundary* — a spy service proves an invalid payload never reaches scoring, so it can never mutate card state |
| E6 | Create `src/monitoring/drift_reporter.py` so `make monitor` works | `src/monitoring/drift_reporter.py`, `src/monitoring/prediction_log.py`, `src/serving/metrics.py`, `docs/RUNBOOK.md` | **DONE (2026-09-07)** — `make monitor` runs end-to-end against real artifacts: 250 transactions scored through the real service, 171 features compared, drift detected (47.4% > 30%), HTML + JSON written, exit code 1. E6 also closed a loop nobody had noticed: **`serving.log_file` was configured from Phase 0 but nothing ever wrote it**, so the default monitoring path had no window to compare — `PredictionLogWriter` now writes it and `load_prediction_window` reads it, with a round-trip test so the two halves cannot drift apart. The reporter **raises** rather than reporting 'no drift' on an empty, all-null, or disjoint window, and reports (not silently drops) features missing from the window. Data drift only: performance reports need `y_true`, and labels arrive on a ~30-day lag (F6) |

**Third review pass (2026-09-07, `ecc:mle-reviewer` + `ecc:code-reviewer` +
`ecc:agent-evaluator`, run in parallel).** Verdicts: APPROVE WITH WARNINGS /
APPROVE / **4.8/5 "deliver as-is"**. All five previously-open items were
confirmed genuinely closed. `agent-evaluator` verified the claims against a
running process rather than the self-report (479 tests, `/health` returning
`known_cards: 482` and the documented config hashes). Four new defects found and
fixed, each with a test that fails against the pre-fix code:

| Sev | Finding | Fix |
|---|---|---|
| HIGH | **Categorical drift classification rested on an untested third-party fallback.** `InferenceService` logs `features.iloc[0].to_dict()`; a pandas Series is homogeneously typed, so one row out of a mixed-dtype frame upcasts every scalar to float — `ProductCD=5` (int32) logged as `5.0`. Measured on the real artifact: **44 of 171 columns** affected (26% of the vector). Evidently's dtype fallback happened to rescue it, so nothing failed — but a version bump could silently reclassify a quarter of the feature set from chi-square to Wasserstein | `load_prediction_window(reference=...)` restores reference dtypes, refusing to coerce non-integral values. Verified: 44 mismatches → 0, values preserved. **The drift verdict changed 47.4% → 52.0%**, confirming the classification was load-bearing |
| HIGH | The drift tests were **green by construction** — every fixture was `.astype(float)`, so none exercised a low-cardinality categorical and the suite could not have caught the above | New `TestDtypeFidelityThroughTheRealPath` builds the window through the actual `PredictionLogWriter` → `load_prediction_window` pair, not hand-built floats |
| MEDIUM | `drift_share` is computed over COMPARED columns only, so features a pipeline bug drops leave both numerator and denominator — a whole feature group could vanish while the report said "no drift" | `schema_incomplete` alerts in its own right past `monitoring.missing_share_threshold` (0.20); `alert_reason` names which condition fired |
| MEDIUM | `_json_safe` duck-typed on a callable `.item`, so any object with an unrelated `item()` serialized to whatever it returned (repro: a class returning `"not-a-number"` was written verbatim) — contradicting its own contract | `isinstance(value, np.generic)` |

Also applied: `FeatureStateStore.bind_metrics()` replaces the private-attribute
poke that only handled the `None` case (a store built with its *own* metrics
still split the counters, and a future Redis adapter would have defeated the
check silently); log-rotation recorded as a known gap in `docs/RUNBOOK.md` §6;
and a stale "Carry-over, not blocking" note struck through rather than deleted —
`agent-evaluator` flagged that leaving a contradictory historical note in an
actively-read plan risks someone re-doing completed work.

**Phase E is CLOSED.** Suite: **485 passed** (356 at the start of this work).

---

### PRD Phase 4 — SHAP Explainability (P2, unblocked by Phase E)

TreeSHAP over the **XGBoost component only** (0.692 of the blend, ADR-001 §3.4 —
KernelSHAP over the calibrated blend was rejected on latency). Contributions are
in raw-margin (log-odds) space; every result carries `explained_model="xgb"` and
the blend weight so no consumer can mistake it for an explanation of the ensemble
decision.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| P4-1/P4-2 | `FraudExplainer` — `explain_single` / `explain_batch` / `as_api_contributions`; TreeSHAP on the `XGBClassifier`; additivity `base_value + Σcontributions == margin` exact to ~1e-6 | `src/explainability/shap_explainer.py`, `tests/unit/test_shap_explainer.py` (16 tests) | **DONE (2026-09-07)** — `shap==0.44` installed into the `fraudx` env. Gotcha fixed: `TreeExplainer.expected_value` for an `XGBClassifier` is off by a small constant vs the booster intercept, so `_calibrate_base_value()` derives `base_value = margin_ref − Σshap_ref` from one zero-row reference at construction |
| P4-3/P4-4 | Wire the explainer into the serving path — built once at startup from the loaded booster, populates `explanation` / `explained_model` / `explained_weight` on the response, failure-isolated (a raising explainer never affects the score) | `src/serving/registry.py`, `src/serving/inference.py`, `src/serving/metrics.py`, `src/config.py`, `config/config.yaml`, `src/api/schemas/prediction.py`, `tests/unit/test_serving_explainability.py` (12 tests) | **DONE (2026-09-07)** — `ExplainabilityConfig` (`enabled`, `top_k`), defaulted so a pre-Phase-4 config still validates. `ModelRegistry._build_explainer` returns `None` when disabled and **raises at startup** when enabled but no `xgb` booster loaded — it does not serve never-explainable predictions. `InferenceService._explain` runs after the read-then-write state update; on any exception it logs a warning, increments `serving_metrics.explanation_failures` on `/health`, and returns an empty explanation. Explains the engineered vector, not the raw request (same contract as the prediction log) |
| P4-5 | Standalone SHAP dashboard — one self-contained HTML (global importance, per-transaction SHAP distribution, top-feature table), all figures inlined as base64 PNGs, no CDN | `src/explainability/dashboard_generator.py`, `scripts/generate_shap_dashboard.py`, `tests/unit/test_shap_dashboard_generator.py` (9 tests) | **DONE (2026-09-07)** — `build_dashboard_data` reduces a TreeSHAP batch to mean(\|SHAP\|) descending; `generate_dashboard` renders `monitoring/shap_dashboard.html`. Empty sample raises rather than emitting a blank page. Verified against the frozen artifacts over a 2000-row test sample (top factor `C13`, 222 KB, 2 inlined figures). CLI reads the XGBoost blend weight from `models/ensemble.json`; a missing key is non-fatal (page omits the percentage) |
| P4-6 | `notebooks/04_shap_analysis.ipynb` — global importance, beeswarm, TP/FN/FP waterfalls, `TransactionAmt` dependence, regulatory-language markdown; figures to `reports/figures/` | `notebooks/04_shap_analysis.ipynb` | **DONE (2026-09-07)** — 19 cells, executed end-to-end via `nbconvert --execute` (0 error outputs), additivity `max\|base+Σshap−margin\| = 2.6e-5`. 6 figures written to `reports/figures/shap_*.png` |
| P4-7 | Docs — this section; `docs/RUNBOOK.md` §4.1 (dashboard run) and §6 gap #3 (SHAP scope, no longer "not implemented"); `reports/RESULTS.md` §6 explanation-coverage note | `docs/IMPLEMENTATION_PLAN.md`, `docs/RUNBOOK.md`, `reports/RESULTS.md` | **DONE (2026-09-07)** |
| P4-8 | Parallel review round (`ecc:mle-reviewer` + `ecc:code-reviewer` + `ecc:python-reviewer`); fix findings with pre-fix-failing tests | `src/explainability/*`, `src/serving/inference.py`, `src/config.py`, `config/config.yaml` | **DONE (2026-09-07)** — all three APPROVE WITH WARNINGS, **0 CRITICAL**. Fixes applied with pre-fix-failing tests: **(HIGH, mle)** unbounded ~140 ms synchronous SHAP on every `/predict` → `explainability.timeout_ms` (default 50), enforced on a 1-worker pool; on timeout the explanation is dropped + `explanation_failures` increments, score unaffected. **(HIGH, code)** `explanation_failures` was incremented but never asserted → counter assertions added to `TestExplainerFailureIsolation`. **(HIGH, python)** `black --check` failure on `dashboard_generator.py` → reformatted. **(MEDIUM, mle)** no runtime additivity guard → `FraudExplainer._assert_additive` on the served row + a 3-row constant-offset check at construction, raising `ValueError` (caught by `_explain`). **(MEDIUM)** `_calibrate_base_value` docstring described the wrong `shap` bug (`expected_value` is `[0.]`, not a small drift) → corrected. **(MEDIUM)** `build_inference_service` built `ModelRegistry` 3× → single instance; staleness anchor now from the same registry. **(MEDIUM, python)** `_figure_to_data_uri` leaked the figure on `savefig` error → `try/finally` + module-level `Agg`. Deferred (recorded): coverage-rate denominator + Prometheus alert on `explanation_failures`, `docs`/notebook literal `0.692` |

---

### PRD Phase 6 — Real-Time Inference Simulation (Kafka) (P2, unblocked by Phase E)

The Kafka consumer **reuses `src/serving/InferenceService`** — the transport-free
scoring path ADR-001 was written to make possible. HTTP `POST /predict` and a
streamed message construct the same service and share one
`InMemoryFeatureStateStore`; the read-then-write ordering, `TransactionID`
idempotency guard, and `RLock` that landed in Phase E (2026-09-01 review) cover
the two concurrent writers with no new shared state. The consumer runs as an
asyncio background task **inside `fraud-api`** (PRD §6.3) — no separate
container.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| P6-1 | `FraudDetectionConsumer` — polls `transactions`, scores each row through the shared `InferenceService`, publishes every FRAUD decision to `fraud_alerts` keyed by `card1`. Blocking `KafkaConsumer.poll` runs via `asyncio.to_thread` so the event loop is never blocked; one poisoned message is counted and skipped, never fatal. `kafka` imported lazily so the clean-env import-smoke test needs no broker | `src/streaming/consumer.py`, `tests/unit/test_streaming_consumer.py` (7 tests) | **DONE (2026-09-07)** — spy-service + in-memory Kafka fakes; asserts routing through the shared service, FRAUD-only alerting, poison-message survival, `stop()` teardown, and the missing-dependency error |
| P6-2 | Flesh out the `producer.py` stub — `TransactionProducer` streams RAW `data/raw/test_transaction.csv` (left-joined with `test_identity.csv`, mirroring `DataLoader.load_raw`), chunked to bound memory, keyed by `card1`, rate-limited per message. **Raw, not `test_features.parquet`**: the consumer runs the full training transform, so engineered rows would double-transform | `src/streaming/producer.py`, `tests/unit/test_streaming_producer.py` (7 tests) | **DONE (2026-09-07)** — `--rate`/`--limit`/`--config` CLI; tests use a synthetic CSV + fake producer and assert key=`card1`, `--limit`, per-message rate call, NaN→absent-key, and the raw-file-missing error |
| P6-3 | Wire the consumer into the FastAPI `lifespan` (the `# TODO Phase 6` markers). Startup builds it over `app.state.inference_service` and launches `asyncio.create_task(consumer.run())`; shutdown calls `stop()` then cancels the task. A broker unreachable at startup is logged, not raised — HTTP scoring is unaffected and `/health` shows `kafka_consumer_running: false`. `FRAUD_API_DISABLE_KAFKA=1` skips it entirely | `src/api/main.py` | **DONE (2026-09-07)** — additive to the Phase E file; the HTTP path is untouched. Existing API tests set `FRAUD_API_SKIP_MODEL_LOAD`, so `inference_service` is `None` and the consumer never starts there |
| P6-4 | `kafka_consumer_running` on `/health` (PRD Phase 6 done-when), plus `kafka_messages_processed` / `kafka_alerts_published` / `kafka_consumer_errors` | `src/api/schemas/prediction.py`, `src/api/main.py` | **DONE (2026-09-07)** — four optional fields with defaults on `HealthResponse` (additive; existing schema tests unaffected). Health handler reads `consumer.snapshot()` |
| P6-5 | `scripts/run_streaming_demo.sh` + `scripts/streaming_stats.py` + `make streaming-demo`. The demo brings up the 4-service stack, waits for `/health`, runs the producer, drains, and prints a summary. `streaming_stats.py` reads `/health` only — no Kafka client, no extra dependency | `scripts/run_streaming_demo.sh`, `scripts/streaming_stats.py`, `Makefile` | **DONE (2026-09-07)** — both picked up by the clean-env import-smoke test |

**Not changed:** `docker-compose.yml` already defines the `kafka` (KRaft, no
Zookeeper) and `fraud-api` services exactly as PRD §6.1 specifies;
`kafka-python==2.0.2` is already pinned in `requirements.txt`. `config.kafka`
(`input_topic` / `output_topic` / `consumer_group` / `bootstrap_servers`) was
in place from Phase 0.

**Status: PRD Phase 6 CLOSED (implementation).** Suite: **545 passed** (530 at
the start of this work; +13 streaming tests, +2 import-smoke parametrizations).
`docs/RUNBOOK.md` §4.2 documents the run, the health signals, and the
broker-down behaviour. End-to-end verification against a live broker
(`make streaming-demo`, alerts on `fraud_alerts`, consumer-lag < 500) is a
manual step requiring Docker — the code path is unit-covered with fakes.

---

### PRD Phase 7 — Monitoring & Drift Detection (P2, unblocked by Phase 5/6)

Two gaps against PRD §5.7 / §7 remained after E6 built the one-shot drift
reporter:

1. **`GET /metrics` did not exist.** `ServingMetrics` (E6) surfaces the
   silent-degradation counters as JSON on `/health`, but
   `monitoring/prometheus.yml` scrapes `/metrics` and the Grafana dashboard
   (committed in Phase 6) queries `fraud_predictions_total`,
   `fraud_model_version`, `fraud_drift_detected`, and
   `http_request_duration_seconds` — none of which were served.
2. **No scheduled drift loop.** `make monitor` is one-shot; PRD §7.4 asks for a
   loop that reconstructs the window from the prediction log, writes a durable
   alert file, and sleeps `drift_check_interval_hours`.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| P7-1 | `GET /metrics` in Prometheus text format via `prometheus-fastapi-instrumentator`, exposing `http_request_duration_seconds` (the P95 latency panel) automatically. `/metrics` and `/health` excluded from their own histogram | `src/api/main.py` (`_configure_metrics_endpoint`), `requirements.txt` | **DONE (2026-09-07)** — the instrumentator is imported lazily inside `create_app`; a missing dependency logs an error but does not stop `/predict` and `/health`. **`requirements.txt` corrected**: `prometheus-fastapi-instrumentator==6.4.0` does not exist on PyPI (versions jump 6.1.0 → 7.0.0) → pinned `7.0.0`, and `prometheus-client==0.20.0` is now pinned explicitly |
| P7-2 | The custom `fraud_*` collectors the dashboard already names — `fraud_predictions_total{decision}` (Counter), `fraud_model_version{version}` (info-metric idiom: labelled gauge at 1), `fraud_drift_detected` (0/1 gauge). One process-wide singleton on `prometheus_client`'s default registry; `reset_for_testing()` for isolation | `src/serving/prometheus_metrics.py`, `tests/unit/test_prometheus_metrics.py` | **DONE (2026-09-07)** — both decision labels initialised at 0 so `rate()` is defined from the first scrape; `set_model_version` clears the previous label so an in-process redeploy does not leave two versions at 1 |
| P7-3 | Feed `fraud_predictions_total` from `InferenceService.predict`, not the HTTP route, so an HTTP request and a Kafka message land on the same counter (ADR-001: one scoring path). `prometheus` is an optional constructor kwarg — unit tests and the training-equivalence harness build the service without touching the global registry; `main.create_app` wires the real singleton and publishes `fraud_model_version` at startup | `src/serving/inference.py`, `src/api/main.py` | **DONE (2026-09-07)** — additive kwarg, defaulted `None`; all 551 pre-existing tests still pass |
| P7-4 | `DriftCheckScheduler` (PRD §7.4): read the last N predictions from `serving.log_file` → `DriftReporter.generate_data_drift_report` → on drift write `monitoring/alerts/drift_alert_{ts}.json` **and** set `fraud_drift_detected=1` → sleep `drift_check_interval_hours`. `--once` / `--max-iterations` / `--interval-seconds` bound the run. One raising iteration is logged and the loop continues — a monitoring job that dies on the first bad window is the exact failure this area exists to prevent | `src/monitoring/drift_scheduler.py`, `config/config.yaml` (`monitoring.alerts_dir`), `src/config.py` (`MonitoringConfig.alerts_dir`), `Makefile` (`drift-scheduler`) | **DONE (2026-09-07)** — CLI verified end-to-end against the real training reference + real prediction log: `--max-iterations 2 --interval-seconds 1` produced 2 reports, wrote a timestamped alert (55.6% of 171 features drifted vs the training distribution — expected, the log is old ad-hoc test traffic), exit 0. The alert file and the gauge are both written because the gauge is lost on restart and the file is the audit trail |
| P7-5 | `tests/integration/test_monitoring.py` (PRD Phase 7 names this file): the drift report generates without error, the scheduler loop produces ≥ 1 report, the alert file is timestamped and names the drifted features, and `fraud_drift_detected` tracks the latest verdict. `pytest.importorskip("evidently")` so a clean env skips cleanly | `tests/integration/test_monitoring.py` | **DONE (2026-09-07)** — uses the real `PredictionLogWriter` → `load_prediction_window` pair, not hand-built fixtures; `sleep_fn` injected so the loop test does not wait |

**Not changed:** `monitoring/prometheus.yml`, `monitoring/grafana/` (dashboard +
datasource provisioning), and the `prometheus` / `grafana` services in
`docker-compose.yml` were all in place from Phase 6 and already reference the
metric names P7-2 now serves.

**Status: PRD Phase 7 CLOSED and verified against the live stack (2026-09-07).**
New tests: **12** `test_prometheus_metrics` + **9** `test_monitoring`, all green
alongside the existing suite.

`docker compose up -d` brings all four services healthy. Verified end-to-end:

- `GET /metrics` on the running container serves `fraud_predictions_total{decision}`,
  `fraud_model_version{version}` (=1), `fraud_drift_detected` (=0), and
  `http_request_duration_seconds_bucket{handler="/predict"}`. Three live
  `POST /predict` calls moved `fraud_predictions_total{decision="FRAUD"}` 0 → 3.
- **Prometheus** `http://localhost:9090/api/v1/targets` shows `job=fraud-api`
  → `health=up` (scraping `http://fraud-api:8000/metrics`).
- **Grafana** `http://localhost:3000` (admin/admin) auto-provisioned the
  Prometheus datasource and the "Fraud Detection Dashboard"; all five panel
  queries return data from Prometheus.

**One pre-existing bug fixed to get there** (Phase 0/6, not Phase 7):
`docker-compose.yml` had `CLUSTER_ID: "fraud-detection-cluster-01"`, copied
literally from the PRD — `cp-kafka:7.5.0` rejects it ("not a valid UUID", the
PRD's own §15 risk table flagged this). Replaced with a valid 22-char
base64url UUID; the KRaft broker now starts and becomes healthy.

**Second pre-existing bug fixed (Phase 6):** the in-process Kafka consumer
connected to `localhost:9092` (from `config.kafka.bootstrap_servers`) and
ignored the compose env `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` — the env var was
declared in `docker-compose.yml` but nothing read it, so under compose the
consumer could not reach the broker and `/health` showed
`kafka_consumer_running: false`. `load_settings` now overlays a small
`_ENV_OVERRIDES` table (`{("kafka","bootstrap_servers"): "KAFKA_BOOTSTRAP_SERVERS"}`)
onto the parsed YAML before validation: a set, non-empty env var wins; unset or
blank leaves the YAML value, so local and CI runs are unaffected. It is the one
code path `load_settings` has, so API / producer / consumer all see the same
value, and `config_hash` records the broker the process actually used.
`tests/unit/test_settings.py::TestKafkaBootstrapEnvOverride` (5 tests) pins it.

**Verified against the live stack (2026-09-08):** after the fix, `docker
compose up -d` → `/health` shows `kafka_consumer_running: true`
(`bootstrap=kafka:9092` in the logs). 5 messages published to `transactions` →
`kafka_messages_processed: 5`, `kafka_alerts_published: 5`, 0 errors; the
alerts (with SHAP explanations) landed on `fraud_alerts`; and
`fraud_predictions_total{decision="FRAUD"}` moved to 5 — the Kafka path
increments the **same** Prometheus counter as HTTP (ADR-001 one scoring path).

---

### Phase F — Hygiene (P2, opportunistic)

| # | Task |
|---|------|
| F1 | Resolve LightGBM: delete, or document as a rejected baseline — **DONE (2026-08-20, Phase F-Audit)**: kept, and as of the same-day addendum below, part of the deployed 3-way ensemble (test PR-AUC 0.5165 standalone post-F6; ensemble 0.5375) — not merely a documented rejected baseline |
| F2 | Route both trainers through `ImbalanceHandler`, or delete it and its config block |
| F3 | Split `FeatureEngineer` into composable per-family transformers before it crosses 800 lines |
| F4 | Vectorize the md5 hash and decimal-length hot paths |
| F5 | Lazy sequence windowing to remove the ~3 GB materialization; rename the misleading "efficient" method |
| F6 | Update `.gitignore` (`*.db`, `catboost_info/`, `scratch_*`); move `analyze_optuna.py` into `scripts/` |
| F7 | Add tests for `ensemble.py` and `api/main.py`; populate `tests/integration/` |
| F8 | Fix `ReduceLROnPlateau(verbose=True)` deprecation |
| F9 | Add `dropna=False` to the six `groupby("card1")` calls in `create_card_aggregates` (`feature_engineering.py:214,217,227,231`) and `create_velocity_features` (`:427`) — same NaN-key defect fixed in `create_target_encoding` under A6. Latent, not live: `card1` is non-null throughout IEEE-CIS, so this is a consistency fix. Pair it with a NaN-`card1` regression test rather than changing the numerics blind |

---

### PRD Phase 6 follow-up — consumer lag was never measured (P2)

**Closed 2026-09-08.** Phase 6's last done-when item is *"Consumer lag stays
< 500 messages throughout the demo (verify via Prometheus metric)"*, and no
metric existed to verify it against: `snapshot()` counted messages *processed*,
which cannot show a backlog — a consumer can be busy and falling behind at the
same time.

| # | Task | Files | Status |
|---|------|-------|--------|
| P6-L1 | `fraud_consumer_lag` gauge on the default registry, alongside the three existing custom collectors | `src/serving/prometheus_metrics.py` | **DONE** — `-1` (`LAG_UNKNOWN`) is the initial value, not `0`: an unmeasured backlog must not read as healthy and let the < 500 gate pass vacuously |
| P6-L2 | Sample `sum(highwater - position)` over the assignment once per poll | `src/streaming/consumer.py` | **DONE** — computed on the poll thread (the client's own state is not safe to read from the event loop mid-poll), wrapped so a diagnostic failure can never stop scoring, and reset to `-1` on `stop()` so a dead consumer stops advertising a stale number |
| P6-L3 | Surface on `/health` and enforce the PRD budget in the demo summary | `src/streaming/consumer.py`, `scripts/streaming_stats.py` | **DONE** — `kafka_consumer_lag` joins the `/health` block; `streaming_stats.py` exits non-zero at or above 500 and reports "not evaluated" (not "passed") when the lag is unknown |
| P6-L4 | Grafana panel with the 500 red threshold | `monitoring/grafana/dashboards/fraud_detection.json` | **DONE** — "Kafka Consumer Lag (messages)" timeseries, `min: -1` so the unknown sentinel is visible |
| P6-L5 | Tests | `tests/unit/test_streaming_consumer.py`, `tests/unit/test_prometheus_metrics.py` | **DONE** — 8 new: unknown-before-first-poll, backlog published to both sinks, unknown when the broker reports no high-water mark, a raising diagnostic does not break polling, and `stop()` clears a stale reading |

**Live verification (2026-09-08).** Against the running stack, Prometheus
recorded the gauge moving through its real states: `-1` (no assignment yet) →
`0` (assigned and caught up), and `kafka_consumer_lag` now appears on `/health`
alongside the existing consumer counters.

**What the verification also exposed — a throughput limit, not a metrics bug.**
Publishing 3,000 messages showed the consumer scoring at roughly **4
messages/sec** (~250 ms each: feature engineering + V-feature PCA + a TFT
sequence rebuild + the 3-way blend + SHAP). The PRD's Phase 6 demo target is
**200 tx/sec** — two orders of magnitude apart. Two consequences worth stating
plainly rather than leaving for someone to rediscover:

1. The Phase 6 done-when item *"`run_streaming_demo.sh` completes end-to-end in
   < 90 seconds"* at 5,000 transactions is **not currently achievable** on one
   worker; that volume takes ~20 minutes to drain.
2. While the loop is saturated it leaves the event loop little room, so
   `/health` and `/metrics` scrapes time out — meaning the lag gauge is
   observable only while lag is small. The metric is correct; the serving
   throughput is the constraint.

Fixing this is a design change (batch scoring, multiple consumer workers, or
dropping SHAP from the streaming path), not a monitoring change, so it is
recorded here rather than patched under a metrics task.

---

### PRD Phase 8 — Business Impact Quantification

**Status: CLOSED (2026-09-08).**

| # | Task | Files | Status |
|---|------|-------|--------|
| P8-1 | Impact arithmetic as a tested module rather than notebook cells | `src/evaluation/business_impact.py` (new) | **DONE** — `CostModel` / `TransactionVolume` / `BusinessImpact` (all frozen dataclasses), `compute_impact`, `threshold_sweep`, `best_threshold`. Rates are measured on the split and applied to the annual volume, so the report does not move with the evaluation split's size |
| P8-2 | Export the deployed blend's test probabilities once | `scripts/export_test_probabilities.py` (new) | **DONE** — reads weights/threshold from `models/ensemble.json` (what serving loads), blends per-model **calibrated** probabilities, writes `reports/ensemble_test_probabilities.npz`. Reproduces `ensemble_results.json` exactly: TP 3881 / FP 58930 / FN 183 / TN 55115, PR-AUC 0.53752838 vs 0.53752837 recorded |
| P8-3 | `notebooks/05_business_impact.ipynb` with outputs committed | `notebooks/05_business_impact.ipynb` (new) | **DONE** — 20 cells, executed via nbconvert, all outputs present, no errors |
| P8-4 | Threshold sensitivity plot | `reports/figures/business_value_vs_threshold.png` (new) | **DONE** — log-spaced x-axis; the deployed threshold (0.0061) sits **below** the PRD's sketched linear 0.01 floor, so a linear grid would have missed the optimum entirely |
| P8-5 | Executive summary with specific dollar figures | notebook §5 | **DONE** |
| P8-6 | Quantify the carried-over cost-model double-count | notebook §6, `business_impact.py` | **DONE** — see below |

**Headline figures** (deployed threshold 0.006123, test split annualised to
1,197,484 transactions at a measured 3.44% fraud rate):

| Metric | Value |
|---|---|
| Fraud caught / missed per year | 39,349 (95.5% recall) / 1,855 |
| Legitimate customers blocked | 597,480 (51.7% FPR) — **15.2 per fraud caught** |
| Net annual value, PRD convention | **$14,972,275** |
| Net annual value, net of principal | **$15,899,975** |
| Improvement over "flag nothing" (PRD) | $35,574,322 |

**P8-6 — the double-count, finally quantified.** `revenue_tp = 480` and
`cost_fn = 500` describe the same recovered principal from two directions, so
one fraud swings $980 — about twice what is at stake — while a blocked customer
stays priced at $5. Flagged in ADR-001 §6.3 as affecting "every threshold this
ADR treats as frozen"; this is the first time the effect has a number:

- Optimal threshold, PRD convention: **0.004640** (96.7% recall, 672,345 blocked/yr)
- Optimal threshold, each fraud counted once: **0.010183** (92.3% recall, 434,945 blocked/yr)

The double-count lowers the recommended threshold **2.2x**, buying +4.4% recall
for **237,400 additional blocked customers per year**. The notebook reports both
conventions side by side rather than picking one — which `revenue_tp` was meant
to mean is a business decision, not an engineering one.

**Not done, deliberately:** the threshold in `models/ensemble.json` is
unchanged. Retuning a frozen operating point is a deployment decision requiring
the business to settle the cost model first. The notebook's §6 states the
recommendation; it does not act on it.

**A related observation, not fixed here:** `scripts/run_ensemble_eval.py` in the
working tree is still the **2-way** (XGB+TFT) version and writes no
`ensemble_results.json`, yet the committed artifact is 3-way with LightGBM. The
3-way code that produced it is not in the tree. `export_test_probabilities.py`
sidesteps this by reading `models/ensemble.json` rather than re-deriving
weights, but the eval script itself remains stale relative to the artifact.

---

### PRD Phase 9 — Precision Improvement (Recall-Preserving)

**Status: CLOSED (gate not met, formally stopped by ADR-004, 2026-09-09).**
9.0–9.4 ran; the best result (step 9.4) reached 19.17% precision at recall
≥80%, 10.8 points short of the 30% target band. ADR-004 shows in closed form
that step 9.5 (the two-stage cascade) cannot close that gap and accepts the
9.4 operating point as final — see `docs/adr/ADR-004-operating-point-selection.md`.
No further Phase 9 steps are planned. 9.0, 9.1, 9.2, 9.3 done — all
missed the gate. 9.2's mle-review returned BLOCK; its fixes (C1 dataset-hash,
H1 promote-guard, H2 no_tft mode, M1–M4 UID) were folded into the **9.3 cycle**:
one `preprocess.py` re-run (UID refactor, feature count 176 → 177) + retrain of
all three models + re-blend, which also carried the imbalance re-tuning
(`scale_pos_weight` 29 → 1, SMOTE re-tested and rejected at −0.057 PR-AUC).
**Best result so far: the promoted 3-way 9.3 ensemble — precision 10.31% at
recall 90.0%, test PR-AUC 0.5394, ROC-AUC 0.9123, 8.7 FP per fraud, net
$16.18M/yr — still 19.69 points short of the 30% precision target.** 9.4
(multi-window RFM/velocity + `dist1`) is next. Per-step log at the end of this
section. See `docs/prd.md` §10
Phase 9 for the requirement-level version (done-when checklist, target band,
FR-05 conflict).

**Why this phase was added.** Phase 8 quantified the deployed operating point
at **95.5% recall / 6.2% precision** (15.2 false positives per fraud caught).
A 2026-09-08 conversation asked for approaches to reach ~30% precision without
losing recall; literature research (same session) grounded the target and
produced the sequence below. That research is preserved here since it is the
justification for the step ordering and the explicit rejection of SMOTE as a
default (FR-05 conflict).

**Target, stated precisely.** Precision ≥ 30% at recall ≥ 80%, measured at the
model's deployed operating threshold via `src/evaluation/business_impact.py`.
**Not** 95% recall simultaneously — the arithmetic doesn't support it (95%
recall at 30% precision implies ~4.5% FPR on this dataset's class balance; the
model currently needs 51.7% FPR to reach 95% recall) and no published IEEE-CIS
result combines those two numbers. The best evidenced comparable point found
was P=0.3447 / R=0.8275 (AUPRC 0.7373) — methodology unverified (source
paywalled), treated as a directional target, not a guarantee.

**Research summary (2026-09-08 web research, full citations below).**

| Finding | Source | Implication for this phase |
|---|---|---|
| Kaggle 1st place: `UID = card1_addr1_D1n` (D1n = floor(day − D1)) + 47 aggregation features → local AUC 0.9363 → 0.9472. UID itself excluded from training to avoid overfitting to an identifier | NVIDIA blog, Kaggle writeup | **P9-2**: highest-priority feature addition; this pipeline currently aggregates by `card1` alone, not the richer client identifier |
| SMOTE/ADASYN **improve recall but deteriorate PR-AUC and precision** for XGBoost/CatBoost | search-result synthesis (unnamed study) | Conflicts with FR-05, which lists SMOTE as required. Treated as a candidate to test and report, not a default |
| Resampling methods showed **no substantial improvement** on a real (non-benchmark) imbalanced card-payments dataset | de la Bourdonnaye & Daniel, arXiv:2206.13152 | Corroborates the above on non-synthetic data; do not expect resampling alone to close the gap |
| Multi-window transaction aggregation (1/3/6/12/18/24/72/168h) + periodic (von Mises) features | Whitrow et al.; Bahnsen et al., *Feature engineering strategies for credit card fraud detection*, Expert Systems with Applications 2016 | **P9-4**: extend `create_velocity_features`, currently narrower than this window set — verify before adding |
| Two-stage cascade (high-recall stage 1 → high-precision reranker on flagged set only) is the standard production pattern for this exact precision/recall shape | general retrieval/reranking literature | **P9-5**: largest lever, sequenced last because it is the most expensive change (new model, new serving hop, ADR update) |
| GNN benchmark on card fraud reported AUROC 0.8599 | arXiv:2503.22681 (detectGNN) | Below this ensemble's current 0.9066 ROC-AUC — **not a demonstrated win**, explicitly out of scope for this phase |
| Precision-at-fixed-recall as a direct training objective (Lagrangian relaxation) | Eban et al. (Google, research.google.com/pubs/archive/45573.pdf); Kumar et al., arXiv:2107.10960 | Plausible for the TFT specifically; no established GBDT formulation found — out of scope for this phase, candidate for a future one |
| Precision-at-fixed-recall as a **reporting metric** ("what's our precision at 70% recall") — distinct from the training objective above | evaluated against this repo's `ModelEvaluator` on 2026-09-08 | **P9-0**: genuinely missing — `compute_metrics_at_threshold` needs a threshold, `find_optimal_threshold` optimizes cost, neither answers a fixed-recall query directly |
| RFM-style fixed rolling windows (10min/1h/24h counts) + short-window-vs-baseline comparison (24h std vs. 30-day mean) | general fraud-detection practice, corroborated by Whitrow/Bahnsen above | Folded into **P9-4** — distinct from the existing `time_since_last_tx` (point gap) and `tx_count_per_card` (expanding, not fixed-window) |
| `dist1`/`dist2` as a fraud signal — user-proposed, checked directly against `data/raw/train_transaction.csv` 2026-09-08 | — | **Folded into P9-4, `dist1` only.** `dist1` non-null **only for `ProductCD=='W'`** (0% elsewhere); the naive overall present-vs-absent fraud-rate gap (2.0% vs 4.5%) is a `ProductCD` mix confound (product C alone is 11.7% fraud, never has `dist1`) and disappears within W (2.00% vs 2.09%). Real signal: fraud rate roughly **doubles in `dist1`'s top quintile within W** (1.6–1.8% → 2.95% at `dist1>36`) — a genuine, if modest, value gradient. `dist2` checked the same way and rejected: its presence rate and presence→fraud direction both vary by `ProductCD` and **flip sign** across products (H: present > absent; R: present < absent) — no non-confounded signal beyond what `ProductCD` already gives the model |
| IP-based velocity (count by IP/time, IP-geolocation distance) | — | **Ruled out 2026-09-08**: IEEE-CIS has no IP address or lat/long columns. Vesta never documented what `dist1`/`dist2` measure between — the common "billing↔shipping" reading is unconfirmed community speculation, not a Vesta fact |

**Sequence and gate (mirrors `docs/prd.md` §10 Phase 9 exactly — do not let the
two documents diverge if either is edited later):**

| # | Step | Change type | Models retrained | Gate check after |
|---|------|-------------|-------------------|-------------------|
| P9-0 | `ModelEvaluator.precision_at_recall(y_true, y_prob, target_recall)` — walk `precision_recall_curve`, return precision at the point with the **largest threshold whose recall is still ≥ target** (tightest constraint-satisfying operating point; the naive "first point ≥ target" reading is the recall-1.0 corner and is rejected). Precision on the curve is sawtoothed, so this is not a guaranteed lower bound. | Evaluation tooling, no retraining | None | Run once against the current model as the baseline, before P9-1; re-run as a standard line in every later step's eval, not a gate itself |
| P9-1 | Resolve the `revenue_tp`/`cost_fn` double-count (ADR-001 §6.3 open item); re-derive the operating threshold | Config / threshold only | None | **Check first** — thresholds 0.05–0.10 already reach 20–33% precision on the *current* model per the Phase 8 sensitivity table, at 65–75% recall |
| P9-2 | `UID` client-identifier aggregation features | Feature engineering | XGBoost, LightGBM (TFT's existing `card1` sequence grouping evaluated, not assumed to need change) | After retrain + full eval |
| P9-3 | Re-tune `scale_pos_weight` / focal-loss `alpha`,`gamma`; test SMOTE/ADASYN and report the delta rather than assuming benefit (FR-05) | Training config | XGBoost, LightGBM, TFT | After retrain + full eval |
| P9-4 | Multi-window RFM/velocity aggregates: fixed rolling-window counts (10min/1h/24h) per card, short-window-vs-baseline comparisons (e.g. 24h std vs. 30-day mean), plus `log1p(dist1)` + top-quintile `dist1_high` flag scoped to `ProductCD=='W'` | Feature engineering | XGBoost, LightGBM | After retrain + full eval |
| P9-5 | Two-stage cascade (stage 2 trained only on stage-1-flagged transactions) | Architecture (new model + ADR update) | New stage-2 model | After retrain + full eval — only reached if P9-1–P9-4 insufficient |

**Stop condition:** if any step's post-retrain measurement meets the target
band, later steps are not executed. The step that met the target is recorded
as the final configuration and Phase 10 proceeds against it. P9-0 is not
gated — it is measurement tooling, run every time regardless of outcome.

**Not in this phase:** GNNs; precision-at-fixed-recall as a **training
objective** (Eban/Kumar Lagrangian relaxation — distinct from P9-0, which is
reporting only, not a loss function); IP-based velocity features (no IP/geo
columns in IEEE-CIS). All three explicitly deferred or ruled out rather than
silently omitted.

**Definition of done for this tracking section:** each step gets its own dated
entry here (mirroring the P6/P7/P8 sections above) recording what was
implemented, the files touched, the retrain run id, before/after PR-AUC and
precision/recall at the operating threshold, and whether the gate stopped the
sequence at that step.

---

### Phase 9 — step log

**Status: CLOSED (2026-09-09). 9.0–9.4 done, gate not met; formally stopped by
ADR-004 — see the section status line above.**

#### 9.0 — precision-at-fixed-recall tooling + baseline (2026-09-08, no retrain)

Files: `src/evaluation/evaluator.py` (`precision_at_recall`),
`src/evaluation/phase9_report.py` (new — the fixed per-step battery),
`scripts/run_phase9_eval.py` (new CLI), `scripts/export_test_probabilities.py`
(now also writes `reports/ensemble_val_probabilities.npz` for per-step
threshold re-derivation), `tests/unit/test_evaluator.py` (+9),
`tests/unit/test_phase9_report.py` (new, 12), `.gitignore`
(`reports/ensemble_*_probabilities.npz`).

`precision_at_recall` returns precision at the **largest-threshold** curve
point whose recall still meets the floor — reconciled into `docs/prd.md` §10
9.0 and the P9-0 table row (the PRD previously said "first point ≥ target",
which is the recall-1.0 corner and useless).

Baseline — deployed 3-way ensemble, mlflow run `0fb89a9b…`, unchanged from
Phase 8 (test PR-AUC **0.537528**, ROC-AUC 0.906604):

| | precision | recall | threshold |
|---|---|---|---|
| deployed operating point | 6.18% | 95.50% | 0.006123 |
| precision @ recall ≥ 70% | 27.97% | 70.00% | 0.074555 |
| precision @ recall ≥ 80% | 15.95% | 80.02% | 0.031844 |
| precision @ recall ≥ 90% | 9.46% | 90.01% | 0.013338 |

**Gate: NOT MET** (target P ≥ 30% @ R ≥ 80% at the deployed threshold).
Precision shortfall 23.82 pts at the deployed point; even the R≥80% curve
point is 15.95%, ~half the target. **Sequence continues to 9.1.**

Reviewed by `mle-reviewer` — approve with warnings; 3 required fixes applied
(docstring rationale, RESULTS.md baseline row, PRD/plan wording).

#### 9.1 — cost-model resolution + threshold re-derivation (2026-09-08, no retrain)

Files: `docs/adr/ADR-001-inference-orchestration.md` (§6.3 resolution),
`reports/RESULTS.md` (9.1 subsection). No code/config change — the raw
`revenue_tp`/`cost_fn` stay in `config/config.yaml`; only the *combining
convention* is fixed.

**Decision: `net_of_principal`** — each recovered fraud counted once (FN term
= $0). The PRD-literal formula double-counts the principal (~$980 swing on a
$500 fraud).

Frozen model: deployed 3-way ensemble, mlflow run `0fb89a9b…` (no retrain).
Threshold re-derived on the **validation** blend under the adopted convention:
`0.006123 → 0.014740` (`threshold_sweep(..., convention=net_of_principal)` on
`reports/ensemble_val_probabilities.npz`). Frozen **test** metrics at 0.014740:
precision **10.14%**, recall **88.88%** (TP 3612 / FP 32022 / FN 452 /
TN 82023), ~8.9 FP per fraud (was 15.2). Test PR-AUC unchanged: 0.537528.

**Gate: NOT MET, and judged explicitly at both thresholds.** 9.1 proposes
0.014740 as the operating threshold; the gate is evaluated there (P 10.14% /
R 88.88%) → precision 19.86 pts short of 30%. At the still-deployed 0.006123
the gate also fails (P 6.18% / R 95.50%). The threshold is not promoted into
`models/ensemble.json`. The cost model (`cost_fp $5` vs `cost_fn $500`) will
not select a threshold near the ~0.10 needed for 30% precision, and that point
costs ~30 pts of recall anyway. `config/config.yaml` / `models/ensemble.json`
unchanged. **Sequence continues to 9.2 (UID client features).** Artifact
`reports/phase9_step9_1.json`. Reviewed by `mle-reviewer` — approve with
warnings, all applied.

#### 9.2 — UID client-identifier aggregates (2026-09-08, XGB + LGBM retrain)

Files: `src/data/feature_engineering.py` (`create_uid_features` +
`_uid_maps`/`_uid_global` state + persistence), `src/data/preprocess.py`,
`src/serving/transform.py`, `src/data/sequence_builder.py` (TFT excludes
`uid_*`), `scripts/run_ensemble_eval.py` (rebuilt as a real 3-way driver;
writes `models/ensemble.json`), `scripts/run_phase9_eval.py`
(`--derive-threshold-from`), `scripts/export_test_probabilities.py` (val
export), `Makefile` (`ensemble`, `phase9-eval`). Feature count 171 → 176.
Retrain runs: xgb `dec764c1…`, lgbm `40f8a65f…`; TFT not retrained.

`UID = card1_addr1_D1n`; five train-fitted aggregates (freq, amt mean/std,
amt-vs-cohort ratio, D1 mean), unseen UID → global fallback, id itself never
fed to the model. ~51% of val / ~64% of test rows are on a UID unseen in train
(a `D1n` that shifts daily makes UID near-per-transaction) — the feature
carries signal for ~40% of rows.

Standalone: XGB test PR-AUC 0.5115 → **0.5479** (+0.036); LGBM 0.5014 → 0.5284
(+0.027). Ensemble re-blend: **min-lift gate drops LightGBM** (3-way 0.552974
vs 2-way 0.548937, lift +0.004 < 0.005). Deployed blend now **2-way XGB 0.874
/ TFT 0.126**, threshold **0.010664** (net_of_principal, val).

| | pre-9.2 | 9.2 |
|---|---|---|
| test PR-AUC | 0.5375 | 0.5489 (+0.0114) |
| P / R @ deployed threshold | 6.18% / 95.50% | 8.57% / 90.11% |
| FP per fraud | 15.2 | 10.7 |
| P @ recall≥80% (curve) | 15.95% | 16.15% |

**Gate: NOT MET.** Recall clears 80%; precision 8.57% at the deployed
threshold is 21.43 pts short of 30%. UID aggregates improve ranking and halve
the FP ratio but do not move precision into the band. **Sequence continues to
9.3.** Artifacts `reports/phase9_step9_2.json`, `reports/ensemble_results.json`.

**mle-review verdict: BLOCK.** Modeling idea sound, but 9.2 left a
non-loading `models/ensemble.json` (C1: mixed-vintage dataset_hash — TFT not
retrained), an irreproducible config state (C2), a promotion-discipline
violation (H1: `ensemble.json` written on a gate-fail), a dropped `no_tft`
fallback mode (H2), and UID feature issues (M1 ratio docstring, M2 singleton
self-inclusion, M3 `-999` bucket collapse). **All addressed** — see
`reports/RESULTS.md` "9.2 mle-review" table. Key code changes: `--promote`
guard on `run_ensemble_eval.py`, `no_tft` mode always emitted +
`apply_min_lift_gate`/`build_spec` pure functions + schema-round-trip test,
`_UID_MIN_COUNT=3` floor + `uid_count` feature + distinct `naA`/`naD` missing
tokens, TFT feature-dim assertion, MLflow ensemble run, real-artifact load
smoke test. The UID refactor changes feature values → the fix + 9.3 are one
clean preprocess + 3-model retrain cycle (feature count 176 → 177).


#### 9.3 — imbalance re-tuning + 9.2-review fixes (2026-09-08, all-3-model retrain)

Files: `src/training/train_xgb.py` (`resolve_scale_pos_weight` — makes the
config key live; it was read but ignored), `src/training/train_lgbm.py`,
`src/training/train_tft.py` (focal γ 2→1, α 0.25→0.10), `config/config.yaml`
(`scale_pos_weight` 29→1), `src/data/feature_engineering.py` (`_UID_MIN_COUNT=3`,
`uid_count`, `naA`/`naD` tokens), `scripts/run_ensemble_eval.py` (`--promote`
guard, `no_tft` always emitted, MLflow `ensemble_blend` run),
`tests/integration/test_real_ensemble_artifact_loads.py` (new). Feature count
176 → **177**. One combined preprocess + 3-model retrain, `dataset_hash
431cda76…`, mlflow ensemble run `0692d881…`.

Imbalance grid picked `scale_pos_weight=1` on **validation PR-AUC**, which
rises monotonically as the weight falls (0.5648 at spw=29 → 0.5907 at spw=1,
+0.026 — the strongest signal in the grid and the metric `run_ablation.py`
selects on). **SMOTE re-tested and rejected: −0.057 validation PR-AUC**
against class-weighting alone — reported as evidence rather than silently
skipped, per the FR-05 conflict noted in the research table. `run_ablation.py`
never loads the test split, so no selection here touches test.

**Ensemble: LightGBM returns.** Min-lift gate now keeps it (3-way 0.539449 vs
2-way 0.530481, lift +0.008968 ≥ 0.005; 9.2 had measured +0.004 and dropped it).
Promoted blend **XGB 0.560 / LGBM 0.402 / TFT 0.038**, threshold **0.012831**.

| | 9.0 | 9.2 (2-way) | 9.3 (3-way) |
|---|---|---|---|
| test PR-AUC | 0.5375 | 0.5489 | 0.5394 |
| test ROC-AUC | 0.9066 | 0.9017 | **0.9123** |
| P / R @ deployed threshold | 6.18% / 95.5% | 8.57% / 90.1% | **10.31% / 90.0%** |
| FP per fraud | 15.2 | 10.7 | **8.7** |
| P @ recall≥80% (curve) | 15.95% | 16.15% | **18.29%** |
| net annual (net_of_principal) | $15.90M | $15.84M | **$16.18M** |

**Gate: NOT MET.** Recall clears 80%; precision 10.31% is **19.69 pts short**
of 30%. Test PR-AUC dips vs 9.2; the likely cause is the M2 UID self-inclusion
fix removing a train-partition artifact that had inflated 9.2's ranking, but
three changes landed in one retrain cycle and **no ablation isolates them**, so
that attribution is a hypothesis (mle-review 9.3, MEDIUM) — 0.5489 is an
unresolved upper bound, not a disproven number. ROC-AUC, P@R≥80%, FP ratio and
net value all improve. C1 (mixed-vintage ensemble) verified fixed — the
real-artifact load test passes against the real `models/` tree.
Full suite: **650 passed**. **Sequence continues to 9.4.** Artifacts
`reports/phase9_step9_3.json`, `reports/ensemble_results.json`.

**mle-review verdict: APPROVE WITH WARNINGS.** Every 9.2 BLOCK finding
(C1, H1, H2, H3, M1–M4) verified fixed against real source and artifacts, not
just the claims table; the integration test was confirmed to actually run
(not skip) against `models/`. No weights, thresholds or gate decisions are
selected on test. UID features confirmed leakage-free (`_UID_MIN_COUNT` is a
hardcoded constant; `_uid_maps` populated only under `fit=True`, which only
train receives). Findings: one HIGH outside 9.3's scope — the working-tree
`.gitignore` had flipped `data/raw/`/`data/processed/` to `!`-negated, leaving
1.5 GB of Kaggle data stageable, and had commented out the `.ai/.claude/
.cursor/.ecc/` rules; **fixed**. Two documentation errors corrected above
(val-vs-test mislabel on the SMOTE delta; "PR-AUC flat" understated the
monotonic val-PR-AUC evidence that in fact makes `spw=1` *better* supported).
One MEDIUM left open by choice: the unisolated causal claim, now flagged as a
hypothesis in `reports/RESULTS.md` rather than asserted.
---

## Recommended Next Action

Start with **A1** — write the leakage tests before touching the pipeline. They will fail
against the current code, which both proves the finding and gives Phase A an objective
exit criterion. Phases A and B together are the minimum to make any published metric from
this project defensible.

Do not begin Phase E implementation before ADR-002 is written: the real-time feature
question determines whether the current batch feature set is even servable, and it may
force changes back into Phase A's feature engineering.

**Update 2026-08-19:** Phases **A, B, C and D are closed** (D3's re-tuning run is the one
deliberate carry-over — the traceability mechanism landed, but no Optuna study has been run
against the leakage-free features, so `config/tuned/` is empty and the shipped
hyperparameters are still the pre-Phase-A study's). The full pipeline was re-run from raw
CSV on 2026-08-19: preprocessing, XGBoost (MLflow `9a149f6267f443538f4ed68ff7d0829c`) and
TFT. `reports/RESULTS.md` is regenerated from that run and its provenance table is
transcribed from `models/xgb_model.manifest.json`.

Next action is **E1/E2** — the two ADRs. Do not begin Phase E implementation before
ADR-002 is written: the real-time feature question determines whether the current batch
feature set is servable at all.

**Superseded 2026-08-26:** both ADRs are written and Accepted (see the E1/E2 rows above
and `docs/adr/`). ADR-002's feature-by-feature audit answers the gating question —
**the batch feature set is servable; Phase A feature engineering does not change** —
so the block on Phase E implementation is lifted. Next action is **E3**.

**Update 2026-09-01:** **E3, E4 and E5 are DONE** (rows above). A new transport-free
`src/serving/` package implements ADR-001's design — `ModelRegistry` (fail-closed
cross-artifact validation), `EnsembleSpec` (`models/ensemble.json`, checksummed, with a
pre-registered `no_tft` fallback carrying its own threshold), `InMemoryFeatureStateStore`,
`ServingFeatureTransformer` and `InferenceService` — plus a FastAPI `POST /predict`
adapter over it. `scripts/run_ensemble_eval.py` now emits the serving spec, and
`preprocess.py` dual-writes transformers to `models/transformers` because the fraud-api
container does not mount `./data`. Suite: **452 passed** (356 → 452, 96 new). Remaining in
Phase E: **E6** (`drift_reporter.py`). SHAP explanations (PRD Phase 4) and the Kafka
consumer (PRD Phase 6) are deliberately out of scope here; `PredictionResponse.explanation`
exists and stays empty until Phase 4.

**Review pass (`mle-reviewer` + `python-reviewer`, both initially BLOCK — now resolved):**
Two real defects were found and fixed, each with a regression test verified to fail
against the pre-fix code:
1. **CRITICAL (thread-safety).** `ServingFeatureTransformer` swapped `fe._card_agg_state`
   in place for the duration of a transform. `/predict` is a sync route, so Starlette runs
   it in a threadpool over one shared `FeatureEngineer` — two concurrent requests could
   clobber each other's scoped history and score one card against another's aggregates,
   silently. Fixed by passing carried state down as an explicit `card_state` argument;
   the engineer is now read-only during serving. `TestConcurrentTransformsAreIsolated`.
2. **CRITICAL (ADR-002 §5.6 unwired).** `FeatureStateStore.sequence()`/`append_sequence()`
   were defined and never called, so TFT — a sequential model trained on a 10-transaction
   window — was scored on a length-1 sequence for every card, with no degraded flag.
   Fixed by replaying the stored window as `history_X` and appending the transformed
   vector in the same read-then-write step, plus the automatic `no_tft` fallback ADR-001
   §4.6 mandates. Confirmed on the real artifacts: raw TFT moves 0.1848 → 0.1091 with six
   prior transactions. `tests/unit/test_serving_sequence_window.py` (9 tests).

Two further defects surfaced only by running the full path end-to-end against the real
14 MB artifacts, neither reachable from batch data: `create_email_features` crashed on a
single-row frame whose email column is all-null (float dtype, `.str` accessor raises), and
`handle_missing_values` classified all-null numeric columns as categorical for a one-row
request — filling `"MISSING"` where batch filled `-999.0`, which XGBoost then rejected
outright. Both are genuine train/serve skew that a 590k-row frame can never expose.

**Second review pass (2026-09-01, `ecc:mle-reviewer` + `ecc:code-reviewer` +
`ecc:python-reviewer`, run in parallel).** Verdicts: APPROVE WITH WARNINGS / APPROVE with
follow-ups / Warning. All three confirmed the two earlier BLOCK fixes hold. `mle-reviewer`
additionally traced the TFT representation end-to-end and confirmed **no double-scaling**:
`append_sequence` stores the aligned, *unscaled* vector, which is exactly what training
passes as `history_X`, and `card1` survives `_align` un-encoded so `SequenceBuilder`'s
grouping matches training. Four further defects found and fixed, each with a regression
test verified to fail against the pre-fix code (`tests/unit/test_serving_hardening.py`,
12 tests):

| Sev | Finding | Fix |
|---|---|---|
| HIGH | `InMemoryFeatureStateStore.observe`/`_claim` did unlocked read-then-write. ADR-002 §5.3's "Kafka partitions by `card1`, one writer" does not cover HTTP — the sync `/predict` route runs in a threadpool with no per-card affinity, and the same store serves both. **Measured**: with `sys.setswitchinterval(1e-9)` and 16 threads, 299/300 trials lost an update and 9/300 double-claimed. Rare at the default interval, which is what made it dangerous: the aggregates are *expanding*, so a lost or doubled observation is permanent | One `RLock` around every compound mutation. Contention is negligible next to the transform and three forward passes |
| HIGH | `append_sequence` ran unconditionally even when `observe` rejected a duplicate, so a redelivered transaction pushed a second copy into the bounded ring buffer, evicting real history — the accumulators right, the TFT window quietly wrong | Gated on the store's idempotency verdict; `_observe` now returns it |
| MEDIUM | `except Exception` around `_score` masked registry-invariant violations (unloaded model, missing calibrator) as routine degradation — a 200 with `degraded: true` instead of a failure | New `ServingInvariantError`, re-raised before the fallback |
| MEDIUM | The 422 `detail` returned internal feature names (`pca_v_3`, `amount_zscore_per_card`), contradicting `ErrorResponse`'s "no internal detail" docstring and letting a caller enumerate the trained feature set | Generic client message; full detail logged server-side |
| LOW | `pd.to_numeric(errors="coerce")` silently turned unparseable input into NaN, later imputed to -999.0 | Raises when a non-null value fails to parse |

Also applied: `sequence()` typed `List[np.ndarray]`, a named `CARD_STATE_SCALAR_COUNT`
constant, and a client-safe message on the (currently unreachable) `KeyError`→503 branch.

One test-quality note worth recording: the first version of the invariant tests **passed
against the reverted fix** — the fixture's fallback used the same single model, so it
failed identically and the error propagated either way. The fixture now drops to a
different, healthy model, which is what makes those tests discriminating.

**Closed 2026-09-07 (E6 pass).** The three deferred observability items are now
built and wired, each with a test that fails against the pre-fix code:
`src/serving/metrics.py` exposes `cards_scored_with_partial_sequence`,
`out_of_order_transactions`, `duplicate_transactions`,
`target_encoding_state_age_seconds` (ADR-002 §5.5) and `config_hash_mismatch`
through a new `observability` block on `/health`. Verified against real
artifacts: 41 partial-sequence scorings after a cold start, 1 duplicate caught,
17.4-day state age (anchored to the artifact mtime, not process start, so a
restart cannot reset it), and the LightGBM config-hash split surfaced with the
differing hashes named. `docs/RUNBOOK.md` adds the rollback procedure
`mle-workflow` §6 requires — artifact, config, data dependency and traffic
switch — plus a table mapping each signal to an operator action. Suite:
**479 passed**.

**Superseded — the pre-E6 statement of those gaps:** the TFT sequence buffer is not persisted, so
every card warms up from empty after a restart (`mle-reviewer` MEDIUM); ADR-002 §5.3/§5.5
promise two Prometheus gauges that do not exist yet; and SHAP (ADR-001 §3.4) is still
unimplemented with `PredictionResponse.explanation` defaulting to `[]`. The first two
belong with **E6**; SHAP is PRD Phase 4.

**Documented deviation from ADR-001 §4.2:** the ADR lists `config_hash` equality as a
fail-closed startup check. It is implemented as a WARNING, because `config_hash` covers the
whole config file: the shipped artifacts show XGBoost/TFT at `ad07f4af8389` and then
LightGBM/ensemble at `24e774db15ac` after the documented `early_stopping_rounds` edit
between runs. Failing closed there would reject a valid deployment. `dataset_hash`
equality and spec↔model agreement remain fatal. A scoped per-section config hash would let
this be restored as fail-closed.

**~~Carry-over, not blocking~~ — RESOLVED 2026-09-07.** This note said the on-disk
transformers predated ADR-002 (no `card_agg_state`, every card cold-starting) and that
`models/ensemble.json` did not exist. Both were true when written on 2026-09-01 and are
no longer: the artifacts were regenerated during the E6 pass and verified in a running
process — `/health` reports `known_cards: 482` and the API boots cleanly. Kept rather than
deleted so the audit trail stays intact, struck through so a reader does not act on it.
(Found by `ecc:agent-evaluator`, which flagged that leaving a contradictory historical note
in an actively-read planning doc risks someone re-doing completed work.)

**Update 2026-08-20 (afternoon):** Phase F-Audit (above) is closed, including its
addendum — the second, previously undocumented "2026-08-19 metrics audit" (findings F1,
F2, F6) is now tracked in this document, Audit-F6 has been exercised by a full pipeline
re-run (every model's test PR-AUC dropped by the expected amount), and LightGBM has been
reviewed (`mle-reviewer`) and added to the deployed ensemble behind a minimum-lift gate
that passed. Test count is now **356 passed**. `reports/RESULTS.md` is regenerated from
all four fresh MLflow runs (XGBoost, TFT, LightGBM, Ensemble), all sharing one
`dataset_hash`. No carry-overs remain from the morning update. Next action is unchanged:
**E1/E2**, the two ADRs.

One item for a **business** decision rather than an engineering one: the cost model in
`config/config.yaml` credits `revenue_tp=480` on a catch *and* charges `cost_fn=500` on a
miss, which appear to be two encodings of the same recovered amount. The pipeline
faithfully optimizes the objective as written, which is why the cost-optimal operating
point flags 46% of transactions at 6.7% precision. See `reports/RESULTS.md` §6.

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
findings remain open except the one covered by Phase A. TFT's double imbalance correction
and disabled focal loss (Phase B) remain BLOCK. Test count is now 77 passed.

**Update 2026-08-19:** all 9 HIGH findings are now closed. The two BLOCK items —
TFT's double imbalance correction and the silently-disabled focal loss — are fixed under
B1/B2 and exercised by a real training run (`Using FocalLoss (gamma=2.0, alpha=0.25)` in
the run log). Threshold-on-test (C1), no-calibration (C2), grid-boundary (C3) and the
serving-parity/provenance findings (C4, D1–D4, D6, D7) are closed and verified against a
full re-run of the pipeline. Test count is now **328 passed**.

One defect was found by this verification pass and fixed:
`tests/unit/test_model_manifest.py::test_build_manifest_populates_all_fields` asserted
`manifest.git_sha is None` while passing `git_sha=None` — which `build_manifest` documents
as "resolve it from git". The test passed only while the working tree had no commits, and
began failing the moment one landed. It is now hermetic (the resolver is stubbed), which is
what its own comment always claimed it was.

**Remaining verdict:** the blocking findings are cleared. Phase E (serving readiness) is
gated on ADR-001/ADR-002, not on any unresolved correctness issue.

**Update 2026-08-26:** ADR-001 and ADR-002 are written and Accepted (`docs/adr/`), so
that gate is now clear too. ADR-002 additionally promotes **F9** from opportunistic
hygiene to a Phase E prerequisite: the online feature-state store needs one NaN-entity-key
policy, and batch is currently inconsistent with itself (`create_card_aggregates` and
`create_velocity_features` drop NaN `card1` groups while `create_target_encoding` keeps
them via `dropna=False`). Latent today because `card1` is non-null throughout IEEE-CIS.

**Update 2026-08-20:** Phase F-Audit closed three findings (F1, F2, F6) that had landed in
the working tree as code-only fixes with no tracked write-up — see the Phase F-Audit
section above for the full table. Its same-day addendum then exercised Audit-F6 for real
(full pipeline re-run, every model's test PR-AUC dropped as expected) and added LightGBM
to the ensemble behind an `mle-reviewer`-reviewed minimum-lift gate. Test count is now
**356 passed**. This update exists specifically so this document does not go stale
relative to the code again.

---

## Phase G — 4-Agent ML Review & Remediation (2026-09-09)

**Trigger:** the user asked for a deep-dive review of the ML architecture, training
lifecycle, hyperparameters, and ensembling — deployed in parallel across `mle-reviewer`,
`architect`, `code-architect` and `code-reviewer`, with `reports/RESULTS.md` explicitly
flagged as possibly stale. This is a heavier-weight version of the Phase F-Audit pattern
above: four independent agents, seeded with the same seven candidate discrepancies found
by direct artifact inspection before dispatch, then cross-verified against each other.

**What the pre-dispatch inspection found, verified by `ls`/`cat` on the actual files:**
the deployed `models/ensemble.json` (frozen 2026-09-08 18:29, mlflow run `7f2084b7…`) was
weighting a TFT artifact that had since been **overwritten** by a 2026-09-09 05:41 retrain
(P9-6, run `c03fccb9…`) — different dropout/weight_decay/sampling config, different
calibrator, different probability scale. `dataset_hash` matched (P9-6 changed only
training knobs), so the one hard-fail check in `ModelRegistry._validate_consistency`
(`src/serving/registry.py`) passed cleanly on a spec that no longer described the model it
weighted. `config_hash` — the field that WOULD have caught it — is deliberately
warning-only, because the whole-file hash also trips on unrelated serving/monitoring edits
(a real false-positive problem the code correctly reasons about, just not the one that
actually fired here).

**Findings, ranked, after cross-agent verification (each corrected at least one of the
other three's premises — see the session transcript for the full back-and-forth):**

| # | Finding | Severity | Source | Fixed this session? |
|---|---|---|---|---|
| G1 | Min-lift gate (`apply_min_lift_gate`) decided LightGBM's ensemble membership on **test** PR-AUC, re-consulted across ≥4 evaluation runs (9.2 drop, 9.3/9.4 re-add) | CRITICAL | mle-reviewer, architect, code-architect, code-reviewer (independently, all four) | **Yes** — `scripts/run_ensemble_eval.py` now gates on `val2`/`val3`; test lift kept only as a logged diagnostic (`lgbm_lift_test_pr_auc_diagnostic`). Regression test added (`test_min_lift_gate_call_site_uses_validation_not_test`, source-inspects the call site so a future revert is caught even though the pure function's own unit test can't see it) |
| G2 | TFT's `QuantileTransformer` was fit on **train+val concatenated**, not train alone, despite the code comment claiming otherwise — proven at runtime (quantile range reached a val-only value; 20 samples fit instead of 10) | HIGH | code-reviewer (found), verified independently by re-running the proof | **Yes** — new `SequenceBuilder.fit_scaler_on(X)` fits on an explicit frame only; `TFTTrainer.train` now calls it on `X_train` before the train+val combined sequence build (`fit_scaler=False`). 4 new tests, incl. a source-inspection guard on the `train()` call site |
| G3 | Deployed ensemble spec had no `config_hash`/`git_sha`/`created_at`/per-model run-id lineage, though `_model_run_ids()` computed exactly that map 15 lines away for the *reports* manifest only | CRITICAL (structural) | architect, code-architect | **Partially** — the immediate instance is resolved by re-promoting (all three `model_run_ids` in the fresh promoted spec's manifest now match the on-disk model manifests exactly, verified by direct comparison). The structural fix (stamping this into `models/ensemble.json` itself, not just the reports manifest) is **not yet implemented** — recorded here as follow-up work, not done |
| G4 | `make train` silently omits LightGBM (`Makefile`), so the documented "retrain everything" command leaves a model carrying up to 0.38 ensemble weight at a stale vintage by construction | HIGH | architect (found), confirmed by direct read | **Yes** — `Makefile`'s `train:` target now runs all three trainers |
| G5 | `train_lgbm.py` hardcodes `models/lgbm_model.pkl` instead of reading `serving.lightgbm_model_path` from config, unlike its two sibling trainers | MEDIUM | architect, code-reviewer (independently) | **No** — confirmed still present; not fixed this session (not on the user's agreed 5-item list) |
| G6 | LightGBM's `early_stopping_rounds` was disabled (`0`) as a 2026-08-19 diagnostic workaround for a since-removed failure condition (`scale_pos_weight=29`, now `1`) | MEDIUM | mle-reviewer | **Yes** — re-enabled to `100`. Retrain confirmed no regression (best iteration 1196/1200 — the model simply uses its full capacity either way; test PR-AUC unchanged to 4 decimal places) |
| G7 | `reports/RESULTS.md` documents Phase 9 only through step 9.3 and never mentions 9.4, ADR-004, or the 2026-09-09 P9-6 TFT change | MEDIUM | mle-reviewer (corrected the user's own initial framing — RESULTS.md doesn't just lag 9.3, it never mentions Phase 9's later steps at all) | **Not yet** — deferred to the last step of the 5-item remediation list, after the fresh retrain settles |
| G8 | The imbalance ablation (`scripts/run_ablation.py`) hardcodes `CONFIG_SPW = 29.0` with a comment claiming that's still the shipped value; it is not (shipped value is now `1`) — so the grid's labels are inverted relative to what's actually deployed | MEDIUM | code-reviewer | **No** — confirmed still present; not on the agreed 5-item list |

**Remediation carried out this session (user-directed, sequential, one item verified before
the next — not the agents' full recommendation set, a deliberately scoped subset):**

1. **G1 fixed and verified** — 15/15 `test_ensemble_eval.py` pass, full suite 667/667.
2. **G2 fixed and verified** — 33/33 relevant tests pass (`test_sequence_builder.py`,
   `test_tft_boundary_history.py`), full suite 667/667.
3. **All three models retrained** on the unchanged `dataset_hash d20f03c0…`, then
   `scripts/run_ensemble_eval.py --promote` re-run. The retrain surfaced a real regression
   the review had not anticipated: combining G2's fix with the still-active P9-6
   regularisation change (dropout 0.4, weight_decay 1e-2, no oversampling) **collapsed**
   TFT's test PR-AUC to 0.2902 (val/test gap 0.314, worse than any prior TFT run). Reverted
   P9-6's three settings (dropout 0.3, weight_decay 1e-5, oversample restored) and
   retrained TFT a second time: **test PR-AUC 0.4670, overfit gap 0.164 — the best TFT
   result in this project's history**, confirming P9-6 (not G2's fix) had been the cause
   of the collapse. `models/ensemble.json` now matches `reports/ensemble_results.json`
   exactly and its `model_run_ids` match the on-disk model manifests — the specific
   incoherent-deployment state that triggered this whole review no longer exists.
4. **G6 fixed and folded into the retrain above** (user's explicit sequencing choice, to
   avoid a second full retrain cycle).
5. **RESULTS.md regeneration — not yet done as of this entry.** The fresh ensemble weights
   (`xgb 0.618 / tft 0.008 / lgbm 0.374`, test PR-AUC 0.5502) land TFT back at the same
   near-zero weight that originally motivated this review, this time backed by a tight
   bootstrap CI (mean 0.051, p05=p95=0.05) confirming it as a genuine flat-objective
   instability (TFT correlates with XGBoost at 0.824) rather than a bug. User's decision on
   this — see Phase 12 below — was to stop investing further in TFT rather than force its
   weight up or down, and pursue a different architecture next.

**Superseded 2026-09-09 (continuing the exclusion trail from PRD Phase 9 §"Explicitly out
of scope for this phase"):** the GNN exclusion recorded there cited a 2026-09-08 benchmark
at AUROC 0.86 (arXiv:2503.22681) as "not a demonstrated win." A different, more recent
paper (Uddin & Aziz, arXiv:2604.14231, submitted 2026-04-14) reports a GNN-GraphSAGE model
at **AUC-ROC 0.9248 / PR-AUC 0.6334 / F1 0.6013** on the same IEEE-CIS dataset (590,540
transactions, 118,108-row held-out test) — above this project's current ensemble (ROC-AUC
0.9154, PR-AUC 0.5502, fresh 2026-09-09 numbers above). Read directly (not from an
abstract) before citing here; see `docs/prd.md` PRD Phase 12 for the four caveats that
first-hand read surfaced (split methodology not confirmed temporal; new graph-construction
infrastructure required, not a drop-in trainer; the paper's own authors flag the
topology-vs-smoothing attribution as unresolved) and the planned steps. TFT is not deleted:
its artifacts, manifests, and this session's just-improved result stay in the repo as a
completed, evaluated trial, and `src/models/ensemble.py`'s N-model design already supports a
4th candidate without restructuring.

---

## Phase 12 — GNN-GraphSAGE, implemented and rejected (2026-09-10)

**Trigger:** the user directed Phase 12 to be applied end-to-end, with a specific
agent/skill mapping (code-architect for the module blueprints, `ecc:mle-reviewer` for the
graph-construction leakage review *before* code, `ecc:tdd-guide` for tests-first, ADR-005
for the write-up).

**What was done, in order:**

1. **12.2.0 — split methodology.** The paper (arXiv:2604.14231) describes 5-fold stratified
   CV + an 80/20 held-out split with SMOTE-Tomek applied in-fold; it does **not** state the
   held-out split is time-ordered. The +0.083 PR-AUC expectation was revised down before
   any code, anchored to Phase A's measured 0.0076 non-temporal→temporal delta on this exact
   dataset.
2. **Dependency.** `torch-geometric==2.5.3` + `torch-scatter` + `torch-sparse` (compiled for
   `torch 2.2.0+cu121`) installed in the `fraudx` env and added to `requirements.txt` with
   an install note.
3. **12.2.1 — `src/data/graph_builder.py` (design → review → build).** `code-architect`
   blueprinted it against `sequence_builder.py` conventions. `ecc:mle-reviewer` reviewed the
   design **before implementation** for the single highest-risk question — does
   `NeighborLoader` 2-hop sampling let a train node aggregate features from a val/test
   neighbour? Verdict: **APPROVE WITH CHANGES**, findings R1–R9. All implemented:
   - **R1** phase-scoped neighbour windowing: the ≤10 `card1` / ≤5 `(addr1,ProductCD)` caps
     are computed within each phase's eligible node prefix, so a test row's position can
     never change which train↔train edges survive.
   - **R2** one static *undirected* `edge_index` + three boolean edge masks; the train
     loader samples on `edge_index[:, edge_mask_train]` (both endpoints in the train span),
     so a train seed reaches only train nodes at any hop. Symmetric everywhere (train loader,
     val loader, every eval forward — `GNNTrainer._edge_index_for(phase)` is the one source
     of truth). No `directed_past_to_any` refinement.
   - **R3** the model has no normalization layers (enforced at construction) — an eval-mode
     full-graph forward spans val/test nodes, so a `BatchNorm`/`LayerNorm` running stat
     would be a silent leak.
   - **R5** the `QuantileTransformer` is fit on the train node span only (regression test
     compares its `quantiles_` to a train-only reference).
   - **R6** `predict_proba(X)` resolves the split by exact row count, raising on zero/
     ambiguous matches — whole-split scoring only; Phase 12 is offline eval, not serving.
   - **R7** a regression test asserts permuting `y_test` before `train()` leaves the trained
     `state_dict` byte-identical, and no train/val `NeighborLoader` batch reaches a test
     node.
   - **R8** rows on the `addr1` imputation sentinel (`-999.0`, ~11% of rows, 65,706 in the
     real run) are excluded from composite edges — otherwise one dominating clique.
   - **R4 / R9** pre-registered in ADR-005 **before the run**: hyperparameters frozen from
     the paper; GNN proceeds to 12.2.4 iff standalone test PR-AUC > `0.5502 + 0.005`; the
     gate is a one-shot decision on frozen artifacts (no retune-and-recheck — that is the
     G1 failure mode); plausible band 0.52–0.60, anything ≥ 0.6334 to be audited for a leak.
4. **12.2.2 — `src/models/gnn_model.py` + `src/training/train_gnn.py`.** `GraphSAGEModel`:
   2×`SAGEConv` (128→64) + 3-layer MLP head → 1 logit. `GNNTrainer`: full interface parity
   with the three existing trainers (`build_model`/`train`/`predict_proba`/
   `predict_proba_calibrated`/`predict`/`set_threshold`/`set_calibrator`/`save`/`load`),
   3-file checksummed artifact (state_dict only, no pickle of the model), `build_manifest`
   lineage, class-weighted BCE (`pos_weight` = empirical train neg/pos = 27.43), isotonic
   calibrator + val-selected frozen threshold. `model.gnn` config block +
   `src/config.py::GNNConfig` (Optional-defaulted). `Makefile` `train-gnn` target.
   `.gitignore` fixed (`models/` → `/models/`, which had also been ignoring the
   `src/models/` source package).
5. **TDD — 32 new unit tests** (`test_graph_builder.py` 12, `test_gnn_model.py` 10,
   `test_gnn_trainer.py` 10), including the R5/R7 leakage regression tests. Full suite went
   670 → **702 passing**, no regressions.
6. **12.2.3 — standalone train + evaluate.** The first run hung at epoch 36 in a full-graph
   eval forward that thrashed the 8 GB card near-OOM; fixed by batching `_score_nodes` via
   `NeighborLoader` with a wide deterministic fan-out (`eval_num_neighbors [40, 20]`,
   `eval_batch_size 4096`), still phase-scoped so R2/R7 hold. Re-run (mlflow
   `6615a0fc…`, seed 42, unchanged `dataset_hash`, hyperparameters frozen): **test PR-AUC
   0.4410**, val 0.5260, train 0.6564, ROC-AUC 0.8849, overfit gap 0.2155.
   - 0.109 below the 0.5502 ensemble baseline; 0.026 below TFT's best standalone (0.4670);
     0.192 below the paper's 0.6334.
   - **Val 0.5260 → test 0.4410 is an 0.085 drop across the temporal boundary** — ~11× the
     Phase-A leakage-fix delta, confirming the split-methodology caveat: a large part of the
     paper's number is a non-temporal-split artifact. And even on validation the GNN does
     not clear the baseline.
7. **12.2.4 — NOT triggered.** `0.4410 < 0.5502 + 0.005`. `models/ensemble.json` and the
   deployed 3-way blend are untouched; TFT is not replaced.
8. **12.2.5 — `docs/adr/ADR-005-gnn-architecture-evaluation.md`**, Status **Accepted**
   (rejected direction, following the ADR-004 precedent). `docs/adr/README.md`,
   `reports/RESULTS.md` §11, and `docs/prd.md` §12.3 updated.

**Disposition:** all Phase 12 code, tests, config, `models/gnn_model.*`,
`reports/gnn_results.json`, the graph cache (`data/processed/graph/`, gitignored), and
`make train-gnn` stay in the tree as a completed, evaluated trial — same as TFT. Leading
explanation for the gap to the paper (its own authors' unresolved question): GraphSAGE-mean
aggregation over `card1`/`(addr1,ProductCD)` neighbours is feature smoothing that borrows
cross-boundary signal on a stratified split; the GBDT ensemble already carries the
non-leaking part of that as explicit `card1`/`uid` aggregate features (Phase 9.2/9.4). No
follow-up scheduled; the only cheap next probe (does adding a few graph-style mean-aggregate
features to the GBDTs move the baseline?) needs no graph pipeline and is noted in ADR-005 §6.

---

### PRD Phase 10 — Testing Strategy (2026-09-24)

**Status: PARTIALLY CLOSED.** The three test files the PRD names as missing
were added; the suite's own pass/fail state now honestly reports a real gap
against the PRD's own P95 latency NFR rather than being silent about it.

| # | Task | File | Status |
|---|---|---|---|
| T1 | `POST /predict` / `GET /health` TestClient coverage: response schema, explanation shape, decision-vs-threshold consistency | `tests/integration/test_api.py` (new, 9 tests) | **DONE** — uses a fake `InferenceService` injected via `app.dependency_overrides` (the existing pattern from `tests/unit/test_api_prediction_schemas.py`), so it runs without `models/` present. Boundary-rejection coverage already existed and is not duplicated here. One premise fixed during writing: `GET /health` reads `app.state.inference_service` directly, not through `Depends()`, so `dependency_overrides` cannot make it report a loaded service — the test asserts what the route actually does, not what a first draft assumed |
| T2 | End-to-end train on a small subset | `tests/integration/test_training_pipeline.py` (new, 3 tests) | **DONE** — chains the real feature-engineering orchestration (`FeatureEngineer` + `_derive_causal_features`, the same sequence `preprocess.run_pipeline` uses) into a real `XGBTrainer.train()`/`predict_proba()` on a small seeded synthetic raw frame (400 rows). No existing test chained feature engineering through to a trained, scoring model — `test_train_serve_equivalence.py` covers the feature-engineering half only |
| T3 | `tests/performance/test_latency.py`: 100 predictions, assert P95 < 100ms | `tests/performance/test_latency.py` (new, 2 tests) | **DONE, and FAILING against the real deployed service — left failing, not adjusted.** Measures `InferenceService.predict()` directly (not the HTTP round trip) over 100 real raw transactions sampled from `data/raw/train_transaction.csv`, against the real artifacts in `models/`, explainability disabled (SHAP is ~140ms alone by design, off the base-scoring budget per ADR-001/P4-8). Skips when artifacts or raw data are absent, mirroring `test_real_ensemble_artifact_loads.py`'s pattern |

**T3's measured result (2026-09-24, this machine): P95 in the 500-560ms
range across repeated runs (501.5ms and 554.3ms observed; p50 454-475ms,
max 699-970ms) — roughly 5x the PRD's 100ms budget.** This is a genuine,
previously unmeasured finding, not a test defect: the correctness half of the
same fixture (`test_all_sampled_predictions_return_a_valid_probability`)
passes, so the service is answering correctly, just slowly. The user was
asked how to treat the failing assertion and chose to leave it failing and
document the gap rather than relax the threshold or mark it `xfail` — the
suite's default `pytest tests/` will therefore show this test red when real
artifacts are present, by design, until the underlying performance work is
done. Likely contributors, consistent with the already-documented PRD Phase 6
throughput finding (`docs/IMPLEMENTATION_PLAN.md` "PRD Phase 6 follow-up" —
~4 msg/sec vs. 200 tx/sec, ~250ms/scoring): the 3-model ensemble (XGBoost +
TFT + LightGBM) and the TFT sequence rebuild on every request are the
suspected dominant costs, not measured further here. No fix attempted in this
pass — recorded as a known, unresolved limitation.

**Bug found and fixed while writing this test:** the first version of
`_sample_transactions` called `pd.read_csv(_RAW_TX).tail(n)`, which reads and
materialises all ~590,540 rows (V1-V339 alone is ~1.6 GiB as float64) before
discarding all but the last `n`. Run alongside the already-resident XGBoost +
TFT + LightGBM artifacts the `real_service` fixture loads, this triggered a
`numpy.core._exceptions._ArrayMemoryError` on the second read (the module's
two tests each sampled independently). Fixed by counting rows once and
reading only the tail via `skiprows`, and by caching the sample at
module scope (`sampled_transactions` fixture) so the file is read exactly
once for both tests. Caught by running the full suite once before considering
this phase done — the isolated `pytest tests/performance/` run alone did not
reproduce it, since it never ran both tests back to back against a still-resident
service the way `pytest tests/` does.

`tests/integration/test_monitoring.py` already existed and needed no changes.
Full suite: **756 passed, 1 failed** (`test_p95_latency_under_100ms_for_100_predictions`,
expected per the finding above) as of `pytest tests/ -q` on 2026-09-24
(confirmed by three consecutive full-suite runs, 755-756 passed each time —
the single-test variance is unrelated to this work), +14 tests (9 + 3 + 2)
over the pre-Phase-10 baseline.

#### Coverage report (2026-09-24)

`pytest tests/ --ignore=tests/performance --cov=src --cov-report=term-missing
--cov-report=html` (performance tests excluded — they measure latency, not
coverage, and are slow against real artifacts):

**Total: 78% (5,726 statements, 1,277 missed) — meets the >75% bar the user
set for this checkbox; 2 points short of the PRD's original ≥80% figure.**
Per the user's explicit instruction, the remaining gap to 80% is recorded and
not chased further in this pass.

The gap is concentrated, not diffuse — five modules account for most of the
missed lines, and in every case the missed lines are the `main()`/argparse
CLI block and `if __name__ == "__main__":` orchestration, not the class or
function logic those scripts wrap. The logic itself is well covered by
existing unit tests (`test_xgb_trainer.py`, `test_tft_trainer.py`,
`test_lgbm_trainer.py`, `test_tune_xgb.py`) that exercise `XGBTrainer`,
`TFTTrainer`, `LGBMTrainer`, etc. directly; the CLI wrapper around them reads
real config, loads real 590k-row parquet files, and writes real MLflow runs —
running it under test would mean either mocking most of what the test is
supposed to verify, or genuinely running the full pipeline, which is what
`make train`/`make reproduce` are for, not `pytest`.

| Module | Coverage | Missed lines (mostly) |
|---|---|---|
| `src/data/data_loader.py` | 18% | `load_raw()`'s real-CSV read path (PyArrow + pandas-chunked fallback) |
| `src/training/tune_tft.py` | 19% | Optuna study orchestration / CLI |
| `src/data/download_data.py` | 25% | Kaggle API download script |
| `src/training/train_lgbm.py` | 43% | `main()` (lines 286-555) |
| `src/training/train_tft.py` | 44% | `main()` (lines 1020-1329) |
| `src/training/train_xgb.py` | 47% | `main()` (lines 334-628) |
| `src/training/tune_xgb.py` | 51% | Optuna study orchestration / CLI |
| `src/training/losses.py` | 58% | `FocalLoss` edge-case branches |
| `src/api/main.py` | 67% | Kafka-consumer-disabled branches, some lifespan edge cases |

Modules directly exercised by unit/integration tests are high: `src/serving/`
(94-100% across all files), `src/utils/` (96-100%), `src/models/ensemble.py`
(100%), `src/models/gnn_model.py` (100%), `src/training/manifest.py` (100%),
`src/serving/metrics.py` (100%).

HTML report written to `htmlcov/` (gitignored, not committed — regenerate
with the command above).

---

### PRD Phase 11 — Packaging, Documentation & Portfolio Polish (2026-09-24)

**Trigger:** user directed implementation of PRD Phase 11 (docs/prd.md
§10.1-10.5, "Done When" checklist at the phase's end), with `ecc:tdd-guide`
and `mle-reviewer` used to verify the work rather than self-report it.

**Gap found before any change:** `README.md` was a one-line stub (`# Fraud-XAI`)
— everything else the phase's action items call for (docker-compose.yml,
Dockerfile, `reports/RESULTS.md`) was already substantially done by prior
sessions, well beyond the PRD's original template.

**What was done:**

1. **README.md rewritten** to the full PRD §10.1 spec: architecture diagram,
   quick start, `curl` API example with real request/response shapes, results
   table (drawn from `reports/RESULTS.md` §6's verified numbers), project
   structure tree, technical-decisions section, dataset instructions,
   screenshots (SHAP beeswarm + waterfall, referencing existing
   `reports/figures/*.png`), and a Known Limitations section stating the P95
   latency and precision gaps plainly rather than omitting them.
2. **Notebook outputs kept, not cleared** — a deliberate deviation from the
   PRD's literal §10.4 instruction ("cleared outputs committed, clean diff").
   User's explicit call: executed outputs (plots, SHAP charts) render on
   GitHub without cloning, which serves the portfolio purpose better than a
   clean diff. Recorded here so the deviation is visible, not silent.
3. **`make notebooks` target added** (`Makefile`) — `01_eda`,
   `03_model_comparison`, `04_shap_analysis` now have an automated
   `nbconvert --execute` path, matching the pattern `05_business_impact`
   already had via `make business-impact`. Before this, 3 of 4 notebooks had
   no automated "executes cleanly" verification at all — found by the
   `ecc:tdd-guide` review below, not assumed.

**Verification (both run as independent background reviews, not self-checked):**

- **`mle-reviewer`** fact-checked every technical claim in the new README
  against the actual codebase (not against the README's own text): result
  numbers vs. `reports/RESULTS.md` §6, API request/response shape vs.
  `src/api/schemas/prediction.py`, architecture claims vs.
  `src/api/main.py`/`docker-compose.yml`, every `make` target vs. `Makefile`,
  Known Limitations figures vs. `RESULTS.md` §13, and every referenced file
  path. **Verdict: no material inaccuracies found.**
- **`ecc:tdd-guide`** verified the Phase 11 "Done When" checklist item by
  item against the repo, and re-ran `pytest tests/ -q --ignore=tests/performance`
  via the `fraudx` conda env to confirm the "756 passed, 1 known failure"
  claim in `RESULTS.md` is still current, not stale: **755 passed, 0 failed**
  in 124.78s (the "1 known failure" is the P95 latency assertion, which lives
  in the excluded `tests/performance/` — consistent, not a regression).

**Checklist result (docs/prd.md Phase 11 "Done When"):**

| # | Item | Status |
|---|---|---|
| 1 | `docker compose up` brings 4 services online <2min cold start | **PARTIAL** — 4 services confirmed in `docker-compose.yml`; `fraud-api`/`kafka` have healthchecks, `prometheus`/`grafana` do not (only `depends_on`, no readiness gate). Not run end-to-end this session (Docker daemon not exercised). |
| 2 | `docker compose ps` shows exactly 4 containers | **PASS** (structural) — `fraud-api`, `kafka`, `prometheus`, `grafana`, explicit `container_name` on each |
| 3 | Public README renders correctly | **DONE** this session, verified by `mle-reviewer` |
| 4 | All notebooks execute cleanly | **FIXED this session** — was 1/4 automated (`05_business_impact` only); `make notebooks` now covers the other 3. Not re-run end-to-end after adding (needs `make data` + `make train` artifacts present) |
| 5 | `reports/RESULTS.md` has real numbers, no TBD | **PASS** (pre-existing, confirmed by `ecc:tdd-guide`) |
| 6 | No committed data/model/`.env` files | **PASS** — `git ls-files` filtered to `data/`, `models/`, `.env` returns empty |
| 7 | Pre-commit hooks pass on all files | **UNVERIFIED** — `.pre-commit-config.yaml` exists (black/isort/flake8, all pinned), but running `pre-commit run --all-files` fetches hook repos over the network on first run; not executed this session to avoid that dependency. Flagged, not silently assumed passing. |

**Explicitly not done this session:**
- `docker compose up` was not actually run end-to-end to confirm the <2min
  cold-start budget (Docker was available on this machine but not exercised
  for this phase — see item 1 above).
- `prometheus`/`grafana` healthchecks were not added — item 1's gap is
  recorded, not closed. Would need an image with a usable health endpoint
  (Grafana ships `/api/health`; Prometheus ships `/-/healthy`) wired into
  `docker-compose.yml`.
- `pre-commit run --all-files` was not executed (item 7).
- The scratch/debug files at repo root (`__tmp_verify.py`, `scratch_eval.py`,
  `scratch_test.py`, `test_tuning.db`, `test_tuning_2.db`, `tft_tuning.db`,
  `analyze_optuna.py`, `optuna_analysis.txt`, `gnn_arm_search_run.log`) were
  noticed but left untouched — out of this session's agreed scope, and
  removing them without confirming they're not still-in-use working files
  risked discarding in-progress work.
