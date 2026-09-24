# Phase 2 Results — Leakage-Free Pipeline, Phase C/D Evaluation Integrity, PRD Phase 9 Precision Work, and the 2026-09-09 4-Agent ML Review

<!-- Regenerated: 2026-09-09, after the 4-agent ML review (mle-reviewer +
     architect + code-architect + code-reviewer) and its remediation. This
     revision supersedes the 2026-09-08 revision that stopped at Phase 9 step
     9.3 and never mentioned 9.4, ADR-004, or the 2026-09-09 P9-6 TFT change —
     that gap was itself one of the review's findings (G7, docs/
     IMPLEMENTATION_PLAN.md "Phase G"). All headline numbers below are
     transcribed from the manifests and reports/ensemble_results.json
     written by the 2026-09-09 retrain, not carried forward from any earlier
     revision of this document. See §1 for what changed and why, and the new
     §9 for the full review writeup. -->

**Provenance**

| Field | Value |
|---|---|
| MLflow run ID (XGBoost) | `9d60068e5507449e8dd90ccd9e576071` |
| MLflow run ID (TFT) | `64c3745f50a34cf882d1571fcb7c9f3f` |
| MLflow run ID (LightGBM) | `990e44b3a86b4cc488cacd2f135a7892` |
| MLflow run ID (Ensemble) | `69d3a4d5c5ff420a8e119dd3febdc4c3` |
| MLflow experiment ID | `775946513825380473` |
| Tracking URI | `file:./mlruns` |
| Date | 2026-09-09 (post-review retrain, all 3 models + ensemble) |
| Pipeline | `preprocess.py` (unchanged since 2026-09-08) → `train_xgb.py` → `train_tft.py` → `train_lgbm.py` → `scripts/run_ensemble_eval.py --promote` |
| Dataset | IEEE-CIS `train_transaction.csv` + `train_identity.csv` (590,540 rows, 24.4% identity match) |
| Test suite at time of run | `pytest tests/ -q` → **667 passed** |
| Git SHA | `f73b8af1cb0d1a77a7f9228b91c51bd26055b92d` (working tree has uncommitted changes on top — see `git status`; this SHA is what every manifest records, not a claim that the tree is clean) |
| Config hash (XGBoost) | `a58c639797622173eceec176a8a882b4894e311c234b3ad9b436142097bf8926` |
| Config hash (TFT/LightGBM/Ensemble) | `9d647d076538a1c070e1ce103035877f29d3ffa9d875224ca2735c56efb1f9fe` |
| Dataset hash (all four) | `d20f03c02e8de9b1358025a639374c1913a0dce8f8c129b576839e01ebed06c6` |
| Model manifests | `models/xgb_model.manifest.json`, `models/tft_model.manifest.json`, `models/lgbm_model.manifest.json`, `reports/ensemble_results.manifest.json` |
| Deployed ensemble spec | `models/ensemble.json` (checksummed; `model_run_ids` inside `reports/ensemble_results.manifest.json` match the three model manifests' `mlflow_run_id` exactly, verified by direct comparison — this is the property that was broken going into this session, see §9) |

Every field above except the date/pipeline/dataset-description rows is
recorded *in the artifact itself* by `src/training/manifest.py` — transcribed
from the four manifests above, not maintained by hand. The two config hashes
differ only because `model.lightgbm.early_stopping_rounds` was changed
*between* the XGBoost run and the TFT/LightGBM runs (see §9); no XGBoost
hyperparameter differs as a result. All four manifests carry the identical
`dataset_hash`, which is the provenance property that actually matters for
comparing the four numbers against each other — preprocessing was not re-run
this session.

Inspect any run with `mlflow ui --backend-store-uri ./mlruns` or
`mlflow.get_run("<run id above>")`.

---

## 1. What changed and why this report exists

This revision exists because the previous one (regenerated 2026-08-20,
touched only in prose since) had drifted far enough from the code and
artifacts that a subsequent 4-agent review flagged it as a finding in its own
right (G7, `docs/IMPLEMENTATION_PLAN.md` "Phase G"): it stopped at PRD Phase 9
step 9.3, never mentioned step 9.4, the two-stage cascade rejection (step 9.5
/ ADR-004), or the 2026-09-09 P9-6 TFT regularisation change — not merely
*out of date*, but silent about three real events.

**The headline change in this revision:** on 2026-09-09, the user requested a
deep-dive review of the ML architecture, hyperparameters, preprocessing, and
ensembling, run in parallel across four ECC agents (`mle-reviewer`,
`architect`, `code-architect`, `code-reviewer`). Before dispatching them, a
direct inspection of the on-disk artifacts found that **the deployed ensemble
spec (`models/ensemble.json`, frozen 2026-09-08 18:29) was blending a TFT
artifact that had since been overwritten** by a same-day-morning retrain
(P9-6, 2026-09-09 05:41) — a different model, with a different calibrator, on
a different probability scale, than the one the spec's weights were fitted
against. All four agents converged on this and three related findings; the
full table is in §9. **This revision's provenance table above describes the
state after that review's remediation was carried out and verified** — the
deployed spec, the evaluation report, and the three model manifests now all
describe the same training generation, checked field-by-field (§9.3).

The rest of this section preserves the two earlier revisions' history, as
originally written, for the audit record.

### 2026-08-20 revision (preserved)

`features.target_encoding_label_lag_days` (Audit-F6) had landed in code
2026-08-19 evening but was not exercised by any training run until then —
`preprocess.py` was re-run, producing a new `dataset_hash`, and all three
models were retrained on it. Test PR-AUC dropped for every model at the time:
XGBoost 0.5602 → 0.5289 (raw), TFT 0.4906 → 0.4137, LightGBM's post-fix
0.5349 → 0.5165. This was the expected direction and the reason F6 existed —
see the 2026-08-20 revision's full text in git history for the complete
writeup; it is not repeated here because every number it reported has since
been superseded by PRD Phase 9's feature and imbalance changes (§4, §6) and
is no longer the honest current baseline.

### 2026-08-07/2026-08-19 leakage fix (preserved)

The original report described a model trained on features whose transformers
— IncrementalPCA, frequency maps, label encoders, imputation constants, and
the target-encoding prior — had all been fitted on the complete dataset
*before* the train/val/test split. Test rows therefore informed the feature
representation. Phase A of `docs/IMPLEMENTATION_PLAN.md` restructured
`run_pipeline()` so the split is decided first and every stateful transformer
is fitted on train rows only, with `fit=False` on val and test. **This
leakage-free split discipline is unchanged and has been re-verified by every
retrain since**, including this session's (§9.4, mle-reviewer's leakage
review: "no leakage found, and this is the strongest part of the repo").

---

## 2. Data and splits

Time-based 70/10/20 split on `TransactionDT`, no shuffle, decided before any
transformer is fitted. **Unchanged since 2026-08-19** — preprocessing was not
re-run this session; the numbers below are read from the current
`data/processed/*.parquet` files, not carried forward from an earlier report.

| Split | Rows | Fraud rate | Temporal boundary |
|---|---|---|---|
| Train | 413,378 | 3.52% | ends `DT=10,437,996` |
| Validation | 59,053 | 3.49% | ends `DT=12,192,743` |
| Test | 118,109 | 3.44% | starts `DT=12,192,842` |

**Final feature count: 184** — up from 171 in the 2026-08-20 revision. The
growth is PRD Phase 9's feature engineering, in order: 9.2 added 5 UID
(`card1_addr1_D1n`) aggregate columns (171→176), the 9.2-review fix added
`uid_count` (176→177), and 9.4 added the multi-window RFM/velocity aggregates
and `dist1` engineering described in `docs/prd.md` PRD Phase 9 §9.4
(177→184). V1–V339 are still reduced to 30 IncrementalPCA components,
**fitted on the 413,378 train rows only**.

---

## 3. Feature engineering

Summarised by leakage class, verified current by the mle-reviewer pass in
§9.4 (a fresh review, not a re-statement of the 2026-08-20 text):

**Stateless / strictly backward-looking** — computed over the full temporally
ordered frame, deliberately. Each reads only a transaction's own past, which
is exactly what production has.

- Cyclical (sin/cos) encodings of transaction hour and day of week
- `amount_log` and amount decomposition features
- Email domain parsing; D-column and C-column summaries
- Card-level expanding aggregates via `shift(1)` — `tx_count_per_card`,
  `mean_amount_per_card`, `max_amount_per_card`, `std_amount_per_card`
- Anomaly signals: `amount_vs_mean_ratio`, `amount_zscore_per_card`
- Velocity: `tx_sum_per_card`; address, device and browser parsing
- Interactions such as `interaction_hour_amount`
- **(PRD Phase 9.4, new)** Fixed rolling-window transaction counts per card
  (trailing 10min/1h/24h) and short-window-vs-long-window amount deviation,
  distinct from the expanding aggregates above — see `docs/prd.md` §9.4 for
  the Whitrow/Bahnsen citations and the worked example of why an expanding
  count cannot express a recent burst.
- **(PRD Phase 9.4, new)** `log1p(dist1)` and a train-fit top-quintile
  `dist1_high` flag, scoped to `ProductCD == 'W'` rows only (where `dist1` is
  populated) — `dist2` was investigated and explicitly **not** turned into a
  feature; its presence→fraud direction flips by `ProductCD` and the
  aggregate correlation is a product-mix confound, not a per-product signal.

**Stateful** — fitted on train only, applied to val/test with `fit=False`:
card-hash frequencies, expanding target encoding (train-only global prior,
with a 30-day label-lag exclusion — Audit-F6, unchanged), imputation fill
values, categorical label/frequency encoders, and the PRD Phase 9.2 UID
aggregate maps (`uid_freq`, `uid_amt_mean`, `uid_amt_std`, `uid_amt_ratio`,
`uid_d1_mean`, `uid_count`; `_UID_MIN_COUNT=3` floor to avoid self-inclusion
on singleton cohorts — the fix for the 9.2 mle-review BLOCK).

Null-count meta features are computed on the **raw** frame before PCA, so
V-column missingness is captured rather than erased.

---

## 4. Class imbalance

Fraud rate is ~3.5%. **`scale_pos_weight` is now 1 (no class weighting) for
both XGBoost and LightGBM** — changed from 27.43/29 by PRD Phase 9 step 9.3,
on evidence from `scripts/run_ablation.py` (`reports/imbalance_ablation_results.json`):

| `scale_pos_weight` | Val PR-AUC | Val best F1 @ threshold |
|---|---|---|
| **1.0 (none)** | **0.5907** | **0.578 @ 0.220** |
| 7.25 (25% of old config) | 0.5784 | 0.561 @ 0.595 |
| 14.5 (50%) | 0.5726 | 0.563 @ 0.667 |
| 21.75 (75%) | 0.5648 | 0.555 @ 0.730 |
| 29 (100%, old config) | 0.5648 | 0.557 @ 0.789 |
| SMOTE (no spw) | 0.5338 | 0.517 |
| SMOTE + spw=27.43 | 0.5343 | 0.517 |

**Monotonic** — lower `scale_pos_weight` → higher val PR-AUC and best F1.
SMOTE degrades val PR-AUC by ~0.057 vs. the spw=1 point and does not help
combined with class weighting. Selected on **validation only** (this script
never loads test), which the mle-reviewer's 2026-09-09 pass re-confirmed by
reading `run_ablation.py` directly.

> **Caveat carried forward from the 2026-09-09 review, not previously
> recorded here (finding on this ablation, not a blocking one — it was not
> re-run this session):** the ablation grid runs at `n_estimators=100,
> max_depth=6` as a fast proxy for the production model's `n_estimators=887,
> max_depth=10`, with no early stopping in any arm, and was never run for
> LightGBM directly (the LightGBM change "mirrors" the XGBoost result per the
> config comment, without its own ablation arm). The monotonic *direction* is
> credible evidence; the exact magnitude at production capacity is not
> independently confirmed. Separately, `scripts/run_ablation.py`'s own
> `CONFIG_SPW = 29.0` constant and its label `"spw_29 (100%, current)"` are
> now **stale** — the shipped config value is 1, not 29 — so the grid's
> labels are inverted relative to what is actually deployed. Neither issue
> was on the user's agreed 5-item remediation list this session and neither
> has been fixed; recorded here so the gap is visible rather than silent.

For TFT: **focal loss with gamma=1.0, alpha=0.10** (reduced from 2.0/0.25 by
step 9.3, mirroring the GBDT `scale_pos_weight` reduction), and — as of this
session's remediation (§9.3) — **`sampling_strategy: oversample`** is back in
place (`WeightedRandomSampler`, ~27x replication of the fraud class per
epoch). This was briefly set to `"none"` by PRD Phase 9 step P9-6
(2026-09-09 05:41) to fight a measured train/val memorisation gap
(train 0.9519 vs val 0.4975); that change is **reverted** in this session's
remediation because, combined with a separate scaler-leak fix (§9.2),
oversample-off produced the worst TFT result in this project's history. See
§9.3 for the full trace.

---

## 5. Hyperparameters actually used

These are the values read back from this session's MLflow runs — not
aspirational config.

### XGBoost

| Parameter | Value |
|---|---|
| `n_estimators` (ceiling) | 887 |
| Best iteration (this run) | **659** |
| `max_depth` | 10 |
| `learning_rate` | 0.09696200612789875 |
| `subsample` | 0.8382111670615123 |
| `colsample_bytree` | 0.6186070548544907 |
| `min_child_weight` | 4 |
| `reg_alpha` | 0.014571229075512624 |
| `reg_lambda` | 1.1091336812755855e-07 |
| `gamma` | 5.543434625011016e-08 |
| `scale_pos_weight` | **1** (P9-3, was 27.43) |
| `random_state` | 42 |
| `tree_method` / `device` | `hist` / `cuda` |
| `early_stopping_rounds` | 100 (on validation, `aucpr`) |

### LightGBM

| Parameter | Value |
|---|---|
| `n_estimators` (ceiling) | 1200 |
| Best iteration (this run) | **1196** — "Did not meet early stopping," per the training log; validation `average_precision` was still improving marginally at the ceiling |
| `max_depth` | -1 (unlimited) |
| `num_leaves` | 64 |
| `learning_rate` | 0.02 |
| `min_data_in_leaf` | 50 |
| `reg_alpha` / `reg_lambda` | 0.1 / 1.0 |
| `scale_pos_weight` | **1** (P9-3, was 29) |
| `early_stopping_rounds` | **100** — **re-enabled this session** (was 0/disabled since 2026-08-19, a diagnostic workaround for a training failure whose root cause, `scale_pos_weight=29` interacting with `min_data_in_leaf=50`/`reg_lambda=1.0`, no longer exists now that `scale_pos_weight=1`). Re-enabling did not change the result to 4 decimal places (0.54176 → 0.54180 test PR-AUC) — the model simply uses its full budget either way. |

> **Tuning debt, still open.** `config/tuned/` remains **empty** — no Optuna
> study has been run against any version of the leakage-free, current
> 184-feature representation. The hyperparameters above are still the
> pre-Phase-A study's values, carried forward across every feature-set and
> imbalance change since. Re-tuning is a modelling task, not a provenance
> one, and the mle-reviewer's 2026-09-09 pass flagged this as HIGH: "tuning
> debt is material: it is the single largest untested lever on the precision
> gap."

### TFT

| Parameter | Value | Changed this session? |
|---|---|---|
| `hidden_size` | 16 | no |
| `attention_head_size` | 4 | no |
| `num_lstm_layers` | 1 | no |
| `dropout` | **0.3** | **reverted** from 0.4 (P9-6, 2026-09-09 morning) |
| `hidden_continuous_size` | 32 | no |
| `learning_rate` | 0.0023288031819030802 | no |
| `weight_decay` | **1e-5** | **reverted** from 1e-2 (P9-6) |
| `max_epochs` / `patience` | 100 / 10 | no |
| `use_amp` | true | no |
| `imbalance.sampling_strategy` | **`oversample`** | **reverted** from `none` (P9-6) |
| `imbalance.focal_loss_gamma` / `alpha` | 1.0 / 0.10 | no (unchanged since P9-3) |

Training log for this run: 32 epochs, early stopping triggered ("no
improvement for 10 epochs"), 91.0 minutes wall-clock on an NVIDIA GeForce RTX
4060 Laptop GPU. Restored best-epoch weights (val PR-AUC 0.5393 at that
epoch) before final evaluation.

---

## 6. Model performance

All three models share the identical `dataset_hash` above and were trained
in immediate succession this session (2026-09-09), so the comparison below is
apples-to-apples — not assembled from different vintages, which is precisely
the property that was broken before this session's remediation (§9).

### Standalone test PR-AUC (calibrated probabilities, the space the ensemble blends on)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Overfit gap (train−test) |
|---|---|---|---|---|
| **XGBoost** | 0.6703 | 0.5382 | 0.9091 | 0.4618 |
| **TFT** | 0.5317 | **0.4670** | 0.8771 | **0.1638** |
| **LightGBM** | 0.6559 | 0.5418 | 0.9128 | 0.3875 |
| GNN-GraphSAGE *(Phase 12, rejected — not in the ensemble)* | 0.5260 | 0.4410 | 0.8849 | 0.2155 |

The **GNN-GraphSAGE** row is a PRD Phase 12 architecture trial, recorded here
per this project's norm of documenting rejected approaches; it is **not part of
the deployed ensemble**. See §11 below and `docs/adr/ADR-005` for the full
evaluation. Short version: it scores 0.4410 test PR-AUC — below every GBDT
standalone and below TFT's best — with a 0.085 val→test drop that confirms the
external paper's 0.6334 was substantially a non-temporal-split artifact.

**TFT's 0.4670 test PR-AUC and 0.164 overfit gap are the best TFT result in
this project's history** — every prior TFT run (dating back to the original
2026-08-07 baseline) had a test PR-AUC between 0.29 and 0.43 and an overfit
gap ≥ 0.21. This session found and fixed a genuine scaler-fit-scope leak in
the TFT training path (§9.2) and, separately, discovered that PRD Phase 9's
P9-6 regularisation change had been making things worse rather than better
(§9.3) — reverting it, on top of the leak fix, produced this result. TFT is
still the weakest of the three standalone, and its overfit gap being the
smallest of the three is a genuinely different property (it generalises
better, it just generalises to a lower ceiling) — see §9.5 for why this does
not translate into a large ensemble weight.

LightGBM (0.5418) now narrowly **exceeds** XGBoost (0.5382) standalone —
worth stating plainly since earlier revisions of this report described
XGBoost as "the strongest single model"; that is no longer the most accurate
reading, though the two are close enough (0.0036 apart) that neither should
be read as clearly dominant.

### The 3-way ensemble (XGBoost + TFT + LightGBM)

**Pairwise validation-probability correlation** (the pre-registered diversity
check, computed fresh this session — not the same numbers as any prior
revision, since all three underlying models changed):

| Pair | Correlation |
|---|---|
| XGBoost ↔ TFT | 0.8242 |
| XGBoost ↔ LightGBM | **0.9440** |
| TFT ↔ LightGBM | 0.8652 |

Same qualitative pattern as every prior revision: XGBoost and LightGBM
(both GBDTs on the identical feature set) correlate more with each other than
either does with TFT. TFT remains the architecturally decorrelated input.

**Min-lift gate — now scored on validation, not test (2026-09-09 fix, §9.1).**
Prior revisions of this report, and the code itself, gated LightGBM's
ensemble membership on the *test*-split PR-AUC lift. This is fixed as of this
session:

| | Weights (val-optimized) | Val PR-AUC | Test PR-AUC |
|---|---|---|---|
| 2-way baseline (XGB+TFT) | 0.912 XGB / 0.088 TFT | 0.6767 | 0.5408 |
| 3-way challenger (+LightGBM) | 0.618 XGB / 0.008 TFT / 0.374 LGBM | 0.6840 | **0.5502** |

**Gate lift (validation): +0.007321 ≥ 0.005 threshold → KEEP LightGBM.**
Test-split lift is now recorded only as a diagnostic
(`lgbm_lift_test_pr_auc_diagnostic: 0.009393`) and never influences the
keep/drop decision — see §9.1 for why this matters (LightGBM's membership had
flipped across at least four evaluation runs under the old test-gated logic).

**Bootstrap weight-stability diagnostic** (50 resamples of the validation
set, re-run the weight search each time):

| Weight | Mean | Std | 90% CI (p05–p95) |
|---|---|---|---|
| `w_xgb` | 0.607 | 0.044 | [0.55, 0.70] |
| `w_tft` | **0.051** | 0.007 | **[0.05, 0.05]** |
| `w_lgbm` | 0.342 | 0.044 | [0.25, 0.40] |

**Read this table carefully — it is the evidence behind the Phase 12
decision (§9.5), not a red flag on its own.** The *point-optimized* search
below assigns TFT a weight of 0.008; the bootstrap distribution above
(a coarser, more stable diagnostic search) puts its mean at 0.051 with an
essentially zero standard deviation. Both are correct answers to slightly
different questions: TFT sits in a **flat region of the blend's objective
surface** (a direct consequence of its 0.82 correlation with XGBoost — moving
weight between two correlated inputs barely moves PR-AUC), so the exact point
the fine-grained search lands on (0.008 here) is close to arbitrary within
that flat region, not evidence of an error. This is a materially different
situation from the one this review started with — where the deployed 0.008
sat *below* the bootstrap's 5th percentile because it came from a stale,
already-overwritten model (§9). Here, 0.008 and 0.051 are two honest readings
of the same, currently-valid model generation; they simply disagree by
design, because the search that's more expensive is also more sensitive to
exactly where the flat region's optimum sits.

**Final deployed ensemble** — test PR-AUC **0.5502**, ROC-AUC **0.9154**,
threshold **0.012831** (per-model-calibrated probability scale, selected on
validation under the `net_of_principal` cost convention, frozen):

| Metric | Value at frozen threshold (test) |
|---|---|
| Precision | 10.30% |
| Recall | 90.03% |
| F1 | 0.1848 |
| Accuracy | 72.67% |

Confusion matrix (118,109 test rows, 4,064 actual fraud):

| | Predicted legit | Predicted fraud |
|---|---|---|
| **Actual legit** | 82,165 (TN) | 31,880 (FP) |
| **Actual fraud** | 405 (FN) | 3,659 (TP) |

### Reading the model ranking (2026-09-09)

- **LightGBM and XGBoost are close, not XGBoost-dominant.** 0.5418 vs 0.5382
  standalone test PR-AUC — a 0.0036 gap, within the range where re-running
  with a different seed could plausibly flip the ordering. Both remain far
  ahead of TFT (0.4670) and highly correlated with each other (0.944).
- **TFT is kept in the deployed blend, at a small and admittedly unstable
  weight, on the strength of its decorrelation** (0.82 vs XGBoost, the
  lowest pairwise correlation of the three) rather than its standalone
  ranking quality. §9.5 records the decision **not** to keep pushing this
  further: TFT's own best-ever result this session (0.4670 test PR-AUC,
  §9.2/§9.3) still only translates into a ~0.05-weight, flat-region
  contribution to the blend. The user's decision, recorded in `docs/prd.md`
  PRD Phase 12, is to stop optimizing TFT specifically and evaluate a
  different architecture (GNN-GraphSAGE) next, on evidence that architecture
  reports a materially higher PR-AUC on the same underlying dataset.
- **LightGBM's ensemble membership is decided validation-first now.** The
  min-lift gate previously alternated LightGBM in and out across evaluation
  runs (dropped at step 9.2, re-added at 9.3/9.4) partly because it was
  reading test-split noise as a real signal each time. This session's fix
  (§9.1) removes that specific failure mode; whether LightGBM's *val*-gated
  membership itself remains stable across future retrains has not yet been
  observed across more than one cycle.

---

## 7. Artifacts produced by this run

| Artifact | Path |
|---|---|
| XGBoost model (booster, UBJ) + metadata + checksums + manifest | `models/xgb_model.ubj`, `.meta.joblib`, `.checksums.json`, `.manifest.json` |
| TFT weights (`state_dict` only) + metadata + checksums + manifest | `models/tft_model.weights.pt`, `.meta.joblib`, `.checksums.json`, `.manifest.json` |
| LightGBM model (native text format) + metadata + checksums + manifest | `models/lgbm_model.txt`, `.meta.joblib`, `.checksums.json`, `.manifest.json` |
| Deployed ensemble spec + checksum | `models/ensemble.json`, `models/ensemble.checksums.json` |
| Ensemble evaluation results + manifest | `reports/ensemble_results.json`, `reports/ensemble_results.manifest.json` |
| Fitted transformers | `data/processed/transformers/` and `models/transformers/` (dual-written per ADR-001 §3.5) |
| Processed splits | `data/processed/{train,val,test}_{features,labels}.parquet` (unchanged this session) |

> **Stale artifacts, not regenerated this session — flagged rather than
> silently presented as current:** `reports/slice_metrics.{csv,md}` (still
> shows the frozen threshold `0.0004`, which is no longer XGBoost's
> current optimal threshold, `0.000963`), `reports/imbalance_ablation_results.json`
> (from the 2026-09-08 P9-3 run — its findings are still directionally valid
> per §4, but it predates this session's models), and every
> `reports/phase9_step9_*.json` file (all from 2026-09-08, predate this
> session's retrain entirely). None of these were rebuilt because doing so
> was not part of the user's agreed 5-item remediation scope for this
> session; re-running `scripts/run_slice_metrics.py` and
> `scripts/run_phase9_eval.py` against the current models is natural
> follow-up work, not done here.

---

## 8. Status against the remediation plan

Phases A, B, C, D, E, and PRD Phases 4, 6, 7, 8 remain closed as recorded in
prior revisions and `docs/IMPLEMENTATION_PLAN.md`. **PRD Phase 9 is closed**
(ADR-004, 2026-09-09: the ≥30% precision at ≥80% recall target band is
infeasible on this dataset with the current feature set — see
`docs/adr/ADR-004-operating-point-selection.md` for the closed-form
argument). This section covers only what changed in the current session.

---

## 9. The 2026-09-09 4-Agent ML Review and its remediation

**Trigger.** The user requested a deep-dive review of the ML architecture,
hyperparameter setup, data preprocessing, and training/ensembling lifecycle,
deployed in parallel across four ECC review agents (`mle-reviewer`,
`architect`, `code-architect`, `code-reviewer`), after flagging that this
report's accuracy was uncertain. A pre-dispatch inspection of the on-disk
artifacts (before any agent ran) found the trigger finding directly: the
deployed `models/ensemble.json` (frozen 2026-09-08 18:29) was blending a TFT
artifact that had since been silently overwritten by a same-day 05:41 retrain
(PRD Phase 9 step P9-6) — different training config, different calibrator,
different probability scale, same `dataset_hash` (so the one hard-fail
consistency check passed) but a different model entirely. All four agents,
seeded with this and six related discrepancies, converged on (and in three
cases, corrected each other's initial framing of) the following findings.
Full detail, including each agent's exact language and the cross-agent
corrections, is in `docs/IMPLEMENTATION_PLAN.md` "Phase G"; this section is
the condensed, results-oriented version.

| # | Finding | Severity | Fixed this session? |
|---|---|---|---|
| G1 | LightGBM's ensemble membership was decided by a min-lift gate reading **test**-split PR-AUC, re-consulted across ≥4 evaluation runs | CRITICAL | **Yes** — §9.1 |
| G2 | TFT's `QuantileTransformer` was fit on **train+val concatenated**, contradicting its own code comment ("Scaler is fit once, on train only") | HIGH | **Yes** — §9.2 |
| G3 | The deployed ensemble spec carried no `config_hash`/`git_sha`/`created_at`/per-model run-id lineage, making the stale-TFT state (above) undetectable by inspection alone | CRITICAL (structural) | **Partially** — the immediate instance is resolved by this session's re-promote (§9.4); the structural fix (stamping this into `models/ensemble.json` itself) is not yet implemented |
| G4 | `make train` silently omitted LightGBM, so the documented "retrain everything" command left a 0.37-weight model at a stale vintage by construction | HIGH | **Yes** — `Makefile` now runs all three trainers |
| G5 | `train_lgbm.py` hardcodes its model path instead of reading `serving.lgbm_model_path` from config, unlike its two sibling trainers | MEDIUM | No — not on the agreed scope for this session |
| G6 | LightGBM's `early_stopping_rounds` had been disabled since 2026-08-19 as a workaround for a failure condition that no longer exists | MEDIUM | **Yes** — §9.3, re-enabled, confirmed no regression |
| G7 | This report itself had drifted (stopped at step 9.3, silent on 9.4/9.5/ADR-004/P9-6) | MEDIUM | **Yes** — this regeneration |
| G8 | `scripts/run_ablation.py`'s `CONFIG_SPW = 29.0` constant and its label are now stale relative to the shipped `scale_pos_weight=1` | MEDIUM | No — not on the agreed scope; recorded in §4 |

### 9.1 — Min-lift gate fixed to read validation, not test

`scripts/run_ensemble_eval.py`'s `apply_min_lift_gate` call site now passes
`val2, val3` (validation PR-AUC for the 2-way and 3-way blends) instead of
`test2, test3`. Test-split lift is still computed and logged
(`lgbm_lift_test_pr_auc_diagnostic`) but no longer influences the keep/drop
decision. A regression test (`test_min_lift_gate_call_site_uses_validation_not_test`
in `tests/unit/test_ensemble_eval.py`) source-inspects the call site directly,
since the gate function's own unit test — which only exercises its pure
arithmetic — cannot see which split feeds it in production. 15/15 tests in
that file pass; full suite 667/667.

### 9.2 — TFT scaler-fit-scope leak fixed

`train_tft.py`'s `_build_sequences_for_splits` correctly concatenates train
and validation rows before building sequences (Phase B6 — a val card's first
transaction needs its real train history, not a padded window at the split
boundary). The bug was that `fit_scaler=True` was passed straight into that
combined build, so `SequenceBuilder`'s `QuantileTransformer` was fit over
train+val together rather than train alone — reproduced and proven at
runtime by the reviewing agent (a disjoint-range test: the fitted scaler's
quantile boundary reached a value that existed only in the val-only rows).

Fix: a new `SequenceBuilder.fit_scaler_on(X)` method fits the scaler on an
explicit frame with no sequence-building side effect. `TFTTrainer.train` now
calls `fit_scaler_on(X_train)` before the train+val combined build, which now
runs with `fit_scaler=False` so it only transforms with the already-fitted,
train-only scaler. Four new tests cover this (two proving `fit_scaler_on`'s
isolation and the reuse path, two source-inspection guards on the call
sites — one in `test_ensemble_eval.py`'s style, one in
`test_tft_boundary_history.py`). 33/33 relevant tests pass; full suite
667/667.

### 9.3 — Full retrain, a real regression found and reverted, then promoted

All three models were retrained on the unchanged `dataset_hash` and
`scripts/run_ensemble_eval.py --promote` was re-run. The first TFT retrain
(with the scaler fix from §9.2, but with PRD Phase 9 step P9-6's
regularisation settings — `dropout=0.4`, `weight_decay=1e-2`,
`sampling_strategy="none"` — still active) produced a materially **worse**
result than any prior TFT run: test PR-AUC 0.2902, val/test gap 0.314. This
was investigated rather than accepted or silently worked around: comparing
against P9-6's own prior measurement (which had already shown P9-6 making
test PR-AUC worse, 0.4269→0.4009, without ever being re-blended into the
ensemble) pointed at P9-6 itself, not the scaler fix, as the likely cause.

P9-6's three settings were reverted to their pre-P9-6 values (`dropout=0.3`,
`weight_decay=1e-5`, `sampling_strategy="oversample"`) and TFT was retrained
a second time, with the scaler fix still in place. Result: **test PR-AUC
0.4670, overfit gap 0.164 — the best TFT result in this project's history**,
confirming P9-6 (not the scaler fix) had caused the collapse. LightGBM was
also retrained in this cycle with `early_stopping_rounds` re-enabled (§9.3 of
the G-table above; best iteration 1196/1200, no measurable change vs. the
disabled-early-stopping run).

The ensemble was then re-evaluated and promoted:
`models/ensemble.json`'s weights (`xgb 0.618 / tft 0.008 / lgbm 0.374`,
threshold `0.012831`) now match `reports/ensemble_results.json` exactly, and
`reports/ensemble_results.manifest.json`'s `model_run_ids` field
(`xgb: 9d60068e…`, `tft: 64c3745f…`, `lgbm: 990e44b3…`) match the three
current model manifests' `mlflow_run_id` fields exactly, verified by direct
string comparison — the specific incoherent-deployment state that motivated
this whole review (§9, opening paragraph) no longer exists. Integration test
`tests/integration/test_real_ensemble_artifact_loads.py` (loads the real,
promoted artifact through `ModelRegistry`) passes; full suite 667/667.

### 9.4 — Leakage review: no changes, strongest part of the codebase

The mle-reviewer's fresh 2026-09-09 pass explicitly re-verified (not merely
re-stated) the time-based split discipline, the train-only fit/apply
contract for every stateful transformer, the self-exclusion arithmetic in the
lagged target encoding, and the backward-only sequence windows in
`SequenceBuilder`. Verdict, quoted directly: **"no leakage found, and this is
the strongest part of the repo."** No code changes resulted from this part of
the review.

### 9.5 — TFT's weight, and the pivot to Phase 12

The fresh ensemble evaluation (§6, §9.3) landed TFT back at a small weight
(0.008 point estimate, 0.051 bootstrap mean) — the same symptom that
originally motivated deeper investigation, but this time backed by a tight
bootstrap CI confirming it as a genuine flat-objective instability (TFT
correlates with XGBoost at 0.824, so the exact split between them barely
moves the blend's PR-AUC) rather than the artifact of a stale, mismatched
model that it was before this session's fixes.

**Decision (user's, recorded here and in `docs/prd.md` PRD Phase 12):** stop
investing further in TFT specifically. This is explicitly **not** a verdict
that TFT "failed" — the repair work in §9.2/§9.3 produced its best-ever
result, and it is kept in the deployed blend, in the codebase, and in this
report as a completed, evaluated trial. The decision to look elsewhere next
is driven by a newer piece of evidence: Uddin & Aziz (arXiv:2604.14231,
2026-04-14) report a **GNN-GraphSAGE** model at AUC-ROC 0.9248 / PR-AUC
0.6334 / F1 0.6013 on the same IEEE-CIS dataset (verified by reading the
paper directly, not the citation alone) — above this project's current
ensemble (ROC-AUC 0.9154, PR-AUC 0.5502). `docs/prd.md` PRD Phase 12 records
the four caveats that direct verification surfaced (split methodology not
confirmed temporal; new graph-construction infrastructure required; the
paper's own authors flag the topology-vs-smoothing attribution as unresolved)
and six planned, **not yet started**, evaluation steps. No GNN code exists in
this repository as of this report.

---

## 10. What is, and is not, done as of this report

**Done and verified this session:** G1, G2, G4, G6 (table, §9); full retrain
of all three models; ensemble re-promoted with matching provenance across
`models/ensemble.json`, `reports/ensemble_results.json`, and the three model
manifests; 667/667 tests passing; this report regenerated from current
artifacts rather than hand-carried forward.

**Explicitly not done this session, and not silently implied above:**
- G3's structural fix (stamping full provenance into `models/ensemble.json`
  itself, not just the reports manifest) — the immediate symptom is resolved,
  the underlying gap that allowed it is not.
- G5 (`train_lgbm.py`'s hardcoded path) and G8 (`run_ablation.py`'s stale
  `CONFIG_SPW` constant) — both confirmed still present, neither fixed.
- `reports/slice_metrics.*`, `reports/imbalance_ablation_results.json`, and
  `reports/phase9_step9_*.json` were not regenerated against the current
  models (§7).
- A tuning study against the current 184-feature representation
  (`config/tuned/` is still empty).

*(Update 2026-09-10: PRD Phase 12 — GNN-GraphSAGE — has since been implemented
end-to-end and evaluated. It was **rejected**: see §11 below.)*

---

## 11. PRD Phase 12 — GNN-GraphSAGE architecture exploration (2026-09-10)

Phase 12 was implemented end-to-end (12.2.0 through 12.2.5) and the GNN was
**rejected on evidence**. Full record: `docs/adr/ADR-005`.

**What was built** (kept in the tree as a completed trial, same as TFT):
`src/data/graph_builder.py` (transaction graph: nodes = transactions, edges
from shared `card1` ≤10 and composite `(addr1,ProductCD)` ≤5, phase-scoped so
message passing never crosses the temporal split — reviewed by `ecc:mle-reviewer`
before any code, findings R1–R9 all implemented), `src/models/gnn_model.py`
(2×`SAGEConv` 128→64 + 3-layer MLP head, no normalization layers by design),
`src/training/train_gnn.py` (`GNNTrainer`, full interface parity with the other
trainers), `model.gnn` config block, `torch-geometric` deps, 32 unit tests
(leakage regression tests included: permuting `y_test` before `train()` leaves
the trained weights byte-identical). Full suite 670 → **702 passing**.

**Standalone result** (mlflow `6615a0fc…`, seed 42, unchanged `dataset_hash`,
hyperparameters frozen from the paper per the pre-registered ADR-005 rule):

| | GNN-GraphSAGE | 3-way ensemble baseline | Paper (arXiv:2604.14231 Table II) |
|---|---|---|---|
| Test PR-AUC | **0.4410** | 0.5502 | 0.6334 |
| Test ROC-AUC | 0.8849 | 0.9154 | 0.9248 |
| Val PR-AUC | 0.5260 | — | — |
| Overfit gap (train−test) | 0.2155 | — | — |

- **0.109 below the ensemble baseline**, 0.026 below TFT's best standalone
  (0.4670), 0.192 below the paper's headline.
- **Val PR-AUC 0.5260 vs. test 0.4410 — an 0.085 drop across the temporal
  boundary.** That is ~11× this project's Phase-A leakage-fix delta (0.0076)
  and directly confirms the caveat this phase was opened with: the paper's
  0.6334 came from a split that is not confirmed time-ordered, and a large part
  of the gap to it is a non-temporal-split artifact, not model quality.
- Even on validation — measured identically to the baseline's 0.5502 — the GNN
  does not clear the baseline. It is a weaker ranker than the GBDT ensemble on
  this dataset, and its 0.216 train→test gap says the 2-layer GraphSAGE is
  memorising rather than finding transferable topological structure.

**12.2.4 (ensemble integration) was NOT triggered.** The pre-registered gate —
proceed iff standalone test PR-AUC > `0.5502 + 0.005` — evaluates to
`0.4410 < 0.5552`. `models/ensemble.json` and the deployed 3-way blend are
untouched. TFT's disposition is unchanged; the GNN does not replace it.

**Leading explanation for the gap to the paper** (the paper's own authors flag
it as unresolved): GraphSAGE-mean aggregation over `card1`/`(addr1,ProductCD)`
neighbours is a form of feature smoothing, and on a stratified split that
smoothing borrows signal across the train/test boundary in a way a temporal
split forbids. The GBDT ensemble already carries explicit `card1`/`uid`
aggregate features (Phase 9.2/9.4) that capture the non-leaking part of that
signal without a graph.

---

## 12. ADR-006 — GNN revisit: richer edges, deeper architecture, hyperparameter search (2026-09-12 to 2026-09-23)

A user-directed follow-on to §11, combining three hypotheses ADR-005 §6 left
open: edge types beyond `card1`/`(addr1,ProductCD)`, a deeper/regularized
architecture to address the 0.216 overfit gap, and a proper hyperparameter
search rather than paper-frozen values. Full record:
`docs/adr/ADR-006-gnn-revisit-edges-depth-hpo.md` (Status: **Rejected**).

Reviewed before implementation by `architect` and `ecc:mle-reviewer`. Both
rated the odds of success as poor — the architect's structural finding: the
only edge keys that survive leakage scrutiny (`card2`/`card3`/`card5`,
`(addr1,card1)`) are refinements of the existing `card1` key, since the
columns that would add a genuinely new linking modality
(`P_emaildomain`/`R_emaildomain`/`DeviceInfo`/`id_31`/`id_33`) are already
frequency-encoded and non-injective in the processed frame. The user directed
proceeding anyway; both reviews' mitigations (arm-separated attribution so no
single result is uninterpretable, and a val-search/val-confirm discipline
with test never opened during search) were adopted as preconditions.

### 12.1 — Arms A/B (richer edges; deeper/regularized architecture)

| Arm | Scope | Outcome |
|---|---|---|
| A — richer edges, frozen architecture | `card_full = (card1,card2,card3,card5)` and `addr_card = (addr1,card1)`, each new numeric key column given its own `-999.0` sentinel exclusion | No individual run cleared the incumbent's validation PR-AUC (0.5260) by a margin worth searching around alone; `addr_card` was the strongest single direction and was carried into Arm C |
| B — deeper/regularized architecture, frozen (legacy) graph | `weight_decay` swept `1e-5`→`1e-3`, `dropout` up to 0.6, input dropout, `F.normalize` (no `BatchNorm`/`LayerNorm`/`GraphNorm` at any depth — the leakage constraint from ADR-005 R3 held unmodified), 3-layer variant with residual connections | No individual run cleared the incumbent alone |

### 12.2 — Arm C: hyperparameter search

30-trial Optuna study (`TPESampler` + `MedianPruner`), search space spanning
both arms' axes plus `learning_rate`, `aggr`, `pos_weight_scale`, and
`edge_spec_set` (categorical over the pre-built graph variants from Arm A).
The search entry point structurally never opens `test_labels.parquet` — the
objective function returns validation PR-AUC only.

**Trial 14** (`addr_card` edges, 3-layer `[128,64,32]` residual GraphSAGE,
`dropout=0.2946`, `input_dropout=0.1476`, `learning_rate=0.001688`,
`weight_decay=1.121e-05`, `pos_weight=32.598`) reached **val PR-AUC 0.5505 at
epoch 25** — the only trial across both ADR-005 and ADR-006 to clear the
≥0.5502 stop-before-test gate — before crashing with a CUDA allocation error.
A crash-resume continuation ran 15 more epochs without improving on that
value.

### 12.3 — Gate 2 confirmatory run

First attempt (mlflow `e3feda1d…`) was **voided**: the config was populated
from the HPO log's free-text "best params" summary line, which reported the
study's overall-best trial (7), not trial 14's own sampled parameters —
producing a hybrid configuration never actually evaluated in the search.
Preserved at `reports/gnn_results.voided_e3feda1d.json` for the record, not
used as the ADR's result.

**Corrected run** (hyperparameters re-verified directly against the Optuna
study database, not the log text; full epoch budget per the ADR's protocol;
mlflow `d6133250654241c7bf833e9f2d8f14e3`):

| Metric | Value |
|---|---|
| Train PR-AUC | 0.7179 |
| Val PR-AUC | 0.5411 |
| **Test PR-AUC** | **0.4630** |
| Test ROC-AUC | 0.8905 |
| Overfit gap (train − test) | 0.2549 |
| Test P / R / F1 @ threshold 0.1076 | 0.0521 / 0.9668 / 0.0989 |

**Gate 2 (adoption, test > 0.5552): FAIL.** 0.4630 is 0.0922 below the
adoption threshold and below the un-margined 0.5502 baseline. Gate 3 (4-way
ensemble min-lift) and Gate 4 (diversity check) were not triggered — both are
conditional on Gate 2 passing.

**Val→test drop across every GNN run to date:**

| Run | Val PR-AUC | Test PR-AUC | Drop |
|---|---|---|---|
| ADR-005 (paper-frozen) | 0.5260 | 0.4410 | 0.0850 |
| ADR-006 Gate 2, voided | 0.5495 | 0.4797 | 0.0698 |
| ADR-006 Gate 2, corrected | 0.5411 | 0.4630 | 0.0781 |

The drop is consistently 0.070–0.085 across three independent runs spanning
two ADRs, two architectures, and three different edge-spec/hyperparameter
combinations — a genuine temporal-distribution-shift effect, not run-to-run
noise. The overfit gap also did not shrink across the revisit
(0.2155 → 0.2346 → 0.2549) despite Arm B's regularization changes targeting
exactly that.

### 12.4 — Post-close diagnostic: 3-way swap (non-gating)

After Gate 2 failed, a narrower question was checked: does a **3-way swap**
({XGBoost, LightGBM, GNN} replacing TFT) do better than a 4-way add, read
once on test. This is explicitly non-gating and does not reopen ADR-006's
Rejected status.

| Ensemble | Weights | Val PR-AUC | Test PR-AUC |
|---|---|---|---|
| Deployed (xgb+tft+lgbm) | xgb 0.618 / tft 0.008 / lgbm 0.374 | 0.6840 | **0.5502** |
| Swap (xgb+lgbm+gnn) | xgb 0.654 / lgbm 0.340 / gnn 0.006 | 0.6840 | **0.5495** |

Test PR-AUC delta (swap − deployed): −0.0007 — a statistically negligible
step backward. The weight search assigned the GNN a blend weight of 0.006,
essentially the same near-zero weight TFT already holds (0.008). Pairwise
validation-probability correlation: GNN↔XGBoost 0.834, GNN↔TFT 0.895,
GNN↔LightGBM 0.877 — below the >0.9 flag threshold, but not offering
materially more diverse signal than the models already in the blend either.

### 12.5 — Disposition

**ADR-006 closes as Rejected.** All three hypotheses (richer edges,
deeper/regularized architecture, hyperparameter search) were tested and none
reversed ADR-005's verdict. **The deployed 3-way XGBoost + TFT + LightGBM
ensemble (test PR-AUC 0.5502, §6 above) is unchanged and remains the
production configuration.** All Phase 12/ADR-006 code, tests, config,
`models/gnn_model.*`, `reports/gnn_results.json` (+ the voided-run copy), the
graph cache variants, and the HPO/search scripts stay in the repo as a
completed, evaluated trial.

A bug in `GraphBuilder.from_state_dict()` was found and fixed during the
post-close diagnostic (§12.4): it could not reconstruct edge specs stored as
a `Tuple` rather than a `list`, which would have silently blocked loading any
GNN artifact built from Arm A's `card_full` or `addr_card` edge specs —
including trial 14, this ADR's own Gate 2 subject. Fixed to accept both
container types; verified against the existing GraphBuilder/GNNTrainer test
suite and by the diagnostic itself successfully loading and scoring the
model.

---

## 13. PRD Phase 10 — Testing Strategy: real P95 latency measured for the first time (2026-09-24)

The three test files the PRD names as missing (`tests/integration/test_api.py`,
`tests/integration/test_training_pipeline.py`, `tests/performance/test_latency.py`)
were added. Full record: `docs/IMPLEMENTATION_PLAN.md` "PRD Phase 10".

**Headline finding: the deployed service's P95 latency against real
`/predict`-equivalent scoring is 500-560ms, roughly 5x the PRD's 100ms NFR
(§6.1).** Measured directly against `InferenceService.predict()` (not the
HTTP layer) over 100 real raw transactions sampled from the tail of
`data/raw/train_transaction.csv`, against the real deployed artifacts in
`models/`, with explainability disabled so the measurement isolates base
scoring cost from SHAP's separately-documented ~140ms (ADR-001 §3.4, P4-8).

| Metric | Value (range across repeated runs) |
|---|---|
| P95 | 501.5–554.3 ms |
| p50 | 454.4–475.4 ms |
| max | 698.6–969.5 ms |
| Budget (PRD §6.1) | 100 ms |

This was never measured before — no existing test exercised the real service
end-to-end for timing. It is consistent with, and gives a service-level
number to, the already-documented PRD Phase 6 streaming-throughput finding
(~4 msg/sec vs. the 200 tx/sec target, ~250ms/scoring measured from the
Kafka side): the 3-model ensemble (XGBoost + TFT + LightGBM) plus the TFT
sequence rebuild on every request are the suspected dominant costs, not
isolated further here. **The test is left failing rather than adjusted or
marked `xfail`** — an explicit choice to keep the gap visible in the suite's
default output rather than have `pytest tests/` report green over a known
NFR miss. No performance work was attempted in this pass.

Full suite: **756 passed, 1 known failure** (the latency assertion above) —
`pytest tests/ -q`, 2026-09-24, confirmed by three consecutive runs (755-756
passed each time — a single test's count varies run to run, unrelated to
this work; not chased further). `pytest tests/ -q --ignore=tests/performance`
alone: 755/755 passed, confirming the one failure is isolated to the new
latency assertion and nothing else regressed.

**Coverage: 78% overall on `src/`** (`--cov=src --cov-report=term-missing`,
performance tests excluded) — **meets the >75% bar**, 2 points short of the
PRD's original ≥80% figure. The remaining gap to 80% concentrates in the
`main()`/CLI orchestration of the training and tuning scripts (`train_xgb.py`
47%, `train_tft.py` 44%, `train_lgbm.py` 43%, `tune_tft.py` 19%,
`data_loader.py`'s real-CSV-read path 18%) — code that is exercised through
its classes and functions by existing unit tests but whose
`if __name__ == "__main__":` block only runs meaningfully against the real
590k-row dataset. Modules under direct unit/integration test are high
(`src/serving/` 94-100%, `src/models/ensemble.py` and `src/models/gnn_model.py`
100%). Not investigated or improved further in this pass, per explicit
instruction. Full per-module breakdown: `docs/IMPLEMENTATION_PLAN.md` "PRD
Phase 10".

A bug was found and fixed while writing the latency test itself: the first
version read the entire ~590,540-row raw CSV (`pd.read_csv(...).tail(n)`,
~1.6 GiB for the V-columns alone) before discarding all but the sampled rows,
which OOM'd on this machine when run alongside the already-resident model
artifacts. Fixed to count rows once and read only the tail via `skiprows`,
and to sample once per test module rather than per test.

**Not done this pass:** a repo-wide coverage report against the PRD's ≥80%
target on `src/` was not generated; the underlying latency gap was not
investigated or fixed, per an explicit decision to document rather than
chase a performance fix in this pass.
