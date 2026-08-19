# Phase 2 Results — Re-baselined on the Leakage-Free Pipeline

<!-- Regenerated: 2026-08-07 under IMPLEMENTATION_PLAN.md task A7. Supersedes the
     pre-Phase-A report, whose headline test PR-AUC of 0.5678 was produced by a
     pipeline that fitted every transformer on train+val+test before splitting. -->

**Provenance**

| Field | Value |
|---|---|
| MLflow run ID | `74b8460d68ae400889fb5b128db51751` |
| MLflow experiment ID | `775946513825380473` |
| Tracking URI | `file:./mlruns` |
| Run name | `xgb_enhanced_features` |
| Date | 2026-08-07 |
| Pipeline | `python src/data/preprocess.py` → `python src/training/train_xgb.py` |
| Dataset | IEEE-CIS `train_transaction.csv` + `train_identity.csv` (590,540 rows, 24.4% identity match) |
| Test suite at time of run | `pytest tests/ -q` → 77 passed |
| Git SHA | *unavailable — the working tree is not a git repository (see D4)* |

Inspect the run with `mlflow ui --backend-store-uri ./mlruns` or
`mlflow.get_run("74b8460d68ae400889fb5b128db51751")`.

---

## 1. What changed and why this report exists

The previous version of this report described a model trained on features whose
transformers — IncrementalPCA, frequency maps, label encoders, imputation
constants, and the target-encoding prior — had all been fitted on the complete
dataset *before* the train/val/test split. Test rows therefore informed the
feature representation, and the reported test PR-AUC of **0.5678** was not a
clean held-out estimate.

Phase A of `docs/IMPLEMENTATION_PLAN.md` restructured `run_pipeline()` so the
split is decided first and every stateful transformer is fitted on train rows
only, with `fit=False` on val and test. This report is regenerated from that
corrected pipeline.

**Headline: the honest test PR-AUC is 0.5602, down from 0.5678 — a drop of
0.0076 absolute (1.3% relative).**

That is a modest move, and it is the expected magnitude. The leakage was almost
entirely *unsupervised* — distributional information (PCA basis, category
frequencies, fill values) rather than labels. The one label-bearing channel, the
target-encoding global prior, is a single scalar smoothed across 590k rows. The
reason to fix this was methodological defensibility, not a suspected inflated
score, and the small delta is consistent with that diagnosis rather than
evidence the fix was unnecessary.

### Defect found and fixed during this run

Re-running preprocessing end-to-end crashed on the test split with
`InvalidIndexError: Reindexing only valid with uniquely valued Index objects`.

Root cause: the target-encoding carried state keys per-entity `(sum, count)`
totals in a dict, and rows with a missing entity key were stored under a literal
`NaN`. Because `nan != nan` and each frame's `groupby` emits a fresh NaN object,
every `update_state=True` call wrote a *new* missing-key entry instead of
accumulating onto the previous one. Two splits still read back the first entry
and looked correct; the third call — the `train → val → test` order the pipeline
actually uses — found a duplicated key and could not align it.

Two consequences, one loud and one silent: preprocessing died on the test split,
and val's missing-key history had silently replaced train's rather than adding
to it. This matters because target encoding runs before imputation, and `addr1`,
`card2` and `R_emaildomain` are missing on a large share of IEEE-CIS rows.

Fixed by canonicalising missing entity keys to a single `MISSING_ENTITY_KEY`
sentinel in `_carried_totals` and `_accumulate_entity_totals`, with a regression
test (`test_missing_key_history_accumulates_over_three_splits`) that exercises
all three splits — the existing two-split test could not catch it.

---

## 2. Data and splits

Time-based 70/10/20 split on `TransactionDT`, no shuffle, decided before any
transformer is fitted.

| Split | Rows | Fraud rate | Temporal boundary |
|---|---|---|---|
| Train | 413,378 | 3.52% | ends `DT=10,437,996` |
| Validation | 59,053 | 3.49% | ends `DT=12,192,743` |
| Test | 118,109 | 3.44% | starts `DT=12,192,842` |

Final feature count: **171**. V1–V339 are reduced to 30 IncrementalPCA
components, **fitted on the 413,378 train rows only** and applied to all rows in
batches — this preserves the memory optimisation (434 → ~128 columns before the
temporal sort) without letting val/test rows into the fit.

Measured variance retained by the train-only PCA fit in this run: **100.0%**
(logged by `reduce_v_features`). Note that `notebooks/01_eda.ipynb` asserts
">85%" for the same reduction; that notebook has never been executed and its
claim is uncomputed. Task D8 covers reconciling it — treat the figure above,
which comes from this run's log, as the measured one.

---

## 3. Feature engineering

Unchanged in substance from the previous report; what changed is *when* each
piece is fitted. Summarised by leakage class:

**Stateless / strictly backward-looking** — computed over the full temporally
ordered frame, deliberately. Each reads only a transaction's own past, which is
exactly what production has; computing them per split would instead zero out
each card's history at the split boundary and create a train/serve mismatch.

- Cyclical (sin/cos) encodings of transaction hour and day of week
- `amount_log` and amount decomposition features
- Email domain parsing; D-column and C-column summaries
- Card-level expanding aggregates via `shift(1)` — `tx_count_per_card`,
  `mean_amount_per_card`, `max_amount_per_card`, `std_amount_per_card`
- Anomaly signals: `amount_vs_mean_ratio`, `amount_zscore_per_card`
- Velocity: `tx_sum_per_card`; address, device and browser parsing
- Interactions such as `interaction_hour_amount`

**Stateful** — fitted on train only, applied to val/test with `fit=False`:
card-hash frequencies, expanding target encoding (train-only global prior),
imputation fill values, and categorical label/frequency encoders.

Null-count meta features are computed on the **raw** frame before PCA, so
V-column missingness is captured rather than erased (task A5).

---

## 4. Class imbalance

Fraud rate is ~3.5%. XGBoost uses cost-sensitive learning with
`scale_pos_weight = 27.43`, computed at runtime as train `neg/pos`
(398,840 / 14,538).

> **Not re-baselined.** The SMOTE-vs-`scale_pos_weight` comparison that
> originally justified this choice selected its winner on the *test* set and was
> never re-run on the corrected pipeline. `scale_pos_weight` is retained here
> because it is what this run used, not because the comparison has been
> re-validated. Task C6 covers re-running that ablation on validation.

---

## 5. Hyperparameters actually used

These are the values read back from the MLflow run — not aspirational config.

| Parameter | Value |
|---|---|
| `n_estimators` | 887 (best iteration 885) |
| `max_depth` | 10 |
| `learning_rate` | 0.09696200612789875 |
| `subsample` | 0.8382111670615123 |
| `colsample_bytree` | 0.6186070548544907 |
| `min_child_weight` | 4 |
| `reg_alpha` | 0.014571229075512624 |
| `reg_lambda` | 1.1091336812755855e-07 |
| `gamma` | 5.543434625011016e-08 |
| `scale_pos_weight` | 27.434310083918007 |
| `random_state` | 42 |
| `tree_method` / `device` | `hist` / `cuda` |
| `early_stopping_rounds` | 100 (on validation, `aucpr`) |

> **Provenance gap.** The previous report listed a different set of "best
> Optuna" values (`n_estimators=897`, `lr=0.0893`, `colsample_bytree=0.911`,
> `min_child_weight=7`). Those do not correspond to the shipped config and
> cannot be traced to any recorded study, so they have been removed rather than
> reconciled. The table above is what `config/config.yaml` contained at run time
> and what the model was actually fitted with. Tuning has **not** been re-run
> against the leakage-free features — these hyperparameters were selected under
> the old pipeline and are carried forward unchanged so the PR-AUC delta
> isolates the leakage fix. Task D3 covers linking tuned params to a run ID.

---

## 6. Model performance

### PR-AUC (threshold-independent — the honest headline)

| Split | PR-AUC |
|---|---|
| Train | 1.0000 |
| Validation | 0.6922 |
| **Test** | **0.5602** |

Test ROC-AUC: **0.9095**. Overfitting gap (train − test): **0.4398**.

The train PR-AUC of 1.0 is memorisation, not a result. The meaningful pair is
val 0.6922 versus test 0.5602: a 0.13 drop across the temporal boundary, on a
split where test is strictly in the future. That gap is a property of the data —
fraud patterns drift — and it is the number to watch as later phases land.

### Comparison to the pre-fix baseline

| | Pre-fix (leaky) | Re-baselined | Δ |
|---|---|---|---|
| Test PR-AUC | 0.5678 | **0.5602** | −0.0076 (−1.3%) |

### Operating point

| Metric | Value at threshold 0.0100 |
|---|---|
| Precision | 21.61% |
| Recall | 75.54% |
| F1 | 0.3361 |
| Accuracy | 89.73% |

Confusion matrix at that threshold (118,109 test rows, 4,064 actual fraud):

| | Predicted legit | Predicted fraud |
|---|---|---|
| **Actual legit** | 102,908 (TN) | 11,137 (FP) |
| **Actual fraud** | 994 (FN) | 3,070 (TP) |

Best achievable F1 on test is **0.5520** at threshold **0.2060**, which is a
substantially better operating point than the cost-based one below.

> **Three caveats — this row is not an out-of-sample estimate.**
>
> 1. **The threshold is still selected on the test set.**
>    `find_optimal_threshold(y_test, y_prob_test, ...)` picks 0.0100 using test
>    labels, and precision/recall/F1 are then reported at that threshold on the
>    same data. This is optimistically biased. Phase A fixed the *feature*
>    leakage; this is a separate defect and task **C1** remains open. The
>    validation split exists and is not yet used for this purpose.
> 2. **0.0100 is the first point of the search grid** (`linspace(0.01, 0.99, 99)`).
>    An optimiser returning its own boundary means the true optimum lies outside
>    the search space — the search is clipped, not converged (task C3).
> 3. **The probabilities are uncalibrated.** The 500 FN / 5 FP cost model is only
>    valid on calibrated `p(fraud|x)`, and `scale_pos_weight=27.43` systematically
>    distorts the output scale. No calibration step or Brier score exists yet
>    (task C2).
>
> Treat PR-AUC as this run's trustworthy result and the operating-point row as
> provisional until Phase C lands.

### LightGBM

> **Not re-baselined.** The previous report's LightGBM row (test PR-AUC 0.3206)
> came from the leaky pipeline and has not been retrained here — this task
> covered XGBoost only. The figure is withheld rather than restated, since it is
> not comparable to the 0.5602 above. LightGBM remains an orphaned baseline
> (task F1).

### Regularisation experiment

> **Not re-baselined.** The earlier finding that heavy regularisation
> (`max_depth=4`, `reg_alpha=10`, `reg_lambda=10`) *degraded* test PR-AUC to
> 0.508 was measured under the leaky pipeline and has not been repeated. Its
> conclusion — that this temporal split rewards high model capacity — is
> plausible but currently unsupported by any clean run.

---

## 7. Artifacts produced by this run

| Artifact | Path |
|---|---|
| Model | `models/xgb_model.pkl` |
| Fitted transformers | `data/processed/transformers/` |
| Processed splits | `data/processed/{train,val,test}_{features,labels}.parquet` |
| PR curve | `reports/figures/xgb_pr_curve.png` |
| ROC curve | `reports/figures/xgb_roc_curve.png` |
| Confusion matrix | `reports/figures/xgb_confusion_matrix.png` |
| Threshold vs business value | `reports/figures/xgb_threshold_value.png` |

All four figures are also logged as MLflow artifacts under the run ID above.

---

## 8. Status against the remediation plan

Phase A (eliminate leakage) is complete: A1–A6 were verified previously, and
this run closes **A7**. The acceptance criterion — "no transformer observes
val/test rows during fit; test PR-AUC re-baselined and documented as the honest
number, whatever it turns out to be" — is met at **0.5602**.

Still open and directly affecting numbers in this report:

| Task | Effect on this report |
|---|---|
| C1 | Threshold selected on test — §6 operating point is biased |
| C2 | No calibration — cost-based threshold rests on distorted probabilities |
| C3 | Threshold grid boundary — 0.0100 is clipped, not optimal |
| C6 | Imbalance ablation never re-run on validation — §4 justification is stale |
| D3 | Tuned hyperparameters not traceable to a study — §5 provenance gap |
| D4 | No model manifest and no git SHA — provenance is MLflow-only |
| D8 | PCA variance claim in notebook 01 unreconciled — §2 |
| F1 | LightGBM orphaned — §6 comparison withheld |
