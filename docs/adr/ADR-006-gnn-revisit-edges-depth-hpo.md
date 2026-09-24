# ADR-006: GNN revisit — richer edges, deeper/regularized architecture, hyperparameter search

- **Status:** **Rejected** (2026-09-23). Arms A and B produced no validation
  improvement over the incumbent worth searching around further; Arm C (HPO)
  found one trial (14) that cleared Gate 1 on validation (0.5505 > 0.5502)
  after a crash-resume; the Gate 2 confirmatory run using trial 14's verified
  hyperparameters scored **test PR-AUC 0.4630**, short of the 0.5552 adoption
  threshold. See §7 for the full execution record and §8 for the close-out
  disposition.
- **Date:** 2026-09-12 (opened) — 2026-09-23 (closed)
- **Extends:** [ADR-005](ADR-005-gnn-architecture-evaluation.md), which
  rejected the paper-frozen GNN-GraphSAGE (test PR-AUC 0.4410, run
  `6615a0fc6f5e4a55a1a3a084d952cbe8`, `reports/gnn_results.json`) and closed
  Phase 12 with "no follow-up scheduled ... would need a materially different
  hypothesis."
- **Trigger:** User-directed revisit combining three hypotheses named in
  ADR-005 §6 open questions: (1) edge types beyond `card1`/`(addr1,ProductCD)`,
  (2) a deeper/regularized architecture to fight the 0.216 overfit gap, (3) a
  proper hyperparameter search instead of paper-frozen values.
- **Reviewed by:** `architect` and `ecc:mle-reviewer` sub-agents, 2026-09-12,
  before any implementation. Both reports are summarized in §1 and their full
  findings are the basis for §3 (execution discipline) and §5 (risk).

---

## 1. Review summary (read before the rest of this document)

Both reviewers were given the ADR-005 record and the full proposed scope and
asked to assess independently, without being told to justify proceeding.

**Architect's structural finding:** the "richer edges" hypothesis is largely
foreclosed by the existing preprocessing pipeline. `GraphBuilder` consumes the
*processed* parquet frames, and the columns that would add a genuinely new
identity-linking modality — `P_emaildomain`, `R_emaildomain`, `DeviceInfo`,
`id_31`, `id_33` — are already **frequency-encoded** into non-injective floats
(`feature_engineering.py` `encode_categoricals`), with every train-unseen value
collapsing to `0.0`. Building edges on them would connect unrelated
transactions through a single mega-clique — a larger, silent version of the
`addr1` sentinel-clique bug ADR-005 already found and fixed (R8). The only
edge keys that survive scrutiny (`(card1,card2,card3,card5)`,
`(addr1,card1)`) are **refinements of the existing `card1` key**, not new
signal. Separately, the 0.216 overfit gap splits into a train→val portion
(0.130, capacity-shaped) and a val→test portion (0.085, distribution-shift
shaped); depth/regularization addresses only the first. Structural read: a
GraphSAGE-mean over `card1` neighbours is a lower-fidelity, leakage-starved
approximation of aggregate features the GBDTs already compute exactly
(`create_card_aggregates`, `create_uid_features`). Optimistic HPO-only ceiling
estimate: validation ~0.54–0.57, projecting to **test ~0.46–0.49 after the
same ~0.085 temporal-boundary drop the incumbent showed — still below the
0.5502 baseline.**

**mle-reviewer's execution-risk finding:** PROCEED-WITH-CONDITIONS. Two
CRITICAL findings — (a) an Optuna search with no pre-registered trial budget
or validation/test discipline is a slow-motion version of the exact
repeated-test-consultation failure ("G1") ADR-005 names; (b) new edge keys on
high-missingness identity columns need an R8-style sentinel exclusion or they
create a label-correlated leak, worse than a merely uninformative one, because
missingness in these columns correlates with fraud. Three HIGH findings —
depth creates real pressure to add normalization layers, which would reopen
the R3 leakage argument (independently converges with the architect's
recommendation: use `F.normalize`, never a running-stats module); no compute
budget evidence exists for sizing a search; bundling all three hypotheses in
one search makes any single result uninterpretable (can't attribute a gain,
or a loss, to a specific cause).

**Decision on how to proceed, given both reviews:** the user has directed
proceeding with the full combined scope despite both reviewers rating the
odds of success as poor. This ADR adopts every mitigation both reviews
required as a **precondition**, not a nice-to-have — in particular, sequencing
the three hypotheses as independently-attributable arms (mle-reviewer's fix
for the bundling problem) and the val-search/val-confirm + zero-test-during-
search discipline (mle-reviewer's fix for the HPO leak risk).

---

## 2. Decision drivers (unchanged from ADR-005, restated for this revisit)

- **D1 — Comparability unchanged.** Only *this revisit's* standalone test
  PR-AUC vs. the 0.5502 ensemble baseline decides adoption. The paper's 0.6334
  remains a non-target; any arm scoring ≥0.60 gets audited for a leak before
  being believed (same heuristic that caught findings G1/G2).
- **D2 — Leakage discipline carries over unchanged.** Every new edge key
  passes through the same phase-scoped, symmetric edge-masking machinery
  (`GraphBuilder._assert_invariants`) with no exceptions.
- **D3 — Interface parity unchanged.** No `GNNTrainer` public method signature
  changes. `run_ensemble_eval.py`'s per-model loop is untouched until (and
  unless) Gate 2 passes.
- **D4′ — This revisit's pre-registered decision rule.** See §4. Fixed here,
  before the first trial, exactly as ADR-005's D4 was fixed before its run.
- **D5 — No normalization layers, unchanged, and now explicitly extended.**
  `FORBIDDEN_MODULE_TYPES` in `src/models/gnn_model.py` stays the sole gate.
  Any depth/regularization change uses `F.normalize` (a pure per-node,
  buffer-free function — passes `_assert_no_norm_layers` because it holds no
  state) instead of `LayerNorm`/`BatchNorm`/`GraphNorm`/`GATConv`-with-norm/
  `GINConv`-with-a-BatchNorm-MLP. Relaxing this requires a re-review of the
  leakage argument in ADR-005 §4.2, not a code comment.
- **D6 — Arm attribution (new for this ADR).** The three hypotheses are
  evaluated as **separately attributable arms**, each against the same frozen
  incumbent, so a result can be traced to a specific cause. No single search
  mixes edge-set choice, architecture family, and full hyperparameter ranges
  in one undifferentiated space.

---

## 3. Execution plan

### 3.1 Arm A — richer edges, frozen architecture

**Scope, per the architect's finding:** email/device columns are excluded —
they are frequency-encoded and non-injective in the processed frame, and using
them would recreate the R8 clique bug at a larger, label-correlated scale.
Only genuine refinements of `card1` are tested:

- `edge_spec: card_full = (card1, card2, card3, card5)`, capped neighbours
  matching `card1_max_neighbors` (10) as a starting point.
- `edge_spec: addr_card = (addr1, card1)`, capped at `addr_product_max_neighbors`
  (5).

Each new numeric key column (`card2`, `card3`, `card5`) gets its own sentinel
exclusion (`-999.0`, the same imputation fill value `addr1` uses) — omitting
this recreates R8 one column over, per the mle-reviewer's CRITICAL finding.
Any composite-key column matching `*_target_enc` or the frequency-encoded
blocklist (`P_emaildomain`, `R_emaildomain`, `DeviceInfo`, `id_31`, `id_33`,
`card_hash_freq`) is a hard-fail in `GraphBuildConfig.__post_init__`, not a
convention — this is the leak-prevention mechanism, not documentation of one.

Architecture and hyperparameters: **frozen at ADR-005's paper values**
(`hidden_dims [128, 64]`, paper `pos_weight`, fan-out `[10, 5]`). Only the
graph changes. Two graph variants are built and cached (legacy vs.
`card_full`-extended vs. `addr_card`-extended), each keyed by its own
`dataset_hash`-derived cache path so they don't collide.

**Comparison point:** validation PR-AUC of each new-graph run vs. the
incumbent's validation PR-AUC (0.5260, `reports/gnn_results.json`).

### 3.2 Arm B — deeper/regularized architecture, frozen (legacy) graph

Scope: legacy `card1`/`(addr1,ProductCD)` graph only. Architecture changes,
in order of expected value (mle-reviewer's compute-budget finding: the
current `weight_decay=1e-5` is arguably 4 orders of magnitude too small for a
model with a 0.216 overfit gap — this is the one lever both reviews rate as
having a real, non-zero mechanism):

- `weight_decay`: sweep `1e-5` → `1e-3` (order-of-magnitude steps first).
- `dropout`: `0.3` → up to `0.6`.
- `input_dropout` (new): feature-level dropout before the first `SAGEConv`.
- `l2_normalize` (new): `F.normalize(h, p=2, dim=-1)` after each conv
  activation — R3-compliant scale control, no persisted buffers.
- `residual` (new): additive skip connection when dims match, else a linear
  projection shortcut — stateless, R3-compliant, targets the oversmoothing
  risk of going to 3 layers.
- `num_layers`: `2` (current) vs. `3` (`hidden_dims [256, 128, 64]`,
  matching fan-out length `[15, 10, 5]`) — **the fan-out list length must
  equal `len(hidden_dims)`**; add a build-time assertion in `GNNTrainer` so a
  mismatch fails fast instead of silently truncating sampling.

**Hard constraint carried forward unmodified:** no `BatchNorm`/`LayerNorm`/
`GraphNorm`/any running-stats module, anywhere, at any depth. If a GIN variant
is tried (`conv_type="gin"`), its internal MLP is constructed **without**
`BatchNorm` even though that is GIN's canonical form — flagged explicitly so
this isn't rediscovered as a leak later.

**Comparison point:** validation PR-AUC of each architecture variant vs.
0.5260, on the unchanged legacy graph.

### 3.3 Arm C — hyperparameter search on top of the winning axis/axes

Only runs if Arm A and/or Arm B shows a validation improvement worth
searching around (see Gate 0 below — this is not automatic).

**Search space** (Optuna, `TPESampler` + `MedianPruner`):

| Parameter | Range |
|---|---|
| `learning_rate` | log-uniform `[3e-4, 1e-2]` |
| `weight_decay` | log-uniform `[1e-6, 1e-2]` |
| `dropout` | uniform `[0.2, 0.7]` |
| `input_dropout` | uniform `[0.0, 0.2]` |
| `num_layers` | categorical `{2, 3}` |
| `hidden_dims` preset | categorical `{[128,64], [256,128], [256,128,64], [128,64,32]}` (length-matched to `num_layers`) |
| `residual` | categorical `{True, False}`, conditional on `num_layers==3` |
| `l2_normalize` | categorical `{True, False}` |
| `aggr` | categorical `{mean, max}` |
| `num_neighbors` preset | categorical, length-matched to `num_layers` |
| `pos_weight_scale` | uniform `[0.5, 1.5]` × empirical train neg/pos |
| `edge_spec_set` | categorical over the 2–3 **pre-built, cached** graphs from Arm A (never rebuilt inside a trial) |

**Trial budget:** 30 trials, fixed seed 42 across all trials (seed variance is
not mistaken for hyperparameter signal at these margins), `max_epochs=50`,
`patience=8` during search (reduced from 100/10 — the incumbent's val curve
was flat from epoch ~25 to ~45, gaining only +0.0045 over that span; searching
at full epoch budget wastes compute establishing what's already known).
Full settings (`max_epochs=100`, `patience=10`) are restored **only** for the
single confirmatory run in §3.4.

**Hard rule, enforced in code, not by discipline alone:** the HPO entry point
runs in `--search-mode`, which **does not open `test_labels.parquet` at any
point**. Test features may still load (needed for graph topology only, R7-
sanctioned — labels never enter the loss), but the label file path is never
read during search. The Optuna objective function returns validation PR-AUC
only. This is the mle-reviewer's CRITICAL fix for the "slow-motion G1"
failure mode — it must be structurally impossible for a trial to consult
test, not merely undocumented.

### 3.4 Confirmatory run

Exactly one. The single best-by-validation configuration across all arms that
actually attempted (some arms may be skipped per Gate 0) is retrained with
full epoch budget, evaluated on train/val/test **once**, and the result is
written to `reports/gnn_results.json` (overwriting the ADR-005 baseline
record — the prior run stays recoverable via mlflow run
`6615a0fc6f5e4a55a1a3a084d952cbe8` and this ADR's citation of its numbers).
**No re-running after seeing the test number.** If it misses the gate, the
answer is no and this ADR is closed as Rejected, same disposition as ADR-005.

---

## 4. Pre-registered decision rule (fixed now, before Arm A starts)

- **Baseline:** `B = 0.5502` test PR-AUC (frozen 3-way ensemble, mlflow
  `69d3a4d5…`, `dataset_hash d20f03c0…`).
- **Margin:** `m = 0.005` (`config.ensemble.min_lightgbm_lift`), unchanged —
  gate = **0.5552**.

**Gate 0 — per-arm validation screen (before Arm C runs at all).** Arm C
(HPO) only executes if Arm A and/or Arm B produces a validation PR-AUC
improvement over the incumbent's 0.5260. If neither arm improves validation,
Arm C is skipped and the revisit closes on Arms A/B's negative result — HPO on
a direction with no signal is exactly the wasted-compute failure mode the
mle-reviewer's compute-budget finding warns against.

**Gate 1 — stop-before-test screen (validation only).** Across all arms that
ran, if the single best validation PR-AUC does not reach **0.5502** (the
baseline's own level — the bar the paper-frozen incumbent failed to clear
even on validation, 0.5260 < 0.5502), the revisit **stops here**. No
confirmatory test run executes. This is the primary leak-preventer and cost-
saver: it is checked without ever touching test.

**Gate 2 — standalone adoption (test, read exactly once).** If Gate 1 passes,
run the single confirmatory run (§3.4). Proceed to ensemble integration
**iff** test PR-AUC `> 0.5552`.

**Gate 3 — 4-way ensemble min-lift (validation, once), sequential ordering.**
If Gate 2 passes, add `"gnn"` to `run_ensemble_eval.py`'s per-model loop.
Gates are evaluated **sequentially against whichever blend already survived**
— LightGBM's existing 2-way-vs-3-way gate is unchanged and runs first; the
GNN's 3-way-vs-4-way gate then runs against that surviving blend. The GNN
addition never retroactively re-opens LightGBM's membership decision. Keep
the GNN iff `val_pr_auc(4-way) − val_pr_auc(3-way) >= 0.005`
(`apply_min_lift_gate`, evaluated on validation per the existing 2026-09-09
fix). `test_lift` is logged as a diagnostic only, never decisive.

**Gate 4 — diversity sanity.** Record pairwise validation-probability
correlation with XGBoost/LightGBM/TFT (`pairwise_diagnostics`). A GNN
correlating >0.9 with XGBoost gets TFT's disposition (ADR-005 §1.3): recorded,
kept at a marginal weight if it clears Gates 1–3 at all, not treated as a win
worth extra deployment complexity.

**Failure disposition (any gate fails):** ADR-006 is updated to Status:
Rejected, the negative result (whichever gate stopped it, and at what number)
is recorded in this document and in `reports/gnn_results.json`, all new
infrastructure stays in the tree per the ADR-005 precedent, and no further
GNN work is scheduled without a hypothesis that is materially different from
all three tried here.

---

## 5. Risk assessment (carried forward from the architect's review, not softened)

The gap to close is **0.1092 test PR-AUC** (0.4410 → 0.5552) — roughly 15×
the size of LightGBM's entire admission-worthy gain (+0.0074 validation lift).
Structural reasons both reviews rate the odds as poor:

1. The richer-edges hypothesis cannot deliver a new linking modality — the
   columns that would (email, device) are already destroyed as equality keys
   by upstream frequency encoding. What remains are refinements of a key the
   graph already has.
2. The 0.216 overfit gap is roughly half capacity-shaped (train→val, 0.130)
   and half distribution-shift-shaped (val→test, 0.085). Depth and
   regularization address only the capacity half.
3. The GBDT ensemble already computes the non-leaking part of the `card1`/
   `uid` neighbourhood signal exactly, via expanding aggregates
   (`create_card_aggregates`, `create_uid_features`) — a sampled,
   leakage-starved GraphSAGE-mean is a lossier version of the same
   information, not new information.
4. The leakage discipline (phase-scoped symmetric masking) is precisely what
   removes most of the paper's headline advantage on a temporal split, and it
   is not going to be relaxed — so the realistic ceiling was already measured
   in ADR-005, not newly discovered here.
5. The incumbent's validation curve was flat for its last 20 epochs — it
   converged. The gap is not hiding in more training.

This ADR proceeds anyway per explicit user direction, with Gate 0 and Gate 1
structured specifically to make a negative outcome **cheap and fast** rather
than to avoid finding one.

---

## 6. Open items to resolve during implementation

- Instrument per-epoch wall-clock timing in `train_gnn.py` (absent from the
  ADR-005 run) before sizing the Arm C trial budget for real, rather than by
  the estimate in §3.3.
- Add the `edge_index.shape[1]` ceiling assertion (architect's finding) so a
  degenerate key produces a fast, clear failure instead of a multi-hour OOM.
- Add the legacy-vs-explicit-`edge_specs` regression test (byte-identical
  `edge_index` with `edge_specs=()` vs. the equivalent explicit legacy specs)
  before any new spec is exercised in training.

---

## 7. Execution record (2026-09-13 through 2026-09-23)

**Arms A/B (2026-09-13 to 2026-09-17):** richer-edge variants (`card_full`,
`addr_card`) and deeper/regularized architectures were run against the
incumbent's 0.5260 validation PR-AUC. No individual run in this phase cleared
the incumbent by a margin worth searching around on its own; `addr_card`
(architect's structural read: a genuine refinement of the `card1` key, not a
new modality) was the strongest single direction and became one of Arm C's
categorical `edge_spec_set` choices per Gate 0.

**Arm C — HPO (2026-09-18 to 2026-09-21), study `gnn_arm_c_hpo`:**
Stage 1 (5 trials, `max_epochs=40, patience=6`) ran 2026-09-18/19. Trial 7
was the stage's best-by-Optuna-value result (val 0.5484, legacy edges).
A targeted continuation (5 more trials, `max_epochs=60, patience=10`,
resuming the same study so TPE was informed by Stage 1) ran 2026-09-20/21.
**Trial 14** (`addr_card` edges, 3-layer `[128,64,32]` residual GraphSAGE,
`l2_normalize=False`, `dropout=0.2946`, `input_dropout=0.1476`,
`learning_rate=0.001688`, `weight_decay=1.121e-05`,
`pos_weight_scale=1.1882` → `pos_weight=32.598`) reached **val PR-AUC 0.5505
at epoch 25** — the only trial across both ADRs to clear Gate 1 (≥0.5502) —
before crashing with a CUDA "bad allocation" error at epoch 26. A surviving
per-epoch checkpoint allowed a crash-resume continuation (`scripts/
resume_trial14.py`) that ran 15 more epochs (patience 15) without ever
beating the epoch-25 value, early-stopping at epoch 40 with best-held 0.5505.

**Gate 2 confirmatory run — first attempt, VOIDED (2026-09-22, mlflow
`e3feda1d79684e02a4655d22301b8636`):** `config/config.yaml`'s `model.gnn`
block was populated from the HPO log's "Best params" summary line, which
reports the *study's* best trial (7), not trial 14's own sampled parameters.
The resulting run paired trial 14's architecture with trial 7's optimizer
settings (`learning_rate=0.000803`, `weight_decay=1.482e-04`,
`pos_weight=13.87` instead of trial 14's `pos_weight=32.60` — a 2.35x
difference). This hybrid configuration was never evaluated in the HPO search
and does not represent Gate 1's passing trial. Result (val 0.5495, test
0.4797) is preserved at `reports/gnn_results.voided_e3feda1d.json` for the
record but is **not** the ADR's Gate 2 read.

**Gate 2 confirmatory run — corrected (2026-09-22/23, mlflow
`d6133250654241c7bf833e9f2d8f14e3`):** hyperparameters re-verified directly
against the Optuna study database (`reports/gnn_hpo_study.db`, trial 14's
`.params`), not the log text, before this run. Full budget per §3.4
(`max_epochs=100, patience=10`). Training ran 70 epochs (early-stopped),
best validation at epoch 31 with a plateau around 0.538–0.541 through the
60s–70s (epoch-count/optimizer stochasticity — NeighborLoader sampling order,
CUDA nondeterminism — means an independent run of an identical config is not
expected to reproduce the exact HPO-run epoch-25 value; landing in the same
0.53–0.55 region as trial 14 is the relevant check, and it did).

| Metric | Value |
|---|---|
| Train PR-AUC | 0.7179 |
| **Val PR-AUC** | **0.5411** |
| **Test PR-AUC** | **0.4630** |
| Test ROC-AUC | 0.8905 |
| Overfit gap (train − test) | 0.2549 |
| Test P / R / F1 @ threshold 0.1076 | 0.0521 / 0.9668 / 0.0989 |

**Gate 1 (informational at this point):** the corrected run's own validation
(0.5411) does not re-clear 0.5502 — only the original HPO trial 14 run did,
by 0.0003, before its crash. **Gate 2 (adoption, test > 0.5552): FAIL.**
0.4630 is 0.0922 below the adoption threshold and below even the un-margined
0.5502 baseline.

**Val→test drop, tracked across every GNN run to date:**

| Run | Val PR-AUC | Test PR-AUC | Drop |
|---|---|---|---|
| ADR-005 (paper-frozen) | 0.5260 | 0.4410 | 0.0850 |
| ADR-006 Gate 2, voided | 0.5495 | 0.4797 | 0.0698 |
| ADR-006 Gate 2, corrected | 0.5411 | 0.4630 | 0.0781 |

The drop is consistently 0.070–0.085 across three independent runs spanning
two ADRs, two architectures, and three different edge-spec/hyperparameter
combinations. This matches the architect's original structural diagnosis
(§1): part of this gap is a genuine temporal-distribution-shift effect, not
run-to-run noise, and no amount of further tuning on this side of the split
boundary is expected to close it. The overfit gap also did **not** improve
across the revisit (0.2155 → 0.2346 → 0.2549) — the deeper/regularized
architecture (Arm B's hypothesis) did not fix the capacity-shaped half of the
gap it targeted; if anything the trend across three data points moves the
wrong way.

---

## 8. Disposition

**Gate 2 failed on the correctly-configured confirmatory run. Per §4's
pre-registered failure disposition, this ADR closes as Rejected.** Gate 3
(4-way ensemble min-lift) and Gate 4 (diversity check) are not triggered —
they only run if Gate 2 passes.

All three of ADR-006's hypotheses were tested and none reversed ADR-005's
verdict:
1. **Richer edges (Arm A):** confirmed the architect's prediction — the only
   edge keys that survive leakage scrutiny are refinements of the existing
   `card1` key, not a new linking modality. `addr_card` was the best of the
   three variants but did not, alone or combined with HPO, close the gap.
2. **Deeper/regularized architecture (Arm B):** the overfit gap did not
   shrink; it grew slightly across every subsequent run. `weight_decay`,
   `dropout`, `input_dropout`, `residual`, and 3-layer depth were all
   explored inside Arm C's search space and the best-found combination still
   left a 0.25 train−test gap.
3. **Proper hyperparameter search (Arm C):** found exactly one configuration
   that cleared Gate 1 on validation, by a 0.0003 margin, and it did not
   survive contact with the temporal test boundary — consistent with the
   architect's pre-registered ceiling estimate (val 0.54–0.57 → test
   0.46–0.49) from before any trial ran.

**No further GNN work is scheduled**, per this ADR's own §4 disposition
clause and ADR-005's precedent, **without a hypothesis materially different
from richer edges, deeper/regularized architecture, or broader HPO** — all
three have now been tried. All Phase 12/ADR-006 code, tests, config,
`models/gnn_model.*`, `reports/gnn_results.json` (+ the voided-run copy),
the graph cache variants (`data/processed/graph/`, gitignored), and
`scripts/{run_gnn_arm_search,run_gnn_hpo,resume_trial14}.py` stay in the tree
as a completed, evaluated trial. The `--checkpoint-path` option added to
`train_gnn.py`'s CLI during this ADR's execution (crash-resume support) is a
durable, reusable improvement independent of the ADR's outcome.

**Process note, for future ADRs with an HPO stage:** the voided first Gate 2
attempt was caused by reading a "best trial" summary line from a log instead
of querying the Optuna study database directly for a specific trial number's
own parameters. When more than one trial is discussed by number in a log
(the study's overall best vs. a specific trial of interest), treat the log's
free-text summary as a pointer, not a source — pull hyperparameters
programmatically (`study.trials[i].params`) before writing them into any
config that will produce a test-set-consuming run.

---

## 9. Post-close exploratory diagnostic (2026-09-23, non-gating)

After Gate 2 failed, a user-asked question fell outside anything Gate 3/4
were designed to answer: Gate 3 (never triggered, since it is conditioned on
Gate 2 passing) would only test a **4-way add** (GNN joining the existing
XGBoost+TFT+LightGBM blend). The question actually asked was narrower — does
a **3-way swap** (GNN *replacing* TFT, i.e. {XGBoost, LightGBM, GNN} vs. the
deployed {XGBoost, TFT, LightGBM}) do better, read once on test.

This is explicitly **not** a gate and does **not** reopen or amend this
ADR's Rejected status regardless of its outcome — it is recorded here only
because it is directly relevant context a reader would otherwise have to
reconstruct from `scripts/compare_gnn_swap_ensemble.py`'s git history. Both
candidates' blend weights were found via the same validation-only simplex
weight search `run_ensemble_eval.py` uses, then applied to test exactly
once.

| Ensemble | Weights | Val PR-AUC | Test PR-AUC |
|---|---|---|---|
| Deployed (xgb+tft+lgbm) | xgb 0.618 / tft 0.008 / lgbm 0.374 | 0.6840 | **0.5502** |
| Swap (xgb+lgbm+gnn) | xgb 0.654 / lgbm 0.340 / gnn 0.006 | 0.6840 | **0.5495** |

**Test PR-AUC delta (swap − deployed): −0.0007.** No improvement; the swap is
a statistically negligible step backward. The weight search assigned GNN a
weight of 0.006 — essentially the same negligible weight TFT already held
(0.008) — and validation PR-AUC came out identical to four decimal places
regardless of which near-zero-weight third model was offered, i.e. the
optimizer found the same XGBoost+LightGBM-dominated solution either way.

Pairwise validation-probability correlation (`pairwise_diagnostics`, all
four models): `corr_xgb_gnn=0.834`, `corr_tft_gnn=0.895`,
`corr_lgbm_gnn=0.877`, `corr_xgb_lgbm=0.944`, `corr_xgb_tft=0.824`,
`corr_tft_lgbm=0.865`. GNN does not correlate above the >0.9 threshold §4
Gate 4 would have flagged, but at 0.83–0.90 it is not offering materially
more diverse signal than the models already in the blend either — the same
"marginal weight, not a win" disposition Gate 4 would have assigned it lands
here by a different route.

**Bugfix found and fixed during this diagnostic:** `GraphBuilder.
from_state_dict()` (`src/data/graph_builder.py`) checked
`isinstance(edge_specs, list)` to decide whether to reconstruct nested
`EdgeKeySpec` dataclasses from their serialized dict form, but `GraphBuilder.
state_dict()`'s own `asdict()` call preserves a `Tuple`-typed field as an
actual tuple (not a list) when kept in-memory, which is exactly the form the
`GNNTrainer` artifact's `.meta.joblib` uses. Any GNN model saved with a
non-empty `edge_specs` — i.e. anything built from Arm A's `card_full` or
`addr_card` edge specs, which is what trial 14 (this ADR's Gate 2 subject)
used — could never be reloaded via `GNNTrainer.load()`; `from_state_dict`
would pass raw dicts through to `GraphBuildConfig`, and the first later
access to `spec.name` would raise `AttributeError: 'dict' object has no
attribute 'name'`. Fixed to accept both `list` and `tuple` containers;
verified against the existing 51-test `GraphBuilder`/`GNNTrainer` suite (all
pass) and by this diagnostic successfully loading and scoring the model. This
would have silently blocked any future GNN serving/inference code from
loading a Arm-A-edge-spec GNN artifact, independent of this ADR's outcome.
