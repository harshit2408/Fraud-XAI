# ADR-005: GNN-GraphSAGE architecture evaluation

- **Status:** Accepted (2026-09-10) — **GNN-GraphSAGE rejected on evidence**;
  the evaluation infrastructure is kept, following the ADR-004 precedent of
  recording a rejected direction in full.
- **Date:** 2026-09-09 (decision drivers / design); **2026-09-10** (result)
- **Plan task:** `docs/IMPLEMENTATION_PLAN.md` PRD Phase 12, steps **12.2.1–12.2.5**
- **Extends:** nothing — this is a standalone architecture-exploration record,
  following the precedent of writing the ADR **regardless of outcome** set by
  [ADR-004](ADR-004-operating-point-selection.md) (which recorded a *rejected*
  direction).
- **Gates:** closes PRD Phase 12; a positive 12.2.3 result additionally gates
  12.2.4 (ensemble integration).

---

## 1. Context

### 1.1 Why this phase exists

PRD Phase 9 explicitly excluded GNNs, citing a 2026-09-08 benchmark
(arXiv:2503.22681, "detectGNN") at AUROC 0.86 — below this project's ensemble
ROC-AUC at the time. That exclusion was **superseded on new evidence**: Uddin &
Aziz, *"Shapley Value-Guided Adaptive Ensemble Learning for Explainable
Financial Fraud Detection…"* (arXiv:2604.14231, submitted 2026-04-14), Table II,
reports a **GNN-GraphSAGE** model on the **same IEEE-CIS dataset** (590,540
transactions, 3.5% fraud, 118,108-row held-out test) at:

| Metric | Paper's reported value |
|---|---|
| AUC-ROC | 0.9248 |
| PR-AUC | 0.6334 |
| F1 (τ\*=0.86) | 0.6013 |
| Precision / Recall (τ\*) | 70.6% / 52.4% |

This project's frozen 2026-09-09 ensemble baseline (`reports/ensemble_results.json`,
mlflow run `69d3a4d5…`, `dataset_hash d20f03c0…`) scores **test PR-AUC 0.5502,
ROC-AUC 0.9154**. The paper's 0.6334 PR-AUC would be a **+0.083 absolute** lift
*if it transferred cleanly* — enough to justify a dedicated exploration phase.

### 1.2 Four caveats from reading the paper directly (not the abstract)

1. **Split methodology is not confirmed temporal.** The paper describes 5-fold
   stratified CV + an 80/20 held-out split with SMOTE-Tomek applied *inside*
   training folds. It does not state the held-out split is time-ordered on
   `TransactionDT`. This project's entire Phase A rework exists because a
   non-temporal split on this exact dataset was found to leak distributional
   information; the measured Phase-A delta was **0.0076 absolute** for
   *mostly-unsupervised* leakage, and a random/stratified split leaks more.
   **If the paper's split is not temporal, its 0.6334 is not directly
   comparable to this project's 0.5502**, and the honest expectation for a
   temporal-split reproduction is lower — plausibly by 0.02–0.04 given
   stratified + SMOTE-Tomek together.
2. **New infrastructure, not a drop-in trainer.** GraphSAGE needs an explicit
   transaction-to-transaction graph (`src/data/graph_builder.py`), not a fourth
   `train_*.py`.
3. **Class-weighted loss, not SMOTE, for the GNN** — the paper uses
   `pos_weight ≈ 27.6` class-weighted CE, reasoning synthetic nodes have no
   graph connectivity. Consistent with this repo's own Phase 9.3 finding
   (SMOTE degrades val PR-AUC vs. class-weighting on the GBDTs).
4. **The paper's own authors flag the attribution as unresolved**: whether the
   GNN gain is "genuine topological signal … or … an implicit feature-smoothing
   mechanism on correlated tabular inputs remains an open question." If it is
   smoothing, a cheaper feature-engineering change could capture most of it.

### 1.3 TFT's disposition (recorded so this reads as "moving on", not "TFT failed")

The same session that opened Phase 12 finished repairing the TFT (fixed a
val-set scaler leak — finding G2 — and reverted a regularisation change that had
made things worse) and produced the **best TFT result in the project's
history**: test PR-AUC 0.4670, overfit gap 0.164. TFT is set aside because even
at its best it contributes a marginal, unstable ensemble weight
(~0.008–0.05, inside the flat region of a search dominated by its 0.82
correlation with XGBoost), not because the repair was wasted. TFT stays in the
codebase; its artifacts, manifests and figures remain in `reports/`.

---

## 2. Decision drivers

- **D1 — Comparability.** Only one comparison governs an adoption decision:
  **GNN-on-this-temporal-split vs. the 0.5502 ensemble baseline on the same
  split.** The paper's 0.6334 is a working hypothesis, not a target. A GNN
  result at or above 0.6334 is to be treated as *suspicious* and audited for a
  leak before it is believed (this repo's history — Phase A, findings G1/G2 —
  is a run of "too-good" numbers that were leaks).
- **D2 — Leakage discipline carries over unchanged.** The GNN must inherit the
  same train-only-fit guarantees every other model here has. The graph is built
  over `concat(train, val, test)` for realistic connectivity, but message
  passing is **phase-scoped and symmetric** (see §4.2): a train node can only
  ever aggregate from other train nodes, at any hop.
- **D3 — Interface parity.** `GNNTrainer` matches the
  `XGBTrainer`/`LGBMTrainer`/`TFTTrainer` contract exactly, so it slots into
  `scripts/run_ensemble_eval.py`'s existing per-model loop with no parallel
  evaluation path.
- **D4 — Pre-registration (mle-reviewer R4), fixed BEFORE the run:**
  - GNN hyperparameters are **frozen from the paper** — `hidden_dims [128, 64]`,
    3-layer MLP head, `pos_weight` = empirical train neg/pos (≈ 27.6),
    `NeighborLoader` fan-out `[10, 5]` — or tuned on **validation only**. The
    effective config hash is recorded in the model manifest.
  - The GNN **proceeds to 12.2.4 ensemble integration iff** its standalone
    **test** PR-AUC exceeds `0.5502 + m` with **`m = 0.005`** (the existing
    `ensemble.min_lightgbm_lift`). Without a pre-registered margin, "did it beat
    baseline?" becomes a repeated held-out consultation — the exact G1 failure
    mode.
  - The 4-way min-lift gate and any "replace TFT with GNN" decision are
    evaluated on **validation** PR-AUC, **once**, on frozen artifacts and a
    frozen `dataset_hash`. `test_lift` stays a logged diagnostic. No "retrain
    the GNN and re-check."
- **D5 — No normalization layers in the model (mle-reviewer R3).** The eval-mode
  full-graph forward runs on val/test-spanning edge slices; a `BatchNorm` /
  `LayerNorm` / `GraphNorm` would update shared running statistics from
  val/test activations, reused on the next `.train()` call — a silent leak the
  temporal edge masking cannot catch. `SAGEConv → ReLU → Dropout` only. Any
  future norm-layer addition **requires re-review of this ADR's leakage
  argument**.

---

## 3. Options considered

### 3.1 Leakage control for graph construction

The graph must give val/test nodes realistic connectivity while never letting
post-split information shape a train node's learned representation. Three
mechanisms were considered:

| Option | Mechanism | Verdict |
|---|---|---|
| **(a) Phase-scoped edge masks, symmetric message passing** | One static undirected `edge_index`; three boolean edge masks. For the train phase, `NeighborLoader` samples on `edge_index[:, edge_mask_train]`, whose edges have **both** endpoints in the train span — so a train seed can only reach train nodes at any hop. The per-node neighbour cap is computed **within each phase's eligible node prefix** (train-span for `edge_mask_train`, train+val for `edge_mask_val`, all for `edge_mask_test`), so a later-phase row's position can never change which train↔train edges survive. | **CHOSEN.** Correct, and the standard PyG idiom. Matches `preprocess.py`'s "full-frame causal features, split-scoped fit" pattern. |
| (b) Forward-only directed graph (`src < dst`) | Keep only past→future edges. | **Rejected.** Breaks `SAGEConv`'s in-neighbour aggregation: the earliest transaction of a card has zero in-neighbours and learns nothing from the graph; degree distribution is pathologically skewed; boundary nodes are starved. |
| (c) Full undirected graph, mask only the *loss* to the target split | Every edge live in every batch; compute loss only on train seeds. | **Rejected — this is the actual leak.** With 2-hop `NeighborLoader` sampling, a train node's 1-hop neighbour can be a val/test node, and that node's *features* (post-split distribution) get aggregated into the train node's embedding. Masking the loss does not stop feature information crossing the temporal boundary during message passing. |

A `directed_past_to_any` refinement on top of (a) — keeping only the `src ≤ dst`
copy in train/val phases — was considered and **dropped**: `max(src, dst) ≤ ceiling`
already blocks every cross-split edge, and applying the directed filter to the
loaders but not the eval forward would make val early-stopping and the reported
test number use a *different* message-passing topology than training. The
resolution is **symmetric everywhere** (train loader, val loader, every eval
forward — `GNNTrainer._edge_index_for(phase)` is the single source of truth),
justified because `card1` / `(addr1, ProductCD)` edges are **identity links**,
not temporal-causal ones.

### 3.2 Where `GNNTrainer` lives and how it scores

`predict_proba(X)` in the sibling trainers takes a plain feature frame and
scores it standalone; a GNN node needs its neighbourhood, which only exists in
the built graph. Options: (i) hold the built `Data` on the instance and map
input rows → node ids; (ii) rebuild a subgraph per call. **(i) chosen** — the
trainer holds `self.data` (populated by `train()`, or lazily by `load()` from
the graph cache), and `predict_proba` resolves the split by **exact row count**
against the graph's registered span sizes, raising on a zero or ambiguous match
(mle-reviewer R6). **Limitation, recorded and asserted by a test:** this
supports **whole-split scoring only**. Row-level scoring, arbitrary subsets, and
scoring a transaction absent from the cached graph are **out of scope for Phase
12** — Phase 12 delivers *offline evaluation*, not a serving path. True online
inductive GNN inference (attach node, sample neighbours from a live graph store)
would need its own design.

---

## 4. Decision

### 4.1 Build the GNN-GraphSAGE evaluation infrastructure — done

New modules, all matching existing conventions (interface, config-driven
hyperparameters, 3-file checksummed artifact, `build_manifest` lineage):

- `src/data/graph_builder.py` — `GraphBuilder` + `GraphBuildConfig`. Builds the
  transaction graph: nodes = transactions from `concat(train, val, test)`;
  edges from shared `card1` (≤ 10 neighbours) and composite `(addr1, ProductCD)`
  (≤ 5 neighbours), phase-scoped; rows on the `addr1` imputation sentinel
  (`-999.0`) excluded from composite edges (mle-reviewer R8 — otherwise a
  ~47k-row clique); a `QuantileTransformer` fit on the **train node span only**
  (R5); a `Data` blob cached with a sha256 sidecar + dataset-hash meta.
- `src/models/gnn_model.py` — `GraphSAGEModel`: 2 × `SAGEConv` (128 → 64) +
  3-layer MLP head → 1 logit. No normalization layers (R3, enforced at
  construction).
- `src/training/train_gnn.py` — `GNNTrainer` (interface parity) + `main()`:
  `NeighborLoader` on the phase-scoped symmetric train subgraph, class-weighted
  BCE, early stopping on validation PR-AUC, isotonic calibrator + val-selected
  frozen threshold, standalone eval on all splits, `reports/gnn_results.json`.
- `config/config.yaml` `model.gnn` block; `src/config.py` `GNNConfig`
  (Optional-defaulted for backward compatibility) + `serving.gnn_model_path`;
  `torch-geometric==2.5.3` + `torch-scatter` + `torch-sparse` in
  `requirements.txt`.

### 4.2 Leakage control: phase-scoped symmetric edge masks (Option 3.1(a))

As described in §3.1. Regression tests assert: no `edge_mask_train` edge has a
non-train endpoint; no `edge_mask_val` edge has a test endpoint; `edge_mask_test`
covers all edges; the train↔train edge set is invariant to whether val/test rows
are present; and permuting `y_test` before `train()` leaves the trained
`state_dict` byte-identical (test labels never enter the loss — R7).

### 4.3 Standalone result (PRD 12.2.3) — GNN underperforms the baseline

Run 2026-09-10, `reports/gnn_results.json`, mlflow run
`6615a0fc6f5e4a55a1a3a084d952cbe8`, seed 42, `dataset_hash` unchanged from the
0.5502 baseline. Hyperparameters frozen from the paper (D4): `hidden_dims
[128, 64]`, 3-layer MLP head, `pos_weight` = 27.43 (empirical train neg/pos),
`NeighborLoader` train fan-out `[10, 5]`; scoring batched (`eval_num_neighbors
[40, 20]`, `eval_batch_size 4096`) after a full-graph forward OOM'd the 8 GB
card.

| Metric | GNN-GraphSAGE (this pipeline, temporal split) | Ensemble baseline | Paper (Table II) |
|---|---|---|---|
| **Test PR-AUC** | **0.4410** | 0.5502 | 0.6334 |
| Test ROC-AUC | 0.8849 | 0.9154 | 0.9248 |
| Val PR-AUC | 0.5260 | — | — |
| Train PR-AUC | 0.6564 | — | — |
| Overfit gap (train − test) | 0.2155 | — | — |
| Val → test gap | **0.0850** | — | — |
| Test P / R / F1 @ cost-threshold (0.0981) | 5.6% / 94.8% / 0.107 | — | 70.6% / 52.4% / 0.601 |
| Test Brier (raw → calibrated) | 0.0675 → 0.0246 | — | — |

**The GNN lands at test PR-AUC 0.4410 — 0.109 *below* the 0.5502 ensemble
baseline, 0.026 below standalone TFT's best (0.4670), and 0.192 below the
paper's headline 0.6334.** It is outside the pre-registered "pessimistic" band
(0.52–0.56) on the low side.

Two things the number tells us plainly:

1. **The split methodology accounts for a large part of the gap to the paper
   (§1.2 caveat 1, confirmed).** Val PR-AUC 0.5260 vs. test 0.4410 is an
   **0.085 absolute** drop across the temporal boundary — an order of magnitude
   larger than this project's Phase-A leakage-fix delta (0.0076), and squarely
   in the "stratified + SMOTE-Tomek costs 0.02–0.04, plausibly more" range the
   ADR pre-registered. A model that scores 0.53 on a shuffled/stratified
   held-out set and 0.44 on a temporal one is consistent with the paper's 0.63
   being partly a split artifact.
2. **What remains is genuine underperformance, not just an unfair comparison.**
   Even the *validation* PR-AUC (0.5260) — measured the same way the baseline's
   0.5502 is — does not clear the baseline. The GNN is a weaker ranker than the
   existing GBDT ensemble on this dataset, on its own turf. The 0.216
   train→test overfit gap says the 2-layer GraphSAGE is memorising rather than
   finding transferable topological structure.

### 4.4 Ensemble integration (PRD 12.2.4) — NOT triggered

Per the pre-registered D4 rule: the GNN proceeds to 12.2.4 **iff** standalone
test PR-AUC > `0.5502 + 0.005 = 0.5552`. Actual: **0.4410 < 0.5552**. The
run's own log line records the decision:

```
12.2.4 ensemble integration trigger: test PR-AUC 0.4410 vs baseline+margin 0.5552 -> DO NOT PROCEED
```

No `"gnn"` key is added to `scripts/run_ensemble_eval.py`; no `no_gnn` fallback
mode; the deployed 3-way `models/ensemble.json` is untouched. This is a
one-shot decision on frozen artifacts — there is no "retune the GNN and
re-check" (that would be the G1 failure mode the margin exists to prevent).

---

## 5. Consequences

**GNN-GraphSAGE is rejected for this project.** It does not beat — does not
even match — the 3-way GBDT+TFT ensemble baseline on this project's temporal,
leakage-audited IEEE-CIS split (0.4410 vs. 0.5502 test PR-AUC), and it does not
clear the baseline on validation either. Phase 12 closes here; 12.2.4 and any
ensemble change are not pursued.

- **The paper's 0.6334 does not transfer.** The 0.085 val→test drop under a
  temporal split (§4.3) confirms §1.2 caveat 1 was the right thing to worry
  about. The paper's authors' own unresolved question (§1.2 caveat 4 — "genuine
  topological signal … or … implicit feature-smoothing on correlated tabular
  inputs") is now the leading explanation: a GraphSAGE-mean aggregation over
  `card1`/`(addr1,ProductCD)` neighbours *is* a form of feature smoothing, and
  on a stratified split that smoothing borrows signal across the train/test
  boundary in a way a temporal split forbids. The GBDT ensemble already has
  explicit `card1` / `uid` aggregate features (Phase 9.2/9.4) that capture the
  non-leaking part of that signal without a graph.
- **TFT stays where it is.** This result does not change TFT's disposition
  (§1.3) — TFT remains in the deployed ensemble at its marginal weight; the GNN
  does not replace it.
- **Disposition of the new code.** `src/data/graph_builder.py`,
  `src/models/gnn_model.py`, `src/training/train_gnn.py`, their 32 unit tests,
  the `model.gnn` config block, and the graph cache
  (`data/processed/graph/fraud_graph.pt`, gitignored) all stay in the tree as a
  completed, evaluated trial — same as TFT. `models/gnn_model.*` and
  `reports/gnn_results.json` are the artifact record. `make train-gnn`
  reproduces the run. The three `torch-geometric` deps stay in
  `requirements.txt`.
- **No follow-up is scheduled.** A revisit would need a materially different
  hypothesis (a deeper/regularised GNN to fight the 0.216 overfit gap; edge
  types beyond `card1`/`addr1`; or an inductive serving design) *and* a reason
  to expect it beats a 0.5502 baseline that four GBDT/TFT iterations have not
  moved much. The cheap probe in §6 is the only thing worth doing first, and it
  does not require a graph.

Standing consequences regardless of outcome:

- `torch-geometric` + `torch-scatter` + `torch-sparse` are now project
  dependencies (compiled against `torch 2.2.0+cu121`; install note in
  `requirements.txt`).
- `src/data/graph_builder.py`, `src/models/gnn_model.py`,
  `src/training/train_gnn.py` and their tests remain in the tree as a completed,
  evaluated trial — the same disposition TFT gets — so a future revisit starts
  from working infrastructure, not a blank page.
- The R3 "no normalization layers" constraint is load-bearing for the leakage
  argument and is enforced in code; it must not be relaxed without re-review.

---

## 6. Open questions

- **Smoothing vs. topology (paper caveat 4).** If the reproduction lands below
  the plausible band, is the paper's headline number an artifact of
  neighbourhood aggregation smoothing correlated tabular inputs on a
  non-temporal split? A cheap probe: add a few `card1`/`uid` mean-aggregate
  features to the GBDTs and see if they close part of any observed gap without a
  graph.
- **Denser graph than the paper.** This construction produces more edges than
  the paper's reported 385,018 (sliding-window cap vs. their likely random
  sampling). `NeighborLoader` re-samples per layer at train time so this only
  widens the sampling pool, but a `random_cap` mode would allow closer parity if
  a follow-up wants it.
- **Inductive serving.** Out of Phase 12 scope; would need on-the-fly
  neighbourhood construction from a live graph store, and its own leakage
  review.
