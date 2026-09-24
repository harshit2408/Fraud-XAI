# ADR-004: Operating-point selection under an infeasible target band

- **Status:** Accepted
- **Date:** 2026-09-09
- **Plan task:** `docs/IMPLEMENTATION_PLAN.md` PRD Phase 9, step **9.5**
- **Extends:** [ADR-001](ADR-001-inference-orchestration.md) §3.3, §4.3, §4.5
  — *extends, does not supersede*. ADR-001's orchestration decisions all remain
  correct; this records a scoring-architecture option that was evaluated and
  **rejected**, and settles the operating point instead.
- **Gates:** closes PRD Phase 9 (the target-band stop condition)

---

## 1. Context

### 1.1 What Phase 9 set out to do, and what it achieved

PRD Phase 9 targets **precision ≥ 30% at recall ≥ 80%** at the deployed
threshold. Four steps ran, each measured and recorded:

| Step | Lever | Test PR-AUC | P @ deployed | P @ R≥80% (curve) |
|---|---|---|---|---|
| 9.0 | baseline | 0.5375 | 6.18% @ R 95.5% | 15.95% |
| 9.1 | cost model / threshold | 0.5375 | ~10% | 15.95% |
| 9.2 | UID client features | 0.5489\* | 8.57% @ R 90.1% | 16.15% |
| 9.3 | imbalance re-tune | 0.5394 | 10.31% @ R 90.0% | 18.29% |
| 9.4 | RFM/velocity + `dist1` | 0.5508 | 9.99% @ R 90.5% | 19.17% |

\* leakage-inflated by a UID self-inclusion artifact; not a valid baseline.

Precision at R≥80% moved **+3.2 points total** (15.95% → 19.17%) across four
cycles, with a decelerating per-cycle return (+0.2, +2.1, +0.9). The remaining
gap to the band is **10.8 points**.

### 1.2 Why the cascade cannot close that gap — closed-form, not estimated

PRD §10 step 9.5 proposes a two-stage cascade: stage 2 trained only on
stage-1-flagged transactions. A cascade is a **monotone filter** — stage 2 can
only remove rows from the flagged set, so every point it can reach corresponds
to some subset of that set. The question is therefore purely quantitative.

Measured on the promoted 9.4 model (`reports/ensemble_test_probabilities.npz`,
threshold 0.012251):

```
flagged set  N = 36,816   fraud = 3,677 (9.99%)   negatives = 33,139
stage-1 recall at the gate = 0.9048
```

To deliver **80% end-to-end recall**, stage 2 may keep only
`0.80 / 0.9048 = 88.42%` of the frauds it receives, i.e. TP = 3,251. At that
TP, the false positives a 30%-precision target allows are:

```
FP_allowed = TP x (1 - p) / p = 3,251 x (0.70 / 0.30) = 7,586
```

So stage 2 must **discard 77.1% of the flagged negatives** — an FPR of
**0.229** on the flagged-negative population.

The incumbent single-stage ensemble, asked for the same recall, already
achieves:

```
stage 1 at R = 80%:  precision 19.18%   TP 3,252   FP 13,703
                     FPR on flagged negatives = 0.414
```

**The cascade therefore requires a 44.6% relative FPR reduction, at equal
recall, on precisely the region the incumbent was optimised for, from a model
trained on the same features with 69% fewer negatives.** Lower targets do not
rescue it: 25% precision still needs FPR 0.294, and 20% needs 0.392 — barely
better than the incumbent already delivers.

### 1.3 The PRD's stated rationale does not survive measurement

PRD §10 9.5 justifies the cascade on the grounds that *"stage two sees a much
less imbalanced problem (flagged set is ~6% fraud vs. 3.5% overall)"*.

Two problems. First, the measured flagged-set prevalence is **9.99%**, not 6%.
Second, and decisively: **prevalence lift is not the mechanism that produces
precision at a fixed recall.** Once an operating point is chosen on a curve,
precision-at-recall is a function of ranking quality, not base rate. The
"less imbalanced ⇒ more precise" intuition conflates base-rate precision with
rank-quality precision. Imbalance was already attacked directly in 9.3
(+2.1pt) and SMOTE was rejected on measured evidence (−0.057 validation
PR-AUC). The cascade's stated lever is a third pass at a lever measured twice.

The cascade's *other* rationale — that stage 2 "can afford more expensive
features" — is genuine, but it is a **latency optimisation for expensive
features, not an accuracy mechanism in itself**. No such feature has been
identified. Building the architecture before the lever exists is speculative
generality.

---

## 2. Decision drivers

- **D8 — A recall floor is a hard constraint, not an objective term (new).**
  Every threshold selector in this repo maximises net value, which has no
  recall floor; the band demands one.
- **D9 — Stage 2's training distribution must be the serving distribution
  (new).** Stage 2's training set is defined by stage 1's own predictions, so
  in-sample stage-1 scores would train it on a distribution that never occurs
  in production.
- **D10 — Evidence over sequence completion (new).** Executing 9.5 because
  the plan lists it, against arithmetic saying it cannot succeed, is not
  diligence.
- Inherited: D1 exactness, D3 latency (<100 ms P95 — note this is a *tail*
  metric, and at a ~31% flag rate the P95 request is a flagged one, so the
  whole stage-2 cost lands in the budget).

---

## 3. Options considered

| # | Option | Verdict |
|---|---|---|
| A | Single-stage with new expensive features (no cascade) | Deferred — no candidate feature identified |
| B | Cascade, stage 2 on the 184-feature frame + OOF stage-1 score | **Rejected** (§1.2) |
| C | Cascade with flagged-set-specific expensive features | Rejected — depends on A |
| D | Recalibrate the operating point; no second stage | **Chosen** |

**B — the honest version of the build.** Recorded so a future revisit starts
from the right design rather than repeating the analysis: stage 2 trains on
training rows whose **out-of-fold** stage-1 score clears the gate, using the
184-feature frame **plus the stage-1 blended and per-model probabilities as
explicit inputs** (without them, stage 2 must rediscover the ordering from
fewer negatives and lands *below* stage 1). Folds must be **time-ordered**,
not random — random folds leak future card history through the expanding
aggregates and the ADR-003 trailing windows. Thresholds `(t1, t2)` must be
selected on a **joint** validation grid under an explicit end-to-end recall
floor, and stage-2 recall must always be reported end-to-end, never on the
flagged subset. Cost: K time-ordered replicas of all three models — the most
expensive single step in Phase 9 — for an estimated **+2 to +5 points**
against a 10.8-point gap.

**A comparison trap this ADR exists to prevent:** a cascade's flagged-set
precision must be compared to the single-stage curve **at matched recall**
(19.18% at R≥80%), never to the single-stage *deployed-threshold* precision
(9.99%). The latter comparison overstates the gain by roughly nine points and
would make a failed cascade look like a success.

---

## 4. Decision

### 4.1 Do not build the cascade

PRD Phase 9 step 9.5 is closed as **rejected on measured evidence**, not
deferred. §1.2's arithmetic is not an estimate; only the achievable relative
FPR improvement is, and even generous assumptions there miss the band.

### 4.2 The target band is infeasible on this dataset at R ≥ 80%

Recorded as a finding, not a failure. The measured precision-at-recall curve
for the promoted 9.4 ensemble:

| Recall | Precision | FP per fraud caught |
|---|---|---|
| 70% | 29.17% | ~2.4 |
| 75% | 24.12% | ~3.1 |
| 80% | **19.18%** | ~4.2 |
| 85% | 14.56% | ~5.9 |
| 90% | 10.38% | ~8.6 |

At R≥80% the system catches 80% of fraud at ~4.2 false positives per catch.
The band asks for ~2.3. That point is not on this dataset's frontier with the
current feature set.

### 4.3 The operating point stays where it is, pending a business decision

The R≥70% point measures **29.17%** precision — arithmetically close to the
30% target, at 10 points less recall. It is **not** promoted here, for two
reasons that must not be skipped:

1. **It is test-curve-derived.** Every threshold in this repo is
   validation-selected and test-confirmed (ADR-001 §2). Promoting a
   test-selected point would violate that discipline and inflate the number.
2. **It carries a real error bar.** At R=70%, TP ≈ 2,844 / FP ≈ 6,904 gives a
   binomial standard error of ~0.46pt on 29.17% — ±0.9pt at 2σ from sampling
   alone, before threshold-selection variance. `evaluator.py`'s own docstring
   warns the precision-at-recall curve is sawtoothed and "not a guaranteed
   lower bound". **The honest statement is ≈29% ± ~1pt, and the gap to 30%
   sits inside that interval.** Do not report the target as met at R≥70%.

Choosing R≥70% over R≥90% is a **business trade** — review capacity against
fraud caught — not a modelling decision. It also overrides the
`net_of_principal` cost model settled in ADR-001 §7, under which the operating
threshold is a *derived* quantity. If the business wants R≈70%, that is a
deliberate override and must be recorded as one, then promoted through
`scripts/run_ensemble_eval.py`'s spec-writing path with recall-constrained
selection made explicit — never by hand-editing `models/ensemble.json`, which
would break the checksum contract.

### 4.4 The dependency is inverted for any future revisit

**Feature first, architecture second.** 9.5 becomes worth reconsidering only
if a specific expensive feature with demonstrated standalone lift is found —
at which point option A (add it single-stage) should be measured before
option C (add a stage to afford it). Note GNN-style entity features, the
usual candidate, were already scoped out of this phase on measured evidence
(comparable benchmarks at AUROC 0.86, below this ensemble's 0.9153).

---

## 5. Consequences

**Positive.** Phase 9 closes with a defensible finding instead of an open
step. The most expensive step in the sequence is not built. The precision/
recall frontier is documented, so the operating point becomes an explicit,
revisitable business choice.

**Negative, accepted.** The PRD's target band is not met, and this ADR is the
record of why. Any future cascade work restarts from §3 option B's design
notes rather than from a running implementation.

**Incidental fix shipped alongside.** `parse_ensemble_spec` read
`schema_version` and never validated it, so a future 2.0 spec would have been
silently downgraded to 1.0 with unknown fields dropped — serving a single
stage at what was meant to be a gate threshold, flagging ~31% of traffic while
reporting healthy. Now rejected with a major-version check
(`src/serving/ensemble_spec.py`), with tests for both the rejection and an
additive 1.x bump.

---

## 6. Open questions

1. **Which operating point does the business actually want?** §4.3 frames the
   trade; the answer needs review-capacity data this project does not have.
2. **Is there an expensive feature worth the cascade?** Unanswered, and the
   precondition for revisiting §4.1.
3. **Would a recall floor belong in the threshold selector itself?** Every
   selector here optimises net value unconstrained. If recall floors become a
   recurring requirement, `find_optimal_threshold` should grow a constraint
   argument rather than each caller re-deriving one.
