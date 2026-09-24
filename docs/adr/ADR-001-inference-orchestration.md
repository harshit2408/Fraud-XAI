# ADR-001: Inference orchestration — `ModelRegistry` + `InferenceService`

- **Status:** Accepted
- **Date:** 2026-08-26
- **Amended:** 2026-09-08 — §7 resolves open question 6.3 (cost-model convention), PRD Phase 9 step 9.1
- **Plan task:** `docs/IMPLEMENTATION_PLAN.md` Phase E, task **E1**
- **Gates:** PRD Phase 5 (FastAPI serving), PRD Phase 4 (SHAP in the request path)
- **Depends on:** [ADR-002](ADR-002-realtime-feature-state.md) for the online feature-state port this design consumes

---

## 1. Context

Phases A–D closed a training pipeline that now produces four checksummed,
manifest-backed artifacts sharing one `dataset_hash`
(`4c1059dc…`, see `reports/ensemble_results.manifest.json`):

| Artifact | Produced by | Carries |
|---|---|---|
| `models/xgb_model.pkl` (+ `.metadata`, `.checksums`) | `src/training/train_xgb.py:202` | booster, `feature_names`, frozen threshold, frozen calibrator |
| `models/lgbm_model.pkl` (+ sidecars) | `src/training/train_lgbm.py:215` | same shape as XGB |
| `models/tft_model.pt` (+ sidecars) | `src/training/train_tft.py:839` | `state_dict`, model config, sequence-builder state incl. the fitted `QuantileTransformer`, frozen threshold, frozen calibrator |
| `data/processed/transformers/` | `FeatureEngineer.save_transformers` (`feature_engineering.py:1026`) | label/freq encoders, PCA, imputer fills, card-hash freqs, target-encoding prior + per-entity counters, `null_ratio` denominator |

What does **not** exist is anything that owns them together at serving time.
`src/api/main.py` is still the Phase-0 stub: a `/health` endpoint, a lifespan
handler containing `# TODO Phase 5: Load models into ModelRegistry`, and
`"model_loaded": False` hardcoded in the response. `src/api/routes/`,
`src/api/schemas/` and `src/api/middleware/` are 1-line empty packages.

Three properties of the closed phases make the orchestration question sharper
than "load three models and average them":

1. **The deployed model is an ensemble whose parameters live outside every
   model artifact.** The blend weights (`xgb 0.692 / tft 0.134 / lgbm 0.174`)
   and the operating threshold (`0.006123`) exist only in
   `reports/ensemble_results.json`, written by `scripts/run_ensemble_eval.py`.
   Each per-model artifact carries its *own* frozen threshold, which is the
   wrong threshold for the blend.
2. **The blend is over calibrated probabilities, not raw ones.**
   `run_ensemble_eval.py:70 predict_proba_prefer_calibrated` blends each model's
   `predict_proba_calibrated` output, and the threshold was selected on
   *validation ensemble probabilities built that way*
   (`run_ensemble_eval.py:297`). A serving path that blends raw probabilities
   and applies `0.006123` is applying a threshold to a different quantity.
3. **The codebase has a standing "never invent an operating point" rule.**
   `XGBTrainer.predict` (`train_xgb.py:198`) and `TFTTrainer.predict`
   (`train_tft.py:835`) both raise rather than default when no threshold was
   frozen. Any orchestration layer must inherit that posture rather than quietly
   soften it.

A fourth constraint is operational: `docker-compose.yml` mounts `./src`,
`./models`, `./logs` and `./config` into `fraud-api` — **not `./data`**. The
fitted transformers the inference path cannot function without currently live
under `data/processed/transformers/` and are therefore not present inside the
serving container at all.

---

## 2. Decision drivers

- **D1 — Latency.** PRD §6.1: < 100 ms P95 for a single-transaction prediction;
  Kafka consumer ≥ 500 tx/s (PRD §6.1, Phase 6).
- **D2 — One prediction path, two transports.** HTTP (`POST /predict`, Phase 5)
  and the Kafka consumer (Phase 6) must produce identical decisions for
  identical input. Duplicating orchestration in a route handler and again in a
  consumer loop is the drift mechanism that produced the two divergent ensemble
  implementations `src/models/ensemble.py` was rewritten to eliminate.
- **D3 — Train/serve equivalence is a testable requirement**, not an aspiration
  (task E4: "byte-identical feature vectors").
- **D4 — Traceability.** Every response must name the model version that
  produced it (task E5), and that version must resolve back to a git SHA,
  config hash and dataset hash.
- **D5 — Fail loudly, never silently degrade.** Established by Phases C4/D6:
  checksum verification before deserialization, hard raises on a missing
  threshold or calibrator.

---

## 3. Options considered

### 3.1 Where orchestration lives

**Option A — Logic inside FastAPI route handlers.**
Simplest for Phase 5 alone. Rejected: violates D2. The Kafka consumer (Phase 6)
runs in the same process but not through the HTTP layer, so it would either
re-implement the sequence or have to fabricate a fake `Request`.

**Option B — A single `PredictionPipeline` god-object owning both artifact
loading and per-request scoring.**
Rejected: conflates a process-lifetime concern (load once, validate once, fail
startup) with a per-request concern (transform, score, explain). It also makes
the pipeline untestable without real artifacts on disk, which blocks E4's
train/serve equivalence test from running in unit-test time.

**Option C (chosen) — Two collaborators in a transport-free `src/serving/`
package.** `ModelRegistry` owns load-time concerns and hands out an immutable
snapshot; `InferenceService` owns per-request orchestration and takes that
snapshot plus a feature-state port (ADR-002) as constructor arguments.
`src/api/` is reduced to an HTTP adapter (routes, Pydantic schemas,
middleware); the Kafka consumer constructs the same `InferenceService`.

### 3.2 Which probability the ensemble blends

**Option A — Blend raw probabilities, threshold at `0.006123`.**
Rejected: the threshold was not selected against that quantity (see Context §2).

**Option B (chosen) — Blend per-model *calibrated* probabilities, reproducing
`predict_proba_prefer_calibrated` exactly, and threshold the blend at the frozen
`0.006123`.** This is the only combination for which the shipped threshold is
meaningful.

**Option C — Additionally apply the ensemble-level isotonic calibrator.**
Rejected on evidence: `reports/ensemble_results.json` records
`brier_test_before 0.021731` → `brier_test_after 0.022036`. The ensemble's own
calibration *worsens* test Brier, and it is not persisted as an artifact. It
stays a diagnostic.

### 3.3 Degraded operation when a model is unavailable

**Option A — Renormalize the surviving weights over the simplex and keep
serving at `0.006123`.**
Rejected. Renormalizing changes the score distribution; a threshold selected for
a 3-way blend is not valid for a renormalized 2-way blend. This is precisely the
"invent an operating point" failure the trainers already refuse to commit.

**Option B — Refuse all traffic if any model is unavailable.**
Rejected as unnecessarily brittle: TFT is the heaviest and most failure-prone
component and carries only 0.134 weight.

**Option C (chosen) — Pre-registered degradation modes.** A degradation mode is
servable only if its weights *and its own threshold* were frozen at evaluation
time. `run_ensemble_eval.py` is extended to emit, alongside the 3-way optimum,
a `{xgb, lgbm}` 2-way mode with its own grid-searched weights and its own
cost-optimal threshold. Any mode not in the artifact is not servable, and the
request fails rather than being scored on an improvised operating point. The
response names the mode and the contributing models.

### 3.4 What SHAP explains

**Option A — KernelSHAP over the ensemble scoring function.**
Model-agnostic and explains the actual decision, but costs thousands of blend
evaluations per request — including a TFT forward pass each. Rejected against
D1 by roughly two orders of magnitude.

**Option B (chosen) — TreeSHAP on the XGBoost component only**, built once at
startup from the loaded booster, explaining the component that carries 0.692 of
the blend. The response labels this honestly (`explained_model: "xgb"`,
`explained_weight: 0.692`) rather than presenting it as an explanation of the
ensemble.

**Option C — TreeSHAP on XGB and LightGBM, weight-summed.**
Deferred, not rejected. Both are tree models over an identical feature set, so
their SHAP values are additive over a common basis and this would cover 0.866 of
the blend. It doubles explanation cost for a component whose scores correlate at
0.9253 with XGB's (`reports/ensemble_results.json`), so it is not worth doing
before latency is measured.

### 3.5 Where serving artifacts live

**Option A — Add a `./data:/app/data` mount to `docker-compose.yml`.**
Rejected: mounts the entire raw and processed dataset (gigabytes) into a serving
container to reach one small directory, and couples serving to the training
machine's data layout.

**Option B (chosen) — `save_transformers` also writes to a serving artifact
location under `models/` (`models/transformers/`), which is already mounted.**
Serving reads only from `models/`. `data/processed/` remains a training output.

---

## 4. Decision

### 4.1 Package layout

```
src/serving/
  registry.py       ModelRegistry, LoadedModels (frozen snapshot)
  inference.py      InferenceService
  ensemble_spec.py  load/validate the ensemble artifact
src/api/
  main.py           lifespan builds the registry; app state holds the service
  routes/           HTTP adapter only
  schemas/          Pydantic request/response (task E5)
```

`src/serving/` imports nothing from `src/api/`. The Kafka consumer (Phase 6)
depends on `src/serving/` alone.

### 4.2 `ModelRegistry` — load once, validate once, fail startup

Constructed inside the FastAPI `lifespan` startup block. It:

1. Loads `XGBTrainer.load`, `LGBMTrainer.load`, `TFTTrainer.load` and
   `FeatureEngineer.load_transformers` — each of which already verifies its own
   sha256 manifest before deserializing (Phase D6). No new integrity mechanism
   is introduced.
2. Loads the ensemble specification (§4.3).
3. Runs **cross-artifact consistency validation** and raises on any failure:
   - every model manifest's `dataset_hash` is identical and equals the ensemble
     manifest's;
   - every model's `config_hash` is identical — **amended 2026-09-01 during
     implementation: this is a WARNING, not a hard failure.** `config_hash`
     covers the entire config file, so editing one model's hyperparameters
     changes it for every artifact trained afterwards even though the earlier
     models are untouched. The shipped artifacts show exactly that (XGBoost and
     TFT at `ad07f4af8389`, then LightGBM and the ensemble at `24e774db15ac`
     after the documented `early_stopping_rounds` edit), so failing closed here
     would reject a valid deployment. `dataset_hash` equality stays fatal. A
     per-section config hash would let this be restored as fail-closed;
   - every model has a non-`None` frozen `calibrator` (the blend is defined over
     calibrated probabilities — `predict_proba_prefer_calibrated`'s raw fallback
     is a *training-script* concession to pre-Phase-C4 artifacts and is **not**
     permitted in serving);
   - the ensemble spec's model names are exactly the loaded models' names.
4. Builds the TreeSHAP explainer from the XGBoost booster.
5. Returns a frozen snapshot. Nothing mutates it afterwards; a model change is a
   process restart, not an in-place swap.

Startup failure is the correct outcome for any of these: a mixed-vintage
ensemble silently serving traffic is strictly worse than a container that will
not come up.

### 4.3 The ensemble becomes an artifact, not a report

`reports/ensemble_results.json` is an *output of evaluation*. Serving must not
read from `reports/`. `run_ensemble_eval.py` additionally writes
`models/ensemble.json` + `models/ensemble.checksums.json`, containing only what
serving needs:

```jsonc
{
  "schema_version": "1.0",
  "modes": {
    "full":   { "models": ["xgb","tft","lgbm"],
                "weights": {"xgb": 0.692, "tft": 0.134, "lgbm": 0.174},
                "threshold": 0.006123399927495639 },
    "no_tft": { "models": ["xgb","lgbm"],
                "weights": {"xgb": "<grid-searched>", "lgbm": "<grid-searched>"},
                "threshold": "<cost-optimal on val for THIS blend>" }
  },
  "default_mode": "full",
  "probability_space": "per_model_calibrated",
  "dataset_hash": "4c1059dc…",
  "mlflow_run_id": "0fb89a9beb3f4dbd8883dce82819979f"
}
```

`probability_space` is explicit so the contract cannot be lost: it names which
per-model method the weights and thresholds were fitted against. A registry
loading a spec whose `probability_space` it does not implement raises.

### 4.4 `model_version` — one string, fully resolvable

```
{dataset_hash[:12]}-{config_hash[:12]}-{git_sha[:7]}
```

taken from the ensemble manifest (all three fields are already populated there).
It appears in every `/predict` response, every JSONL prediction log line, and
`/health`. Because every artifact was validated to share these hashes at
startup, one string identifies the whole serving stack.

### 4.5 Per-request orchestration

`InferenceService.predict(transaction)` executes:

1. **Validate** — Pydantic schema (task E5), including range and staleness
   checks. Rejection happens at the boundary; nothing downstream re-validates.
2. **Transform** — `FeatureEngineer` methods with `fit=False` and, for target
   encoding, `update_state=False`. Feature groups requiring online per-entity
   history are supplied by the `FeatureStateStore` port defined in
   [ADR-002](ADR-002-realtime-feature-state.md).
3. **Align** — reindex to the registry's `feature_names` (the XGB artifact's
   persisted list). A missing column raises; this is the loud failure the
   Phase-F audit already noted as fortunate.
4. **Score** — `predict_proba_calibrated` on each model in the active mode. TFT
   additionally receives its per-card sequence window from the state store.
5. **Blend** — `src/models/ensemble.blend` with the mode's weights. The existing
   N-model function is reused unchanged; there is no second blend implementation.
6. **Decide** — `blended >= mode.threshold`.
7. **Explain** — TreeSHAP on the XGB component; top-5 signed contributions.
8. **Log** — structured JSON line to `serving.log_file`, carrying
   `model_version`, mode, per-model probabilities, blend, threshold, decision.

Steps 4–7 are the only per-request cost. Step 5 is arithmetic over three
scalars.

### 4.6 Degradation is declared, never improvised

Mode selection is by artifact, per §3.3. If the state store cannot supply TFT's
sequence window, the service falls back to `no_tft` *only because that mode has
its own frozen threshold in the artifact*. If a mode is absent, the request
fails with 503. The response always carries `mode` and `degraded: true|false`.

Note explicitly: a card with no prior history is **not** a degraded request.
Zero-padded sequences and prior-valued target encodings are exactly what the
training data contains for each card's first transactions (see ADR-002 §5.4).
Cold start is the normal path.

---

## 5. Consequences

**Positive**

- One scoring path serves HTTP and Kafka (D2). E4's train/serve equivalence test
  targets `InferenceService` directly, with no HTTP client involved.
- Startup validation converts the entire class of "ensemble weights from one run
  applied to models from another" bugs into a container that refuses to start.
- The `probability_space` field makes the calibrated-blend contract explicit and
  machine-checkable, closing the most likely way this design gets silently
  broken later.
- The threshold-per-mode rule extends the trainers' existing "never invent an
  operating point" discipline to the ensemble instead of quietly abandoning it
  at the layer where it matters most.

**Negative / accepted costs**

- `run_ensemble_eval.py` must now run a second weight search and threshold
  selection for the `no_tft` mode, and write a serving artifact. This is real
  added work in a script that is already the most complex in `scripts/`.
- `save_transformers` gains a second write location, which is duplication until
  `data/processed/transformers/` is retired.
- SHAP explains 69.2% of the decision, not 100%, and the response has to say so.
  Users who want a whole-ensemble explanation are not served by this ADR.
- The registry snapshot is immutable, so model rollout requires a restart. At
  this project's scale that is the right trade; it would not be at higher scale.

**Follow-on tasks unblocked**

| Task | Now specified by |
|---|---|
| E3 (`load_transformers` production call site) | §4.2 step 1, §3.5 |
| E4 (train/serve equivalence test) | §4.1, §4.5 steps 2–3 |
| E5 (Pydantic schemas, model version in response) | §4.4, §4.5 step 1 |
| PRD Phase 4 (SHAP) | §3.4, §4.5 step 7 |
| PRD Phase 6 (Kafka consumer) | §4.1 — consumer reuses `InferenceService` |

---

## 6. Open questions (not blocking acceptance)

1. **Measured latency.** The < 100 ms P95 budget (D1) is assumed satisfiable
   with a synchronous TFT forward pass at batch size 1 on CPU
   (`XGBTrainer.load` already forces CPU via `resolve_device("auto",
   force_cpu=True)`). This is an assumption until Phase 5 measures it. If it
   fails, §3.4 Option C and a `no_tft` fast path are the levers — both already
   expressible in this design without restructuring it.
2. **Whether `no_tft` is the right second mode.** It is chosen because TFT is
   the component with a hard dependency on sequence state. If measurement shows
   the state store, not TFT, is the fragile part, a different mode set may be
   warranted — the artifact format supports any number of pre-registered modes.
3. **The cost-model question flagged in `IMPLEMENTATION_PLAN.md`** (`revenue_tp`
   and `cost_fn` appearing to double-count the same recovered amount) affects
   every threshold this ADR treats as frozen, including the new per-mode ones.
   It is a business decision, out of scope here, but every threshold this design
   ships inherits it.

---

## 7. Resolution — cost-model convention settled (2026-09-08, PRD Phase 9 step 9.1)

*(Resolves open question 3 in §6.)*

**Decision: adopt the `net_of_principal` convention — each recovered fraud is
counted exactly once.** A caught fraud recovers `revenue_tp`; a missed fraud
recovers nothing (there is no second charge for the same principal), so the
false-negative term is $0 rather than `cost_fn`. This is
`business_impact.CONVENTION_NET`, already implemented and reportable alongside
the PRD-literal `CONVENTION_PRD`.

**Why.** The PRD §8.1 formula as written (`gross = TP·revenue_tp`,
`fn_loss = FN·cost_fn`, with `revenue_tp=480` and `cost_fn=500`) makes the
value swing between catching and missing the *same* ~$500 fraud equal to
`revenue_tp + cost_fn ≈ $980` — nearly twice the amount actually at stake. The
recovered principal is booked once as revenue and again as an avoided loss.
That inflates the fraud side of the trade against the $5 false-positive cost,
which is why the PRD-convention cost-optimal threshold sits at ≈0.006 on the
blended-probability scale and tolerates ~15 false positives per fraud caught.

**What changes, and what does not.**
- The **accounting used for the operating-threshold search and the
  business-impact report** is `net_of_principal` from Phase 9 onward.
- Re-deriving the operating threshold on the **validation** blend under this
  convention (argmax of net annual value over the log-spaced grid) moves it
  from `0.006123` to **`0.014740`**. For orientation, the closed-form
  per-transaction break-even under `net_of_principal` is
  `cost_fp / revenue_tp = 5 / 480 ≈ 0.0104` on the calibrated-probability
  scale; the grid optimum sitting somewhat above that reflects the validation
  PR curve's shape and the annualisation, not a grid-boundary artifact.
  (`ModelEvaluator.find_optimal_threshold` with the FN term zeroed uses the
  *same* grid and objective, so it returns 0.014740 by construction — a
  recomputation, not an independent check.) On the frozen **test** split that
  point gives precision **10.14%**, recall **88.88%** — recall still above the
  80% floor, precision still far below the 30% Phase 9 target. See
  `reports/RESULTS.md` "PRD Phase 9 — 9.1".
- `models/ensemble.json`'s `threshold` is **not changed by step 9.1 alone.**
  Per the Phase 9 stop-gate rules, a step's configuration is promoted only if
  it meets the target band; 9.1 does not, so the sequence continues to 9.2 and
  the deployed threshold moves (if at all) only when a later step's model is
  frozen. `config/config.yaml:thresholds` keeps `revenue_tp: 480` / `cost_fn:
  500` as the raw inputs; the *convention* that combines them is what this
  decision fixes, and it lives here, in `business_impact.py`'s docstring, and
  in the Phase 9 tooling — not as a new config key, since the two raw numbers
  are still both needed (`revenue_tp` for the gross-benefit line under either
  convention).
- The per-mode thresholds this ADR describes as frozen (§3.3, §4.3) are
  unaffected until a Phase 9 step actually replaces the deployed blend.
