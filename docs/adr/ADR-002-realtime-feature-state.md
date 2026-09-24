# ADR-002: Real-time card aggregates and target encodings

- **Status:** Accepted
- **Date:** 2026-08-26
- **Plan task:** `docs/IMPLEMENTATION_PLAN.md` Phase E, task **E2**
- **Gates:** PRD Phase 6 (Kafka streaming) — and, per the plan's standing
  instruction, the servability of the batch feature set as a whole
- **Consumed by:** [ADR-001](ADR-001-inference-orchestration.md) §4.5 step 2

---

## 1. Context

`docs/IMPLEMENTATION_PLAN.md` carries a standing block on Phase E:

> *"Do not begin Phase E implementation before ADR-002 is written: the
> real-time feature question determines whether the current batch feature set is
> even servable, and it may force changes back into Phase A's feature
> engineering."*

That question is answered here by auditing every feature the batch pipeline
produces and classifying it by what it needs at serving time. The audit is the
substance of this ADR; the store design follows from it.

The batch pipeline (`src/data/preprocess.py:run_pipeline`) derives features in
two groups: `_derive_causal_features` (nine stateless-or-causal groups run over
the full temporally-ordered frame) and `_apply_stateful_transforms` (four
groups with fitted state, fit on train and applied with `fit=False` to
val/test). The pipeline's own leakage contract states the causal group is
computed over the full frame *deliberately*, because those transforms "read
strictly past rows relative to each transaction, which is exactly what is
available in production". This ADR tests that claim feature by feature.

---

## 2. The audit: what each feature needs at serving time

### 2.1 Row-wise — no state, trivially servable (verified)

| Feature group | Source | Why it is row-wise |
|---|---|---|
| Temporal (`hour_of_day`, `day_of_week`, sin/cos) | `feature_engineering.py:150` | arithmetic on this row's `TransactionDT` |
| Amount (`amount_log`, `amount_cents`, `amount_is_round`, `amount_decimal_len`) | `:182` | arithmetic on this row's `TransactionAmt` |
| Email (`email_match`, `*_is_free`, `email_both_present`) | `:287` | this row's two domain strings |
| D-column summaries (`D_null_count`, `D_mean`, `D_std`, `D_max`) | `:329` | `axis=1` reductions — **across columns of one row**, not across rows |
| C-column summaries (`C_sum`, `C_max`, `C_nonzero_count`) | `:358` | `axis=1` reductions |
| Address (`addr_present`, `dist_present`) | `:490` | null checks on this row |
| Device (`browser_family`, `device_brand`, `screen_*`) | `:510` | string parsing of this row's `id_31` / `DeviceInfo` / `id_33` |
| Null-count meta (`null_count`, `null_ratio`) | `:451` | per-row null count; the denominator is frozen at fit time and persisted |

The `axis=1` point is worth stating plainly because the method names read like
aggregations: `D_mean` and `C_sum` are **not** cross-transaction aggregates.
Nothing in this table needs history.

### 2.2 Frozen fitted state — servable from `load_transformers` (verified)

| Feature | Source | Persisted by |
|---|---|---|
| V-feature PCA components | `reduce_v_features:866` | `pca.joblib` |
| `card_hash_freq` | `create_card_hash_features:382` | `card_hash_freq.joblib` |
| Frequency encodings (`P_emaildomain`, `R_emaildomain`, `id_31`, `id_33`, `DeviceInfo`) | `encode_categoricals:809` | `freq_encoders.joblib` |
| Label encodings (low-cardinality categoricals) | `encode_categoricals:809` | `label_encoders.joblib` |
| Imputation fill values | `handle_missing_values:985` | `imputer.joblib` |
| `null_ratio` denominator, target-encoding prior | — | `feature_state.joblib` |

These are frozen dictionaries and fitted sklearn objects. Both unseen-value
paths are already correct for serving: unknown high-cardinality values map to
`0.0` frequency, and unknown labels are folded onto the reserved `MISSING`
sentinel (`:850` deliberately reserves it even when training data had no NaNs,
precisely so inference cannot raise).

### 2.3 Requires online per-entity history — **the actual problem**

| Feature | Source | State needed per `card1` |
|---|---|---|
| `tx_count_per_card` | `create_card_aggregates:223` | count `n` |
| `tx_sum_per_card` | `:226` | sum `Σx` |
| `mean_amount_per_card` | `:229` | derived from `n`, `Σx` |
| `max_amount_per_card` | `:236` | running `max` |
| `std_amount_per_card` | `:240-248` | `Σx²` (plus `n`, `Σx`) |
| `amount_vs_mean_ratio` | `:251` | derived |
| `amount_zscore_per_card` | `:257` | derived |
| `interaction_hour_zscore` | `create_interactions:279` | derived (depends on the zscore above) |
| `time_since_last_tx`, `time_since_last_tx_log` | `create_velocity_features:436` | last seen `TransactionDT` |

**Finding:** every feature in this table reduces to **five scalars per `card1`** —
`(n, Σx, Σx², max_x, last_dt)`. The batch code's expanding variance is
`(Σx² − (Σx)²/n)/(n−1)` over strictly-prior rows, which is a pure function of
those accumulators. This is not an approximation: an online store holding five
numbers per card reproduces the batch values **exactly**, in constant time and
constant space per card.

### 2.4 Requires online per-entity history **plus a label feed**

`create_target_encoding` (`:564`) produces six features —
`card1/card2/addr1/P_emaildomain/R_emaildomain/device_brand` `_target_enc` —
each needing a per-entity `(cum_sum, cum_count)` of the **target**. This is
categorically different from §2.3: those counters advance only when a fraud
label is known, and `config/config.yaml` sets
`target_encoding_label_lag_days: 30` precisely because labels are not known at
transaction time (finding F6). The batch implementation
(`_lagged_expanding_totals:704`) counts only rows whose own `TransactionDT` is
at least `lag_seconds` earlier than the row being scored.

Conflating §2.3 and §2.4 into one "feature store" would be the central design
error available here. They differ in trigger (transaction vs. label
confirmation), in latency tolerance (synchronous vs. 30 days), and in failure
mode (stale §2.3 state gives wrong velocity; stale §2.4 state gives a value the
model has already been trained to expect).

### 2.5 Requires a per-card sequence window

`TFTTrainer.predict_proba` builds sequences of `max_encoder_length: 10`
transactions per card, scaled by the fitted `QuantileTransformer` restored from
the model artifact. Serving needs the last 10 **fully-transformed feature
vectors** per `card1` — roughly 10 × ~125 floats, three orders of magnitude more
per-card state than §2.3's five scalars.

### 2.6 Verdict on servability

**The current batch feature set is servable.** Nothing in it requires a value
that is unavailable at prediction time, and nothing requires an unbounded
recomputation. Phase A's feature engineering does **not** need to change. The
one thing that was genuinely unservable — `predict_proba(X, y)` requiring
ground-truth labels — was already fixed under Phase B7, and the target-encoding
state that `save_transformers` once omitted is now persisted in
`feature_state.joblib`.

Two pre-existing defects become serving-relevant and should land before Phase 6:

- **F9** (`groupby("card1")` without `dropna=False` in `create_card_aggregates`
  and `create_velocity_features`): the online store must have an explicit
  NaN-key policy, and it must match batch. Batch currently drops NaN keys here
  while target encoding keeps them (`:677`) — an inconsistency that is latent
  only because `card1` is non-null throughout IEEE-CIS.
- **At-least-once delivery** (Kafka, Phase 6) means the same transaction can be
  presented twice. Batch has no equivalent, so nothing today guards against
  double-counting into `n`/`Σx`.

---

## 3. Decision drivers

- **D1 — Exactness over approximation.** §2.3 showed exact online reproduction
  is achievable in O(1) space. Task E4 demands "byte-identical feature vectors";
  a design that only approximates the batch aggregates fails it by construction.
- **D2 — Freshness where it is load-bearing.** Velocity features exist to catch
  rapid-fire fraud on a card. Their whole value is at second-to-minute
  granularity.
- **D3 — Throughput.** ≥ 500 tx/s consumer-side, < 100 ms P95 per HTTP request
  (PRD §6.1).
- **D4 — The shipped stack has four services** (`fraud-api`, `kafka`,
  `prometheus`, `grafana`). Adding infrastructure needs justification.
- **D5 — Label latency is a modeled assumption, not an accident.** Whatever is
  built must honour the 30-day lag rather than quietly reintroducing the
  same-window leakage F6 removed.

---

## 4. Options considered

**Option A — Online per-entity state store, updated synchronously on the
scoring path.**
Exact (§2.3), fresh (D2), O(1) per transaction. Requires solving ordering,
idempotency and recovery. **Chosen for §2.3 and §2.5.**

**Option B — Micro-batch precomputation.** A scheduled job recomputes a
per-card feature table every N minutes; serving does a key lookup.
Rejected. It fails D2 at exactly the point the features earn their keep: with a
5-minute refresh, `time_since_last_tx` for a burst of card-testing transactions
reads as the gap to the last *batch boundary*, not to the previous transaction —
the rapid-fire pattern the feature exists to detect is the pattern it erases. It
also fails D1: a served value would differ from the batch value by an amount
that varies with where in the refresh cycle the transaction landed.
**Retained for §2.4**, where a staleness of minutes is irrelevant against a
30-day lag.

**Option C — Cold-start degradation only.** Serve with no history at all:
every card looks brand new.
Rejected as a steady state — it discards every card-history feature (§2.3) and
every target encoding (§2.4), and the model was not trained on a population
where every card is new. **Retained as an explicit, declared fallback mode** (ADR-001 §4.6), not
as a design.

**Option D — Recompute per request from a transaction history store.** Query
this card's prior transactions at scoring time and re-run the batch functions.
Exact and needs no incremental logic, and it is genuinely attractive because it
keeps one implementation of the computation. Rejected on two grounds: the query
is unbounded (a card's full history, not a window — the aggregates are
*expanding*, not rolling), and §2.5 needs the last 10 transactions materialised
as a ring buffer regardless, so Option A's store must exist either way. Option D
would be a second mechanism layered on top of one already required.

---

## 5. Decision

### 5.1 A `FeatureStateStore` port with two adapters

Defined in `src/serving/feature_state.py`, depended on by `InferenceService`
(ADR-001 §4.5 step 2). Two implementations:

- **`InMemoryFeatureStateStore`** — dicts, single process. This is what
  `docker compose up` ships, and it is correct there: `fraud-api` is one
  replica hosting both the HTTP app and the Kafka consumer.
- **`RedisFeatureStateStore`** — the same interface over Redis hashes, for a
  multi-replica deployment. **Not added to `docker-compose.yml`** (D4). It
  exists so the single-replica assumption is a deployment choice rather than a
  design constraint baked into the call sites.

The port is the deliverable; the Redis adapter may be deferred.

### 5.2 Read-then-write, mirroring the batch exclusion semantics exactly

Batch excludes the current row by construction: `cumsum() − df[col]` and
`cumcount()` (exclusive). The online rule is therefore:

1. **Read** the card's accumulators.
2. **Compute** features from those accumulators — the current transaction
   contributes nothing.
3. **Score.**
4. **Write** the updated accumulators
   (`n+1`, `Σx+x`, `Σx²+x²`, `max(max,x)`, `last_dt=t`).

Step 4 after step 3 is not an optimization, it is the correctness condition. A
write-then-read ordering would leak the transaction's own amount into its own
z-score.

### 5.3 Idempotency and ordering

- **Idempotency:** each update is keyed by `TransactionID`. The store keeps a
  bounded set of recently applied IDs and ignores repeats. Without this,
  Kafka's at-least-once delivery inflates `n` and `Σx` permanently — an error
  that never self-corrects because the aggregates are expanding.
- **Ordering:** the Kafka input topic is **partitioned by `card1`**. All of one
  card's transactions therefore land on one partition and one consumer, so the
  per-card accumulators are updated by a single writer in offset order. This
  makes the in-memory adapter correct under multi-partition parallelism and
  removes the need for cross-replica locking.
- **Out-of-order within a card** is not defended against beyond partition
  ordering. `time_since_last_tx` computed against a later `last_dt` would go
  negative; the store clamps to the batch's own no-history sentinel (`-1.0`)
  and increments a counter exposed to Prometheus rather than emitting a value
  the model never saw in training.

### 5.4 Cold start is the normal path, not a special case

An unseen `card1` yields `n=0`, and the batch code's own zero-history branches
then produce `tx_count_per_card=0`, `mean_amount_per_card=0.0`,
`amount_vs_mean_ratio=1.0`, `amount_zscore_per_card=0.0`,
`time_since_last_tx=-1.0`, and `<col>_target_enc = prior` (`:685`). These are
**exactly** the values the batch pipeline produces for each card's first
transaction, of which the training data contains hundreds of thousands. No
special-casing, no imputation, and no degraded flag: the model has seen this
input distribution.

The store is **seeded at startup** from `feature_state.joblib`'s
`target_enc_state`, which holds train+val entity totals (test is deliberately
excluded — `preprocess.py:72 SPLITS_THAT_UPDATE_STATE`). Serving therefore
begins from the same carried history that the test split was scored against.
The §2.3 amount accumulators are **not** currently persisted by the batch
pipeline; §6 records this as the one code change this ADR requires.

### 5.5 Target encodings: a separate, label-driven, lag-respecting path

Per-entity state is a triple: `(ready_sum, ready_count, pending[])`, where
`pending` holds `(tx_time, label)` events not yet lag-eligible.

- **Reads** (on the scoring path) use `ready_sum` / `ready_count` only, folded
  into the same `(cum_sum + w·prior)/(cum_count + w)` formula the batch code
  uses. This is a dictionary lookup; it costs nothing against D3.
- **Writes** are driven by a **label feed**, not the transaction stream. When a
  confirmed label arrives, the event is appended to `pending`.
- **Promotion** is a periodic sweep: events with
  `now − tx_time ≥ label_lag_seconds` move from `pending` into the ready
  counters. This is exactly equivalent to `_lagged_expanding_totals`'s
  `searchsorted(times, times − lag)` prefix, expressed incrementally.
- `pending` is bounded by 30 days of traffic per entity, so it does not grow
  without limit.

Because reads tolerate minutes of staleness against a 30-day lag, this path may
be served from a micro-batch refresh (Option B) rather than a synchronous
store — the one place where Option B is the right tool.

**No label feed exists in this project**, which uses a static Kaggle dataset.
The shipped behaviour is therefore: seed from `feature_state.joblib`, never
promote, and expose `target_encoding_state_age_seconds` as a Prometheus gauge so
the staleness is visible rather than assumed away. This is honest about what a
portfolio deployment can do; the promotion mechanism is specified so that a real
label feed is a plug-in, not a redesign.

### 5.6 Sequence windows for TFT

A per-`card1` ring buffer of the last `max_encoder_length` (10) transformed
feature vectors, written in the same step 4 as §5.2. On a cache miss or a short
buffer, the sequence is zero-padded with the mask set — identical to what
`SequenceBuilder` does for a card's first transactions in training. If the
buffer is unavailable entirely (store failure), `InferenceService` drops to the
pre-registered `no_tft` mode per ADR-001 §4.6.

### 5.7 Bounded growth

Both stores carry a per-entity TTL, refreshed on write, defaulting to **180
days** — chosen to exceed the ~6-month span of the IEEE-CIS training window, so
an entity's state is evicted only after a gap longer than any the model was
trained across. A card returning after eviction re-enters through §5.4's
cold-start path, which is a state the model has seen. New config block:

```yaml
serving:
  feature_state:
    backend: "memory"        # memory | redis
    entity_ttl_days: 180
    sequence_window: 10      # must equal model.tft.max_encoder_length
    dedupe_window: 100000    # recently-applied TransactionIDs retained
```

---

## 6. Consequences

**Positive**

- The plan's blocking question is answered: **Phase A feature engineering does
  not change.** Phase E implementation, and Phases 4–7 of the PRD behind it, are
  unblocked.
- Exact train/serve equivalence for the card aggregates is reachable rather than
  aspirational, because the accumulators are sufficient statistics for every one
  of those features. E4 can assert equality, not tolerance.
- The label-driven path is separated from the transaction-driven path, so the
  30-day lag F6 introduced cannot be silently undone by a store that updates
  counters when a transaction arrives.
- Cold start needs no special code, because it is already in the training
  distribution.

**Negative / accepted costs**

- **One required code change:** `save_transformers` must additionally persist
  the per-`card1` `(n, Σx, Σx², max_x, last_dt)` accumulators at the end of the
  batch run, so serving continues the card history the model was trained on
  instead of restarting every card from zero. This is new state on
  `FeatureEngineer` — `create_card_aggregates` currently computes these
  vectorised and keeps nothing. It is a genuine (if small) extension of Phase
  A's feature engineering, and the only one this audit forces.
- Kafka partitioning by `card1` is now a correctness requirement, not a
  throughput tuning knob. It must be documented at the topic definition.
- The in-memory adapter means state is lost on restart and reseeded from the
  training snapshot — acceptable for this project, and the reason the Redis
  adapter is specified even if deferred.
- Target encodings are frozen in the shipped deployment. The gauge makes this
  visible, but a long-running instance's encodings do drift from what a live
  system would have.

**Follow-on tasks**

| Task | Now specified by |
|---|---|
| E3 (persist + load target-encoding state; `load_transformers` call site) | §5.4 seeding; §6's accumulator addition |
| E4 (train/serve equivalence) | §5.2 read-then-write; §2.3 sufficiency argument |
| E6 (drift reporter) | §5.5 / §5.7 expose the gauges it should report on |
| F9 (`dropna=False` consistency) | §2.6 — promote to a Phase E prerequisite |
| PRD Phase 6 (Kafka) | §5.3 partitioning and idempotency |

---

## 7. Open questions (not blocking acceptance)

1. **`entity_ttl_days: 180`** is reasoned from the training window's span, not
   measured. Once serving telemetry exists, the right test is whether evicted-
   and-returning cards score differently from never-seen cards.
2. **The 30-day label lag itself** is documented in `config/config.yaml` as "a
   conservative, documented assumption pending an actual label-latency SLA".
   §5.5's promotion sweep makes the lag a parameter rather than a constant, so a
   real SLA changes a config value and nothing else.
3. **Per-entity write throughput under a hot card** (one `card1` receiving a
   large share of traffic) is serialized by §5.3's partitioning. Whether that
   becomes the throughput ceiling against D3 is a measurement, not a design
   question — and if it does, the accumulators are commutative, which makes a
   sharded-then-merged variant available without changing the feature semantics.
