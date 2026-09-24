# ADR-003: Serving the trailing-window velocity features

- **Status:** Accepted
- **Date:** 2026-09-08
- **Plan task:** `docs/IMPLEMENTATION_PLAN.md` PRD Phase 9, step **9.4**
- **Extends:** [ADR-002](ADR-002-realtime-feature-state.md) §2.3, §2.5,
  §4 (Option D), §5.7 — *extends, does not supersede*: ADR-002 remains correct
  for every feature it audited. This covers a feature class that did not exist
  when it was written.
- **Gates:** PRD Phase 9 step 9.4 promotion; task **E4** (train/serve
  equivalence)

---

## 1. Context

### 1.1 What 9.4 added, and why the existing counts were not enough

PRD Phase 9 step 9.4 adds five per-`card1` features
(`src/data/feature_engineering.py`, `_add_rolling_window_features`):

| Feature | Meaning |
|---|---|
| `tx_count_10min_per_card` | transactions by this card in `[t − 600s, t)` |
| `tx_count_1h_per_card` | same, 1 hour |
| `tx_count_24h_per_card` | same, 24 hours |
| `amt_24h_mean_per_card` | mean amount over the trailing 24h |
| `amt_24h_vs_card_mean_ratio` | that mean ÷ the card's expanding mean |

These are **fixed trailing windows**. The pipeline's existing
`tx_count_per_card` is an **expanding** window running from a card's first
transaction, and the distinction is the whole point: a card with 200
transactions over six months and one with 20 in the last hour can carry the
same expanding count, and only the second is a burst.

### 1.2 Why ADR-002 §2.3's five-scalar sufficiency does not reach them

ADR-002 §2.3 is built on a sufficiency argument: every card-history feature in
the batch set is a pure function of five scalars per card —
`(n, sum_amt, sum_amt_sq, max_amt, last_dt)`, modelled by `CardAggregateState`
— so retaining them reproduces the batch values **exactly, in O(1) space per
card**. That is what made ADR-002 §4's Option A viable at all.

A trailing-window count is **not** such a function. `n` records how many
transactions a card has ever had and `last_dt` when the most recent one
happened; neither can answer "how many fell inside the last 600 seconds".
Answering that requires the individual timestamps. No fixed set of scalars
suffices, because the window boundary moves with each new transaction's `t`.

### 1.3 ADR-002 §4's Option D rejection is now half-void

ADR-002 §4 rejected Option D (recompute per request from a history store) on
two grounds:

> *"the query is unbounded (a card's full history, not a window — the
> aggregates are **expanding**, not rolling), and §2.5 needs the last 10
> transactions materialised as a ring buffer regardless"*

The **first** ground does not apply to 9.4: these aggregates *are* rolling, so
the history they need is bounded by the window span. The **second** still
holds, and is now stronger — a second bounded per-card buffer must exist
regardless, so an external history query would be a *third* mechanism for
state the process already holds. Option D stays rejected, on the surviving
half of its original reasoning.

### 1.4 Measured window occupancy

Measured on `data/raw/train_transaction.csv`, 2026-09-08 — prior transactions
inside a 24h window, per row:

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 21.3 | 5 | 38 | 359 | 654 |

36% of rows have more than 10 prior transactions in 24h, so the existing
10-slot sequence ring buffer (ADR-002 §2.5, built for the TFT) is far too
small to back these features.

---

## 2. Decision drivers

- **D1′ — Exactness.** Inherited from ADR-002 D1 and task E4's "byte-identical
  feature vectors". Non-negotiable; an approximation here would be undetectable
  train/serve skew in exactly the features added to catch bursts.
- **D2 — Freshness.** Inherited. A velocity feature computed from stale state
  is worse than no velocity feature.
- **D3 — Throughput.** Inherited: ≥500 tx/sec, <100 ms P95.
- **D6 — Bounded memory (new).** ADR-002 §2.3's constant-space-per-card
  property is precisely what this decision trades away; the replacement bound
  has to be stated, not assumed.
- **D7 — No silent feature degradation (new).** Any mechanism that can quietly
  change a feature's value under load is disqualified, however convenient.

---

## 3. Options considered

| # | Option | Verdict |
|---|---|---|
| A | Time-trimmed per-card `(dt, amount)` history in the existing store | **Chosen** |
| B | A separate, parallel window store | Rejected |
| C | Count-capped ring buffer | Rejected |
| D | Reduced feature set — drop the 24h window, keep 10min/1h | Rejected |
| E | O(1) exponential-decay approximation of the 24h mean | Rejected (for now) |
| F | Re-open ADR-002 Option D (external history query) | Rejected |

**B — separate store.** ADR-002 §2.4 rightly separated the target-encoding
counters into their own mechanism, because they differ in *trigger* (a label
feed, not a transaction), *latency tolerance* (30 days), and *failure mode*.
The trailing windows differ from the expanding aggregates in **none** of
those: same trigger, same synchronous latency requirement, same failure mode
(wrong velocity). Same trigger ⇒ same store, same lock, same idempotency
claim. Splitting them would create two objects that must be claimed, locked
and trimmed in lockstep, with a second dedupe set free to disagree with the
first.

**C — count cap.** Rejected, and this is the sharpest call in the ADR. A cap
of *C* silently rewrites "this card did 900 transactions today" as "this card
did *C* transactions today" — the feature saturates on exactly the population
it exists to isolate, and emits a plausible in-range value while doing it.
ADR-002 §4 rejected Option B because "the rapid-fire pattern the feature
exists to detect is the pattern it erases"; accepting a count cap here would
be the same error under a different name. See §4.2 for the defensive ceiling
that is *not* a feature-shaping cap.

**D — drop the 24h window.** Tempting on memory (a 1h window holds single-digit
entries typically, ~24× less state). Rejected because it costs **three of the
five features, not one**: `amt_24h_mean_per_card` and
`amt_24h_vs_card_mean_ratio` both die with it, and the ratio is the only
feature in the set expressing *recent normal vs lifetime normal* rather than
a raw count. With Phase 9 still 19.7 points short of its precision target,
discarding 60% of the new feature block to save tens of MB is a bad trade.

**E — decaying accumulator.** An EWMA with a 24h half-life *is* O(1) and
*would* preserve constant space. It is also **not the same feature**: a hard
`[t − 24h, t)` window and an exponential kernel disagree on every row by an
amount that depends on the arrival pattern — which is the signal. Training on
the window and serving the EWMA is uncharacterised skew and a straight E4
failure. Training *and* serving the EWMA is legitimate, but means re-running
9.4's feature build for constant space not currently needed. Revisit if
sustained throughput passes ~50 tx/sec and packed arrays are not enough.

---

## 4. Decision

### 4.1 State shape and location

A frozen `CardWindowState` (`src/data/feature_engineering.py`) holds an
ascending tuple of `(TransactionDT, TransactionAmt)` pairs, carried per
`card1` in the **existing** `InMemoryFeatureStateStore` — same dict key, same
`RLock`, same `_claim` idempotency guard, folded in by the same `observe()`
call. `FeatureStateStore` gains one read method, `window(card_id)`, mirroring
`snapshot(card_id)`.

**The trade, stated plainly:** ADR-002 §2.3's claim that serving needs only
five numbers per card is **no longer true of the full feature set**. Per-card
space is now O(a card's traffic in 24h). Exactness is preserved; constant
space is not. This is the cost this ADR exists to record.

### 4.2 Trim policy: time only

Entries are dropped on write while `oldest_dt < new_dt − WINDOW_24H` — the
incremental form of the batch `searchsorted(dt, dt − span, side="left")`.

**No count cap** (see §3, option C). A memory safety valve, if one is ever
needed, must take the form of a ceiling far above the observed maximum that
**increments a metric and logs** when hit — an alarm, not a silent feature
change — in the same spirit as the existing out-of-order counter.

**Memory bound.** The honest bound is not per-card depth × card count, since
each transaction sits in exactly one card's window: the total is bounded by
transactions in the last 24h across all cards. At the measured ~4 msg/sec
streaming throughput that is ~345k entries (tens of MB); at the PRD's 200
tx/sec target, ~17M entries, at which point the tuple representation should
give way to packed `float64` arrays (~280 MB). Recorded in §6.

### 4.3 Exactness conditions

1. **Read-then-write** (ADR-002 §5.2). `window()` returns state excluding the
   transaction being scored; `observe()` folds it in afterwards. The batch
   code computes `idx − left` where `idx` is the row's own position, so the
   current row is excluded by construction — serving must not append before
   reading, or every count is off by exactly one.
2. **Trim against the incoming row's `t`,** not the last write's, so the
   answer does not depend on write history.
3. **Boundary matches `side="left"`:** an entry exactly `span` seconds old is
   **inside** the window (`prior_dt >= t − span`). `TransactionDT` has heavy
   tie structure, so an off-by-one comparison here fails loudly rather than
   subtly — which is the desired behaviour.
4. **Idempotency:** the window write sits inside the existing `_claim` guard.
   A double append would self-correct after 24h but corrupt every count in
   between.

### 4.4 Backdated transactions

`CardWindowState.observe` **rejects** an append whose `dt` precedes the newest
entry, rather than inserting in place. Ascending order is what makes the
counts correct, and batch never sees out-of-order rows because it sorts
globally first. This mirrors ADR-002 §5.3's existing clamp of
`time_since_last_tx`, and the case is already counted by the store's
out-of-order metric. Insert-in-place was rejected: it turns an O(1) append
into a scan and still cannot repair counts already emitted.

### 4.5 Persistence

`save_transformers` persists `card_window_state` alongside `card_agg_state`.
This is a **second** required code change beyond the one ADR-002 §6 recorded.
Without it a restart pairs a card's full expanding history with an empty
trailing window — `tx_count_per_card` in the thousands beside
`tx_count_24h_per_card == 0` — a combination the training distribution never
contains for an active card, and therefore worse than a clean cold start.
Bounded cost: the last 24h of the training tail.

### 4.6 Cold start and TTL

The window self-evicts far faster than ADR-002 §5.7's 180-day entity TTL: a
card idle for 25 hours presents an empty window on its next transaction, which
is correct — batch counts zero there too. A card holding a large
`CardAggregateState` beside an empty window is the expected pairing for a
returning card, not an inconsistency.

### 4.7 What E4 asserts

The three `tx_count_*` features are integer-valued and are asserted
**bit-exact**. `amt_24h_mean_per_card` and `amt_24h_vs_card_mean_ratio` join
the **existing** `CARRIED_SUM_FEATURES` 1e-12 carve-out in
`tests/integration/test_train_serve_equivalence.py`: batch computes the
windowed mean as a prefix-sum difference while serving sums the carried
entries, which regroups the same additions. This is the float-associativity
class already documented there, with *smaller* error than its existing members
— the window spans ≤ ~650 terms where the carried subtotals span a card's
lifetime. The equivalence test was **not** weakened, given a new tolerance
class, or skipped.

---

## 5. Consequences

**Positive.** All five 9.4 features are served exactly. The store keeps one
key, one lock and one idempotency claim per card. Backdated and truncation
events are observable rather than silent.

**Negative, accepted.** ADR-002 §2.3's constant-space property is retired for
this feature class; serving memory now scales with 24h of traffic. A second
persistence obligation joins ADR-002 §6. The `(dt, amount)` tuple
representation needs replacing with packed arrays somewhere around 20–50
tx/sec sustained.

**Follow-on tasks.**

| Task | Where |
|---|---|
| Packed-array representation if throughput grows | `src/serving/feature_state.py` |
| Redis adapter must model the window, not just scalars | ADR-002 §5.1's deferred adapter |
| Defensive ceiling + metric, if a pathological card is ever observed | `src/serving/feature_state.py` |

---

## 6. Open questions

1. **Redis representation.** A sorted set scored by `TransactionDT` is the
   obvious mapping, with `ZREMRANGEBYSCORE` for the trim — but it makes the
   read-then-write of §4.3 a multi-command sequence needing a Lua script or a
   transaction to stay atomic. Unresolved until the multi-replica deployment
   ADR-002 §5.1 defers actually arrives.
2. **Representation crossover point.** The tuple-of-tuples costs ~120 B per
   entry against ~16 B packed. The crossover was estimated, not measured;
   worth measuring before any throughput increase rather than after.
3. **Window spans are currently hardcoded** (`WINDOW_10MIN`/`_1H`/`_24H`).
   If they ever become configurable, the trim bound must follow the longest
   configured span, not the constant.
