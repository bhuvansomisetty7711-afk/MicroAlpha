# MICROALPHA — QUANTITATIVE BACKTESTING SYSTEM
## Deep Architecture Analysis & Design Blueprint
### Senior Quant Engineer Audit Report

---

# SECTION 1 — GLOBAL SYSTEM ARCHITECTURE

## The Philosophy of a Realistic Backtest

The most dangerous words in quantitative finance are "my backtest shows." A backtest that does not model execution is not a backtest — it is an accounting exercise in fiction. The entire purpose of a realistic backtesting engine is to destroy your confidence in your strategy before the market does it with real money.

Your system exists along a spectrum: at one end is the naive price-series-return backtest (completely unrealistic); at the other end is a full tick-level L3 order-flow reconstruction with actual market impact modeling, queue position tracking, latency races, and adverse selection. Your architecture is pushing toward the realistic end, which is the right direction.

The codebase you have built represents significant learning and genuine engineering sophistication. The analysis below treats it seriously as an engineering artifact.

---

## Layer 0 — Philosophy of Backtesting

**What it is:** The set of assumptions you make before writing a line of code.

**Core axiom:** *You cannot observe your own market impact in a replay.* When you replay historical data, you are replaying a world in which your orders did not exist. Every fill you simulate is counterfactual. The engine's job is to estimate the most plausible counterfactual, not to compute a deterministic one.

**Derived principles from your codebase:**

- The `impact_mode='replay'` flag in `RealisticMatchingEngine` correctly acknowledges this. Historical snapshots are immutable. You log permanent impact but do not apply it forward — because you cannot know how other participants would have responded to your presence.
- The probabilistic queue modeling in both engines acknowledges that you cannot know your exact queue position.
- The `assert_invariants=True` flag in `RealisticMatchingEngine` reflects professional discipline: the engine should catch violations of accounting consistency before they contaminate results.

**The invariant of invariants:** *A backtest that cannot prove its own accounting is not usable.* This is why your four unit tests (roundtrip PnL, cancel-before-arrival race, FIFO partial fills, fee consistency) are the most important thing in the codebase.

---

## Layer 1 — Market Data Replay

**What it is:** The mechanism by which recorded market data is fed into the engine in chronological order, one event at a time.

**Your implementation:**

- `ReplayEngine` (hybrid engine) iterates `df.iterrows()` — iterating pandas rows is slow but correct for a single-threaded replay.
- `BacktestRunner.run()` iterates a list of `MarketSnapshot` objects — better because the snapshot abstraction decouples the data format from the engine logic.
- `MarketBook.from_row()` handles Databento MBP-10 flat column format (bid_px_00...bid_px_09, bid_sz_00...bid_sz_09) converting to numpy arrays.

**What it must guarantee:**
1. Strict monotonic timestamp ordering — never feed a snapshot at time T+δ before T.
2. No lookahead — the strategy must never see future book state when making decisions at time T.
3. Snapshot consistency — if a snapshot has N bid levels, all N must be populated before the engine processes it.

**Design pattern:** Event-driven simulation where the "events" are market snapshots. This is a *synchronous discrete event system* — events arrive at fixed simulation ticks, not in continuous time.

---

## Layer 2 — Order Book Reconstruction

**What it is:** The process of building a coherent L2 or L3 order book from raw market data.

**Your implementation:**

The hybrid engine uses `MarketBook` with numpy arrays for the top 10 bid/ask levels. The `update_from_snapshot` method updates these in-place. The `compute_deltas` method compares old and new book state to estimate consumption.

The `RealisticMatchingEngine` uses `MarketSnapshot` with lists of (price, size) tuples and tracks per-price deques for resting orders.

**Critical architectural insight:** Your `compute_deltas` method in the hybrid engine computes:

```python
delta_bid = old_book.bid_sizes - self.bid_sizes
delta_ask = old_book.ask_sizes - self.ask_sizes
```

This is an L2 delta — the *change* in displayed size at each price level. This is NOT the same as the volume that actually traded at that level. A level can shrink because: (a) trades consumed it, (b) resting limit orders were cancelled, or (c) the price level disappeared and a new level appeared at the same price index (level-index aliasing). Your engine does not distinguish between these cases, which is a significant source of imprecision in queue fill modeling.

**The L3 vs L2 problem:** With MBP-10 (top-of-book, 10 levels), you have L2 data. Real queue position modeling requires L3 (order-by-order) data. Your probabilistic fill model is the correct engineering response to this limitation.

---

## Layer 3 — Matching Engine

**What it is:** The core logic that decides when, how much, and at what price resting orders are filled.

**Two implementations coexist in your codebase:**

**Implementation A: `RealisticMatchingEngine` (matching_engine_realistic.py)**

This is the production-grade engine. Key mechanisms:
- FIFO per-price deques (`bid_queues`, `ask_queues`) keyed by price float
- Probabilistic passive fill via `simulate_passive()` using `p_reach` calculation
- Lognormal latency jitter via `_sample_latency_seconds()`
- Cancel-before-arrival race condition properly handled
- Maker/taker fee differentiation
- Adverse selection scoring via `_compute_adverse_factor()`

**Implementation B: `MatchingEngine` (hybrid_matching_engine_modified.py)**

Simpler engine. Key mechanisms:
- Market orders sweep L2 book mutating `sizes[i]` in-place (BUG: mutates historical snapshot)
- Passive fills use ratio-based probability: `p_fill = (delta_front + trade_component) / qp`
- No per-price deques — single global `StrategyBook` list
- No FIFO between multiple orders at same price

**The correct architecture uses Implementation A.** Implementation B has several critical issues detailed in Section 4.

---

## Layer 4 — Latency Model

**What it is:** The model of network and system latency between when the strategy decides to act and when the exchange receives and processes the action.

**Your implementation in `RealisticMatchingEngine`:**

```python
def _sample_latency_seconds(self):
    base = self.latency_ms / 1000.0
    sigma_log = self.latency_jitter_ms / max(1.0, self.latency_ms)
    mu_log = math.log(max(1e-9, base)) - 0.5 * sigma_log**2
    sampled = float(self.rng.lognormal(mean=mu_log, sigma=sigma_log))
    return max(base * 0.5, min(sampled, base * 10.0))
```

This is excellent. The lognormal distribution is the correct choice for latency modeling because:
1. Latency is always positive (lognormal bounded below at 0).
2. Latency has a long right tail (occasional microbursts, GC pauses).
3. The median is preserved by the mean correction term `- 0.5 * sigma_log**2`.

**What is missing:**
- Asymmetric uplink vs downlink latency (market data feed latency ≠ order submission latency).
- Sequencing number awareness — exchange processes orders in sequence; two orders submitted near-simultaneously may arrive in unexpected order.
- Cancel latency should be independently sampled, not reuse the same distribution as placement latency.

**The latency queue:** Both engines use a priority queue (heapq or pending list sorted by arrival time) to hold orders until their arrival timestamp. This is correct and is the standard discrete-event simulation pattern.

---

## Layer 5 — Queue Position Modeling

**What it is:** The model of where your order sits in the per-price FIFO queue at the exchange, which determines when and how much of your order gets filled.

**This is the hardest layer to get right.** Your engines attempt it with varying sophistication.

**Implementation A (`RealisticMatchingEngine`) approach:**

Upon arrival:
```
qp = max(0, observed_depth + new_ahead - expected_depletion + hidden_extra) * jitter
```
Where:
- `observed_depth` = visible size at price level at time of placement (from snapshot).
- `new_ahead` = Poisson-sampled new orders that joined the queue between placement and arrival.
- `expected_depletion` = estimated volume consumed before arrival.
- `hidden_extra` = hidden liquidity adjustment.

Queue decay during resting:
```
queue_position *= exp(-lambda * dt) + new_arrivals_ahead
```

This exponential decay model represents order arrivals ahead of you (your priority deteriorates as new orders join ahead and active orders cancel from the front).

**Implementation B (hybrid engine) approach:**

```python
qp = max(o.queue_position, 1e-6)
p_fill = min(1.0, (delta_front + trade_component) / qp)
```

Simpler but has a fundamental flaw: `p_fill` is treated as a fill probability on `o.remaining`, producing fractional fills on every tick. Real queues fill in discrete chunks, not continuous fractions.

**Correct queue model:** A resting order at price P with queue position Q gets filled when the total volume traded at price P (from the front of the queue) exceeds Q. This is deterministic given the true queue state, but we only have a probability distribution over queue state due to L2 data limitations.

---

## Layer 6 — Execution Simulation

**What it is:** The translation from fill events into actual trade records with prices, quantities, and costs.

**Your implementation has three execution paths:**

1. **Market order (aggressive):** `execute_market_order()` sweeps L2 levels with temporary + permanent impact adjustment. Returns a list of (price, qty) fills.

2. **Passive limit fill:** `simulate_passive()` allocates consumption to resting orders via FIFO with probabilistic scaling.

3. **Limit order placement/arrival:** Orders go through latency queue, then arrive with computed queue position.

**Fee model in `_apply_fill_to_order()`:**

```
Maker fee = qty * price * (fee_bps + maker_rebate_bps) / 10000 + qty * fee_per_share
Taker fee = qty * price * (fee_bps + taker_fee_bps) / 10000 + qty * fee_per_share
```

This is a correct representation of the US equity exchange fee structure (maker-taker model with ECN rebates). The `maker_rebate_bps` being negative represents a credit.

---

## Layer 7 — Risk Management

**What it is:** The layer that enforces position limits, sizing constraints, and kill switches before orders reach the exchange.

**Your `KellyConstrainedStrategy` in `Runnn_backtest.py`:**

- `max_inventory`: hard position limit (inventory kill switch)
- `kelly_shrink=0.05`: uses 5% of theoretical Kelly fraction — extremely conservative, appropriate for research
- `adv_participation_limit=0.01`: caps at 1% of ADV per trade horizon
- `max_shares_per_order=500`: per-order book-walking prevention
- `min_time_between_orders_ms=50`: order pacing
- `max_orders_per_minute=120`: rate limiting
- `loss_cooldown_ms=200`: cooldown after realized loss

**Avellaneda-Stoikov reservation price:**

```
r = mid - gamma * sigma^2 * tau * inventory
```

Your `compute_reservation_price` implements this correctly with an ML tilt bounded to a small fraction of inventory adjustment. The comment "ML tilt is strictly limited" reflects correct prioritization: inventory management dominates signal following.

**The strange backtest results diagnosed:** The output shows `Avg Actual Size: 0.0 shares` and `Size Reduction: 99.3-100%`. This means Kelly sizing is returning 0 on virtually every tick. The reason is this chain:

1. `ml_score = 0.0` (no model connected, stub returns 0)
2. `adjusted_edge ≈ 0 - latency_loss_frac < 0` (edge is always negative because latency cost exceeds zero-signal edge)
3. `if adjusted_edge <= 0 or size <= 0: return 0`
4. No orders placed

The system is working correctly — it refuses to trade when there is no edge. The "Net PnL: $620" in the first run comes from fills on already-placed orders from earlier in the run, not from ongoing trading.

---

## Layer 8 — PnL Accounting

**What it is:** The authoritative reconstruction of cash, position, realized PnL, unrealized PnL, and fees from the event log.

**`calculate_pnl_enhanced()` in `Runnn_backtest.py` is the best implementation.** It operates on the single-source-of-truth engine log, processes events chronologically, and reconstructs state without relying on any mutable runtime state from the strategy.

**Mark-to-market logic:**

```python
if pos > 0:
    mark = last_bid  # liquidate long => hit bid (correct: liquidation cost)
else:
    mark = last_ask  # liquidate short => lift ask (correct: cover cost)
```

This is the correct liquidation-price mark, not the mid-price mark. This is more conservative and more realistic.

**Conservation invariant (checked at every row):**

```
total_pnl == cash + unrealized_pnl
```

This is the fundamental double-entry accounting check. If this fails, there is a bug in the PnL calculation.

---

## Layer 9 — Strategy Layer

**What it is:** The signal generation, sizing, and order management logic.

**Two strategy implementations:**

**`KellyConstrainedStrategy` (Run_backtest_3.py) — The XGBoost version:**
- XGBoost model inference every 20 ticks (`PREDICT_EVERY=20`)
- Converts model output (probabilities over sell/hold/buy) to edge estimate
- Kelly sizing with confidence weighting
- Hard rounding: `int(size_constrained / 100) * 100` — this forces minimum lot size of 100 shares, which is very large and kills small edge trades

**`KellyConstrainedStrategy` (Runnn_backtest.py) — The signal-agnostic version:**
- ML score is external (injected via `model_infer_fn`)
- More sophisticated edge computation with latency cost deduction
- More sophisticated Kelly sizing with ADV participation cap
- Pacing controls (rate limiting, loss cooldown)
- Clean separation of strategy state from engine state

The second version is architecturally superior. The first version conflates feature computation, model inference, and strategy decision into one class.

---

## Layer 10 — Diagnostics and Validation

**Your unit test suite covers:**
1. Roundtrip PnL (end-to-end integration test)
2. Cancel-before-arrival race (latency ordering invariant)
3. FIFO partial fills (queue ordering invariant)
4. Fee consistency (accounting invariant)
5. Conservation invariant (PnL accounting invariant)

This is an excellent foundation. What is missing is covered in Section 6.

---

# SECTION 2 — DESIGN PATTERNS

## Pattern 1: Event-Driven Discrete-Time Simulation

**Problem it solves:** Market data arrives in chronological order. The engine needs a clean way to advance simulation time, process all events at each time step, and prevent future information from bleeding into past decisions.

**Your implementation:** The outer loop in `BacktestRunner.run()` iterates snapshots chronologically. At each snapshot:
1. Ingest snapshot (update book state).
2. Process pending arrivals (latency queue drain).
3. Call strategy (see current book, not future book).
4. Enqueue strategy orders into latency queue.
5. Simulate passive fills (using deltas from this snapshot).
6. Attribute fills to strategy.
7. Update inventory.

This ordering is critical and mostly correct in your codebase.

**One ordering bug:** In the hybrid engine `ReplayEngine.run()`, the strategy decision at time T uses `self.book` which has already been updated from the current row. This means the strategy sees the book state that includes the current tick's information — correct. But passive fills are also run using `deltas` computed between the current and previous book — these deltas represent what happened *during* the tick, which the strategy should not have used to decide orders. In practice for 5ms latency this is fine because the strategy's orders arrive after the current tick's passive fills anyway.

## Pattern 2: Latency Queue (Priority Queue for Future Events)

**Problem it solves:** Orders submitted at time T must not interact with the book until time T + latency. Without a latency queue, the engine would process strategy decisions instantaneously, creating severe lookahead in execution.

**Your implementation:** `heapq` in hybrid engine, list filtering in `RealisticMatchingEngine`. The heap-based approach is O(log n) vs O(n) for list filtering, but for typical backtests with <1000 resting orders this difference is negligible.

**Pattern:** Orders are timestamped with `arrival_time = submit_time + latency_sample`. On each tick, drain all orders with `arrival_time <= current_time`. This correctly models the *minimum* delay — orders cannot arrive before their latency window expires.

## Pattern 3: FIFO Per-Price Queue

**Problem it solves:** At a given price level, multiple orders compete for fills. Exchange matching is price-time priority: orders at the same price are filled in the order they arrived (FIFO). Without modeling this, resting orders are unrealistically likely to fill (everyone gets fills instantly).

**Your implementation:** `defaultdict(deque)` keyed by price. `deque.popleft()` for consumption. `deque.append()` for new arrivals. This is the correct FIFO data structure.

**Interaction with queue decay:** After a resting order arrives, its queue position decays over time (new orders arrive ahead, old orders cancel from the front). Your exponential decay + Poisson new arrivals model represents this.

## Pattern 4: Probabilistic Fill Model

**Problem it solves:** With L2 data, you cannot know exactly how much of the consumption at a price level reached your order (as opposed to other orders ahead of you). The probabilistic model converts level consumption into a fill probability distribution.

**Your `p_reach` calculation:**

```python
p_reach = 1.0 - math.exp(-expected_consumption / (queue_ahead + eps))
```

This is the CDF of an exponential distribution — it models the probability that the consumption "reaches" your position in the queue. When `expected_consumption >> queue_ahead`, `p_reach → 1.0` (almost certainly filled). When `expected_consumption << queue_ahead`, `p_reach → 0` (unlikely to be reached).

**The stochastic actual_consumed:**

```python
actual_consumed = float(self.rng.poisson(lam=max(0.0, expected_filled_volume)))
```

Using Poisson here is reasonable for modeling discrete order flow but introduces artificial variance. In production engines, this would be calibrated against historical fill rate data.

## Pattern 5: Maker/Taker Fee Differentiation

**Problem it solves:** Exchange fee structures differ significantly between passive (maker) and aggressive (taker) executions. In US equities, makers receive rebates (negative fees) while takers pay fees. Conflating these understates the advantage of passive execution.

**Your implementation correctly separates:**
- `fee_bps`: base fee on notional
- `maker_rebate_bps`: additional credit for passive fills (negative = rebate)
- `taker_fee_bps`: additional surcharge for aggressive fills
- `fee_per_share`: fixed per-share component (models SEC/FINRA fees)

## Pattern 6: Adverse Selection Scoring

**Problem it solves:** When your passive limit order fills, it may be because: (a) a random, uninformed flow hit your price, or (b) an informed participant who knows the price is about to move adversely filled against you. Case (b) is adverse selection — you fill exactly when the market is about to go against you.

**Your `_compute_adverse_factor()` implementation:** Uses recent consumption history and aggressor direction as a heuristic. This is a reasonable first-order model. Production systems would use microstructure features like order flow imbalance, realized volatility, and trade-sign autocorrelation.

## Pattern 7: Single Source of Truth for PnL

**Problem it solves:** If multiple components (strategy, engine, PnL calculator) all maintain their own cash/position state, they will diverge over time due to attribution errors, race conditions, and double-counting.

**Your architecture correctly designates `engine.logs` as the single source of truth** and rebuilds all PnL downstream from those logs via `calculate_pnl_enhanced()`. The strategy's `current_inventory` is maintained separately for real-time decision making but is NOT used for final PnL computation.

## Pattern 8: Inventory-Aware Reservation Pricing (Avellaneda-Stoikov)

**Problem it solves:** A market maker with a large long position must lower their bid/ask to encourage selling and reduce inventory risk. Without inventory-awareness, the strategy mechanically accumulates positions without regard to their hedging cost.

**Formula:**
```
r = S - gamma * sigma^2 * tau * inventory
```

Where:
- `S` = current mid price
- `gamma` = risk aversion coefficient
- `sigma` = short-term volatility
- `tau` = time horizon to close position
- `inventory` = current net position

When inventory is long (+), reservation price is below mid (discourages further buying). When short (-), reservation price is above mid.

---

# SECTION 3 — SYSTEM INVARIANTS

The following invariants must ALWAYS hold. A violation means there is a bug in the engine.

## Timing Invariants

1. All events must be processed in monotonically non-decreasing timestamp order.
2. A strategy order cannot arrive at the exchange before it was submitted.
3. `order.arrival_time >= order.placed_at` for all orders.
4. `cancel.arrival_time >= cancel.submitted_at` for all cancels.
5. Passive fills can only occur at or after the snapshot timestamp that caused them.
6. The strategy's `on_tick()` is called with the book state at time T, not T+1.
7. Model inference uses only features derived from data at or before time T.
8. No fill can occur at a timestamp earlier than the order's arrival_time.
9. Delta computation must use the PREVIOUS snapshot, not the current one, as the baseline.
10. Queue position is assigned at arrival time, not at placement time.

## Queue and Fill Invariants

11. FIFO must be preserved within each price level — no order at a price level skips ahead of earlier-arriving orders.
12. An order cannot fill more than its remaining quantity.
13. `order.remaining >= 0` at all times.
14. An order in status 'cancelled' must not appear in any fill event.
15. A filled order must not appear in any subsequent cancel event (or cancel must be NOOP).
16. Queue position must be non-negative.
17. Fills allocated to a level cannot exceed the delta consumed at that level.
18. A passive fill only occurs if the aggressor is on the opposite side (buy aggressor fills passive sells).
19. Market order fills must consume book levels from best price outward (price-time priority).
20. Partial fills leave the order in 'resting' status with reduced remaining quantity.

## PnL and Accounting Invariants

21. `total_pnl == cash + unrealized_pnl` at every point in time.
22. `cash(T) == initial_cash - sum(buy_fills * price) + sum(sell_fills * price) - sum(fees)`
23. `position(T) == sum(buy_fill_qty) - sum(sell_fill_qty)`
24. `realized_pnl(T) == sum of (fill_price - avg_cost) * close_qty` for all closing fills.
25. `unrealized_pnl(T) == (mark_price - avg_cost) * position` when position != 0.
26. Cumulative fees must be monotonically non-decreasing (fees can be negative due to rebates, so this applies only when rebates are disabled).
27. When position == 0 and no open orders, `total_pnl == realized_pnl + initial_cash - cum_fees`.
28. Average cost is undefined (zero) when position == 0.
29. Average cost must be updated atomically with position — no partial updates.
30. Fee computation must use the same price as the fill event, not mid price.
31. Maker fills get maker rebate; taker fills get taker surcharge. Never mix.
32. `cum_fees(T)` from PnL builder must equal sum of `fee` fields in all fill events attributed to this strategy.
33. A round-trip (buy then sell same qty at same price) must produce `realized_pnl == -fees_paid`.
34. Inventory flipping (long → short) must correctly realize PnL on the closing portion and open a new short position.

## Impact and Book Interaction Invariants

35. In `impact_mode='replay'`, historical snapshot prices must not be mutated by strategy orders.
36. Permanent impact is logged but cannot retroactively change past snapshots.
37. Market impact must be non-negative for buys (cost goes up) and non-negative for sells (proceeds go down).
38. A strategy buying qty shares cannot fill at prices better than the best ask at time of arrival.
39. The engine cannot give better-than-market fills (no price improvement beyond the quoted best).
40. Hidden liquidity, if modeled, must add to effective queue depth, not reduce it.

## Latency and Race Condition Invariants

41. If a cancel arrives before the order arrives, the order must be cancelled (never rests).
42. If a cancel arrives after a partial fill, only the remaining quantity is cancelled.
43. If an order fills completely before the cancel arrives, the cancel is a NOOP (not an error).
44. Two orders submitted simultaneously (same timestamp) must be ordered deterministically (e.g., by order ID).
45. Latency samples must be i.i.d. — one order's latency must not affect another's.
46. The latency clock uses simulation time, not wall-clock time.

## Strategy and Risk Invariants

47. Strategy position (runtime tracking) must be consistent with engine-computed position from fills.
48. Kelly fraction must be positive and ≤ 1.0.
49. Order size must be positive (no zero-quantity orders).
50. Order size must not exceed `max_inventory - abs(current_inventory)`.
51. Edge must exceed transaction costs before a trade is placed.
52. No order is placed while in loss cooldown period.
53. Order rate must not exceed `max_orders_per_minute`.
54. Minimum time between consecutive orders must be respected.
55. ML score must be in [-1, 1] after normalization.

## Data and Reconstruction Invariants

56. Snapshots must be sorted by timestamp before replay begins.
57. Bid prices must be in descending order (best bid first).
58. Ask prices must be in ascending order (best ask first).
59. `best_bid < best_ask` (no crossed book) — if crossed, reject snapshot or treat as data error.
60. All price levels must have positive size (zero-size levels should be removed).
61. Delta computation: `delta = old_size - new_size` (positive delta = consumption).
62. Negative delta (size increased at a level) means new orders arrived, not consumption — do not trigger fills.
63. When a price level disappears, treat all remaining size at that level as consumed (pessimistic assumption).
64. Timestamp must be strictly positive.
65. `last_trade_aggressor` must be in {-1, 0, +1}.

## Engine State Invariants

66. `order_map` must contain all orders ever created (pending, resting, filled, cancelled).
67. A resting order must be in exactly one per-price deque.
68. A pending order must be in `pending_orders` and not in any price deque.
69. A filled/cancelled order must not be in any price deque or `pending_orders`.
70. Total fills allocated to a level ≤ total consumption at that level.
71. Sum of `remaining` quantities for all resting orders ≤ total visible depth at their levels.

---

# SECTION 4 — CODEBASE REVIEW

## 4.1 `matching_engine_realistic.py` — RealisticMatchingEngine

**Architecture:** Single class combining book state management, latency queue, per-price FIFO queues, passive fill simulation, aggressive execution, fee computation, and logging. The design is monolithic but well-organized.

### Good Decisions

**1. Lognormal latency jitter** — Technically correct distribution for network latency. The `mu_log` correction ensures the median matches `base`, not the mean.

**2. FIFO deques per price level** — `defaultdict(deque)` is the right data structure. FIFO is preserved by `appendleft` being prohibited and fills consuming from `popleft`.

**3. Cancel-before-arrival race** — `_process_pending_cancels()` checks order status correctly: if still 'pending', cancel before arrival; if 'resting', remove from queue. This matches real exchange behavior.

**4. `_check_invariants()`** — Running invariant checks during simulation catches bugs early. The check for `remaining < -1e-8` with float tolerance is correct.

**5. `impact_mode='replay'`** — Correctly acknowledges that in replay mode, historical book is immutable. Many backtesting engines silently mutate book data and then wonder why their results are unrealistic.

**6. Adverse selection scoring** — Even the heuristic implementation is valuable because it forces the engineer to think about *when* passive fills are likely adverse.

**7. Single log stream** — `self.logs` as the single event stream makes PnL reconstruction deterministic.

### Weaknesses and Issues

**Issue 1: Float price as dict key (Critical for production)**

```python
self.bid_queues = defaultdict(deque)  # keyed by float
```

Float keys for dictionaries will fail with floating-point precision issues. If the book shows 100.00000001 due to a parsing artifact and your order was placed at 100.0, the FIFO queue lookup will fail silently.

*Fix:* Use `round(price / tick_size)` as integer key, or use `Decimal` with explicit tick rounding.

**Issue 2: Pending orders removal is O(n)**

```python
self.pending_orders = [po for po in self.pending_orders if po.order_id != order_id]
```

This is a list comprehension over all pending orders on every cancel. For high-frequency strategies with many simultaneous pending orders, this becomes O(n²). Use a `dict` keyed by order_id or a `set` of cancelled IDs for O(1) removal.

**Issue 3: Queue decay uses shared `arrival_rate` for all price levels**

```python
arrival_rate = self._estimate_order_arrival_rate()
for qdict in (self.bid_queues, self.ask_queues):
    for price, dq in qdict.items():
        new_ahead = float(self.rng.poisson(lam=max(0.0, arrival_rate * dt)))
```

The same arrival rate is applied to all price levels simultaneously. But the bid side and ask side have different arrival rates, and rates at different price levels differ dramatically (top-of-book is much more active than deep levels). This over-deteriorates deep-level queue positions.

*Fix:* Estimate arrival rate per side per level using historical delta data.

**Issue 4: `_estimate_order_arrival_rate()` uses consumption as arrival proxy**

```python
lam = max(0.5, min(200.0, avg_consumed / 10.0))
```

This uses volume consumed as a proxy for order arrival rate. But consumption and order arrival are different phenomena — a level can have high arrival rate with low consumption (lots of orders adding and cancelling) or low arrival rate with high consumption (a single large sweep). The proxy is weak.

**Issue 5: `execute_market_order()` does not remove consumed levels from book**

Market order execution sweeps levels:
```python
levels = list(self.book.asks) if side == 'buy' else list(self.book.bids)
for price, available in levels:
    take = min(available, remaining)
    fills.append((exec_price, take))
    remaining -= take
```

But it never updates `self.book.bids` or `self.book.asks`. If two market orders are simulated in the same tick, the second one still sees the full book depth. In replay mode, this is explicitly accepted (book is immutable), but it should be documented more clearly as a known limitation.

**Issue 6: Adverse factor always multiplies by 1.2 for any aggression**

```python
if aggressor_side == 'buy':
    factor *= 1.2
elif aggressor_side == 'sell':
    factor *= 1.2
```

Both cases multiply by 1.2 — there is no differentiation based on direction. The factor should be higher when aggressor direction aligns with your short-term loss direction (e.g., buy aggressor when you're resting on the ask = you just sold to someone who thinks it's going up).

**Issue 7: `_remove_from_resting()` uses O(n) linear scan**

```python
for idx, elt in enumerate(dq):
    if elt.order_id == o.order_id:
        dq.remove(elt)
        break
```

`deque.remove()` is O(n). For a deque with many resting orders, this is slow. For typical backtest sizes it's acceptable, but worth noting.

**Issue 8: Hidden liquidity model is too simple**

```python
hidden_extra = base_qp * self.hidden_liquidity_frac * self.rng.uniform(0.0, 1.0)
```

Hidden liquidity (iceberg orders) at a price level affects fill probability differently than visible liquidity. It's not simply a multiplier on queue position. Hidden orders fill passively but do not show in the book depth. A more realistic model would randomly reveal hidden size as the queue is consumed.

---

## 4.2 `hybrid_matching_engine_modified.py` — ReplayEngine / MatchingEngine

**Architecture:** Simpler, single-file engine. Strategy integrated directly. Less separation of concerns.

### Good Decisions

**1. L2 delta computation** — `compute_deltas` comparing `old_book` to current book for bid and ask sides separately is the right approach.

**2. `StrategyBook` persistence** — The comment "No amnesia bug (strategy orders persist across snapshots)" correctly identifies a common failure mode in naive replay engines where resting orders are wiped on each tick.

**3. Heapq latency queue** — Using `heapq` for the latency queue is O(log n) and correct.

### Critical Bugs and Weaknesses

**Bug 1: `execute_market` mutates the historical snapshot (Critical)**

```python
def execute_market(self, book: MarketBook, side: str, qty: float):
    if side == "buy":
        prices, sizes = book.ask_prices, book.ask_sizes
    ...
    sizes[i] -= take  # MUTATES NUMPY ARRAY IN PLACE
```

`book.ask_sizes` is the numpy array from the historical snapshot. Mutating it means:
1. The same snapshot is now corrupted for any subsequent use.
2. Passive fill simulation that runs after market execution will see a depleted book, double-counting consumption.
3. `prev_book = MarketBook.from_row(row)` at the end of the loop reconstructs from the raw `row` dict, so the corruption doesn't persist to the next tick. But during the same tick, the delta computation will be wrong.

*Fix:* Work on a copy: `sizes = sizes.copy()`, or use the `RealisticMatchingEngine` pattern where book is immutable in replay mode.

**Bug 2: `prev_book` is reconstructed from `row`, not from `self.book`**

```python
prev_book = MarketBook.from_row(row)
```

This means `prev_book` at tick T is identical to `self.book` at tick T (not tick T-1). The delta `compute_deltas(prev_book)` is computed as `current_book - current_book = 0` on every tick. No deltas are ever non-zero because `prev_book` tracks the current snapshot.

*Fix:* `prev_book` should be updated at the END of the loop with the current snapshot, and the delta should be computed at the BEGINNING of the next tick between the new snapshot and the saved `prev_book`.

The correct loop structure:
```python
prev_book = initial_book
for each tick:
    self.book.update_from_snapshot(row)  # new book
    deltas = self.book.compute_deltas(prev_book)  # compare to last tick
    # ... process ...
    prev_book = copy(self.book)  # save for next tick
```

**Bug 3: Probabilistic fill generates fractional fills on every tick**

```python
p_fill = min(1.0, (delta_front + trade_component) / qp)
fill_qty = p_fill * o.remaining
o.remaining -= fill_qty
```

This generates small fractional fills on nearly every tick, even when no actual consumption occurred. In the real world, fills are discrete events. The result is that passive orders appear to always be slowly filling, which overstates fill rates dramatically.

Additionally, `fill_qty = p_fill * o.remaining` creates a paradox: as `o.remaining` decreases, `fill_qty` also decreases (it's a fraction of remaining). An order never fully fills in this model — it asymptotically approaches zero remaining. This is mathematically incorrect.

**Bug 4: Level disappearance treated as full fill**

```python
try:
    idx = np.where(levels == o.price)[0][0]
except:
    # Level disappeared → assume fully filled
    fills.append((o.order_id, o.remaining, o.price))
    o.status = "filled"
```

When a price level disappears from the book, the engine assumes the strategy's order was fully filled. But levels disappear for many reasons: mass cancellation, price movement (best bid shifted down), or the level moved in price. Assuming a fill on level disappearance is a major source of fill rate overestimation.

**Bug 5: No FIFO between multiple strategy orders at the same price**

The `StrategyBook` is a flat list. `simulate_passive()` iterates all active orders. If two strategy orders are resting at the same price, they are both independently given a fill probability without respecting their relative priority.

**Bug 6: Queue position semantics are unclear**

```python
o.queue_position -= (delta_front + trade_component)
```

`queue_position` is initialized to the visible depth at the price level at arrival time. Then it is decremented by consumption. But the interpretation is ambiguous: is it "shares ahead of me in queue" or "my priority rank"? When `queue_position <= 0`, the order is marked filled — but this implies your queue position went negative before filling, which can happen erroneously if consumption exceeds the original queue depth without actually filling your order.

**Bug 7: Strategy action processing in run loop has wrong ordering**

```python
# 2. Strategy decision
action = self.strategy.on_tick(self.book, self.strategy_book.active_orders())
# 3. Activate latency orders
ready = self.latency.pop_ready(t)
```

The strategy sees the current book and makes decisions. Then the engine activates orders from the latency queue. But the latency queue activation should happen BEFORE the strategy sees the book, because those orders represent actions the strategy took in the past that are now arriving. The correct order is:

1. Update book
2. Activate pending latency orders (these are from past decisions arriving now)
3. Simulate passive fills (from current tick's market activity)
4. Call strategy (strategy sees book after current market activity and after its past orders have arrived)
5. Enqueue new strategy decisions into latency queue

---

## 4.3 `Runnn_backtest.py` — BacktestRunner + KellyConstrainedStrategy

### Good Decisions

**1. Inventory as single authoritative path** — `_update_strategy_inventory()` is the single place strategy inventory is mutated in the run loop. This prevents double-counting.

**2. Fill attribution** — `_attribute_fill_to_strategy()` uses order_id as primary key and owner string as fallback, with an "ambiguous" log event for mismatches. This is correct defensive programming.

**3. Recent mid returns buffer** — Maintaining a rolling `deque(maxlen=1000)` of mid returns for volatility estimation is the right pattern.

**4. Conservation invariant at every row** — The `assert_invariants=True` in `calculate_pnl_enhanced` checks `total_pnl == cash + unrealized_pnl` at every event. This catches bugs during development.

**5. Replay mode flag** — `strategy.replay_mode = True` prevents the strategy from mutating its own cash in `on_fill()`. Clean separation of strategy state and PnL computation.

### Weaknesses

**Issue 1: Delta computation only uses `last_trade_aggressor` and `last_trade_size`**

```python
if snap.last_trade_aggressor == 1 and snap.last_trade_size:
    deltas['buy'].append((float(top_ask_price), float(snap.last_trade_size)))
```

This assigns ALL consumption at a tick to the top-of-book price level. But a trade that walks the book (sweeps multiple levels) would have `last_trade_size` that exceeds the size at the top level, and consumption at deeper levels is ignored. More importantly, if the snapshot represents multiple trades (aggregated over a time window), the delta computation misses all but the last trade.

*Fix:* Use actual L2 level-by-level comparison between consecutive snapshots for all levels, not just the last trade.

**Issue 2: Kelly sizing produces 0 with neutral ML signal**

With `ml_score = 0.0`, edge = 0 (before latency cost deduction), adjusted_edge < 0 (after deduction), `size = 0`. The strategy never trades unless the ML score exceeds a threshold. This is correct behavior, but the strange backtest results make more sense now: the stub model returning 0 means the strategy correctly abstains.

**Issue 3: Fee audit in `_final_audit()` will fail silently**

```python
if abs(cum_fees_from_df - strategy_fees_from_log) > 1e-6 and abs(...) > 0.05:
    raise AssertionError(...)
```

The 5% tolerance on fee mismatch is too generous for a production audit. In a system with many fills, a systematic 4% fee discrepancy could represent significant money. Use 0.1% tolerance or absolute threshold.

---

## 4.4 `Run_backtest_3.py` — Original Strategy + Hybrid Engine

### Good Decisions

**1. PREDICT_EVERY = 20** — Batching model inference to every 20 ticks is a practical optimization. XGBoost inference is fast but not free.

**2. Transaction cost explicit in strategy parameters** — `transaction_cost_bps=10` as a named parameter forces the developer to think about costs upfront.

**3. Health checks in output** — The diagnostic printout with inventory, cost efficiency, and edge strength checks is excellent for rapid strategy evaluation.

### Critical Issues

**Issue 1: PnL computation uses tuple-indexed DataFrame**

```python
for _, row in df[df[1] == "limit_placed"].iterrows():
    order_side[row[2]] = row[3]
```

The logs from `ReplayEngine.run()` are tuples `(t, event, ...)`, stored in a DataFrame with integer column indices. This is fragile — adding or reordering log fields breaks PnL computation silently. The `Runnn_backtest.py` version uses dict-based structured logs, which is far superior.

**Issue 2: Hard lot size rounding**

```python
final_size = int(size_constrained / 100) * 100
```

Rounding to the nearest 100 shares is too coarse. For a strategy with 2 bps of edge and Kelly fraction 0.25, the theoretical position is small. Rounding to 100 shares means the minimum trade is large relative to edge, leading to either very large trades or no trades at all.

**Issue 3: No cancel/replace logic**

Orders are placed but never cancelled or replaced. In a real market-making strategy, quotes are continuously updated (replace bid/ask as mid moves). Without this, stale orders accumulate in the book at prices that are no longer competitive, creating adverse selection risk.

---

# SECTION 5 — CONNECTING THE DOTS

## The Unified Data Flow

```
Raw Parquet/Feed (Databento MBP-10)
        ↓
    Data Layer (timestamp normalization, column validation)
        ↓
    MarketSnapshot construction (structured, typed)
        ↓
    BacktestRunner.run() main loop
        ↓  (each tick)
    ┌─────────────────────────────────────────┐
    │  1. engine.ingest_snapshot(snap)        │
    │     - Update book state                 │
    │     - Queue decay on resting orders     │
    │     - _process_pending_arrivals()       │
    │     - _process_pending_cancels()        │
    │                                         │
    │  2. model_infer_fn(features) → score   │
    │                                         │
    │  3. strategy.on_tick(engine, snap)      │
    │     - Compute edge, variance, size      │
    │     - Compute reservation price         │
    │     - Return action list                │
    │                                         │
    │  4. Translate actions to engine calls   │
    │     - place_limit_order → latency queue │
    │     - cancel_order → cancel queue       │
    │     - execute_market_order → immediate  │
    │                                         │
    │  5. _compute_deltas(snap) → deltas dict │
    │                                         │
    │  6. engine.simulate_passive(deltas)     │
    │     - Per-price FIFO allocation         │
    │     - p_reach calculation               │
    │     - Fill events logged                │
    │                                         │
    │  7. Attribute fills to strategy         │
    │     - strategy.on_fill(ev)              │
    │     - _update_strategy_inventory(ev)    │
    │                                         │
    │  8. Update mid returns buffer           │
    └─────────────────────────────────────────┘
        ↓
    calculate_pnl_enhanced(engine.logs)
        ↓  (single pass over structured log)
    PnL DataFrame: cash, position, realized, unrealized, fees
        ↓
    _final_audit(): conservation + fee consistency
        ↓
    Output: performance metrics, diagnostics, plots
```

## How the Books Connect

The **BACKTESTING_BOOK** and **BOOK_1** (Avellaneda-Stoikov, microstructure theory) provide the theoretical foundation that your codebases attempt to implement:

- **Reservation price** = the price at which you are indifferent to buying vs selling, given current inventory and volatility. Your `compute_reservation_price()` implements equation `r = S - gamma*sigma²*tau*x`.
- **Kelly criterion** = the fraction of capital to risk given a known edge and variance. Your `compute_kelly_size()` implements `f* = mu/sigma²` with a heavy shrinkage factor.
- **Queue position** = the fundamental source of microstructure alpha for passive market makers. Your probabilistic model approximates this given L2 data.
- **Adverse selection** = the selection bias in passive fill timing. Your `_compute_adverse_factor()` is a heuristic proxy.

The codebases are applying the textbook theory correctly at a high level, but with implementation shortcuts that are appropriate for a research prototype and should be acknowledged before treating results as production-ready.

---

# SECTION 6 — BACKTESTING REALISM CHECKLIST

## Data Quality

- [ ] Timestamps are in nanoseconds or microseconds, not milliseconds (tick data has sub-ms precision)
- [ ] Timestamps are monotonically increasing — check for clock jumps, rollbacks
- [ ] Book is never crossed (best_bid < best_ask) — filter or flag crossed-book snapshots
- [ ] All 10 bid/ask levels are populated — handle sparse books (zero-size levels)
- [ ] `last_trade_aggressor` is populated and correct (not always available in all feeds)
- [ ] Multiple trades between snapshots are captured (aggregated tick data loses intra-snapshot trades)
- [ ] Trading halts, auction periods, and open/close periods are correctly identified and excluded
- [ ] Splits, dividends, and corporate actions are adjusted
- [ ] Data gaps (missing ticks) are handled — do not interpolate book state

## Latency Model

- [ ] Latency applied to all orders — placement, cancellation, and replace
- [ ] Latency is stochastic, not fixed — use lognormal or empirical distribution
- [ ] Uplink latency (your order to exchange) ≠ downlink latency (exchange to you)
- [ ] Cancel latency independently sampled from placement latency
- [ ] Latency model calibrated against actual co-location latency data
- [ ] Burst latency events (network microbursts, GC pauses) modeled via heavy tail
- [ ] Cancel-before-fill race condition correctly implemented
- [ ] Cancel-before-arrival race condition correctly implemented

## Queue Position

- [ ] Initial queue position uses visible depth at price level at time of submission
- [ ] Latency between submission and arrival allows additional orders to join ahead
- [ ] Queue position decays over time as new orders arrive and front-of-queue cancels
- [ ] Hidden liquidity estimated and included in effective queue depth
- [ ] Queue position at fill captured for adverse selection analysis
- [ ] Multiple orders at same price respect FIFO ordering
- [ ] Queue position never goes negative (clamp at zero)
- [ ] Fill probability calibrated against historical fill-rate data at similar queue positions

## Fill Simulation

- [ ] Passive fills only occur when consumption at the order's price level is detected
- [ ] Level disappearance does not automatically trigger fill (may be cancellation, not consumption)
- [ ] Fills are discrete, not fractional (stochastic fill-or-no-fill per consumption event)
- [ ] Fills cannot exceed level consumption
- [ ] Fills cannot exceed order remaining quantity
- [ ] Partial fills leave order resting with reduced remaining
- [ ] Market orders walk the book from best price outward
- [ ] Market orders can fail to fully fill if book depth is insufficient (partial execution)

## Market Impact

- [ ] Permanent impact modeled (price level shifts after your market order)
- [ ] Temporary impact modeled (spread widens during aggressive execution)
- [ ] Square-root impact law used (not linear — empirically validated)
- [ ] Impact scaled by participation rate relative to ADV
- [ ] Impact mode correctly set: replay (immutable history) vs simulated (mutable synthetic book)
- [ ] Passive limit orders have zero immediate impact (but price discovery contribution is not modeled)
- [ ] Self-trading prevention (strategy should not trade against its own resting orders)

## Transaction Costs

- [ ] Exchange fees (maker and taker separately)
- [ ] Regulatory fees (SEC transaction fee, FINRA TAF)
- [ ] Clearing fees
- [ ] Per-share fees (fixed cost component)
- [ ] Basis-point fees (variable cost component)
- [ ] Market impact cost (separate from fees)
- [ ] Spread cost (half-spread paid on aggressive orders)
- [ ] All fees included in PnL computation
- [ ] Fee structure per-venue (different exchanges have different fee structures)
- [ ] Tiered fee structure (volume discounts at higher ADV)

## Strategy Realism

- [ ] No lookahead bias — strategy cannot see future prices
- [ ] Realistic signal decay — ML predictions go stale over time
- [ ] Quote staleness — stale limit orders should be cancelled as market moves
- [ ] Inventory hard limits enforced
- [ ] Loss cooldown after adverse fills
- [ ] Pacing controls (order rate limits, minimum time between orders)
- [ ] ADV participation cap
- [ ] Cancel-replace logic when book moves (not just place-and-wait)
- [ ] Flat at close logic (avoid overnight inventory)

## PnL and Accounting

- [ ] PnL computed from single authoritative log (not from runtime state)
- [ ] Mark-to-market using liquidation price (bid for long, ask for short), not mid
- [ ] Realized PnL computed correctly on position flips (long → short)
- [ ] Unrealized PnL reset to zero when position closes
- [ ] Average cost correctly maintained through partial fills and position changes
- [ ] Conservation invariant checked at every event
- [ ] Fee audit — fees in PnL match fees in fill events

## Diagnostics and Validation

- [ ] Fill rate (fraction of resting orders that fill) tracked and validated against expected
- [ ] Adverse selection rate (fraction of fills that precede unfavorable moves) tracked
- [ ] Queue position at fill distribution tracked
- [ ] Slippage vs benchmark (VWAP, arrival price) computed
- [ ] Turnover computed (2-sided trading volume / capital)
- [ ] Sharpe ratio computed on daily PnL, not intraday (intraday Sharpe is inflated)
- [ ] Maximum drawdown computed
- [ ] Kelly ratio validation (realized Sharpe² should approximate optimal Kelly leverage)
- [ ] Walk-forward validation — no strategy parameters tuned on test data

---

# SECTION 7 — FAILURE MODES

## FM-1: Lookahead Bias (Most Dangerous)

**What it is:** The strategy uses information that was not available at decision time.

**How it happens in your system:**
- If `update_from_snapshot` is called BEFORE `on_tick`, the strategy correctly sees only current state. But if the snapshot update includes trade data (last_trade_price, last_trade_size) that represents the CURRENT tick's trades, the strategy effectively sees the market activity it's trying to profit from BEFORE deciding.
- Feature computation using future returns as labels (in model training, not replay).
- `PREDICT_EVERY = 20` skips ticks, but the features computed at tick T might use `self.fb` which was updated at tick T without waiting for model inference.

**Your system's risk:** Low in replay loop, but the delta computation uses `last_trade_aggressor` from the CURRENT snapshot, which represents a trade that happened DURING the current snapshot period. The strategy's response to this trade (via passive fill simulation) is correct, but if the strategy were to use this trade information in its decision at tick T, that information technically belongs to the interval [T-1, T] and may be slightly stale or perfectly contemporaneous depending on data vendor encoding.

## FM-2: Infinite Liquidity Assumption

**What it is:** Assuming your order fills instantly and completely regardless of order size.

**How it happens:** A naive backtest simply assumes buy at ask, sell at bid, any quantity. Your engine avoids this for passive orders (queue model) but partially for market orders (the market order sweeps as much as available in the L2 book, which is the correct behavior, but the book represents one snapshot in time, not an order-flow series).

**Your system's risk:** Medium. Market orders correctly sweep levels but: (1) do not deplete the book for subsequent orders in the same tick, and (2) do not model the price impact that reduces subsequent available liquidity.

## FM-3: No Queue Model (Fill Rate Overestimation)

**What it is:** Assuming all resting limit orders fill at their price level regardless of queue position.

**Your system's risk:** Low for `RealisticMatchingEngine` (probabilistic queue model implemented). Medium for hybrid engine (queue model has the fractional fill bug described above, which overestimates fill rates).

**How overestimated fill rates harm results:** If you assume you always fill at the bid and sell at the ask (zero queue position), your simulated PnL will be unrealistically high. In live trading, you fill only when you're at the front of the queue, which takes time and may involve adverse selection.

## FM-4: Incorrect Permanent Impact

**What it is:** Ignoring that your own trades move prices, making subsequent executions more expensive.

**Your system's risk:** Low in replay mode (impact is logged but not applied to future snapshots). Medium in "simulated" mode (future work not yet implemented). This is the correct tradeoff — in replay mode, you cannot know how the market would have responded to your orders.

**The deeper issue:** In replay mode, you are implicitly assuming your orders are infinitesimally small relative to market flow. For small ADV participation rates (< 1%), this is acceptable. For larger trades, permanent impact can be material.

## FM-5: Survivorship Bias

**What it is:** Backtesting on instruments that exist in the current universe but ignoring instruments that were delisted, went bankrupt, or were removed during the backtest period.

**Your system's risk:** Not applicable to a single-instrument backtest (AAPL). But relevant if scaled to a multi-stock strategy — any stock selection using current membership would introduce survivorship bias.

## FM-6: Overfitting to Historical Data

**What it is:** Strategy parameters tuned on the same data used to evaluate performance.

**Your system's risk:** High for the XGBoost model (Run_backtest_3.py). The model is trained and evaluated on the same AAPL data. The backtest performance reflects in-sample fit, not out-of-sample predictive power.

**The Kelly paradox:** Kelly sizing requires accurate edge estimates. If edge estimates are from an overfitted model, Kelly will bet large on a model that has no out-of-sample edge. The result is aggressive drawdowns in live trading.

## FM-7: Transaction Cost Underestimation

**What it is:** Simulating lower costs than actually incurred.

**Common underestimations:**
- Ignoring market impact (assuming no price movement from your order)
- Using mid-price for fills instead of bid/ask
- Ignoring slippage on larger orders (walking the book)
- Missing regulatory fees (SEC, FINRA)
- Missing clearing fees
- Ignoring settlement delays (opportunity cost)

**Your system's risk:** Low for the fee model (maker/taker, per-share, regulatory are all parameterized). Medium for market impact (logged but not subtracted from fill prices in replay mode). The test results showing `Total Costs: $85-$426` on `$620 PnL` mean costs are 7-69% of gross PnL — that's already aggressive and realistic.

## FM-8: Stale Quote Risk

**What it is:** A passive limit order is placed at price P. The market moves, and P is no longer a competitive quote. The order sits unexecuted but invisible to the strategy's decision logic.

**Your system's risk:** High. Neither strategy implementation cancels and replaces quotes as the market moves. `on_tick()` places new orders but does not cancel old orders at stale prices. The result is a growing list of resting orders across many price levels.

## FM-9: Cross-Impact

**What it is:** In a multi-instrument strategy, a trade in instrument A impacts the price of instrument B.

**Your system's risk:** Not applicable (single instrument). But critical if scaled.

## FM-10: Regime Change and Non-Stationarity

**What it is:** A strategy calibrated on a historical period may fail when market conditions change (higher volatility, lower volume, change in tick size, new participants).

**Your system's risk:** Medium. The backtest covers one period of AAPL trading. Performance metrics from this period may not extrapolate to other regimes.

---

# SECTION 8 — IDEAL ARCHITECTURE

## Core Philosophy

A production-grade backtesting system is not a single script. It is a layered software system with clear interfaces between components, comprehensive testing, and explicit acknowledgment of every assumption made.

## Module Structure

```
microalpha/
├── data/
│   ├── loaders.py          # Parquet, CSV, Databento, Bloomberg adapters
│   ├── validators.py       # Schema validation, timestamp checks, crossed-book detection
│   ├── normalizers.py      # Corporate actions, splits, tick size normalization
│   └── snapshot.py         # MarketSnapshot dataclass (typed, validated)
│
├── engine/
│   ├── latency.py          # LatencyModel: lognormal jitter, asymmetric up/down
│   ├── queue_model.py      # QueuePositionModel: arrival rate, decay, fill probability
│   ├── matching.py         # MatchingEngine: FIFO, passive fills, market sweeps
│   ├── impact.py           # ImpactModel: square-root law, permanent + temporary
│   └── fees.py             # FeeModel: maker/taker, per-share, regulatory
│
├── book/
│   ├── l2_book.py          # L2Book: typed, immutable snapshots, delta computation
│   └── delta_engine.py     # DeltaEngine: level-by-level consumption extraction
│
├── strategy/
│   ├── base.py             # BaseStrategy interface: on_tick() → List[Action]
│   ├── kelly.py            # KellyConstrainedStrategy
│   ├── avellaneda.py       # Avellaneda-Stoikov market maker
│   └── actions.py          # Action dataclasses: PlaceLimit, Cancel, MarketOrder
│
├── pnl/
│   ├── builder.py          # PnLBuilder: single-pass log reconstruction
│   ├── metrics.py          # Sharpe, drawdown, turnover, fill rate, adverse selection
│   └── attribution.py      # Fill attribution: order_id → strategy → PnL
│
├── runner/
│   ├── backtest.py         # BacktestRunner: orchestrates all layers
│   └── walk_forward.py     # Walk-forward validation framework
│
├── diagnostics/
│   ├── invariants.py       # InvariantChecker: all 70 invariants as callable checks
│   ├── fill_analysis.py    # Fill rate, queue position, adverse selection
│   └── reports.py          # HTML/PDF backtest report generator
│
└── tests/
    ├── unit/               # Per-module unit tests
    ├── integration/        # Full end-to-end backtest scenarios
    └── invariants/         # Automated invariant stress tests
```

## Data Flow (Production)

```
Raw Feed
  ↓ [data.loaders + validators]
Validated MarketSnapshots (monotonic, non-crossed, fully populated)
  ↓ [engine.book]
L2Book updates + delta computation (level-by-level, not just last trade)
  ↓ [engine.latency]
Latency queue drain (arrivals processed in timestamp order)
  ↓ [strategy.on_tick]
Action list (typed PlaceLimit/Cancel/MarketOrder actions)
  ↓ [runner.backtest — action translation]
engine.place_limit / cancel / market_order → latency queue
  ↓ [engine.matching + queue_model]
FIFO fills with probabilistic queue position, p_reach, adverse scoring
  ↓ [engine.fees]
Per-fill fee computation (maker vs taker, per-share, regulatory)
  ↓ [pnl.attribution]
Fill → strategy attribution via order_id
  ↓ [pnl.builder]
PnL DataFrame (single pass over structured log)
  ↓ [diagnostics]
Performance metrics + invariant verification + report generation
```

## Testing Strategy

**Level 1 — Unit Tests (per module):**
- `LatencyModel`: check lognormal parameters, verify median = base, test clamping
- `QueuePositionModel`: verify queue monotonically decays over time, test Poisson arrivals
- `MatchingEngine`: test FIFO, test market sweep, test partial fills
- `FeeModel`: test maker/taker distinction, test per-share component
- `PnLBuilder`: test roundtrip, test inventory flip, test conservation at every row

**Level 2 — Integration Tests:**
- Cancel-before-arrival race (your test passes)
- FIFO with two orders (your test passes)
- Fee consistency roundtrip (your test passes)
- Conservation invariant (your test passes)
- Market + limit order interaction in same tick
- Inventory hard limit enforcement
- Latency queue with jitter: order 2 submitted later but arrives earlier

**Level 3 — Stress Tests:**
- 100,000 tick replay with 1,000 simultaneous resting orders
- High-frequency order placement (1 order per tick) under rate limits
- Extreme market conditions: zero liquidity, large spread, crossed book
- Latency burst scenario: spike to 100ms latency for 100ms

**Level 4 — Calibration Tests:**
- Fill rate comparison: simulated fill rate at queue position Q vs historical fill rate
- Adverse selection score calibration: score vs actual post-fill returns
- Impact model calibration: simulated impact vs empirically observed impact

**Level 5 — Validation Tests:**
- Walk-forward: train on period A, evaluate on period B (never overlap)
- Paper trading comparison: compare backtest vs paper trading on live data
- Regime robustness: evaluate across different market regimes (high/low vol, trending/mean-reverting)

## Key Architectural Principles for Ideal System

**1. Price as Integer** — Never use floats as dictionary keys. Round all prices to integer tick units at ingestion. `price_ticks = round(price / tick_size)`. This eliminates an entire class of floating-point bugs.

**2. Immutable Snapshots** — Market data snapshots are immutable once created. Strategy actions never modify historical data. This makes the replay loop side-effect-free and testable.

**3. Structured Log as Single Truth** — All engine outputs flow to a structured log. PnL, fills, fees, queue events — everything. PnL is never computed incrementally by the engine; it is reconstructed in a single deterministic pass over the log.

**4. Actions are Typed** — Strategy returns typed dataclass actions, not raw dicts. This catches interface errors at compile time (or with mypy).

**5. Invariant Checking is Always On** — The 70 invariants in Section 3 are implemented as fast assertions and run on every test execution. In production, they run on a sample of events.

**6. Walk-Forward by Default** — No strategy parameter is ever tuned on the evaluation period. The framework enforces this by maintaining separate train/eval date ranges.

**7. Explainable Assumptions** — Every simplification in the engine (e.g., using `last_trade_size` as a proxy for level consumption) is documented with a comment explaining what it is approximating and what it ignores.

---

# EXECUTIVE SUMMARY

Your system represents serious, thoughtful engineering that is far ahead of naive "buy at ask, sell at bid" backtesting. The `RealisticMatchingEngine` is the foundation of a professional system. The unit tests demonstrate real engineering discipline.

**Immediate fixes needed:**

1. Fix the `prev_book` delta computation bug in the hybrid engine — currently all deltas are zero.
2. Fix the market order mutation of historical snapshots in the hybrid engine.
3. Replace float dict keys with integer tick-unit keys.
4. Fix the fractional fill bug in the hybrid engine's passive fill model.
5. Add cancel/replace logic to the strategy — quotes go stale without it.

**The strange backtest results are not bugs** — they reflect a strategy correctly refusing to trade when `ml_score = 0.0` means no edge above latency costs. Connect a real model or use a non-zero signal and the strategy will produce more interesting (if still cautious) results.

**The path forward:**

You have built the foundation. The next steps are: (1) L3 data for true queue position reconstruction, (2) walk-forward validation framework, (3) model calibration against realized fill rates and adverse selection scores, and (4) multi-instrument framework with cross-impact.

The architecture is right. The engineering is disciplined. The theory is correctly applied. This is a research prototype ready for serious signal development.
