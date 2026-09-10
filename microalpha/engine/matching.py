"""
microalpha/engine/matching.py
==============================
Core Matching Engine: the heart of the MICROALPHA backtesting system.

This is the production-grade engine that orchestrates:
  - FIFO per-price queues (using integer tick keys — no float-key bugs)
  - Lognormal latency queue (pending orders and cancels)
  - Queue position assignment and decay
  - Probabilistic passive fill simulation
  - Aggressive (market) order execution with impact
  - Maker/taker fee computation per fill
  - Single structured event log (single source of truth)
  - 70+ invariant checks

CRITICAL ORDERING within each tick (from architecture doc Section 2, Pattern 1):
  1. Ingest snapshot → update book state, record previous snapshot
  2. Queue decay → erode resting order priorities (time has passed)
  3. Process pending arrivals → move 'pending' → 'resting' for arrived orders
  4. Process pending cancels → cancel before or after arrival
  5. Strategy sees the book (called by BacktestRunner)
  6. Strategy actions enqueued (called by BacktestRunner)
  7. simulate_passive(deltas) → FIFO passive fills from market activity
  8. BacktestRunner attributes fills to strategy

All critical bugs from the architecture document are fixed here:
  Bug 1 (hybrid): no snapshot mutation (replay mode immutable)
  Bug 2 (hybrid): prev_book correctly tracks PREVIOUS snapshot, not current
  Bug 3 (hybrid): fills are discrete Poisson events, not continuous fractions
  Bug 4 (hybrid): level disappearance does NOT auto-fill
  Bug 5 (hybrid): FIFO is enforced via per-price deques
  Bug 6 (hybrid): queue_position has clear semantics (shares ahead)
  Issue 1 (realistic): float price keys → integer tick keys
  Issue 2 (realistic): pending_orders is a dict for O(1) lookup
  Issue 6 (realistic): adverse_factor differentiates direction
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from microalpha.book.l2_book import L2Book, LevelDelta, price_to_tick, tick_to_price
from microalpha.data.snapshot import (
    FillEvent, FillType, MarketSnapshot, Side
)
from microalpha.engine.fees import FeeConfig, FeeModel
from microalpha.engine.impact import ImpactConfig, ImpactModel
from microalpha.engine.latency import LatencyConfig, LatencyModel
from microalpha.engine.queue_model import QueueModelConfig, QueuePositionModel


# ---------------------------------------------------------------------------
# Order lifecycle object stored inside the engine
# ---------------------------------------------------------------------------

@dataclass
class LimitOrder:
    """
    Internal representation of a resting or pending limit order.
    Not exposed directly to strategy code — strategy receives typed actions
    and fill events only.

    Invariants:
      - remaining >= 0 (invariant #12, #13)
      - arrival_time >= placed_at (invariant #3)
      - status transitions: PENDING → RESTING → FILLED/CANCELLED (invariant #14, #15)
    """
    order_id:                  str
    owner:                     str
    side:                      Side
    price:                     float       # float for logging / display
    price_tick:                int         # integer tick key for dict lookup
    qty:                       float
    remaining:                 float
    placed_at:                 float       # simulation time at placement
    arrival_time:              float       # simulation time the order reaches exchange
    observed_depth:            float = 0.0 # visible depth at price at placement time
    queue_position:            float = 0.0 # shares ahead of us in FIFO queue
    status:                    str   = "pending"  # pending → resting → filled/cancelled
    cancel_requested:          bool  = False
    cancel_arrival_time:       Optional[float] = None
    fill_events:               List[FillEvent] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status in ("pending", "resting")

    def is_fully_filled(self) -> bool:
        return self.remaining <= 1e-9

    def mark_filled(self) -> None:
        self.status = "filled"
        self.remaining = 0.0

    def mark_cancelled(self) -> None:
        self.status = "cancelled"


# ---------------------------------------------------------------------------
# Main Matching Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Production-grade matching engine for L2 replay backtesting.

    Combines:
      - LatencyModel (lognormal jitter, independent for placement vs cancel)
      - QueuePositionModel (initial position + decay + p_reach)
      - FeeModel (maker/taker, per-share)
      - ImpactModel (square-root law, replay-mode safe)
      - L2Book (integer tick keys, level-by-level deltas)
      - FIFO per-price deques (keyed by integer tick)
      - Structured event log as single source of truth

    Usage:
        engine = MatchingEngine(...)
        for snap in snapshots:
            prev_snap = engine.current_snapshot  # save before update
            engine.ingest_snapshot(snap)
            # ... strategy calls engine.place_limit_order(), engine.cancel_order()
            deltas = engine.compute_deltas(prev_snap)
            engine.simulate_passive(deltas)
        pnl_df = PnLBuilder().build(engine.logs)
    """

    def __init__(self,
                 latency_config:     LatencyConfig    = None,
                 queue_config:       QueueModelConfig  = None,
                 fee_config:         FeeConfig         = None,
                 impact_config:      ImpactConfig      = None,
                 tick_size:          float             = 0.01,
                 num_levels:         int               = 10,
                 rng_seed:           int               = 42,
                 assert_invariants:  bool              = True):
        """
        tick_size:         minimum price increment (used for integer price keys)
        num_levels:        max L2 depth to track per side
        rng_seed:          master seed; sub-models derive from sequential seeds
        assert_invariants: run invariant checks every tick (recommended during research)
        """
        self.tick_size         = float(tick_size)
        self.num_levels        = num_levels
        self.assert_invariants = assert_invariants

        # Sub-models (each gets a unique derived seed)
        rng = np.random.RandomState(rng_seed)
        seeds = rng.randint(0, 2**31, size=4)
        self.latency_model = LatencyModel(latency_config or LatencyConfig(), int(seeds[0]))
        self.queue_model   = QueuePositionModel(queue_config or QueueModelConfig(), int(seeds[1]))
        self.fee_model     = FeeModel(fee_config or FeeConfig())
        self.impact_model  = ImpactModel(impact_config or ImpactConfig())
        self.l2_book       = L2Book(tick_size=tick_size, num_levels=num_levels)

        # Engine state
        self.time:              Optional[float]       = None
        self.current_snapshot:  Optional[MarketSnapshot] = None
        self._prev_snapshot:    Optional[MarketSnapshot] = None

        # Order lifecycle containers
        # pending_orders: dict[order_id → LimitOrder] for O(1) lookup (fixes Issue 2)
        self._pending_orders:   Dict[str, LimitOrder] = {}
        # pending_cancels: dict[order_id → cancel_arrival_time]
        self._pending_cancels:  Dict[str, float]      = {}
        # resting_queues: dict[side → dict[price_tick(int) → deque[LimitOrder]]]
        # Integer tick keys eliminate float-dict-key bug (fixes Issue 1)
        self._resting: Dict[Side, Dict[int, deque]] = {
            Side.BUY:  defaultdict(deque),
            Side.SELL: defaultdict(deque),
        }
        # order_map: ALL orders ever created (invariant #66)
        self._order_map: Dict[str, LimitOrder] = {}
        self._order_counter: int = 0

        # Structured event log — the single source of truth (Pattern 7)
        self.logs: List[Dict[str, Any]] = []

        # Engine-level metrics
        self.metrics: Dict[str, int] = {
            "total_fills": 0,
            "passive_fills": 0,
            "aggressive_fills": 0,
            "cancels": 0,
            "rejections": 0,
            "ticks_processed": 0,
        }

    # =========================================================================
    # Snapshot ingestion (step 1 in tick ordering)
    # =========================================================================

    def ingest_snapshot(self, snap: MarketSnapshot) -> None:
        """
        Feed the next market snapshot into the engine.

        This must be called FIRST in each tick, before any strategy actions.
        It:
          1. Saves previous snapshot for delta computation
          2. Updates internal book state (L2Book with integer keys)
          3. Applies queue decay to all resting orders
          4. Processes pending arrivals (latency queue drain)
          5. Processes pending cancels

        NOTE: This method does NOT call simulate_passive(). The runner must call
        compute_deltas() then simulate_passive() AFTER strategy actions are enqueued.
        This ensures strategy orders arrive AFTER the tick's market activity.

        Invariant #1: strictly monotonic timestamp ordering enforced here.
        """
        if self.time is not None and snap.ts < self.time - 1e-12:
            self._log({
                "event": "snapshot_order_violation",
                "time": snap.ts,
                "prev_time": self.time,
                "message": "Non-monotonic timestamp — snapshot rejected",
            })
            return  # drop non-monotonic snapshot

        # 1. Save previous snapshot for delta computation
        self._prev_snapshot = self.current_snapshot

        # 2. Update book state
        self.current_snapshot = snap
        self.time = snap.ts
        self.l2_book.update(snap)

        # 3. Queue decay: erode queue positions of all resting orders
        dt = (snap.ts - self._prev_snapshot.ts) if self._prev_snapshot else 0.0
        if dt > 0:
            self._apply_queue_decay(dt)

        # 4. Process pending arrivals (latency queue drain)
        self._process_pending_arrivals()

        # 5. Process pending cancels
        self._process_pending_cancels()

        self.metrics["ticks_processed"] += 1

        if self.assert_invariants:
            self._check_invariants()

    def compute_deltas(self) -> List[LevelDelta]:
        """
        Compute level-by-level consumption deltas between current and previous snapshot.

        Returns list of LevelDelta objects. Each delta with consumed > 0 represents
        market activity that may trigger passive fills.

        Must be called AFTER ingest_snapshot() and AFTER strategy orders are
        enqueued (but orders only start affecting queues at arrival, so timing
        of placement does not affect delta computation).

        Fixes Bug 2 from the hybrid engine: we use _prev_snapshot (previous tick's
        snapshot), not the current snapshot, as the baseline.
        """
        return self.l2_book.compute_deltas(self._prev_snapshot)

    # =========================================================================
    # Strategy-facing order API
    # =========================================================================

    def place_limit_order(self,
                          owner: str,
                          side:  Side,
                          price: float,
                          qty:   float,
                          now_ts: Optional[float] = None) -> str:
        """
        Strategy-facing: place a new resting limit order.
        Order enters the latency queue and arrives after a lognormal delay.

        Returns: order_id (str)

        Invariants checked:
          #2, #3: arrival_time >= placed_at
          #49: qty > 0
        """
        if qty <= 0:
            self._log({"event": "order_rejected_zero_qty", "owner": owner,
                       "side": str(side), "price": price, "qty": qty})
            self.metrics["rejections"] += 1
            return ""

        now = now_ts if now_ts is not None else self.time
        if now is None:
            now = 0.0

        latency_s = self.latency_model.sample_placement()
        arrival_time = now + latency_s

        # Get visible depth at this price at placement time
        observed_depth = 0.0
        if self.current_snapshot:
            observed_depth = self.current_snapshot.visible_size_at_price(side, price)

        price_tick = price_to_tick(price, self.tick_size)
        order_id = self._next_order_id()

        order = LimitOrder(
            order_id=order_id,
            owner=owner,
            side=side,
            price=price,
            price_tick=price_tick,
            qty=qty,
            remaining=qty,
            placed_at=now,
            arrival_time=arrival_time,
            observed_depth=observed_depth,
            status="pending",
        )
        self._pending_orders[order_id] = order
        self._order_map[order_id] = order

        self._log({
            "event":          "limit_placed_pending",
            "time":           now,
            "order_id":       order_id,
            "owner":          owner,
            "side":           str(side.value),
            "price":          price,
            "qty":            qty,
            "observed_depth": observed_depth,
            "arrival_expected": arrival_time,
            "latency_ms":     latency_s * 1000.0,
        })
        return order_id

    def cancel_order(self,
                     order_id: str,
                     now_ts:   Optional[float] = None) -> bool:
        """
        Strategy-facing: cancel a pending or resting order.

        Cancel itself has a latency delay (independently sampled — invariant #45).
        If cancel arrives before order: order never rests (cancel wins — invariant #41).
        If cancel arrives after full fill: NOOP (not an error — invariant #43).
        If cancel arrives after partial fill: remaining qty cancelled (invariant #42).

        Returns True if cancel was accepted (order exists), False if order unknown.
        """
        now = now_ts if now_ts is not None else self.time
        if now is None:
            now = 0.0

        order = self._order_map.get(order_id)
        if order is None:
            self._log({"event": "cancel_unknown", "time": now, "order_id": order_id})
            self.metrics["rejections"] += 1
            return False

        if order.status in ("filled", "cancelled"):
            # Already done: cancel is a NOOP (invariant #43)
            self._log({"event": "cancel_noop", "time": now, "order_id": order_id,
                       "status": order.status})
            return True

        # Schedule cancel with independent latency sample (invariant #45)
        cancel_latency_s = self.latency_model.sample_cancel()
        cancel_arrival = now + cancel_latency_s

        order.cancel_requested = True
        order.cancel_arrival_time = cancel_arrival
        self._pending_cancels[order_id] = cancel_arrival

        self._log({
            "event":             "cancel_requested",
            "time":              now,
            "order_id":          order_id,
            "cancel_arrival":    cancel_arrival,
            "cancel_latency_ms": cancel_latency_s * 1000.0,
        })
        return True

    def execute_market_order(self,
                             owner:  str,
                             side:   Side,
                             qty:    float,
                             now_ts: Optional[float] = None) -> List[FillEvent]:
        """
        Aggressive market sweep: consume book levels from best price outward.

        Invariants enforced:
          #35: historical snapshot NOT mutated (working on a copy of levels)
          #37: impact non-negative
          #38: buys cannot fill below best_ask
          #39: no better-than-market fills
          #19: levels consumed best-price-first

        Returns: list of FillEvent objects for each micro-fill.
        """
        now = now_ts if now_ts is not None else self.time
        if now is None:
            now = 0.0

        if not self.current_snapshot:
            return []

        # Get book levels (read-only — do NOT mutate the snapshot)
        if side == Side.BUY:
            levels_read = [(lv.price, lv.size) for lv in self.current_snapshot.asks]
        else:
            levels_read = [(lv.price, lv.size) for lv in self.current_snapshot.bids]

        # Sweep using ImpactModel (which does NOT mutate the snapshot)
        fills_raw, unfilled = self.impact_model.sweep_book(levels_read, qty, side)

        fill_events: List[FillEvent] = []
        total_filled_qty = 0.0
        total_fee = 0.0

        for exec_price, fill_qty in fills_raw:
            fee = self.fee_model.compute_fee(fill_qty, exec_price, FillType.AGGRESSIVE)
            evt = FillEvent(
                order_id="MKT",
                owner=owner,
                side=side,
                fill_type=FillType.AGGRESSIVE,
                filled_qty=fill_qty,
                price=exec_price,
                fee=fee,
                ts=now,
            )
            fill_events.append(evt)
            total_filled_qty += fill_qty
            total_fee += fee
            self.metrics["aggressive_fills"] += 1

        # Estimate permanent impact (log only — invariant #36)
        impact_estimate = None
        if fills_raw:
            ref_price = fills_raw[0][0]
            impact_estimate = self.impact_model.estimate(total_filled_qty, ref_price, side)

        self._log({
            "event":             "market_order_executed",
            "time":              now,
            "owner":             owner,
            "side":              str(side.value),
            "requested_qty":     qty,
            "filled_qty":        total_filled_qty,
            "unfilled_qty":      unfilled,
            "fills":             [{"price": e.price, "qty": e.filled_qty} for e in fill_events],
            "avg_price":         (sum(e.price * e.filled_qty for e in fill_events) / total_filled_qty
                                  if total_filled_qty > 0 else 0.0),
            "fee":               total_fee,
            "maker":             False,
            "permanent_impact":  impact_estimate.permanent_shift if impact_estimate else 0.0,
            "impact_mode":       self.impact_model.config.impact_mode,
        })
        self.metrics["total_fills"] += len(fill_events)
        return fill_events

    # =========================================================================
    # Passive fill simulation (step 7 in tick ordering)
    # =========================================================================

    def simulate_passive(self, deltas: List[LevelDelta]) -> None:
        """
        Simulate passive (maker) fills resulting from market activity in this tick.

        Called AFTER strategy orders have been enqueued and AFTER ingest_snapshot().
        For each level that was consumed (delta.consumed > 0), allocate fills to
        resting orders in FIFO priority order.

        Fixes vs hybrid engine:
          - Bug 3: fills are discrete Poisson events, not continuous fractions
          - Bug 4: level disappearance does NOT auto-fill (consumed = 0 for such levels
                   unless old_size > 0 and new_size = 0 which IS treated as consumed)
          - Bug 5: FIFO enforced via per-price deques
        """
        for delta in deltas:
            consumed = delta.consumed
            if consumed <= 0:
                continue

            # A buy aggressor consumed ASK levels → fills passive SELL orders
            # A sell aggressor consumed BID levels → fills passive BUY orders
            # delta.side is the BOOK side that was consumed:
            #   delta.side == Side.SELL (ask) → buy aggressor → our resting SELL orders may fill
            #   delta.side == Side.BUY  (bid) → sell aggressor → our resting BUY orders may fill
            resting_side = Side.SELL if delta.side == Side.SELL else Side.BUY
            aggressor_side = Side.BUY if delta.side == Side.SELL else Side.SELL

            # Record consumption for queue model (arrival rate estimation)
            self.queue_model.record_consumption(aggressor_side, consumed, self.time)

            price_tick = delta.tick_key
            dq = self._resting[resting_side].get(price_tick)
            if not dq:
                self._log({
                    "event":    "level_consumed_no_resting",
                    "time":     self.time,
                    "price":    delta.price,
                    "consumed": consumed,
                    "side":     str(delta.side.value),
                })
                continue

            # Total queue ahead (all resting orders at this level before us)
            total_queue = sum(o.remaining for o in dq)
            if total_queue <= 0:
                continue

            # FIFO allocation: iterate queue from front
            remaining_budget = consumed  # how much consumption we can allocate
            orders_to_pop: List[LimitOrder] = []

            for resting_order in list(dq):  # copy to allow modification
                if remaining_budget <= 1e-9:
                    break

                queue_ahead = resting_order.queue_position

                # p_reach: probability consumption reaches this order's position
                p_reach = self.queue_model.compute_p_reach(remaining_budget, queue_ahead)

                # Sample fill volume (discrete, not continuous fraction)
                fill_qty = self.queue_model.sample_fill_volume(
                    p_reach=p_reach,
                    consumed=remaining_budget,
                    queue_ahead=queue_ahead,
                    max_fill=resting_order.remaining,
                )

                if fill_qty > 1e-9:
                    self._apply_passive_fill(resting_order, fill_qty, delta.price,
                                             aggressor_side)
                    remaining_budget -= fill_qty

                # If fully filled, mark for removal from deque
                if resting_order.is_fully_filled():
                    orders_to_pop.append(resting_order)

            # Remove fully filled orders from deque (FIFO semantics: only front orders)
            for o in orders_to_pop:
                if dq and dq[0] is o:
                    dq.popleft()
                else:
                    # Order not at front — this shouldn't happen in correct FIFO
                    self._remove_from_resting(resting_side, o)

            # Clean empty deques
            if not dq:
                del self._resting[resting_side][price_tick]

            self._log({
                "event":          "level_consumption_allocated",
                "time":            self.time,
                "price":          delta.price,
                "consumed":       consumed,
                "allocated":      consumed - remaining_budget,
                "queue_at_level": total_queue,
                "aggressor":      str(aggressor_side.value),
                "level_appeared": delta.appeared,
                "level_vanished": delta.disappeared,
            })

        if self.assert_invariants:
            self._check_invariants()

    # =========================================================================
    # Internal: pending arrivals and cancels
    # =========================================================================

    def _process_pending_arrivals(self) -> None:
        """
        Drain orders from the pending (latency) queue whose arrival_time <= now.

        Assigns queue position probabilistically at arrival time (invariant #10).
        Moves orders from pending_orders dict to resting_queues.

        Invariant #10: queue position is assigned at arrival, not at placement.
        Invariant #2:  no fill before arrival_time.
        """
        now = self.time
        arrived_ids = [
            oid for oid, order in self._pending_orders.items()
            if order.arrival_time <= now
        ]

        for oid in arrived_ids:
            order = self._pending_orders.pop(oid)

            # If cancel arrived before order: order stays cancelled
            if order.status == "cancelled":
                continue

            # Compute probabilistic queue position at arrival (invariant #10)
            latency_s = max(1e-6, order.arrival_time - order.placed_at)
            qp = self.queue_model.compute_initial_queue_position(
                observed_depth=order.observed_depth,
                latency_s=latency_s,
                side=order.side,
            )
            order.queue_position = qp
            order.status = "resting"

            # Insert into per-price FIFO deque (integer tick key)
            self._resting[order.side][order.price_tick].append(order)

            self._log({
                "event":          "limit_arrived",
                "time":            now,
                "order_id":        order.order_id,
                "owner":           order.owner,
                "side":            str(order.side.value),
                "price":           order.price,
                "qty":             order.qty,
                "queue_position":  order.queue_position,
                "observed_depth":  order.observed_depth,
            })

    def _process_pending_cancels(self) -> None:
        """
        Process cancels whose arrival_time <= now.

        Cases:
          - Order still PENDING: cancel before arrival → order will never rest (inv #41)
          - Order RESTING: remove from queue, cancel partial fill if any (inv #42)
          - Order FILLED: NOOP (inv #43)
        """
        now = self.time
        processed_ids = []

        for oid, cancel_arr in list(self._pending_cancels.items()):
            if cancel_arr > now:
                continue

            processed_ids.append(oid)
            order = self._order_map.get(oid)
            if order is None:
                continue

            if order.status == "pending":
                # Cancel wins the race — order never rests (invariant #41)
                order.mark_cancelled()
                # Remove from pending dict (may have already arrived — but let's be safe)
                self._pending_orders.pop(oid, None)
                self._log({"event": "cancel_before_arrival", "time": now, "order_id": oid})
                self.metrics["cancels"] += 1

            elif order.status == "resting":
                # Cancel a resting order (possibly partial fill already done)
                self._remove_from_resting(order.side, order)
                order.mark_cancelled()
                self._log({
                    "event":     "cancel_resting",
                    "time":      now,
                    "order_id":  oid,
                    "filled_so_far": order.qty - order.remaining,
                })
                self.metrics["cancels"] += 1

            else:  # filled or already cancelled
                self._log({"event": "cancel_noop", "time": now, "order_id": oid,
                           "status": order.status})

        for oid in processed_ids:
            self._pending_cancels.pop(oid, None)

    # =========================================================================
    # Internal: queue decay
    # =========================================================================

    def _apply_queue_decay(self, dt: float) -> None:
        """
        Apply exponential decay + Poisson arrivals to all resting order queue positions.

        dt: time elapsed since last snapshot (seconds).

        This models the continuous erosion of queue advantage:
          - Old orders at the front cancel (exponential decay)
          - New orders arrive and join ahead (Poisson additions)
        """
        for side, price_map in self._resting.items():
            for price_tick, dq in price_map.items():
                for order in dq:
                    order.queue_position = self.queue_model.decay_queue_position(
                        order.queue_position, dt, side
                    )

    # =========================================================================
    # Internal: fill application
    # =========================================================================

    def _apply_passive_fill(self,
                            order:         LimitOrder,
                            fill_qty:      float,
                            fill_price:    float,
                            aggressor_side: Side) -> None:
        """
        Apply a passive fill to a resting order. Updates order state, computes fee,
        computes adverse selection score, logs the fill event.

        Invariants enforced:
          #12: fill_qty <= order.remaining
          #20: partial fills leave order resting with reduced remaining
          #30: fee uses fill_price not mid
          #31: maker fee (passive) applied
        """
        # Clamp to remaining (invariant #12)
        fill_qty = min(fill_qty, order.remaining)
        if fill_qty <= 1e-9:
            return

        order.remaining -= fill_qty
        if order.is_fully_filled():
            order.status = "filled"

        # Adverse selection score
        recent = self.queue_model._ask_consumption if aggressor_side == Side.BUY else self.queue_model._bid_consumption
        adverse_score = self.queue_model.compute_adverse_score(
            aggressor_side=aggressor_side,
            our_side=order.side,
            recent_consumption=recent,
        )

        # Fee computation (maker = passive fill, invariant #31)
        fee = self.fee_model.compute_fee(fill_qty, fill_price, FillType.PASSIVE)

        # Per-fill diagnostics
        mid_price = self.l2_book.mid
        fill_to_mid = (fill_price - mid_price) if mid_price is not None else 0.0
        fill_latency = self.time - order.placed_at if order.placed_at is not None else 0.0

        fill_event = FillEvent(
            order_id=order.order_id,
            owner=order.owner,
            side=order.side,
            fill_type=FillType.PASSIVE,
            filled_qty=fill_qty,
            price=fill_price,
            fee=fee,
            ts=self.time,
            adverse_score=adverse_score,
            queue_pos_at_fill=order.queue_position,
            fill_latency_s=fill_latency,
        )
        order.fill_events.append(fill_event)

        self._log({
            "event":             "limit_fill",
            "time":              self.time,
            "order_id":          order.order_id,
            "owner":             order.owner,
            "side":              str(order.side.value),
            "filled_qty":        fill_qty,
            "price":             fill_price,
            "fee":               fee,
            "maker":             True,
            "remaining":         order.remaining,
            "aggressor_side":    str(aggressor_side.value),
            "adverse_score":     adverse_score,
            "fill_to_mid":       fill_to_mid,
            "queue_pos_at_fill": order.queue_position,
            "fill_latency_s":    fill_latency,
            "best_bid":          self.l2_book.best_bid_price,
            "best_ask":          self.l2_book.best_ask_price,
        })

        self.metrics["total_fills"] += 1
        self.metrics["passive_fills"] += 1

    # =========================================================================
    # Internal utilities
    # =========================================================================

    def _remove_from_resting(self, side: Side, order: LimitOrder) -> None:
        """Remove an order from its resting deque (for cancel/fill cleanup)."""
        dq = self._resting[side].get(order.price_tick)
        if dq is None:
            return
        # O(n) deque scan — acceptable for typical queue sizes < 100 in research
        for idx, o in enumerate(dq):
            if o.order_id == order.order_id:
                del dq[idx]  # O(n) deque delete
                break
        if not dq:
            del self._resting[side][order.price_tick]

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"O{self._order_counter:09d}"

    def _log(self, entry: Dict[str, Any]) -> None:
        """Append to the authoritative event log. Ensures 'time' is present."""
        if "time" not in entry:
            entry["time"] = self.time
        self.logs.append(entry)

    # =========================================================================
    # Invariant checker
    # =========================================================================

    def _check_invariants(self) -> None:
        """
        Fast invariant checks run every tick when assert_invariants=True.

        Checks a subset of the 70+ invariants that are cheaply verifiable:
          #12, #13: remaining >= 0 for all resting orders
          #3:       arrival_time >= placed_at for all pending orders
          #16:      queue_position >= 0 for all resting orders
          #67:      resting orders are in exactly one price deque
        """
        for side, price_map in self._resting.items():
            for price_tick, dq in price_map.items():
                for o in dq:
                    if o.remaining < -1e-8:
                        raise AssertionError(
                            f"[INV #12] Order {o.order_id} has negative remaining: {o.remaining}"
                        )
                    if o.queue_position < -1e-8:
                        raise AssertionError(
                            f"[INV #16] Order {o.order_id} has negative queue_position: {o.queue_position}"
                        )
                    if o.status != "resting":
                        raise AssertionError(
                            f"[INV #67] Non-resting order {o.order_id} (status={o.status}) "
                            f"found in resting queue at price_tick {price_tick}"
                        )

        for oid, o in self._pending_orders.items():
            if o.arrival_time < o.placed_at - 1e-9:
                raise AssertionError(
                    f"[INV #3] Pending order {oid} arrival_time {o.arrival_time} "
                    f"< placed_at {o.placed_at}"
                )

    # =========================================================================
    # Public helpers for strategy / diagnostics
    # =========================================================================

    @property
    def best_bid(self) -> Optional[float]:
        return self.l2_book.best_bid_price

    @property
    def best_ask(self) -> Optional[float]:
        return self.l2_book.best_ask_price

    @property
    def mid(self) -> Optional[float]:
        return self.l2_book.mid

    @property
    def spread(self) -> Optional[float]:
        return self.l2_book.spread

    def active_resting_orders(self, owner: Optional[str] = None) -> List[LimitOrder]:
        """Return all currently resting orders, optionally filtered by owner."""
        result = []
        for side, price_map in self._resting.items():
            for dq in price_map.values():
                for o in dq:
                    if owner is None or o.owner == owner:
                        result.append(o)
        return result

    def pending_orders_for(self, owner: str) -> List[LimitOrder]:
        """Return all pending (in-flight) orders for a given owner."""
        return [o for o in self._pending_orders.values() if o.owner == owner]

    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        """Look up any order by ID (invariant #66: order_map contains all orders)."""
        return self._order_map.get(order_id)

    def dump_state(self) -> Dict[str, Any]:
        """Serializable snapshot of engine state for debugging."""
        return {
            "time":         self.time,
            "best_bid":     self.best_bid,
            "best_ask":     self.best_ask,
            "mid":          self.mid,
            "pending":      list(self._pending_orders.keys()),
            "pending_cnt":  len(self._pending_orders),
            "resting_bid":  {tick_to_price(t, self.tick_size): len(dq)
                             for t, dq in self._resting[Side.BUY].items()},
            "resting_ask":  {tick_to_price(t, self.tick_size): len(dq)
                             for t, dq in self._resting[Side.SELL].items()},
            "metrics":      dict(self.metrics),
        }
