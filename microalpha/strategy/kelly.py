"""
microalpha/strategy/kelly.py
=============================
Kelly-Constrained Strategy: conservative Kelly sizing with inventory awareness.

This is the production-grade version of the strategy:
  - ML score is EXTERNAL (injected via model_infer_fn, not baked in)
  - Edge computed conservatively: raw signal edge - latency cost
  - Kelly sizing with extreme shrinkage (5% of theoretical f*)
  - ADV participation cap
  - Per-order hard size cap
  - Pacing controls: minimum interval, per-minute cap, loss cooldown
  - Stale quote cancellation (fixes FM-8: stale quote risk)
  - Reservation price via Avellaneda-Stoikov formula (inventory coercion)

Architecture doc issue fixed:
  - Hard lot-size rounding to 100 shares removed (Issue 2 of Run_backtest_3.py)
  - No conflation of ML inference with strategy logic (separate concern)
  - Cancel/replace logic added (stale quote risk FM-8)
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from microalpha.data.snapshot import (
    CancelAction, MarketSnapshot, PlaceLimitAction, Side
)
from microalpha.strategy.base import BaseStrategy, StrategyAction

if TYPE_CHECKING:
    from microalpha.engine.matching import MatchingEngine


class KellyConstrainedStrategy(BaseStrategy):
    """
    Conservative Kelly-positioned market-making strategy with inventory awareness.

    The strategy:
      1. Receives an ML score in [-1, 1] from an external model (injected by runner)
      2. Computes edge = ml_tilt - latency_cost
      3. If edge > threshold: compute Kelly position size
      4. Compute Avellaneda-Stoikov reservation price (inventory coerces this)
      5. Place limit order at tick-aligned price
      6. Cancel stale orders that are no longer at competitive prices

    Parameters:
        capital:                  Strategy capital in dollars
        max_inventory:            Maximum net position in shares (hard limit)
        kelly_shrink:             Fraction of Kelly f* to use (0.05 = 5%)
        transaction_cost:         Per-trade cost as fraction of price
        latency_ms:               Expected latency in milliseconds
        tick_size:                Minimum price increment
        adv_daily_notional:       Estimated daily ADV in dollars
        adv_participation_limit:  Max fraction of ADV per trade horizon
        max_shares_per_order:     Hard per-order size cap
        min_edge_threshold:       Minimum fractional edge to trade
        min_time_between_orders_ms: Minimum inter-order interval
        max_orders_per_minute:    Order rate cap
        ml_confidence_halflife_ms: Exponential decay for ML score staleness
        loss_cooldown_ms:         Cooldown after detected realized loss
        stale_price_ticks:        Cancel and replace if book has moved by N ticks
        gamma:                    Avellaneda-Stoikov risk aversion coefficient
    """

    def __init__(self,
                 name:                         str   = "kelly_strategy",
                 capital:                      float = 100_000.0,
                 max_inventory:                float = 1000.0,
                 kelly_shrink:                 float = 0.05,
                 transaction_cost:             float = 0.0002,
                 latency_ms:                   float = 5.0,
                 tick_size:                    float = 0.01,
                 adv_daily_notional:           float = 1_000_000.0,
                 adv_participation_limit:      float = 0.01,
                 max_shares_per_order:         int   = 500,
                 min_edge_threshold:           float = 1e-4,
                 min_time_between_orders_ms:   float = 50.0,
                 max_orders_per_minute:        int   = 120,
                 ml_confidence_halflife_ms:    float = 500.0,
                 loss_cooldown_ms:             float = 200.0,
                 stale_price_ticks:            int   = 2,
                 gamma:                        float = 1e3,
                 rng_seed:                     int   = 42):
        super().__init__(name=name)
        self.capital                    = float(capital)
        self.max_inventory              = float(max_inventory)
        self.kelly_shrink               = float(kelly_shrink)
        self.transaction_cost           = float(transaction_cost)
        self.latency_ms                 = float(latency_ms)
        self.tick_size                  = float(tick_size)
        self.adv_daily_notional         = float(adv_daily_notional)
        self.adv_participation_limit    = float(adv_participation_limit)
        self.max_shares_per_order       = int(max_shares_per_order)
        self.min_edge_threshold         = float(min_edge_threshold)
        self.min_time_between_orders_ms = float(min_time_between_orders_ms)
        self.max_orders_per_minute      = int(max_orders_per_minute)
        self.ml_confidence_halflife_ms  = float(ml_confidence_halflife_ms)
        self.loss_cooldown_ms           = float(loss_cooldown_ms)
        self.stale_price_ticks          = int(stale_price_ticks)
        self.gamma                      = float(gamma)

        self.rng = np.random.RandomState(rng_seed)
        self.ml_score: float = 0.0   # externally injected by runner
        self._last_avg_cost: float = 0.0   # rough tracker for loss detection

        # Track our active orders for stale-quote detection: {order_id: placed_price}
        self._active_order_prices: Dict[str, float] = {}

    # =========================================================================
    # Core strategy computation
    # =========================================================================

    def estimate_short_vol(self, horizon_seconds: float = 0.5) -> float:
        """
        Estimate short-horizon return volatility (std dev) from recent mid returns.
        Scaled to the given horizon using the square-root rule.
        """
        arr = list(self.recent_mid_returns)
        if len(arr) < 3:
            return 0.001  # conservative placeholder
        sigma_per_sample = float(np.std(arr))
        # Each sample corresponds to one tick interval; scale to horizon
        tick_interval_s = 0.005  # assume ~5ms per tick (common in US equities)
        samples_per_horizon = max(1.0, horizon_seconds / tick_interval_s)
        return max(1e-6, sigma_per_sample * math.sqrt(samples_per_horizon))

    def compute_edge_and_variance(self,
                                  ml_score:        float,
                                  mid_price:       float,
                                  spread:          float,
                                  latency_seconds: float
                                  ) -> tuple[float, float, float]:
        """
        Conservative edge and variance computation.

        Returns: (adjusted_edge_frac, variance, sigma)
          adjusted_edge_frac: expected return fraction after latency cost
          variance:           inflated variance (4x for conservatism)
          sigma:              raw short vol for reservation price formula
        """
        sigma = self.estimate_short_vol(horizon_seconds=latency_seconds)

        # ML tilt: at most 25% of spread in price units
        ml_tilt_price = float(ml_score) * 0.25 * spread
        edge_price = ml_tilt_price  # gross signal edge

        # Latency cost: expected adverse move during our latency window
        arr = list(self.recent_mid_returns)
        if arr:
            p80 = np.percentile(np.abs(arr[-min(len(arr), 50):]), 80)
        else:
            p80 = sigma
        latency_loss_frac = float(p80)

        # Net edge (fraction of mid price)
        edge_frac = edge_price / max(1e-8, mid_price)
        adjusted_edge = edge_frac - latency_loss_frac

        # Conservative variance (4x inflation — arch doc recommends this)
        variance = (sigma ** 2) * 4.0

        return adjusted_edge, variance, sigma

    def compute_kelly_size(self,
                           adjusted_edge:       float,
                           variance:            float,
                           mid_price:           float,
                           latency_seconds:     float) -> int:
        """
        Compute share size using conservative Kelly sizing.

        Invariants enforced:
          #48: Kelly fraction in (0, 1]
          #49: size > 0
          #50: size <= max_inventory - abs(current_inventory)
        """
        if adjusted_edge <= 0 or variance <= 0:
            return 0

        # Theoretical Kelly fraction f* = mu / sigma^2
        f_star = adjusted_edge / max(1e-12, variance)

        # Heavy shrinkage + cap (invariant #48: f ≤ 1.0)
        f_shrunk = max(0.0, min(f_star * self.kelly_shrink, 0.20))

        # Dollar risk allowed
        dollar_risk = f_shrunk * self.capital

        # Risk per share estimate: tick + half-spread + transaction cost
        risk_per_share = max(
            self.tick_size,
            0.5 * (mid_price * 0.0001)   # baseline micro-slippage
        ) + mid_price * self.transaction_cost
        risk_per_share_notional = risk_per_share * mid_price

        if risk_per_share_notional <= 0:
            return 0

        shares_by_risk = int(max(0, dollar_risk / risk_per_share_notional))

        # ADV participation cap
        adv_per_sec = self.adv_daily_notional / (24 * 3600)
        max_notional_by_adv = adv_per_sec * latency_seconds * self.adv_participation_limit
        shares_by_adv = int(max_notional_by_adv / max(1e-8, mid_price))

        # Inventory headroom cap (invariant #50)
        headroom = int(max(0, self.max_inventory - abs(self.current_inventory)))

        # Final size
        size = min(shares_by_risk, shares_by_adv, headroom, self.max_shares_per_order)
        return max(0, size)

    def compute_reservation_price(self,
                                  mid_price:            float,
                                  sigma:                float,
                                  time_horizon_seconds: float,
                                  ml_tilt_price:        float) -> float:
        """
        Avellaneda-Stoikov reservation price.

        r = S - gamma * sigma² * tau * inventory

        Where inventory coercion dominates and ML tilt is strictly limited.
        When inventory is positive (long), reservation price < mid (we want to sell).
        When inventory is negative (short), reservation price > mid (we want to buy).
        """
        inv = self.current_inventory
        inventory_penalty = self.gamma * (sigma ** 2) * time_horizon_seconds * inv
        r_base = mid_price - inventory_penalty

        # ML tilt strictly limited to small fraction of inventory adjustment
        max_ml_tilt = max(abs(inventory_penalty) * 0.10, 0.0005 * mid_price)
        ml_tilt_limited = max(-max_ml_tilt, min(max_ml_tilt, ml_tilt_price))

        return r_base + ml_tilt_limited

    # =========================================================================
    # Pacing control
    # =========================================================================

    def _check_pacing(self, ts: float) -> bool:
        """Returns True if we can place an order now (pacing rules allow it)."""
        # Minimum inter-order interval (invariant #54)
        dt_ms = (ts - self._last_order_time) * 1000.0
        if dt_ms < self.min_time_between_orders_ms and self._last_order_time > 0:
            return False

        # Per-minute order cap (invariant #53)
        if ts - self._minute_start_time > 60.0:
            self._orders_this_minute = 0
            self._minute_start_time = ts
        if self._orders_this_minute >= self.max_orders_per_minute:
            return False

        # Loss cooldown (invariant #52)
        dt_loss_ms = (ts - self._last_loss_time) * 1000.0
        if dt_loss_ms < self.loss_cooldown_ms and self._last_loss_time > 0:
            return False

        return True

    def _tick_align(self, price: float) -> float:
        """Align a price to the nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    # =========================================================================
    # Stale quote cancellation
    # =========================================================================

    def _get_cancel_actions(self,
                             engine: "MatchingEngine",
                             best_bid: float,
                             best_ask: float) -> List[CancelAction]:
        """
        Cancel stale resting orders: those whose price has drifted more than
        stale_price_ticks away from the current best bid/ask.

        This fixes FM-8 (stale quote risk) from the architecture document.
        Without this, old orders accumulate at uncompetitive prices creating
        adverse selection risk.
        """
        cancels = []
        threshold = self.stale_price_ticks * self.tick_size

        for order in engine.active_resting_orders(owner=self.name):
            if order.side == Side.BUY:
                # Our bid: cancel if it's too far below current best bid
                if abs(order.price - best_bid) > threshold:
                    cancels.append(CancelAction(order_id=order.order_id, owner=self.name))
                    self._active_order_prices.pop(order.order_id, None)
            else:  # SELL
                # Our ask: cancel if it's too far above current best ask
                if abs(order.price - best_ask) > threshold:
                    cancels.append(CancelAction(order_id=order.order_id, owner=self.name))
                    self._active_order_prices.pop(order.order_id, None)

        return cancels

    # =========================================================================
    # Main strategy tick
    # =========================================================================

    def on_tick(self,
                engine:   "MatchingEngine",
                snapshot: MarketSnapshot) -> List[StrategyAction]:
        """
        Called once per snapshot. Returns list of typed strategy actions.

        Invariants respected:
          #6:  strategy sees book at time T, not T+1 (guaranteed by runner ordering)
          #7:  ML score from features available at T only
          #55: ML score in [-1, 1] after normalization
          #51: edge must exceed transaction costs before placing order
          #52: no order during loss cooldown
          #53: order rate limit enforced
          #54: minimum time between orders enforced
        """
        actions: List[StrategyAction] = []
        ts = snapshot.ts

        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        if best_bid is None or best_ask is None:
            return actions  # empty book — nothing to do

        mid = snapshot.mid
        spread = snapshot.spread or self.tick_size

        # --- 1. Stale quote cancellation (FM-8 fix) --------------------------
        cancel_actions = self._get_cancel_actions(engine, best_bid, best_ask)
        actions.extend(cancel_actions)

        # --- 2. Pacing check -------------------------------------------------
        if not self._check_pacing(ts):
            return actions  # may still have cancel actions

        # --- 3. ML score decay (signal becomes stale over time) --------------
        # invariant #55: ml_score in [-1, 1]
        ml_score = max(-1.0, min(1.0, self.ml_score))
        dt_ms = (ts - self._last_order_time) * 1000.0 if self._last_order_time > 0 else 0.0
        tau = max(1.0, self.ml_confidence_halflife_ms)
        if dt_ms > 0:
            ml_score *= math.exp(-dt_ms / tau)

        # --- 4. Edge and variance computation --------------------------------
        latency_seconds = max(1e-3, self.latency_ms / 1000.0)
        adjusted_edge, variance, sigma = self.compute_edge_and_variance(
            ml_score, mid, spread, latency_seconds
        )

        # --- 5. Size computation (invariant #51: edge > transaction costs) ---
        if abs(adjusted_edge) < self.min_edge_threshold:
            return actions

        size = self.compute_kelly_size(adjusted_edge, variance, mid, latency_seconds)
        if size <= 0:
            return actions  # not enough edge for any shares

        # --- 6. Reservation price (Avellaneda-Stoikov) ----------------------
        ml_tilt_price = ml_score * 0.25 * spread
        r_price = self.compute_reservation_price(
            mid_price=mid,
            sigma=sigma,
            time_horizon_seconds=latency_seconds * 2.0,
            ml_tilt_price=ml_tilt_price,
        )

        # --- 7. Choose side and price ----------------------------------------
        if adjusted_edge > 0:
            # Positive edge → want to buy (place bid)
            target_price = self._tick_align(min(r_price, best_bid))
            target_price = max(target_price, best_bid - self.tick_size)  # inside or at best bid
            action = PlaceLimitAction(
                side=Side.BUY,
                price=target_price,
                qty=float(size),
                owner=self.name,
                tag=f"ml={ml_score:.3f},edge={adjusted_edge:.5f}",
            )
        else:
            # Negative edge → want to sell (place ask)
            target_price = self._tick_align(max(r_price, best_ask))
            target_price = min(target_price, best_ask + self.tick_size)  # inside or at best ask
            action = PlaceLimitAction(
                side=Side.SELL,
                price=target_price,
                qty=float(size),
                owner=self.name,
                tag=f"ml={ml_score:.3f},edge={adjusted_edge:.5f}",
            )

        actions.append(action)

        # --- 8. Update pacing state ------------------------------------------
        self._last_order_time = ts
        self._orders_this_minute += 1

        return actions

    def on_fill(self, fill_log: dict) -> None:
        """Detect realized losses for cooldown trigger."""
        super().on_fill(fill_log)
        side = fill_log.get("side", "")
        price = float(fill_log.get("price", 0.0))

        # Loss cooldown: if we close a position at a worse price than entry
        if self._last_avg_cost > 0 and abs(self.current_inventory) > 1e-9:
            if self.current_inventory > 0 and "sell" in str(side).lower():
                if price < self._last_avg_cost:
                    self._last_loss_time = fill_log.get("time", 0.0)
            elif self.current_inventory < 0 and "buy" in str(side).lower():
                if price > self._last_avg_cost:
                    self._last_loss_time = fill_log.get("time", 0.0)

        self._last_avg_cost = price  # rough tracker
