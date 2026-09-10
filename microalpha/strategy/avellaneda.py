"""
microalpha/strategy/avellaneda.py
==================================
Avellaneda-Stoikov market-making strategy.

Implements the classical Avellaneda-Stoikov (2008) optimal market-making
model for a continuous-time setting approximated in discrete simulation time.

The key formula:
    r = S - gamma * sigma² * tau * q          (reservation price)
    delta_bid = gamma * sigma² * tau + ln(1 + gamma/kappa) / gamma + ...
    delta_ask = gamma * sigma² * tau + ln(1 + gamma/kappa) / gamma + ...

In our simplified discrete implementation:
    reservation_price = mid - gamma * sigma² * tau * inventory
    bid = reservation_price - spread_half
    ask = reservation_price + spread_half
    where spread_half is calibrated to be at least tick_size / 2

The strategy continuously:
  1. Computes optimal bid and ask prices around the reservation price
  2. Places/replaces limit orders on both sides
  3. Cancels any stale orders that have drifted beyond threshold
  4. Enforces inventory hard limits: no new orders if at max position
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from microalpha.data.snapshot import (
    CancelAction, MarketSnapshot, PlaceLimitAction, Side
)
from microalpha.strategy.base import BaseStrategy, StrategyAction

if TYPE_CHECKING:
    from microalpha.engine.matching import MatchingEngine


class AvellanedaStoikovStrategy(BaseStrategy):
    """
    Optimal market-making strategy based on Avellaneda-Stoikov (2008).

    Places symmetric bid/ask quotes around the inventory-adjusted reservation
    price. Continuously replaces stale quotes as the market moves.

    Parameters:
        gamma:          risk aversion parameter (higher → tighter inventory mgmt)
        kappa:          order arrival rate intensity (higher → tighter spreads)
        sigma:          initial volatility guess (updated from data)
        tau:            time horizon for position closure (seconds)
        base_qty:       default order quantity per quote
        max_inventory:  absolute position limit in shares
        tick_size:      minimum price increment
        quote_lifetime_ms: cancel and replace quotes after this many ms
        min_spread_ticks:  minimum spread in ticks from mid
    """

    def __init__(self,
                 name:                str   = "as_mm",
                 gamma:               float = 0.1,
                 kappa:               float = 1.5,
                 sigma:               float = 0.002,
                 tau:                 float = 60.0,
                 base_qty:            float = 100.0,
                 max_inventory:       float = 1000.0,
                 tick_size:           float = 0.01,
                 quote_lifetime_ms:   float = 200.0,
                 min_spread_ticks:    int   = 1,
                 rng_seed:            int   = 99):
        super().__init__(name=name)
        self.gamma            = float(gamma)
        self.kappa            = float(kappa)
        self._sigma           = float(sigma)
        self.tau              = float(tau)
        self.base_qty         = float(base_qty)
        self.max_inventory    = float(max_inventory)
        self.tick_size        = float(tick_size)
        self.quote_lifetime_ms = float(quote_lifetime_ms)
        self.min_spread_ticks = int(min_spread_ticks)
        self.rng = np.random.RandomState(rng_seed)

        # Track active bid and ask order IDs for replacement
        self._bid_order_id: Optional[str] = None
        self._ask_order_id: Optional[str] = None
        self._last_quote_ts: float = 0.0

    def _estimate_sigma(self) -> float:
        """Update sigma from recent mid returns."""
        arr = list(self.recent_mid_returns)
        if len(arr) < 5:
            return self._sigma
        return max(1e-6, float(np.std(arr[-50:])))

    def _reservation_and_spread(self, mid: float, sigma: float) -> tuple[float, float]:
        """
        Compute Avellaneda-Stoikov reservation price and optimal half-spread.

        r    = mid - gamma * sigma² * tau * inventory
        delta = gamma * sigma² * tau + (2/gamma) * ln(1 + gamma/kappa)
                (optimal spread width — each side is delta/2)
        """
        inv = self.current_inventory
        reservation = mid - self.gamma * (sigma ** 2) * self.tau * inv

        # Optimal spread from A-S solution
        if self.kappa > 0:
            optimal_half_spread = (
                self.gamma * (sigma ** 2) * self.tau / 2.0
                + (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
            )
        else:
            optimal_half_spread = self.gamma * (sigma ** 2) * self.tau

        # Enforce minimum spread in ticks
        min_half_spread = (self.min_spread_ticks * self.tick_size) / 2.0
        half_spread = max(optimal_half_spread, min_half_spread)

        return reservation, half_spread

    def _tick_align(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _cancel_if_exists(self, order_id: Optional[str]) -> Optional[CancelAction]:
        if order_id is not None:
            return CancelAction(order_id=order_id, owner=self.name)
        return None

    def on_tick(self,
                engine:   "MatchingEngine",
                snapshot: MarketSnapshot) -> List[StrategyAction]:
        """
        Each tick: cancel stale quotes and place fresh ones at optimal prices.
        """
        actions: List[StrategyAction] = []
        ts = snapshot.ts

        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        if best_bid is None or best_ask is None:
            return actions

        mid = snapshot.mid

        # Update volatility estimate
        sigma = self._estimate_sigma()
        self._sigma = sigma

        # Compute optimal quotes
        reservation, half_spread = self._reservation_and_spread(mid, sigma)

        our_bid = self._tick_align(reservation - half_spread)
        our_ask = self._tick_align(reservation + half_spread)

        # Ensure bid < ask and both are within visible book
        our_bid = min(our_bid, best_bid)
        our_ask = max(our_ask, best_ask)
        if our_bid >= our_ask:
            return actions  # degenerate: skip

        # Check quote staleness: replace if quote lifetime exceeded
        dt_ms = (ts - self._last_quote_ts) * 1000.0
        needs_refresh = dt_ms >= self.quote_lifetime_ms or self._last_quote_ts == 0.0

        if needs_refresh:
            # Cancel existing quotes
            for cancel_action in [
                self._cancel_if_exists(self._bid_order_id),
                self._cancel_if_exists(self._ask_order_id),
            ]:
                if cancel_action:
                    actions.append(cancel_action)
            self._bid_order_id = None
            self._ask_order_id = None

            # Determine available quantities (inventory-constrained)
            inv = self.current_inventory

            # Cap buy qty: don't exceed max_inventory
            buy_qty = min(self.base_qty, max(0, self.max_inventory - inv))
            # Cap sell qty: don't exceed max_inventory on short side
            sell_qty = min(self.base_qty, max(0, self.max_inventory + inv))

            if buy_qty > 0 and inv < self.max_inventory:
                bid_action = PlaceLimitAction(
                    side=Side.BUY,
                    price=our_bid,
                    qty=buy_qty,
                    owner=self.name,
                    tag=f"r={reservation:.4f},hs={half_spread:.4f}",
                )
                actions.append(bid_action)

            if sell_qty > 0 and inv > -self.max_inventory:
                ask_action = PlaceLimitAction(
                    side=Side.SELL,
                    price=our_ask,
                    qty=sell_qty,
                    owner=self.name,
                    tag=f"r={reservation:.4f},hs={half_spread:.4f}",
                )
                actions.append(ask_action)

            if actions:
                self._last_quote_ts = ts

        return actions
