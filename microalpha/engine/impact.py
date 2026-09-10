"""
microalpha/engine/impact.py
============================
Market impact model: square-root law for permanent and temporary impact.

Based on Almgren-Chriss (2001) and subsequent empirical studies showing that:
- Temporary impact (price moves while you trade): ~ σ * (qty/ADV)^0.5
- Permanent impact (price shift after trade):     ~ η * σ * (qty/ADV)^0.5

In replay mode (impact_mode='replay') the historical book is IMMUTABLE — we can
only LOG the estimated impact. This is the correct behavior:
  "You cannot observe your own market impact in a replay." (Architecture doc Layer 0)

The impact is computed and stored in fill events for attribution but is NOT
subtracted from future snapshot prices (that would require knowing how market
participants would have responded — counterfactual).

In future 'simulated' mode, a synthetic book would be maintained and mutated.

Architecture invariants enforced:
  #35  — in replay mode, historical snapshot prices are never mutated
  #36  — permanent impact logged but not applied retroactively
  #37  — market impact is non-negative (buys move price up, sells move down)
  #38  — buys cannot fill below best_ask
  #39  — no better-than-market fills (no price improvement beyond quotes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from microalpha.data.snapshot import MarketSnapshot, Side


@dataclass
class ImpactConfig:
    """
    Parameters for the square-root market impact model.

    perm_impact_coeff:  permanent impact coefficient (η in Almgren-Chriss)
    temp_impact_coeff:  temporary impact coefficient (ε in Almgren-Chriss)
    adv_estimate:       estimated average daily volume in dollar notional
    impact_mode:        'replay' (immutable history) or 'simulated' (future work)
    """
    perm_impact_coeff: float = 0.02
    temp_impact_coeff: float = 0.01
    adv_estimate:      float = 1_000_000.0
    impact_mode:       str   = "replay"

    def __post_init__(self):
        if self.impact_mode not in ("replay", "simulated"):
            raise ValueError(f"impact_mode must be 'replay' or 'simulated', got {self.impact_mode!r}")
        if self.adv_estimate <= 0:
            raise ValueError("adv_estimate must be positive")


@dataclass
class ImpactEstimate:
    """Result of an impact computation."""
    permanent_shift:  float  # price shift after trade (dollars), signed by direction
    temp_price_adj:   float  # within-trade price adjustment (always worsens execution)
    mode:             str    # 'replay' or 'simulated'


class ImpactModel:
    """
    Square-root market impact model for aggressive (market) orders.

    Impact scales as sqrt(participation_rate) where:
        participation_rate = trade_notional / adv_estimate

    In replay mode: impact is estimated and logged but NOT applied to future snapshots.
    The fill prices returned by execute_market_order() DO include temporary impact
    (the immediate within-trade cost), but permanent impact is only logged.

    Usage:
        model = ImpactModel(ImpactConfig())
        estimate = model.estimate(qty=1000, ref_price=150.0, side=Side.BUY)
        exec_price = ref_price + estimate.temp_price_adj  # for buy orders
    """

    def __init__(self, config: ImpactConfig = None):
        self.config = config or ImpactConfig()

    def estimate(self,
                 qty:       float,
                 ref_price: float,
                 side:      Side) -> ImpactEstimate:
        """
        Estimate impact for an aggressive order of `qty` shares at `ref_price`.

        Returns an ImpactEstimate with:
          - temp_price_adj: immediate cost per share (positive for buys, positive for sells)
          - permanent_shift: signed price shift (positive for buys lifting the book)
        """
        notional = qty * ref_price
        adv = max(1.0, self.config.adv_estimate)

        # Square-root law: sqrt(notional / ADV)
        sqrt_participation = math.sqrt(notional / adv)

        # Temporary impact: immediate within-trade execution cost
        temp_impact = self.config.temp_impact_coeff * sqrt_participation * ref_price

        # Permanent impact: lasting price shift after the trade
        perm_impact = self.config.perm_impact_coeff * sqrt_participation * ref_price

        # Direction: buys move price up, sells move price down (invariant #37)
        direction = +1.0 if side == Side.BUY else -1.0

        return ImpactEstimate(
            permanent_shift=direction * perm_impact,
            temp_price_adj=temp_impact,    # always positive (cost, not directional)
            mode=self.config.impact_mode,
        )

    def apply_to_fill_price(self,
                            base_price: float,
                            estimate:   ImpactEstimate,
                            side:       Side) -> float:
        """
        Adjust a fill price by the temporary impact component.

        Buys:  exec_price = base_price + temp_impact (we pay more)
        Sells: exec_price = base_price - temp_impact (we receive less)

        This models the within-trade slippage from sweeping multiple levels.
        """
        direction = +1.0 if side == Side.BUY else -1.0
        return base_price + direction * estimate.temp_price_adj

    def sweep_book(self,
                   levels:    List[Tuple[float, float]],
                   qty:       float,
                   side:      Side) -> Tuple[List[Tuple[float, float]], float]:
        """
        Sweep the book greedily (price-time priority), applying temporary impact.

        Returns:
            fills: list of (exec_price, qty) micro-fills
            unfilled: quantity that could not be filled (book depth insufficient)

        In replay mode, this reads the snapshot levels WITHOUT mutating them.
        The caller must NOT mutate the snapshot book (invariant #35).
        """
        remaining = float(qty)
        fills: List[Tuple[float, float]] = []

        if not levels or remaining <= 0:
            return fills, remaining

        # Compute impact estimate on full order (approximation; in reality impact
        # accrues as the order sweeps deeper, but this is standard practice)
        ref_price = levels[0][0]
        estimate = self.estimate(qty=qty, ref_price=ref_price, side=side)

        for level_price, level_size in levels:
            if remaining <= 1e-9:
                break
            take = min(level_size, remaining)
            exec_price = self.apply_to_fill_price(level_price, estimate, side)
            fills.append((exec_price, take))
            remaining -= take

        unfilled = max(0.0, remaining)
        return fills, unfilled
