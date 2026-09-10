"""
microalpha/engine/queue_model.py
=================================
Queue position model: probabilistic estimation of where your order
sits in the per-price FIFO queue at the exchange.

This is the hardest layer to get right with L2 data.
With MBP-10 data we cannot know exact queue position — we can only
estimate a probability distribution.

Architecture:
  1. INITIAL QUEUE POSITION (at order arrival):
     qp = max(0, observed_depth + new_ahead - expected_depletion + hidden_extra) * jitter
     Where:
       observed_depth    = visible size at price level at placement time
       new_ahead         = Poisson(lambda * dt_latency) — orders that joined ahead during latency
       expected_depletion = estimated volume consumed before arrival
       hidden_extra      = fraction of observed_depth for hidden (iceberg) liquidity

  2. QUEUE DECAY (while resting between snapshots):
     queue_position *= exp(-lambda * dt) + new_arrivals_ahead
     This exponential decay + Poisson arrivals models:
       (a) Old orders at front of queue cancel
       (b) New orders arrive and join ahead of us

  3. FILL PROBABILITY (p_reach):
     p_reach = 1 - exp(-consumed / (queue_ahead + eps))
     CDF of exponential distribution — probability consumption "reaches" our position.

  4. ACTUAL FILL ALLOCATION:
     actual_consumed ~ Poisson(p_level_fill * consumed)
     Then FIFO allocation: front-of-queue orders get filled first.

Architecture doc fixes implemented here:
  - Fixed: arrival rate uses per-side (bid vs ask) estimation, not one global rate
  - Fixed: queue position is clamped to >= 0 always (invariant #16)
  - Fixed: fill is discrete event (Poisson sample), not continuous fraction (bug #3)
  - Fixed: level disappearance does NOT auto-trigger fill (bug #4)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from microalpha.data.snapshot import Side


@dataclass
class QueueModelConfig:
    """
    Configuration for queue position modeling.

    decay_lambda:       exponential decay rate (per second) for queue advantage
                        Higher = faster queue position erosion (more aggressive market)
    hidden_liquidity_frac: fraction of additional hidden depth to add to qp
    arrival_rate_min:   minimum order arrival rate (orders/second, per level)
    arrival_rate_max:   maximum order arrival rate (clip ceiling)
    """
    decay_lambda:            float = 1.0
    hidden_liquidity_frac:   float = 0.0
    arrival_rate_min:        float = 0.5
    arrival_rate_max:        float = 200.0

    def __post_init__(self):
        if self.decay_lambda < 0:
            raise ValueError(f"decay_lambda must be >= 0, got {self.decay_lambda}")


class QueuePositionModel:
    """
    Models per-price FIFO queue position for resting limit orders.

    This class is used by MatchingEngine to:
      1. Assign initial queue position when an order arrives
      2. Decay queue positions each tick (new mkt activity erodes priority)
      3. Compute fill probability given level consumption
      4. Sample actual fill quantity (stochastic Poisson model)

    Usage:
        model = QueuePositionModel(QueueModelConfig())
        # When order arrives:
        qp = model.compute_initial_queue_position(observed_depth, latency_s)
        # Each tick:
        new_qp = model.decay_queue_position(current_qp, dt_seconds, side)
        # To compute fill probability:
        p_reach = model.compute_p_reach(consumed, queue_ahead)
        filled = model.sample_fill_volume(p_reach, consumed, queue_ahead)
    """

    def __init__(self, config: QueueModelConfig = None, rng_seed: int = 42):
        self.config = config or QueueModelConfig()
        self.rng = np.random.RandomState(rng_seed)
        # Recent consumption window for arrival rate estimation (per side)
        # (ts, consumed_volume) pairs
        self._bid_consumption: deque = deque(maxlen=200)
        self._ask_consumption: deque = deque(maxlen=200)
        # EWMA-based arrival rate state (adaptive, replaces crude avg/10 heuristic)
        self._ewma_bid_rate: float = self.config.arrival_rate_min
        self._ewma_ask_rate: float = self.config.arrival_rate_min
        self._ewma_halflife_events: float = 50.0  # half-life in number of events

    # ---------------------------------------------------------------------------
    # Arrival rate estimation (per side)
    # ---------------------------------------------------------------------------

    def record_consumption(self, side: Side, volume: float, ts: float) -> None:
        """Record a consumption event for arrival rate estimation.

        Updates both the rolling window AND the EWMA-based rate estimator.
        """
        if side == Side.BUY:
            self._ask_consumption.append((ts, volume))  # buy aggressor consumed asks
            # EWMA update: alpha = 1 - 0.5^(1/halflife)
            alpha = 1.0 - 0.5 ** (1.0 / max(1.0, self._ewma_halflife_events))
            self._ewma_ask_rate = (1 - alpha) * self._ewma_ask_rate + alpha * volume
        else:
            self._bid_consumption.append((ts, volume))
            alpha = 1.0 - 0.5 ** (1.0 / max(1.0, self._ewma_halflife_events))
            self._ewma_bid_rate = (1 - alpha) * self._ewma_bid_rate + alpha * volume

    def _estimate_arrival_rate(self, side: Side) -> float:
        """
        Estimate per-level order arrival rate (shares/second proxy).

        Uses EWMA-smoothed consumption volume as a proxy for order flow rate.
        The EWMA adapts to the actual data feed characteristics rather than
        relying on the arbitrary hardcoded "avg_volume / 10" heuristic.

        With L2 data we cannot observe individual order arrivals, so consumption
        remains the best available proxy (architecture doc acknowledges this).
        """
        ewma_vol = self._ewma_bid_rate if side == Side.BUY else self._ewma_ask_rate
        buf = self._bid_consumption if side == Side.BUY else self._ask_consumption

        if not buf:
            return self.config.arrival_rate_min

        # Estimate event rate from recent timestamps (events per second)
        if len(buf) >= 2:
            dt_total = buf[-1][0] - buf[0][0]
            if dt_total > 1e-6:
                events_per_sec = (len(buf) - 1) / dt_total
            else:
                events_per_sec = 1.0
        else:
            events_per_sec = 1.0

        # Rate = EWMA volume per event * events per second
        # This naturally adapts to both the volume profile and snapshot frequency
        rate = ewma_vol * events_per_sec
        return max(self.config.arrival_rate_min,
                   min(self.config.arrival_rate_max, rate))

    def _estimate_depletion(self, side: Side, dt_seconds: float) -> float:
        """Estimate how many shares are consumed ahead of us during latency dt.

        Uses the event-rate-weighted volume estimator rather than a hardcoded
        snapshot interval assumption.
        """
        buf = self._bid_consumption if side == Side.BUY else self._ask_consumption
        if not buf or dt_seconds <= 0:
            return 0.0

        # Event rate from recent history
        if len(buf) >= 2:
            dt_total = buf[-1][0] - buf[0][0]
            if dt_total > 1e-6:
                events_per_sec = (len(buf) - 1) / dt_total
            else:
                events_per_sec = 1.0
        else:
            events_per_sec = 1.0

        # EWMA volume per event
        ewma_vol = self._ewma_bid_rate if side == Side.BUY else self._ewma_ask_rate

        # Expected depletion during dt
        return ewma_vol * events_per_sec * dt_seconds

    # ---------------------------------------------------------------------------
    # Initial queue position assignment
    # ---------------------------------------------------------------------------

    def compute_initial_queue_position(self,
                                       observed_depth: float,
                                       latency_s:      float,
                                       side:           Side) -> float:
        """
        Compute the initial queue position when an order transitions
        from 'pending' to 'resting'.

        observed_depth: visible book size at the price level at placement time
        latency_s:      time between placement and arrival (from LatencyModel)
        side:           BID or ASK side of our order

        Returns: queue position (shares ahead of us; 0 = front of queue)
        """
        observed = max(0.0, observed_depth)
        arrival_rate = self._estimate_arrival_rate(side)

        # New orders that joined the queue ahead of us during latency
        lam = max(0.0, arrival_rate * latency_s)
        new_ahead = float(self.rng.poisson(lam=lam))

        # Expected volume consumed/depleted ahead of us before arrival
        expected_depletion = self._estimate_depletion(side, latency_s)

        # Base queue position: depth at placement + new arrivals - depletion
        base_qp = max(0.0, observed + new_ahead - expected_depletion)

        # Hidden liquidity: iceberg orders add to effective queue depth
        hidden_extra = base_qp * self.config.hidden_liquidity_frac * self.rng.uniform(0.0, 1.0)

        # Small jitter for uncertainty in our position estimate
        jitter = 1.0 + 0.05 * (self.rng.uniform() - 0.5)

        qp = max(0.0, (base_qp + hidden_extra) * jitter)
        return qp

    # ---------------------------------------------------------------------------
    # Queue decay between snapshots
    # ---------------------------------------------------------------------------

    def decay_queue_position(self,
                             current_qp: float,
                             dt_seconds: float,
                             side:       Side) -> float:
        """
        Decay queue position between snapshots.

        Models two competing forces:
          (a) Exponential decay: old front-of-queue orders cancel → our priority improves
          (b) Poisson new arrivals: new orders join ahead → our priority deteriorates

        Net queue position after dt:
            new_qp = current_qp * exp(-lambda * dt) + new_arrivals_ahead

        Clamped to [0, ∞) — invariant #16.
        """
        if dt_seconds <= 0:
            return current_qp

        # Exponential decay of priority (new orders filling in front cancel out)
        decayed = current_qp * math.exp(-self.config.decay_lambda * dt_seconds)

        # Poisson new arrivals ahead
        arrival_rate = self._estimate_arrival_rate(side)
        lam = max(0.0, arrival_rate * dt_seconds)
        new_ahead = float(self.rng.poisson(lam=lam))

        return max(0.0, decayed + new_ahead)

    # ---------------------------------------------------------------------------
    # Fill probability computation
    # ---------------------------------------------------------------------------

    def compute_p_reach(self, consumed: float, queue_ahead: float) -> float:
        """
        Probability that market consumption reaches our queue position.

        Uses the CDF of an exponential distribution:
            P(reach) = 1 - exp(-consumed / (queue_ahead + eps))

        Interpretation:
          - consumed ≫ queue_ahead  →  p_reach ≈ 1.0 (almost certainly filled)
          - consumed ≪ queue_ahead  →  p_reach ≈ 0.0 (unlikely to be reached)
        """
        if consumed <= 0:
            return 0.0
        eps = 1e-9
        return 1.0 - math.exp(-consumed / (max(0.0, queue_ahead) + eps))

    def sample_fill_volume(self,
                           p_reach:      float,
                           consumed:     float,
                           queue_ahead:  float,
                           max_fill:     float) -> float:
        """
        Sample the actual fill volume to allocate, given p_reach.

        This implements a DISCRETE fill model (not continuous fraction):
          - With probability p_reach, some volume reaches our order
          - The actual volume is Poisson-sampled around expected_filled
          - Capped by: consumed, queue_ahead, max_fill (order.remaining)

        The architecture doc specifically calls out Bug #3 in the hybrid engine:
        fractional fills on every tick. This implementation avoids that by using
        a proper stochastic model.

        Returns: fill_qty ≥ 0
        """
        if p_reach <= 0 or consumed <= 0 or max_fill <= 0:
            return 0.0

        # Expected volume that fills our order
        # (p_reach * consumed represents expected volume at our level)
        expected_filled = p_reach * min(consumed, max_fill + queue_ahead)

        # Poisson sample — models discrete order flow
        actual = float(self.rng.poisson(lam=max(0.0, expected_filled)))

        # Caps: can't fill more than consumed, can't fill more than our order remaining
        actual = min(actual, consumed, max_fill)
        return max(0.0, actual)

    # ---------------------------------------------------------------------------
    # Adverse selection score
    # ---------------------------------------------------------------------------

    def compute_adverse_score(self,
                              aggressor_side:      Side,
                              our_side:            Side,
                              recent_consumption:  deque) -> float:
        """
        Heuristic adverse selection multiplier.

        When a passive order fills, it can be:
          (a) Benign random flow (noise trader) — factor ≈ 1.0
          (b) Informed aggressor who knows price is moving adversely — factor > 1.0

        Heuristic: buy aggressors lifting asks are moderately informative.
        High recent consumption intensity → higher adverse fraction.

        Returns a score in [0.1, 3.0] where > 1.0 suggests adverse selection.
        """
        factor = 1.0

        # Aggressor-direction signal:
        # If buy aggressor lifting ask and we are selling passively → they bought knowing it goes up
        # If sell aggressor hitting bid and we are buying passively → they sold knowing it goes down
        if aggressor_side != our_side:
            # Opposite sides: this is how passive fills happen, moderately informative
            factor *= 1.15
        else:
            # Same side (shouldn't normally happen in FIFO model): suspicious
            factor *= 1.05

        # Recent consumption intensity: high volume = more informed flow
        if recent_consumption:
            recent_vols = [v for _, v in list(recent_consumption)[-20:]]
            recent_total = sum(recent_vols)
            # Scale: 1.0 → 2.0 range based on volume
            factor *= 1.0 + min(0.8, recent_total / (recent_total + 1000.0))

        # Small stochastic jitter
        factor *= 1.0 + 0.1 * (self.rng.uniform() - 0.5)

        return max(0.1, min(factor, 3.0))
