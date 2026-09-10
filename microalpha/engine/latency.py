"""
microalpha/engine/latency.py
============================
Latency model: realistic one-way order submission latency.

Architecture requirements (from MICROALPHA doc):
- Lognormal distribution: positive, right-tailed (models GC pauses, microbursts)
- Median = base latency (via mu correction: mu_log = log(base) - 0.5*sigma²)
- Independent samples per order (i.i.d. — invariant #45)
- Separate sampling for: order placement, cancel, replace
- Clamping to avoid pathological outliers
- Simulation time used, NOT wall-clock time (invariant #46)

The lognormal is the textbook choice for latency:
   X ~ LogNormal(mu, sigma)  ←  log(X) ~ Normal(mu, sigma)
   median(X) = exp(mu)
   E[X] = exp(mu + sigma²/2)

By setting mu_log = log(base) - 0.5*sigma_log², the median is exactly `base`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LatencyConfig:
    """
    Configuration for a LatencyModel instance.

    base_ms:          median one-way latency in milliseconds
    jitter_ms:        lognormal shape parameter in ms (controls right-tail width)
    cancel_base_ms:   median cancel latency (default: same as base_ms)
    cancel_jitter_ms: cancel jitter (default: same as jitter_ms)
    min_fraction:     minimum latency as fraction of base (clamp floor)
    max_multiple:     maximum latency as multiple of base (clamp ceiling)
    """
    base_ms:          float = 5.0
    jitter_ms:        float = 2.0
    cancel_base_ms:   Optional[float] = None   # None → same as base_ms
    cancel_jitter_ms: Optional[float] = None   # None → same as jitter_ms
    min_fraction:     float = 0.5
    max_multiple:     float = 10.0

    def __post_init__(self):
        if self.base_ms <= 0:
            raise ValueError(f"LatencyConfig.base_ms must be > 0, got {self.base_ms}")
        if self.jitter_ms < 0:
            raise ValueError(f"LatencyConfig.jitter_ms must be >= 0, got {self.jitter_ms}")
        # Fill in defaults
        if self.cancel_base_ms is None:
            object.__setattr__(self, "cancel_base_ms", self.base_ms)
        if self.cancel_jitter_ms is None:
            object.__setattr__(self, "cancel_jitter_ms", self.jitter_ms)


class LatencyModel:
    """
    Samples realistic one-way latencies using a lognormal distribution.

    Invariants enforced:
    - Each call returns an i.i.d. sample (invariant #45)
    - All latencies are positive (lognormal bounded below at 0)
    - Returns seconds (not milliseconds) for direct timestamp arithmetic
    - Samples are clamped to [base*min_fraction, base*max_multiple]

    Usage:
        model = LatencyModel(LatencyConfig(base_ms=5.0, jitter_ms=2.0), rng_seed=42)
        latency_s = model.sample_placement()
        cancel_latency_s = model.sample_cancel()
    """

    def __init__(self, config: LatencyConfig = None, rng_seed: int = 42):
        self.config = config or LatencyConfig()
        self.rng = np.random.RandomState(rng_seed)

    def _sample_lognormal_seconds(self, base_ms: float, jitter_ms: float) -> float:
        """
        Core lognormal sampler. Returns latency in SECONDS.

        Uses the median-preserving parameterization:
            sigma_log = jitter_ms / max(1, base_ms)  [dimensionless]
            mu_log    = log(base_s) - 0.5 * sigma_log²
        So that exp(mu_log) = base_s = median(sample).
        """
        base_s = base_ms / 1000.0

        if jitter_ms <= 0.0:
            return base_s  # deterministic: no jitter

        # sigma_log controls the spread of the lognormal
        sigma_log = jitter_ms / max(1.0, base_ms)
        # mu_log tuned so that exp(mu_log) = base_s (median preservation)
        mu_log = math.log(max(1e-12, base_s)) - 0.5 * sigma_log ** 2

        sampled = float(self.rng.lognormal(mean=mu_log, sigma=sigma_log))

        # Clamp to reasonable range: [base * min_fraction, base * max_multiple]
        lo = base_s * self.config.min_fraction
        hi = base_s * self.config.max_multiple
        return max(lo, min(sampled, hi))

    def sample_placement(self) -> float:
        """Sample latency for a new limit order placement. Returns seconds."""
        return self._sample_lognormal_seconds(
            self.config.base_ms, self.config.jitter_ms
        )

    def sample_cancel(self) -> float:
        """Sample latency for a cancel request. Returns seconds. Independently sampled."""
        return self._sample_lognormal_seconds(
            self.config.cancel_base_ms, self.config.cancel_jitter_ms
        )

    def sample_market_order(self) -> float:
        """Sample latency for a market (aggressive) order. Same as placement."""
        return self.sample_placement()

    @property
    def median_placement_s(self) -> float:
        """Theoretical median placement latency in seconds."""
        return self.config.base_ms / 1000.0

    @property
    def median_cancel_s(self) -> float:
        """Theoretical median cancel latency in seconds."""
        return self.config.cancel_base_ms / 1000.0
