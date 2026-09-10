"""
microalpha/strategy/base.py
============================
BaseStrategy abstract interface.

All strategies must subclass BaseStrategy and implement on_tick().
The on_tick() method receives the current engine state and snapshot,
and returns a list of typed StrategyAction objects.

Design principles (from architecture doc):
  - Strategy returns TYPED actions (PlaceLimitAction, CancelAction, MarketOrderAction)
    NOT loose dictionaries (invariant from arch doc Section 8, Key Principle 4)
  - Strategy does NOT mutate cash or position directly in replay mode
  - Strategy maintains runtime inventory tracking for decision making only
    (PnL is computed from engine logs, not strategy state — Pattern 7)
  - No lookahead: on_tick receives book state AT time T, not T+1 (invariant #6, #7)
  - on_fill is called by the runner when a fill is attributed to this strategy
"""

from __future__ import annotations

import abc
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, List, Optional

from microalpha.data.snapshot import (
    CancelAction, MarketOrderAction, PlaceLimitAction, MarketSnapshot
)

if TYPE_CHECKING:
    from microalpha.engine.matching import MatchingEngine


# Union type for all strategy actions
StrategyAction = PlaceLimitAction | CancelAction | MarketOrderAction


class BaseStrategy(abc.ABC):
    """
    Abstract base strategy interface.

    To implement a strategy:
      1. Subclass BaseStrategy
      2. Implement on_tick() returning a list of typed actions
      3. Optionally override on_fill() to react to fills
      4. Set self.name to identify fills in the log

    The runner will:
      - Call on_tick() each snapshot
      - Translate returned actions to engine calls
      - Call on_fill() for each fill attributed to self.name
      - Update self.current_inventory (the runner owns this mutation)
    """

    def __init__(self, name: str = "strategy"):
        self.name = name

        # Runtime inventory tracking (for strategy decisions only — NOT PnL source)
        # The runner is the single authority for mutating this (Pattern 7)
        self.current_inventory: float = 0.0

        # Rolling mid-return buffer for volatility estimation
        # Populated by BacktestRunner each tick
        self.recent_mid_returns: Deque[float] = deque(maxlen=1000)
        self._last_mid: Optional[float] = None

        # Fills log (strategy-level copy for convenience; engine logs are authoritative)
        self.fills_log: List[Any] = []

        # Pacing controls (subclasses can override defaults)
        self._last_order_time: float = 0.0
        self._last_loss_time:  float = 0.0
        self._orders_this_minute: int = 0
        self._minute_start_time: float = 0.0

    @abc.abstractmethod
    def on_tick(self,
                engine:   "MatchingEngine",
                snapshot: MarketSnapshot) -> List[StrategyAction]:
        """
        Called once per market snapshot.

        Must return a list of strategy actions. May return an empty list.

        Constraints:
          - MUST NOT access any future snapshots
          - MUST NOT mutate engine.logs
          - MUST NOT mutate engine book state
          - Actions are executed by the runner AFTER this method returns
        """

    def on_fill(self, fill_log: dict) -> None:
        """
        Called by the runner when a fill is attributed to this strategy.

        In replay mode: log the fill but do NOT update cash directly.
        Subclasses can override to implement loss detection, cooldowns, etc.
        """
        self.fills_log.append(fill_log)

    def update_mid_return(self, mid: float) -> None:
        """Called by BacktestRunner to update the recent mid return buffer."""
        if self._last_mid is not None and self._last_mid > 0:
            ret = (mid - self._last_mid) / self._last_mid
            self.recent_mid_returns.append(ret)
        self._last_mid = mid
