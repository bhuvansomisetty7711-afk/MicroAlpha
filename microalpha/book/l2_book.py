"""
microalpha/book/l2_book.py
==========================
L2 order book representation with integer price keys.

Key design decisions:
1. Prices stored as INTEGER TICK UNITS internally (price / tick_size rounded to int).
   This eliminates the float-dict-key bug described in the architecture doc (Issue 1).
   Example: price $150.25, tick_size $0.01 → key = 15025 (int).

2. Snapshots are IMMUTABLE — the book object is replaced, not mutated.
   Strategy actions must NEVER modify L2Book state.

3. Level-by-level delta computation: compares two consecutive snapshots price-by-price
   to compute exact consumption at each level, handling level appearance/disappearance
   correctly (the L2 aliasing problem described in section 2).

The key insight for delta computation: when a price level disappears, it could be
   (a) fully consumed (pessimistic: treat as consumed)
   (b) cancelled out (optimistic: treat as zero consumption)
We follow arch doc invariant #63 and use the PESSIMISTIC assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from microalpha.data.snapshot import L2Level, MarketSnapshot, Side


# ---------------------------------------------------------------------------
# Integer tick-unit price helpers
# ---------------------------------------------------------------------------

TICK_PRECISION = 1e-9  # float equality tolerance for tick rounding


def price_to_tick(price: float, tick_size: float) -> int:
    """Convert a float price to an integer tick key. Rounds to nearest tick."""
    return round(price / tick_size)


def tick_to_price(tick: int, tick_size: float) -> float:
    """Convert an integer tick key back to a float price."""
    return tick * tick_size


# ---------------------------------------------------------------------------
# Per-level delta: what happened at a single price level between two snapshots
# ---------------------------------------------------------------------------

@dataclass
class LevelDelta:
    """
    The change in a single price level between two consecutive snapshots.

    price:      float price of the level
    tick_key:   integer tick-unit key
    side:       BID or ASK
    old_size:   size in previous snapshot (0.0 if level was absent)
    new_size:   size in current snapshot (0.0 if level disappeared)
    consumed:   estimated consumption (old_size - new_size, capped to [0, ∞))
                Negative delta (new_size > old_size) means new orders arrived — NOT consumption.
    appeared:   True if level was absent in previous snapshot but exists now
    disappeared: True if level existed in previous snapshot but is absent now
    """
    price:      float
    tick_key:   int
    side:       Side
    old_size:   float
    new_size:   float

    @property
    def consumed(self) -> float:
        """Estimated volume consumed (capped at 0 — size increases are NOT consumption)."""
        return max(0.0, self.old_size - self.new_size)

    @property
    def appeared(self) -> bool:
        return self.old_size == 0.0 and self.new_size > 0.0

    @property
    def disappeared(self) -> bool:
        return self.old_size > 0.0 and self.new_size == 0.0

    @property
    def new_orders_arrived(self) -> bool:
        """True if new resting orders arrived (size increased)."""
        return self.new_size > self.old_size


# ---------------------------------------------------------------------------
# L2Book: current book state + delta computation
# ---------------------------------------------------------------------------

class L2Book:
    """
    L2 order book with integer tick-unit price keys.

    Holds the current best-N bid and ask levels as dicts keyed by integer tick.
    Provides level-by-level delta computation against a previous snapshot.

    Usage:
        book = L2Book(tick_size=0.01)
        book.update(snapshot)
        deltas = book.compute_deltas(previous_snapshot)
        consumed_at_ask = [(d.price, d.consumed) for d in deltas if d.side == Side.SELL and d.consumed > 0]
    """

    def __init__(self, tick_size: float = 0.01, num_levels: int = 10):
        """
        tick_size:  minimum price increment (e.g. 0.01 for US equities)
        num_levels: maximum depth to track per side
        """
        self.tick_size  = float(tick_size)
        self.num_levels = num_levels
        # bid/ask: dict[tick_key(int) → size(float)]
        self._bids: Dict[int, float] = {}
        self._asks: Dict[int, float] = {}
        self._current_snap: Optional[MarketSnapshot] = None

    def update(self, snap: MarketSnapshot) -> None:
        """Replace internal book state with this snapshot. Does not compute deltas."""
        self._bids = {
            price_to_tick(lv.price, self.tick_size): lv.size
            for lv in snap.bids[:self.num_levels]
        }
        self._asks = {
            price_to_tick(lv.price, self.tick_size): lv.size
            for lv in snap.asks[:self.num_levels]
        }
        self._current_snap = snap

    def compute_deltas(self, prev_snap: Optional[MarketSnapshot]) -> List[LevelDelta]:
        """
        Compute per-level consumption deltas between prev_snap and current state.

        This is the CORRECT implementation solving the L2 aliasing problem:
        - We compare level-by-level by price (tick key), not by array index.
        - A level that disappears is treated as fully consumed (pessimistic, arch #63).
        - A level whose size increased is categorized as new-orders-arrived, not consumption.
        - All 10 levels on both sides are checked.

        Returns a list of LevelDelta objects for ALL changed levels.
        """
        if prev_snap is None:
            return []

        prev_bids = {
            price_to_tick(lv.price, self.tick_size): lv.size
            for lv in prev_snap.bids[:self.num_levels]
        }
        prev_asks = {
            price_to_tick(lv.price, self.tick_size): lv.size
            for lv in prev_snap.asks[:self.num_levels]
        }

        deltas: List[LevelDelta] = []

        # Process all price ticks that appeared in either snapshot (bids)
        all_bid_ticks = set(prev_bids.keys()) | set(self._bids.keys())
        for tick in all_bid_ticks:
            old_sz = prev_bids.get(tick, 0.0)
            new_sz = self._bids.get(tick, 0.0)
            if old_sz != new_sz:
                deltas.append(LevelDelta(
                    price=tick_to_price(tick, self.tick_size),
                    tick_key=tick,
                    side=Side.BUY,
                    old_size=old_sz,
                    new_size=new_sz,
                ))

        # Process all price ticks for asks
        all_ask_ticks = set(prev_asks.keys()) | set(self._asks.keys())
        for tick in all_ask_ticks:
            old_sz = prev_asks.get(tick, 0.0)
            new_sz = self._asks.get(tick, 0.0)
            if old_sz != new_sz:
                deltas.append(LevelDelta(
                    price=tick_to_price(tick, self.tick_size),
                    tick_key=tick,
                    side=Side.SELL,   # ASK side uses Side.SELL (the enum is BUY/SELL)
                    old_size=old_sz,
                    new_size=new_sz,
                ))

        return deltas

    # -- Convenience properties -----------------------------------------------

    @property
    def best_bid_price(self) -> Optional[float]:
        if not self._bids:
            return None
        return tick_to_price(max(self._bids.keys()), self.tick_size)

    @property
    def best_ask_price(self) -> Optional[float]:
        if not self._asks:
            return None
        return tick_to_price(min(self._asks.keys()), self.tick_size)

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid_price, self.best_ask_price
        if b is not None and a is not None:
            return 0.5 * (b + a)
        return None

    @property
    def spread(self) -> Optional[float]:
        b, a = self.best_bid_price, self.best_ask_price
        if b is not None and a is not None:
            return a - b
        return None

    def size_at_price(self, side: Side, price: float) -> float:
        """Return visible size at an exact price level (0.0 if absent)."""
        tick = price_to_tick(price, self.tick_size)
        book = self._bids if side == Side.BUY else self._asks
        return book.get(tick, 0.0)

    def levels_sorted(self, side: Side) -> List[Tuple[float, float]]:
        """Return list of (price, size) sorted by best-first."""
        book = self._bids if side == Side.BUY else self._asks
        if side == Side.BUY:
            return [(tick_to_price(t, self.tick_size), s)
                    for t, s in sorted(book.items(), reverse=True)]
        else:
            return [(tick_to_price(t, self.tick_size), s)
                    for t, s in sorted(book.items())]
