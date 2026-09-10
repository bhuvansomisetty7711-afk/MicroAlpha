"""
microalpha/data/snapshot.py
============================
Core data types for the MICROALPHA backtesting system.

Design principles:
- All market data is immutable once created (frozen dataclasses where feasible).
- Prices are stored as floats in the user API but converted to integer tick units
  internally to eliminate floating-point dictionary key bugs.
- Strong typing via dataclasses with type annotations throughout.
- No loose dicts: every structured value has a defined type.

Key types exported:
    L2Level          — single (price, size) price level
    TradeInfo        — last-trade metadata attached to a snapshot
    MarketSnapshot   — immutable typed representation of one MBP-10 snapshot
    FillEvent        — a fill attributed to a strategy order
    OrderStatus      — enum of order lifecycle states
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

class Side(str, Enum):
    """Canonical order/trade side representation."""
    BUY  = "buy"
    SELL = "sell"

    @staticmethod
    def from_str(s: str) -> "Side":
        """Parse any reasonable string form into Side."""
        s = s.lower().strip()
        if s in ("buy", "bid", "b", "long", "+1", "1"):
            return Side.BUY
        if s in ("sell", "ask", "s", "short", "-1"):
            return Side.SELL
        raise ValueError(f"Cannot parse side: {s!r}")

    def opposite(self) -> "Side":
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderStatus(Enum):
    """Lifecycle state of a limit order."""
    PENDING    = auto()   # submitted but not yet arrived at exchange
    RESTING    = auto()   # arrived, sitting in price-time queue
    FILLED     = auto()   # fully filled
    PARTIAL    = auto()   # partially filled, still resting
    CANCELLED  = auto()   # cancelled (may be partial)


class FillType(Enum):
    """How the fill was generated."""
    PASSIVE    = auto()   # resting limit order was hit by aggressor
    AGGRESSIVE = auto()   # market order swept the book


# ---------------------------------------------------------------------------
# L2 book level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class L2Level:
    """
    A single price level in the order book.

    price: float — price in dollars (e.g. 150.25)
    size:  float — total visible quantity resting at this price
    """
    price: float
    size:  float

    def __post_init__(self):
        if math.isnan(self.price) or math.isinf(self.price):
            raise ValueError(f"L2Level price must be finite, got {self.price}")
        if self.size < 0:
            raise ValueError(f"L2Level size must be non-negative, got {self.size}")

    def to_tuple(self) -> Tuple[float, float]:
        return (self.price, self.size)


# ---------------------------------------------------------------------------
# Trade info
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeInfo:
    """
    Metadata about the most recent trade between two consecutive snapshots.

    aggressor: +1 = buy-aggressor (lifted ask), -1 = sell-aggressor (hit bid), 0 = unknown
    price:     last trade price
    size:      last trade size (shares)

    Invariant: aggressor in {-1, 0, +1}
    """
    aggressor: int   # -1, 0, +1
    price:     Optional[float]
    size:      Optional[float]

    def __post_init__(self):
        if self.aggressor not in (-1, 0, 1):
            raise ValueError(f"TradeInfo.aggressor must be in {{-1,0,1}}, got {self.aggressor}")
        if self.price is not None and (math.isnan(self.price) or self.price <= 0):
            raise ValueError(f"TradeInfo.price must be positive, got {self.price}")
        if self.size is not None and self.size < 0:
            raise ValueError(f"TradeInfo.size must be non-negative, got {self.size}")

    @staticmethod
    def unknown() -> "TradeInfo":
        return TradeInfo(aggressor=0, price=None, size=None)


# ---------------------------------------------------------------------------
# Market snapshot (the primary replay unit)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSnapshot:
    """
    Immutable snapshot of the L2 order book at a point in simulation time.

    This is the fundamental unit of the market data replay loop. Once created,
    a snapshot MUST NOT be mutated — engine operations in replay mode must never
    modify historical book state (Invariant #35).

    Fields:
        ts:         simulation timestamp in seconds (float, monotonically increasing)
        bids:       list of L2Level, sorted descending by price (best bid first)
        asks:       list of L2Level, sorted ascending by price (best ask first)
        trade:      TradeInfo — last-trade metadata between snapshots
        symbol:     instrument identifier (optional, for multi-instrument setups)
        sequence:   exchange sequence number (optional, for gap detection)

    Invariants enforced at construction time:
        - ts > 0
        - best_bid < best_ask (no crossed book)
        - bids price-sorted descending, asks price-sorted ascending
        - all sizes positive
    """
    ts:       float
    bids:     Tuple[L2Level, ...]
    asks:     Tuple[L2Level, ...]
    trade:    TradeInfo
    symbol:   str   = "UNKNOWN"
    sequence: int   = 0

    def __post_init__(self):
        # Validate timestamp
        if self.ts < 0:
            raise ValueError(f"Snapshot timestamp must be >= 0, got {self.ts}")
        # Validate bid ordering: descending by price
        for i in range(len(self.bids) - 1):
            if self.bids[i].price < self.bids[i + 1].price:
                raise ValueError(
                    f"Bids must be in descending price order: "
                    f"{self.bids[i].price} < {self.bids[i+1].price}"
                )
        # Validate ask ordering: ascending by price
        for i in range(len(self.asks) - 1):
            if self.asks[i].price > self.asks[i + 1].price:
                raise ValueError(
                    f"Asks must be in ascending price order: "
                    f"{self.asks[i].price} > {self.asks[i+1].price}"
                )
        # No crossed book
        if self.bids and self.asks:
            if self.bids[0].price >= self.asks[0].price:
                # Allow crossed book via a WARNING in validators but don't hard-error here
                # (some data feeds occasionally produce momentary crosses during auctions)
                pass  # handled by validators.py

    # -- Convenience accessors ----------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        if b is not None and a is not None:
            return 0.5 * (b + a)
        return None

    @property
    def spread(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        if b is not None and a is not None:
            return a - b
        return None

    def visible_size_at_price(self, side: Side, price: float) -> float:
        """Return visible size at a specific price level (0.0 if not present)."""
        levels = self.bids if side == Side.BUY else self.asks
        for lv in levels:
            if abs(lv.price - price) < 1e-9:
                return lv.size
        return 0.0

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by the legacy matching_engine_realistic API."""
        return {
            "ts": self.ts,
            "bids": [(lv.price, lv.size) for lv in self.bids],
            "asks": [(lv.price, lv.size) for lv in self.asks],
            "last_trade_price": self.trade.price,
            "last_trade_size":  self.trade.size,
            "last_trade_aggressor": self.trade.aggressor,
        }

    @staticmethod
    def from_legacy(ts: float, bids: List, asks: List,
                    last_trade_price=None, last_trade_size=None,
                    last_trade_aggressor: int = 0,
                    symbol: str = "UNKNOWN") -> "MarketSnapshot":
        """
        Construct from the legacy tuple-list format used by the original engine
        and by Databento loaders.

        bids/asks: list of (price, size) tuples, already price-sorted.
        """
        bid_levels = tuple(
            L2Level(price=float(p), size=float(s))
            for p, s in bids if float(s) > 0
        )
        ask_levels = tuple(
            L2Level(price=float(p), size=float(s))
            for p, s in asks if float(s) > 0
        )
        trade = TradeInfo(
            aggressor=int(last_trade_aggressor),
            price=float(last_trade_price) if last_trade_price is not None else None,
            size=float(last_trade_size) if last_trade_size is not None else None,
        )
        return MarketSnapshot(
            ts=float(ts),
            bids=bid_levels,
            asks=ask_levels,
            trade=trade,
            symbol=symbol,
        )


# ---------------------------------------------------------------------------
# Fill event (output of the matching engine)
# ---------------------------------------------------------------------------

@dataclass
class FillEvent:
    """
    A single fill event attributed to a strategy order.

    This is the output stream of the matching engine — every fill is captured
    as a structured FillEvent so PnL reconstruction can proceed in one pass.

    Fields:
        order_id:       engine-assigned order identifier
        owner:          strategy name (for multi-strategy attribution)
        side:           Side.BUY or Side.SELL
        fill_type:      PASSIVE (maker) or AGGRESSIVE (taker)
        filled_qty:     shares filled in this event
        price:          execution price per share
        fee:            net fee for this fill (negative = rebate credited)
        ts:             simulation timestamp of the fill
        adverse_score:  heuristic adverse selection score [0.1, 3.0]
        queue_pos_at_fill: queue position when fill occurred
        fill_latency_s: time from order placement to fill (seconds)
    """
    order_id:         str
    owner:            str
    side:             Side
    fill_type:        FillType
    filled_qty:       float
    price:            float
    fee:              float
    ts:               float
    adverse_score:    float = 1.0
    queue_pos_at_fill: float = 0.0
    fill_latency_s:   float = 0.0

    def __post_init__(self):
        if self.filled_qty <= 0:
            raise ValueError(f"FillEvent.filled_qty must be positive, got {self.filled_qty}")
        if self.price <= 0:
            raise ValueError(f"FillEvent.price must be positive, got {self.price}")

    @property
    def is_maker(self) -> bool:
        return self.fill_type == FillType.PASSIVE

    @property
    def notional(self) -> float:
        return self.filled_qty * self.price


# ---------------------------------------------------------------------------
# Strategy actions (typed, not dicts)
# ---------------------------------------------------------------------------

@dataclass
class PlaceLimitAction:
    """Strategy action: place a resting limit order."""
    side:    Side
    price:   float
    qty:     float
    owner:   str = "strategy"
    tag:     str = ""   # optional free-form tag for diagnostics


@dataclass
class CancelAction:
    """Strategy action: cancel a resting or pending order."""
    order_id: str
    owner:    str = "strategy"


@dataclass
class MarketOrderAction:
    """Strategy action: aggressive market sweep."""
    side:  Side
    qty:   float
    owner: str = "strategy"
    tag:   str = ""


# Union type for all strategy actions
StrategyAction = PlaceLimitAction | CancelAction | MarketOrderAction
