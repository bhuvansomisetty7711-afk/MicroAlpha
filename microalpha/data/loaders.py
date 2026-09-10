"""
microalpha/data/loaders.py
===========================
Market data loaders for common formats.

Currently supports:
  - Databento MBP-10 format (Parquet or DataFrame)
      Columns: ts_event, bid_px_00..bid_px_09, bid_sz_00..bid_sz_09,
               ask_px_00..ask_px_09, ask_sz_00..ask_sz_09,
               last_trade_px (optional), last_trade_sz (optional),
               side (optional: A=ask-aggressor, B=bid-aggressor)
  - Generic CSV / DataFrame loader (user-configurable column mapping)
  - Synthetic data generator (for testing, no real market data required)

All loaders return List[MarketSnapshot] sorted by ts ascending.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from microalpha.data.snapshot import (
    L2Level, MarketSnapshot, Side, TradeInfo
)
from microalpha.data.validators import SnapshotValidator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Databento MBP-10 loader
# ---------------------------------------------------------------------------

class DatabentoMBP10Loader:
    """
    Loads Databento MBP-10 (Market-By-Price, top 10 levels) data.

    The column layout expected is the standard Databento flat CSV/Parquet format:
        ts_event        — nanosecond timestamp (int64) or seconds float
        bid_px_00       — best bid price (fixed-point, divide by 1e9 for USD)
        bid_sz_00       — best bid size
        ...
        bid_px_09, bid_sz_09
        ask_px_00..ask_px_09, ask_sz_00..ask_sz_09

    Databento prices are stored as integers (fixed-point, 1e9 scale).
    If prices look like raw integers > 1000, they are auto-detected and divided by 1e9.

    Usage:
        loader = DatabentoMBP10Loader(symbol="AAPL")
        snapshots = loader.load("aapl_mbp10.parquet")
        # or:
        snapshots = loader.from_dataframe(df)
    """

    NUM_LEVELS = 10

    def __init__(self,
                 symbol:         str   = "UNKNOWN",
                 ts_unit:        str   = "auto",   # "ns", "us", "s", "auto"
                 price_scale:    float = None,     # None = auto-detect
                 drop_zero_size: bool  = True,
                 validate:       bool  = True):
        """
        ts_unit:     timestamp unit. "auto" detects nanoseconds vs seconds.
        price_scale: multiply raw prices by this to get USD. None = auto-detect.
        """
        self.symbol         = symbol
        self.ts_unit        = ts_unit
        self.price_scale    = price_scale
        self.drop_zero_size = drop_zero_size
        self.validate       = validate

    def load(self, path: str) -> List[MarketSnapshot]:
        """Load from a Parquet or CSV file."""
        if path.endswith(".parquet") or path.endswith(".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        return self.from_dataframe(df)

    def from_dataframe(self, df: pd.DataFrame) -> List[MarketSnapshot]:
        """Convert a DataFrame into a sorted List[MarketSnapshot]."""
        df = df.copy()
        snapshots = []
        errors = 0

        # Detect timestamp column
        ts_col = self._detect_ts_column(df)
        ts_arr = self._parse_timestamps(df[ts_col].values)

        # Detect price scale
        scale = self._detect_price_scale(df)

        for i in range(len(df)):
            try:
                row = df.iloc[i]
                ts = float(ts_arr[i])

                bids = self._extract_levels(row, "bid", scale)
                asks = self._extract_levels(row, "ask", scale)

                if not bids and not asks:
                    errors += 1
                    continue

                trade = self._extract_trade(row, scale)
                snap = MarketSnapshot(
                    ts=ts,
                    bids=tuple(bids),
                    asks=tuple(asks),
                    trade=trade,
                    symbol=self.symbol,
                    sequence=int(i),
                )
                snapshots.append(snap)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning("Row %d parse error: %s", i, e)

        log.info("Loaded %d snapshots from %d rows (%d errors)",
                 len(snapshots), len(df), errors)

        # Sort by timestamp (enforce monotonicity)
        snapshots.sort(key=lambda s: (s.ts, s.sequence))

        if self.validate:
            validator = SnapshotValidator()
            results = validator.validate_all(snapshots, drop_errors=False)
            snapshots = [r.snapshot for r in results if r.is_ok]
            log.info("After validation: %d valid snapshots. Stats: %s",
                     len(snapshots), validator.stats)

        return snapshots

    def _detect_ts_column(self, df: pd.DataFrame) -> str:
        candidates = ["ts_event", "timestamp", "ts", "time", "datetime"]
        for c in candidates:
            if c in df.columns:
                return c
        # last resort: first column
        return df.columns[0]

    def _parse_timestamps(self, ts_raw) -> np.ndarray:
        """Convert timestamps to seconds (float64)."""
        arr = np.asarray(ts_raw, dtype=np.float64)
        if self.ts_unit == "auto":
            # Heuristic: nanoseconds if median > 1e15, microseconds if > 1e12, else seconds
            med = np.nanmedian(arr[arr > 0])
            if med > 1e15:
                return arr * 1e-9      # nanoseconds → seconds
            elif med > 1e12:
                return arr * 1e-6     # microseconds → seconds
            elif med > 1e9:
                return arr * 1e-3     # milliseconds → seconds
            else:
                return arr             # already seconds
        elif self.ts_unit == "ns":
            return arr * 1e-9
        elif self.ts_unit == "us":
            return arr * 1e-6
        elif self.ts_unit == "ms":
            return arr * 1e-3
        else:
            return arr

    def _detect_price_scale(self, df: pd.DataFrame) -> float:
        if self.price_scale is not None:
            return self.price_scale
        # Check first bid price: if > 10_000, assume fixed-point 1e9
        col = "bid_px_00" if "bid_px_00" in df.columns else None
        if col and len(df) > 0:
            sample = float(df[col].iloc[0])
            if sample > 10_000:
                return 1e-9  # Databento fixed-point
        return 1.0  # already in price units

    def _extract_levels(self, row, side: str, scale: float) -> List[L2Level]:
        """Extract up to NUM_LEVELS price levels for one side."""
        levels = []
        for i in range(self.NUM_LEVELS):
            px_col = f"{side}_px_{i:02d}"
            sz_col = f"{side}_sz_{i:02d}"
            if px_col not in row.index or sz_col not in row.index:
                break
            px = float(row[px_col]) * scale
            sz = float(row[sz_col])
            if math.isnan(px) or math.isnan(sz):
                break
            if px <= 0:
                break
            if self.drop_zero_size and sz <= 0:
                continue
            levels.append(L2Level(price=px, size=sz))
        return levels

    def _extract_trade(self, row, scale: float) -> TradeInfo:
        """Extract last-trade metadata from a row."""
        aggressor = 0
        price = None
        size = None

        # aggressor: Databento 'side' column: A=ask-side aggressor, B=bid-side aggressor
        for col in ["side", "aggressor", "last_trade_aggressor", "trade_side"]:
            if col in row.index:
                val = row[col]
                if isinstance(val, str):
                    val = val.upper()
                    if val in ("A", "ASK", "-1", "SELL"):
                        aggressor = -1
                    elif val in ("B", "BID", "1", "BUY"):
                        aggressor = 1
                elif isinstance(val, (int, float)) and not math.isnan(val):
                    aggressor = int(val)
                break

        for col in ["last_trade_px", "last_trade_price", "trade_price", "last_price"]:
            if col in row.index:
                val = float(row[col])
                if not math.isnan(val) and val > 0:
                    price = val * scale
                break

        for col in ["last_trade_sz", "last_trade_size", "trade_size", "last_size"]:
            if col in row.index:
                val = float(row[col])
                if not math.isnan(val) and val >= 0:
                    size = val
                break

        return TradeInfo(aggressor=aggressor, price=price, size=size)


# ---------------------------------------------------------------------------
# Generic CSV loader with column mapping
# ---------------------------------------------------------------------------

class GenericCSVLoader:
    """
    Flexible loader for custom CSV/DataFrame formats.

    Specify a column_map dict mapping standard field names to your column names:
        {
          "ts": "my_timestamp_column",
          "bid_price_0": "Bid1",
          "bid_size_0": "BidSize1",
          ...
        }
    """

    def __init__(self, symbol: str = "UNKNOWN", column_map: Dict[str, str] = None,
                 ts_unit: str = "s", price_scale: float = 1.0, num_levels: int = 5):
        self.symbol      = symbol
        self.column_map  = column_map or {}
        self.ts_unit     = ts_unit
        self.price_scale = price_scale
        self.num_levels  = num_levels

    def load(self, path: str) -> List[MarketSnapshot]:
        df = pd.read_csv(path) if not path.endswith(".parquet") else pd.read_parquet(path)
        return self.from_dataframe(df)

    def from_dataframe(self, df: pd.DataFrame) -> List[MarketSnapshot]:
        snapshots = []
        cm = self.column_map
        sc = self.price_scale

        def col(name: str):
            return cm.get(name, name)

        for _, row in df.iterrows():
            ts = float(row.get(col("ts"), row.get(col("timestamp"), 0)))
            if self.ts_unit == "ns":  ts *= 1e-9
            elif self.ts_unit == "ms": ts *= 1e-3

            bids, asks = [], []
            for lvl in range(self.num_levels):
                bp = row.get(col(f"bid_price_{lvl}"))
                bs = row.get(col(f"bid_size_{lvl}"))
                ap = row.get(col(f"ask_price_{lvl}"))
                as_ = row.get(col(f"ask_size_{lvl}"))
                if bp is not None and bs is not None:
                    bids.append(L2Level(price=float(bp)*sc, size=float(bs)))
                if ap is not None and as_ is not None:
                    asks.append(L2Level(price=float(ap)*sc, size=float(as_)))

            trade = TradeInfo.unknown()
            snapshots.append(MarketSnapshot(ts=ts, bids=tuple(bids),
                                            asks=tuple(asks), trade=trade,
                                            symbol=self.symbol))

        snapshots.sort(key=lambda s: s.ts)
        return snapshots


# ---------------------------------------------------------------------------
# Synthetic data generator (for tests/demos — no real data required)
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates a realistic synthetic L2 order book feed for testing.

    Simulates a mean-reverting mid price with realistic bid/ask spread,
    level depth, and trade activity. Useful for running complete backtest
    demos without requiring proprietary market data.

    Usage:
        gen = SyntheticDataGenerator(n_ticks=5000, seed=42)
        snapshots = gen.generate()
    """

    def __init__(self,
                 n_ticks:        int   = 5000,
                 initial_mid:    float = 150.0,
                 tick_size:      float = 0.01,
                 spread_ticks:   float = 2.0,
                 vol_per_tick:   float = 0.002,
                 base_depth:     float = 500.0,
                 tick_interval_s: float = 0.005,
                 num_levels:     int   = 5,
                 trade_prob:     float = 0.3,
                 symbol:         str   = "SYNTHETIC",
                 seed:           int   = 42):
        self.n_ticks         = n_ticks
        self.initial_mid     = initial_mid
        self.tick_size       = tick_size
        self.spread_ticks    = spread_ticks
        self.vol_per_tick    = vol_per_tick
        self.base_depth      = base_depth
        self.tick_interval_s = tick_interval_s
        self.num_levels      = num_levels
        self.trade_prob      = trade_prob
        self.symbol          = symbol
        self.rng             = np.random.RandomState(seed)

    def generate(self) -> List[MarketSnapshot]:
        """Generate and return a sorted list of synthetic snapshots."""
        snapshots = []
        mid = self.initial_mid
        ts = 0.0

        for i in range(self.n_ticks):
            # Evolve mid price: random walk with mean reversion
            shock = self.rng.normal(0, self.vol_per_tick * mid)
            mean_rev = -0.01 * (mid - self.initial_mid)  # gentle mean-reversion
            mid = max(self.initial_mid * 0.5, mid + shock + mean_rev)
            mid = round(mid / self.tick_size) * self.tick_size

            half_spread = self.spread_ticks * self.tick_size / 2.0
            best_bid = mid - half_spread
            best_ask = mid + half_spread

            # Generate depth on each side
            bids, asks = [], []
            for lvl in range(self.num_levels):
                bid_px = best_bid - lvl * self.tick_size
                ask_px = best_ask + lvl * self.tick_size
                # Depth decays with level
                depth_mul = self.rng.uniform(0.5, 2.0) * (0.7 ** lvl)
                depth = max(1.0, self.base_depth * depth_mul)
                bids.append(L2Level(price=round(bid_px, 4), size=round(depth)))
                asks.append(L2Level(price=round(ask_px, 4), size=round(depth)))

            # Random trade activity
            trade = TradeInfo.unknown()
            if self.rng.random() < self.trade_prob:
                direction = int(self.rng.choice([-1, 1]))
                trade_price = best_ask if direction == 1 else best_bid
                trade_size = float(self.rng.exponential(scale=100))
                trade = TradeInfo(aggressor=direction, price=round(trade_price, 4),
                                  size=round(trade_size))

            # Timestamp with small jitter
            ts += self.tick_interval_s * (1.0 + self.rng.uniform(-0.1, 0.2))

            snap = MarketSnapshot(
                ts=round(ts, 9),
                bids=tuple(bids),
                asks=tuple(asks),
                trade=trade,
                symbol=self.symbol,
                sequence=i,
            )
            snapshots.append(snap)

        log.info("SyntheticDataGenerator: produced %d snapshots over %.1fs",
                 len(snapshots), snapshots[-1].ts if snapshots else 0.0)
        return snapshots
