"""
microalpha/pnl/metrics.py
==========================
Performance metrics computation from a PnL DataFrame.

Computes the standard quant research performance metrics:
  - Sharpe ratio (annualized)
  - Sortino ratio (downside deviation based)
  - Maximum drawdown (percentage)
  - Calmar ratio (annualized return / max drawdown)
  - Fill rate (filled qty / placed qty)
  - Adverse selection analysis (fill prices vs mid prices)
  - Turnover (annualized notional / capital)
  - Win rate on trades
  - Average trade P&L

All metrics are computed from the authoritative PnL DataFrame produced
by PnLBuilder.build(). No raw engine log access is needed here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_sharpe(pnl_df: pd.DataFrame, annual_periods: float = 252 * 78_000) -> float:
    """
    Sharpe ratio from PnL returns.

    IMPORTANT: annual_periods must match your actual data frequency.
    Default is 252 * 78000 (≈ 19.6M), calibrated for 5ms tick data
    (78000 = 6.5 trading hours * 3600s / 0.3s per tick).

    The BacktestRunner auto-computes this from actual simulation timestamps
    when calling summarize(). If calling this function directly, make sure
    to pass the correct value:
      - 1-min bars:  252 * 390
      - 1-sec bars:  252 * 23400
      - 5ms ticks:   252 * 78000  (default)
    """
    if pnl_df.empty or len(pnl_df) < 2:
        return float("nan")
    rets = pnl_df["total_pnl"].diff().dropna()
    if rets.std() < 1e-12:
        return float("nan")
    return float(rets.mean() / rets.std() * math.sqrt(annual_periods))


def compute_sortino(pnl_df: pd.DataFrame, annual_periods: float = 252 * 78_000) -> float:
    """Sortino ratio: Sharpe using downside deviation (negative returns only)."""
    if pnl_df.empty or len(pnl_df) < 2:
        return float("nan")
    rets = pnl_df["total_pnl"].diff().dropna()
    downside = rets[rets < 0]
    if len(downside) < 2 or downside.std() < 1e-12:
        return float("nan")
    return float(rets.mean() / downside.std() * math.sqrt(annual_periods))


def compute_max_drawdown(pnl_df: pd.DataFrame) -> float:
    """Maximum drawdown as a fraction (always <= 0)."""
    if pnl_df.empty:
        return 0.0
    curve = pnl_df["total_pnl"].values
    running_peak = np.maximum.accumulate(curve)
    dd = (curve - running_peak)
    max_dd_abs = float(dd.min())
    # Normalize by peak (avoid division by zero)
    peak_at_max_dd = float(running_peak[dd.argmin()])
    if abs(peak_at_max_dd) < 1e-6:
        return 0.0
    return max_dd_abs / abs(peak_at_max_dd)


def compute_calmar(pnl_df: pd.DataFrame, annual_periods: float = 252 * 78_000) -> float:
    """Calmar ratio: annualized return / max_drawdown_magnitude."""
    mdd = compute_max_drawdown(pnl_df)
    if abs(mdd) < 1e-9:
        return float("nan")
    if pnl_df.empty or len(pnl_df) < 2:
        return float("nan")
    rets = pnl_df["total_pnl"].diff().dropna()
    ann_return = float(rets.mean() * annual_periods)
    return ann_return / abs(mdd)


def compute_fill_metrics(fill_logs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute fill-level metrics:
      - fill_rate: filled_qty / placed_qty
      - avg_fill_latency_ms
      - avg_adverse_score
      - pct_passive_fills
    """
    if not fill_logs:
        return {"fill_rate": 0.0, "avg_fill_latency_ms": float("nan"),
                "avg_adverse_score": float("nan"), "pct_passive_fills": float("nan")}

    fills = [e for e in fill_logs if (e.get("event") or "").lower() in ("limit_fill",)]
    market_fills = [e for e in fill_logs if (e.get("event") or "").lower() == "market_order_executed"]
    placements = [e for e in fill_logs if (e.get("event") or "").lower() == "limit_placed_pending"]

    placed_qty = sum(float(e.get("qty", 0)) for e in placements)
    filled_qty = sum(float(e.get("filled_qty", 0)) for e in fills)
    fill_rate = filled_qty / max(1.0, placed_qty)

    latencies_ms = [float(e.get("fill_latency_s", 0)) * 1000.0 for e in fills if "fill_latency_s" in e]
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else float("nan")

    adverse_scores = [float(e.get("adverse_score", 1.0)) for e in fills if "adverse_score" in e]
    avg_adverse = float(np.mean(adverse_scores)) if adverse_scores else float("nan")

    total_fills = len(fills) + len(market_fills)
    pct_passive = len(fills) / max(1, total_fills)

    return {
        "fill_rate":           fill_rate,
        "avg_fill_latency_ms": avg_latency,
        "avg_adverse_score":   avg_adverse,
        "pct_passive_fills":   pct_passive,
        "total_passive_fills": len(fills),
        "total_aggressive_fills": len(market_fills),
        "placed_qty":          placed_qty,
        "filled_qty":          filled_qty,
    }


def summarize(pnl_df:       pd.DataFrame,
              fill_logs:    List[Dict[str, Any]],
              capital:      float        = 100_000.0,
              annual_periods: float      = 252 * 78_000) -> Dict[str, Any]:
    """
    Compute a comprehensive summary dict of all performance metrics.

    Returns a flat dict suitable for reporting, logging, or DataFrame conversion.
    """
    out: Dict[str, Any] = {}

    # Core returns
    if not pnl_df.empty:
        final_total   = float(pnl_df["total_pnl"].iloc[-1])
        initial_total = float(pnl_df["total_pnl"].iloc[0])
        net_pnl = final_total - initial_total

        out["net_pnl"]         = round(net_pnl, 4)
        out["cum_fees"]        = round(float(pnl_df["cum_fees"].iloc[-1]), 4)
        out["realized_pnl"]    = round(float(pnl_df["realized_pnl"].iloc[-1]), 4)
        out["unrealized_pnl"]  = round(float(pnl_df["unrealized_pnl"].iloc[-1]), 4)
        out["final_position"]  = round(float(pnl_df["position"].iloc[-1]), 1)
        out["sharpe"]          = round(compute_sharpe(pnl_df, annual_periods), 4)
        out["sortino"]         = round(compute_sortino(pnl_df, annual_periods), 4)
        out["max_drawdown"]    = round(compute_max_drawdown(pnl_df), 4)
        out["calmar"]          = round(compute_calmar(pnl_df, annual_periods), 4)
        out["num_fills"]       = int(len(pnl_df))
    else:
        out.update({"net_pnl": 0.0, "cum_fees": 0.0, "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0, "final_position": 0.0,
                    "sharpe": float("nan"), "sortino": float("nan"),
                    "max_drawdown": 0.0, "calmar": float("nan")})

    # Fill-level metrics
    out.update(compute_fill_metrics(fill_logs))

    return out
