"""
microalpha/diagnostics/reports.py
===================================
Console and text diagnostic reporter for backtest runs.

Produces human-readable reports combining:
  - InvariantReport (pass/fail count, failure details)
  - Performance metrics summary (Sharpe, drawdown, etc.)
  - Engine mechanics stats (fills, cancels, latency distribution)
  - PnL time series statistics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from microalpha.diagnostics.invariants import InvariantReport
from microalpha.pnl.metrics import summarize

if TYPE_CHECKING:
    from microalpha.engine.matching import MatchingEngine


def print_backtest_report(engine:           "MatchingEngine",
                          pnl_df:           pd.DataFrame,
                          inv_report:       InvariantReport,
                          strategy_name:    str = "strategy",
                          capital:          float = 100_000.0,
                          annual_periods:   float = 252 * 78000) -> None:
    """
    Print a comprehensive backtest diagnostic report to stdout.
    """

    metrics = summarize(pnl_df, engine.logs, capital, annual_periods)

    sep = "=" * 70
    sep2 = "-" * 70

    print(f"\n{sep}")
    print(f"  MICROALPHA BACKTEST REPORT — Strategy: {strategy_name}")
    print(f"{sep}")

    # ---- Invariants ---------------------------------------------------------
    print(f"\n{'INVARIANT CHECKS':}")
    print(sep2)
    print(inv_report.summary())

    # ---- PnL summary --------------------------------------------------------
    print(f"\n\nPnL SUMMARY")
    print(sep2)
    if not pnl_df.empty:
        fmt = "  {:35s}: {:>12.4f}"
        print(fmt.format("Net PnL ($)", metrics.get("net_pnl", float("nan"))))
        print(fmt.format("Realized PnL ($)", metrics.get("realized_pnl", float("nan"))))
        print(fmt.format("Unrealized PnL ($)", metrics.get("unrealized_pnl", float("nan"))))
        print(fmt.format("Cumulative Fees ($)", metrics.get("cum_fees", float("nan"))))
        print(fmt.format("Final Position (shares)", metrics.get("final_position", float("nan"))))
    else:
        print("  [No PnL data — no fills occurred]")

    # ---- Risk metrics -------------------------------------------------------
    print(f"\nRISK METRICS")
    print(sep2)
    fmt = "  {:35s}: {:>12.4f}"
    print(fmt.format("Sharpe (annualized)", metrics.get("sharpe", float("nan"))))
    print(fmt.format("Sortino (annualized)", metrics.get("sortino", float("nan"))))
    print(fmt.format("Max Drawdown", metrics.get("max_drawdown", float("nan"))))
    print(fmt.format("Calmar Ratio", metrics.get("calmar", float("nan"))))

    # ---- Fill mechanics -----------------------------------------------------
    print(f"\nFILL MECHANICS")
    print(sep2)
    print(fmt.format("Placed Qty (shares)", metrics.get("placed_qty", 0)))
    print(fmt.format("Filled Qty (shares)", metrics.get("filled_qty", 0)))
    print(fmt.format("Fill Rate", metrics.get("fill_rate", 0)))
    print(fmt.format("Avg Fill Latency (ms)", metrics.get("avg_fill_latency_ms", float("nan"))))
    print(fmt.format("Avg Adverse Score", metrics.get("avg_adverse_score", float("nan"))))
    print(fmt.format("% Passive Fills", metrics.get("pct_passive_fills", float("nan"))))
    print(fmt.format("Total Passive Fills", int(metrics.get("total_passive_fills", 0))))
    print(fmt.format("Total Aggressive Fills", int(metrics.get("total_aggressive_fills", 0))))

    # ---- Engine mechanics ---------------------------------------------------
    print(f"\nENGINE MECHANICS")
    print(sep2)
    m = engine.metrics
    print(fmt.format("Ticks Processed", m.get("ticks_processed", 0)))
    print(fmt.format("Total Fills", m.get("total_fills", 0)))
    print(fmt.format("Passive Fills (engine)", m.get("passive_fills", 0)))
    print(fmt.format("Aggressive Fills (engine)", m.get("aggressive_fills", 0)))
    print(fmt.format("Cancels", m.get("cancels", 0)))
    print(fmt.format("Rejections", m.get("rejections", 0)))

    # ---- PnL time series statistics ----------------------------------------
    if not pnl_df.empty and len(pnl_df) > 1:
        rets = pnl_df["total_pnl"].diff().dropna()
        print(f"\nPnL RETURNS STATISTICS")
        print(sep2)
        print(fmt.format("Mean tick return ($)", float(rets.mean())))
        print(fmt.format("Std tick return ($)", float(rets.std())))
        print(fmt.format("Min tick return ($)", float(rets.min())))
        print(fmt.format("Max tick return ($)", float(rets.max())))
        print(fmt.format("Num positive ticks", int((rets > 0).sum())))
        print(fmt.format("Num negative ticks", int((rets < 0).sum())))

    print(f"\n{sep}\n")
