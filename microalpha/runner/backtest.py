"""
microalpha/runner/backtest.py
==============================
BacktestRunner: the central orchestrator of the MICROALPHA simulation loop.

Correct tick ordering (from architecture doc Section 2, Pattern 1):
  For each snapshot T:
    1. engine.ingest_snapshot(snap)           → updates book, decays queues,
                                                processes arrivals/cancels
    2. strategy.update_mid_return(mid)        → feeds signal to strategy
    3. [optional] model inference              → ml_score = model_fn(features)
    4. strategy.on_tick(engine, snap)         → strategy decision returns actions
    5. Translate actions → engine API calls   → place/cancel/market orders
    6. deltas = engine.compute_deltas()       → level-by-level consumption
    7. engine.simulate_passive(deltas)        → passive fills from mkt activity
    8. Attribute fills → strategy.on_fill()   → strategy reaction
    9. Update strategy.current_inventory      → single authoritative path

After all ticks:
    10. pnl_df = PnLBuilder.build(engine.logs)
    11. InvariantChecker.check_all(engine, pnl_df)
    12. Metrics = summarize(pnl_df, engine.logs)

Key invariants maintained by the runner:
  #6:  strategy.on_tick sees book at T, not T+1 (correct ordering)
  #7:  ML features computed only from data available at T
  #23: position updated by runner's single authority (not strategy)
  #29: avg_cost update is atomic with fill processing
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from microalpha.data.snapshot import (
    CancelAction, MarketOrderAction, MarketSnapshot, PlaceLimitAction, Side
)
from microalpha.diagnostics.invariants import InvariantChecker, InvariantReport
from microalpha.diagnostics.reports import print_backtest_report
from microalpha.engine.fees import FeeConfig
from microalpha.engine.impact import ImpactConfig
from microalpha.engine.latency import LatencyConfig
from microalpha.engine.matching import MatchingEngine
from microalpha.engine.queue_model import QueueModelConfig
from microalpha.pnl.builder import PnLBuilder
from microalpha.pnl.metrics import summarize
from microalpha.strategy.base import BaseStrategy

log = logging.getLogger(__name__)


class BacktestRunner:
    """
    Orchestrates the tick-by-tick simulation loop.

    Usage:
        runner = BacktestRunner.from_configs(...)
        results = runner.run(snapshots, strategy, model_fn=my_ml_model)
        pnl_df  = results["pnl_df"]
        metrics = results["metrics"]
        inv_report = results["invariant_report"]
    """

    def __init__(self,
                 engine:           MatchingEngine,
                 initial_cash:     float = 100_000.0,
                 assert_invariants: bool = True,
                 verbose:          bool = True,
                 log_every_n_ticks: int = 500):
        self.engine             = engine
        self.initial_cash       = float(initial_cash)
        self.assert_invariants  = assert_invariants
        self.verbose            = verbose
        self.log_every_n_ticks  = log_every_n_ticks
        # Watermark for fill-event collection (index into engine.logs).
        # Prevents fills from being double-processed across ticks.
        self._fill_log_watermark: int = 0

    @classmethod
    def from_configs(cls,
                     latency_config:   LatencyConfig   = None,
                     queue_config:     QueueModelConfig = None,
                     fee_config:       FeeConfig        = None,
                     impact_config:    ImpactConfig     = None,
                     tick_size:        float            = 0.01,
                     num_levels:       int              = 10,
                     initial_cash:     float            = 100_000.0,
                     rng_seed:         int              = 42,
                     assert_invariants: bool            = True,
                     verbose:          bool             = True) -> "BacktestRunner":
        """Factory: build a BacktestRunner with a fresh MatchingEngine."""
        engine = MatchingEngine(
            latency_config=latency_config,
            queue_config=queue_config,
            fee_config=fee_config,
            impact_config=impact_config,
            tick_size=tick_size,
            num_levels=num_levels,
            rng_seed=rng_seed,
            assert_invariants=assert_invariants,
        )
        return cls(engine, initial_cash=initial_cash,
                   assert_invariants=assert_invariants, verbose=verbose)

    def run(self,
            snapshots: List[MarketSnapshot],
            strategy:  BaseStrategy,
            model_fn:  Optional[Callable[[MarketSnapshot, MatchingEngine, BaseStrategy], float]] = None
            ) -> Dict[str, Any]:
        """
        Run the full simulation across all snapshots.

        Parameters:
            snapshots:  Sorted list of MarketSnapshot objects
            strategy:   A BaseStrategy subclass instance
            model_fn:   Optional ML model callable:
                            (snapshot, engine, strategy) → float in [-1, 1]
                        If None, strategy uses its internal ml_score attribute.

        Returns a dict with:
            "pnl_df"         — pd.DataFrame from PnLBuilder
            "metrics"        — dict of performance metrics
            "invariant_report" — InvariantReport from InvariantChecker
            "engine"         — the engine reference (for inspection)
            "logs"           — raw engine event log
        """
        log.info("BacktestRunner: starting simulation over %d snapshots", len(snapshots))
        strategy.current_inventory = 0.0
        self._fill_log_watermark = 0    # reset watermark for this run
        n = len(snapshots)

        # =====================================================================
        # Main simulation loop
        # =====================================================================
        for i, snap in enumerate(snapshots):

            # --- 1. Ingest snapshot (decays queue, processes arrivals/cancels) ---
            self.engine.ingest_snapshot(snap)

            # --- 2. Update strategy mid-return buffer -------------------------
            if snap.mid is not None:
                strategy.update_mid_return(snap.mid)

            # --- 3. ML score injection (optional) -----------------------------
            if model_fn is not None:
                try:
                    ml_score = float(model_fn(snap, self.engine, strategy))
                    ml_score = max(-1.0, min(1.0, ml_score))  # invariant #55
                    strategy.ml_score = ml_score
                except Exception as exc:
                    log.warning("model_fn raised at tick %d: %s", i, exc)
                    strategy.ml_score = 0.0

            # --- 4. Strategy decision (sees book at time T, not T+1) ---------
            try:
                actions = strategy.on_tick(self.engine, snap)
            except Exception as exc:
                log.warning("strategy.on_tick raised at tick %d: %s", i, exc)
                actions = []

            # --- 5. Translate actions → engine API calls ---------------------
            for action in (actions or []):
                self._execute_action(action, strategy, snap.ts)

            # --- 6. Compute level-by-level deltas ----------------------------
            deltas = self.engine.compute_deltas()

            # --- 7. Simulate passive fills from market activity --------------
            self.engine.simulate_passive(deltas)

            # --- 8. Attribute fills, call strategy.on_fill() -----------------
            new_fill_logs = self._collect_new_fills(strategy)
            for fill_log in new_fill_logs:
                try:
                    strategy.on_fill(fill_log)
                except Exception as exc:
                    log.warning("strategy.on_fill raised at tick %d: %s", i, exc)

            # --- 9. Update strategy inventory (single authority) -------------
            for fill_log in new_fill_logs:
                self._update_inventory(strategy, fill_log)

            # ---- Progress logging -------------------------------------------
            if self.verbose and (i + 1) % self.log_every_n_ticks == 0:
                log.info("Tick %d/%d | t=%.3fs | mid=%.4f | inv=%.0f | log_sz=%d",
                         i + 1, n,
                         snap.ts or 0.0,
                         snap.mid or 0.0,
                         strategy.current_inventory,
                         len(self.engine.logs))

        log.info("Simulation complete. %d ticks, %d log events.",
                 len(snapshots), len(self.engine.logs))

        # =====================================================================
        # Post-run PnL reconstruction and invariant checks
        # =====================================================================
        pnl_df = PnLBuilder(initial_cash=self.initial_cash).build(
            self.engine.logs, assert_invariants=self.assert_invariants
        )

        inv_report = InvariantChecker().check_all(self.engine, pnl_df)

        # Auto-compute annual_periods from actual simulation time.
        # For 5ms tick data over 25s of sim time: 5000 ticks / 25s * 252 days * 23400s/day
        if n >= 2:
            total_sim_time_s = max(1e-6, snapshots[-1].ts - snapshots[0].ts)
            ticks_per_second = (n - 1) / total_sim_time_s
            trading_seconds_per_day = 6.5 * 3600  # 6.5h trading day
            annual_periods = ticks_per_second * trading_seconds_per_day * 252.0
        else:
            annual_periods = 252 * 78000  # safe default for HFT data

        metrics = summarize(pnl_df, self.engine.logs, capital=self.initial_cash,
                            annual_periods=annual_periods)

        if self.verbose:
            print_backtest_report(
                engine=self.engine,
                pnl_df=pnl_df,
                inv_report=inv_report,
                strategy_name=strategy.name,
                capital=self.initial_cash,
            )

        return {
            "pnl_df":           pnl_df,
            "metrics":          metrics,
            "invariant_report": inv_report,
            "engine":           self.engine,
            "logs":             self.engine.logs,
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _execute_action(self,
                         action:   Any,
                         strategy: BaseStrategy,
                         ts:       float) -> None:
        """Translate a typed strategy action to an engine API call."""
        engine = self.engine

        if isinstance(action, PlaceLimitAction):
            if action.qty <= 0:
                log.warning("Strategy placed 0-qty limit — skipping")
                return
            order_id = engine.place_limit_order(
                owner=action.owner or strategy.name,
                side=action.side,
                price=action.price,
                qty=action.qty,
                now_ts=ts,
            )
            # Track new order IDs in strategy if Kelly strategy (for stale-quote detection)
            if hasattr(strategy, "_active_order_prices") and order_id:
                strategy._active_order_prices[order_id] = action.price

        elif isinstance(action, CancelAction):
            success = engine.cancel_order(
                order_id=action.order_id,
                now_ts=ts,
            )
            if not success:
                log.debug("Cancel for unknown order_id %s", action.order_id)

        elif isinstance(action, MarketOrderAction):
            if action.qty <= 0:
                log.warning("Strategy placed 0-qty market order — skipping")
                return
            fill_events = engine.execute_market_order(
                owner=action.owner or strategy.name,
                side=action.side,
                qty=action.qty,
                now_ts=ts,
            )
            # Market fills are already logged in engine.logs — inventory updated below
            # via _collect_new_fills which includes market_order_executed events

    def _collect_new_fills(self,
                            strategy: BaseStrategy) -> List[Dict[str, Any]]:
        """
        Collect NEW fill events since the last call, attributed to the strategy.

        Uses a watermark (integer index into engine.logs) to guarantee each
        fill is returned exactly once. The watermark advances after every
        collection so that subsequent calls only scan new log entries.

        Attribution logic (in priority order):
          1. event.owner == strategy.name
          2. event.order_id found in strategy._active_order_prices (Kelly strat)

        This fix addresses the critical double-processing bug: without the
        watermark, _collect_new_fills would rescan ALL historical logs each
        tick, causing on_fill and _update_inventory to fire multiple times
        per fill, corrupting position and PnL tracking.
        """
        strategy_name = strategy.name
        new_fills = []

        # Only scan log entries added since the last watermark
        current_len = len(self.engine.logs)
        for idx in range(self._fill_log_watermark, current_len):
            entry = self.engine.logs[idx]
            ev = (entry.get("event") or "").lower()
            if ev not in ("limit_fill", "market_order_executed"):
                continue
            # Attribution by owner
            owner = entry.get("owner", "")
            if owner == strategy_name:
                new_fills.append(entry)
                continue
            # Attribution by order_id (for Kelly strategy stale-quote tracking)
            oid = entry.get("order_id", "")
            if hasattr(strategy, "_active_order_prices") and oid in strategy._active_order_prices:
                new_fills.append(entry)

        # Advance watermark so next call only sees new entries
        self._fill_log_watermark = current_len
        return new_fills

    def _update_inventory(self,
                           strategy:  BaseStrategy,
                           fill_log:  Dict[str, Any]) -> None:
        """
        Update strategy.current_inventory from a fill event.

        This is the single authoritative path for inventory updates (invariant #23).
        The strategy MUST NOT update current_inventory directly.

        Handles:
          - limit_fill: signed fill qty
          - market_order_executed: signed total filled qty
        """
        ev = (fill_log.get("event") or "").lower()

        if ev == "limit_fill":
            side = str(fill_log.get("side", "")).lower()
            qty  = float(fill_log.get("filled_qty") or fill_log.get("qty") or 0.0)
        elif ev == "market_order_executed":
            side = str(fill_log.get("side", "")).lower()
            qty  = float(fill_log.get("filled_qty") or 0.0)
        else:
            return

        if side in ("buy", "bid"):
            strategy.current_inventory += qty
        elif side in ("sell", "ask"):
            strategy.current_inventory -= qty

        # Hard inventory limit enforcement: clip (should never trigger if strategy respects limits)
        if hasattr(strategy, "max_inventory"):
            max_inv = float(strategy.max_inventory)
            strategy.current_inventory = max(-max_inv, min(max_inv, strategy.current_inventory))
