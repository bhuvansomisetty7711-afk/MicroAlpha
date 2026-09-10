"""
tests/integration/test_full_backtest.py
========================================
Integration tests: full end-to-end backtest runs with synthetic data
and validation of invariants, PnL conservation, and fill attribution.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from microalpha.data.loaders import SyntheticDataGenerator
from microalpha.engine.latency import LatencyConfig
from microalpha.engine.fees import FeeConfig
from microalpha.engine.queue_model import QueueModelConfig
from microalpha.runner.backtest import BacktestRunner
from microalpha.strategy.kelly import KellyConstrainedStrategy
from microalpha.strategy.avellaneda import AvellanedaStoikovStrategy


def build_runner(seed=42) -> BacktestRunner:
    return BacktestRunner.from_configs(
        latency_config=LatencyConfig(base_ms=2.0, jitter_ms=0.5),
        queue_config=QueueModelConfig(decay_lambda=0.5),
        fee_config=FeeConfig(fee_bps=0.30, maker_rebate_bps=-0.20, taker_fee_bps=0.30),
        tick_size=0.01,
        initial_cash=100_000.0,
        rng_seed=seed,
        assert_invariants=True,
        verbose=False,  # suppress output in tests
    )


def test_end_to_end_kelly():
    """Full run with Kelly strategy and synthetic data — checks invariants pass."""
    gen = SyntheticDataGenerator(n_ticks=500, seed=42)
    snapshots = gen.generate()

    runner = build_runner()
    strategy = KellyConstrainedStrategy(
        name="kelly_test",
        capital=100_000.0,
        max_inventory=500.0,
        max_shares_per_order=100,
        min_edge_threshold=1e-4,
        tick_size=0.01,
        rng_seed=42,
    )

    # Simple deterministic ML model: slight positive edge
    def trivial_model(snap, engine, strat):
        # Use recent mid returns to generate a signal
        arr = list(strat.recent_mid_returns)
        if not arr:
            return 0.0
        return float(0.1 * (1 if arr[-1] > 0 else -1))

    results = runner.run(snapshots, strategy, model_fn=trivial_model)

    pnl_df = results["pnl_df"]
    inv_report = results["invariant_report"]

    # Check that invariant report ran
    assert len(inv_report.results) > 0

    # Check no timing invariant failures
    timing_failures = [r for r in inv_report.failures if r.category == "timing"]
    assert len(timing_failures) == 0, f"Timing invariant failures: {timing_failures}"

    # Check no accounting invariant failures
    accounting_failures = [r for r in inv_report.failures if r.category == "accounting"]
    assert len(accounting_failures) == 0, f"Accounting invariant failures: {accounting_failures}"

    # Engine processed all ticks
    assert runner.engine.metrics["ticks_processed"] == len(snapshots)


def test_end_to_end_avellaneda():
    """Full run with Avellaneda-Stoikov market maker."""
    gen = SyntheticDataGenerator(n_ticks=300, seed=99)
    snapshots = gen.generate()

    runner = build_runner(seed=99)
    strategy = AvellanedaStoikovStrategy(
        name="as_mm",
        gamma=0.1,
        base_qty=50.0,
        max_inventory=500.0,
        tick_size=0.01,
    )

    results = runner.run(snapshots, strategy)
    inv_report = results["invariant_report"]

    timing_failures = [r for r in inv_report.failures if r.category == "timing"]
    accounting_failures = [r for r in inv_report.failures if r.category == "accounting"]
    assert len(timing_failures) == 0, f"Timing failures: {timing_failures}"
    assert len(accounting_failures) == 0, f"Accounting failures: {accounting_failures}"


def test_pnl_conservation_invariant():
    """Explicitly verify PnL conservation across all rows of a full run."""
    gen = SyntheticDataGenerator(n_ticks=200, seed=7)
    snapshots = gen.generate()

    runner = build_runner(seed=7)
    strategy = AvellanedaStoikovStrategy(
        name="test_as",
        base_qty=100.0,
        max_inventory=300.0,
        tick_size=0.01,
        quote_lifetime_ms=100.0,  # fast refresh → more fills
    )
    results = runner.run(snapshots, strategy)
    pnl_df = results["pnl_df"]

    if pnl_df.empty:
        return  # no fills → trivially conserved

    for i, row in pnl_df.iterrows():
        diff = abs(row["total_pnl"] - (row["cash"] + row["unrealized_pnl"]))
        assert diff < 0.01, f"Conservation violated at row {i}: diff={diff:.6f}"


def test_monotonic_timestamps():
    """Engine only accepts monotonically increasing timestamps."""
    from microalpha.engine.matching import MatchingEngine
    from microalpha.data.snapshot import MarketSnapshot, L2Level, TradeInfo

    engine = MatchingEngine(assert_invariants=True)
    snaps = []
    for t in [1.0, 2.0, 3.0, 2.5, 4.0]:  # 2.5 is out of order
        snaps.append(MarketSnapshot.from_legacy(
            ts=t, bids=[(100.0, 500)], asks=[(100.02, 500)],
        ))

    for s in snaps:
        engine.ingest_snapshot(s)  # should not raise; just drops out-of-order snaps

    # 4 ticks should have been processed (t=2.5 dropped)
    assert engine.metrics["ticks_processed"] == 4, (
        f"Expected 4 ticks processed, got {engine.metrics['ticks_processed']}"
    )


def test_no_lookahead():
    """Strategy cannot access future snapshots — verified by tick ordering contract."""
    # This is an architectural guarantee provided by the runner's tick ordering.
    # We verify that when on_tick is called, engine.current_snapshot.ts == current tick ts.
    gen = SyntheticDataGenerator(n_ticks=50, seed=1)
    snapshots = gen.generate()

    runner = build_runner(seed=1)

    seen_times = []
    class TimestampRecordingStrategy(KellyConstrainedStrategy):
        def on_tick(self, engine, snapshot):
            seen_times.append((snapshot.ts, engine.current_snapshot.ts))
            return []

    strategy = TimestampRecordingStrategy(name="recording")
    runner.run(snapshots, strategy)

    for snap_ts, engine_ts in seen_times:
        assert abs(snap_ts - engine_ts) < 1e-9, (
            f"on_tick snapshot ts {snap_ts} != engine.current_snapshot.ts {engine_ts} — lookahead?"
        )


if __name__ == "__main__":
    test_end_to_end_kelly()
    print("Kelly end-to-end: PASSED")
    test_end_to_end_avellaneda()
    print("Avellaneda end-to-end: PASSED")
    test_pnl_conservation_invariant()
    print("PnL conservation: PASSED")
    test_monotonic_timestamps()
    print("Monotonic timestamps: PASSED")
    test_no_lookahead()
    print("No-lookahead: PASSED")
    print("\nAll integration tests passed!")
