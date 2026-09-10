"""
examples/run_example.py
========================
Complete MICROALPHA backtest example using synthetic market data.

This script demonstrates the full workflow:
  1. Generate synthetic L2 order book data (no real data required)
  2. Configure the matching engine sub-models
  3. Create a Kelly-Constrained strategy with a momentum signal
  4. Run the full backtest simulation
  5. Print the diagnostic report (invariants + PnL + metrics)
  6. Show individual fill events
  7. Run the Avellaneda-Stoikov market maker as a second example

Run from the Backtest_final directory:
    python examples/run_example.py
"""

import logging
import os
import sys

# Ensure the package is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_example")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from microalpha.data.loaders import SyntheticDataGenerator
from microalpha.engine.fees    import FeeConfig
from microalpha.engine.latency import LatencyConfig
from microalpha.engine.impact  import ImpactConfig
from microalpha.engine.queue_model import QueueModelConfig
from microalpha.runner.backtest    import BacktestRunner
from microalpha.strategy.kelly     import KellyConstrainedStrategy
from microalpha.strategy.avellaneda import AvellanedaStoikovStrategy


# ===========================================================================
# EXAMPLE 1 — Kelly-Constrained Strategy with momentum signal
# ===========================================================================
def run_kelly_example():
    print("\n" + "=" * 70)
    print("  EXAMPLE 1: Kelly-Constrained Strategy (Momentum Signal)")
    print("=" * 70)

    # 1. Generate synthetic L2 data
    gen = SyntheticDataGenerator(
        n_ticks=3000,
        initial_mid=150.0,
        tick_size=0.01,
        spread_ticks=2.0,
        vol_per_tick=0.001,
        base_depth=500.0,
        tick_interval_s=0.005,   # 5ms between ticks
        num_levels=5,
        trade_prob=0.4,
        symbol="SYN_AAPL",
        seed=42,
    )
    snapshots = gen.generate()
    log.info("Generated %d synthetic snapshots", len(snapshots))

    # 2. Build runner with realistic engine configs
    runner = BacktestRunner.from_configs(
        latency_config=LatencyConfig(
            base_ms=5.0,           # 5ms median one-way latency
            jitter_ms=3.0,         # lognormal jitter
            cancel_base_ms=4.0,    # slightly faster cancel path
        ),
        queue_config=QueueModelConfig(
            decay_lambda=1.0,               # moderate queue erosion
            hidden_liquidity_frac=0.05,     # 5% hidden (iceberg) depth
        ),
        fee_config=FeeConfig(
            fee_bps=0.30,          # 30c per $1000 notional
            maker_rebate_bps=-0.20, # -20c rebate for passive fills
            taker_fee_bps=0.30,    # 30c surcharge for market orders
            fee_per_share=0.0001,  # SEC proxy fee
        ),
        impact_config=ImpactConfig(
            perm_impact_coeff=0.02,
            temp_impact_coeff=0.01,
            adv_estimate=500_000.0,  # $500K ADV
        ),
        tick_size=0.01,
        initial_cash=100_000.0,
        rng_seed=42,
        assert_invariants=True,
        verbose=True,
    )

    # 3. Create strategy
    strategy = KellyConstrainedStrategy(
        name="momentum_kelly",
        capital=100_000.0,
        max_inventory=500.0,
        kelly_shrink=0.05,
        transaction_cost=0.0002,
        latency_ms=5.0,
        tick_size=0.01,
        max_shares_per_order=200,
        min_edge_threshold=5e-5,   # lower threshold for more trading in demo
        min_time_between_orders_ms=20.0,
        max_orders_per_minute=200,
        ml_confidence_halflife_ms=300.0,
        gamma=5e2,
        rng_seed=42,
    )

    # 4. ML model function: simple momentum signal (sign of last 5 mid returns).
    def momentum_model(snap, engine, strat):
        arr = list(strat.recent_mid_returns)
        if len(arr) < 5:
            return 0.0
        recent = arr[-5:]
        momentum = sum(recent) / max(1e-12, abs(sum(recent)) + 1e-12)
        return float(max(-1.0, min(1.0, momentum * 2.0)))

    # 5. Run the backtest
    results = runner.run(snapshots, strategy, model_fn=momentum_model)

    pnl_df = results["pnl_df"]
    metrics = results["metrics"]

    if not pnl_df.empty:
        log.info("Net PnL: $%.2f | Sharpe: %.3f | Max DD: %.2f%%",
                 metrics.get("net_pnl", 0),
                 metrics.get("sharpe", float("nan")),
                 metrics.get("max_drawdown", 0) * 100)
    else:
        log.info("No fills occurred. Strategy did not find sufficient edge above latency costs.")
        log.info("This is EXPECTED behavior when the signal (momentum) is weaker than latency cost.")
        log.info("Tip: lower min_edge_threshold or increase n_ticks to see more trades.")

    return results


# ===========================================================================
# EXAMPLE 2 — Avellaneda-Stoikov Market Maker
# ===========================================================================
def run_avellaneda_example():
    print("\n" + "=" * 70)
    print("  EXAMPLE 2: Avellaneda-Stoikov Market Maker")
    print("=" * 70)

    gen = SyntheticDataGenerator(
        n_ticks=2000,
        initial_mid=100.0,
        tick_size=0.01,
        spread_ticks=2.0,
        vol_per_tick=0.0015,
        base_depth=300.0,
        tick_interval_s=0.005,
        num_levels=5,
        trade_prob=0.5,
        symbol="SYN_SPY",
        seed=99,
    )
    snapshots = gen.generate()
    log.info("Generated %d synthetic snapshots for A-S demo", len(snapshots))

    runner = BacktestRunner.from_configs(
        latency_config=LatencyConfig(base_ms=3.0, jitter_ms=1.0),
        queue_config=QueueModelConfig(decay_lambda=0.5),
        fee_config=FeeConfig(fee_bps=0.30, maker_rebate_bps=-0.15, taker_fee_bps=0.30),
        tick_size=0.01,
        initial_cash=250_000.0,
        rng_seed=99,
        assert_invariants=True,
        verbose=True,
    )

    strategy = AvellanedaStoikovStrategy(
        name="as_market_maker",
        gamma=0.05,
        kappa=2.0,
        tau=30.0,
        base_qty=100.0,
        max_inventory=500.0,
        tick_size=0.01,
        quote_lifetime_ms=50.0,   # refresh quotes every 50ms
        min_spread_ticks=1,
    )

    results = runner.run(snapshots, strategy)

    pnl_df = results["pnl_df"]
    metrics = results["metrics"]
    inv_report = results["invariant_report"]

    failed = inv_report.num_failed
    log.info("Invariant check: %d passed, %d failed",
             inv_report.num_passed, failed)
    if failed > 0:
        log.warning("Failed invariants:")
        for f in inv_report.failures:
            log.warning("  [%s] %s: %s", f.inv_id, f.desc, f.detail)

    if not pnl_df.empty:
        log.info("Net PnL: $%.2f | Fill Rate: %.2f%% | Passive%%: %.1f%%",
                 metrics.get("net_pnl", 0),
                 metrics.get("fill_rate", 0) * 100,
                 metrics.get("pct_passive_fills", 0) * 100)

    return results


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    log.info("Starting MICROALPHA example runs...")

    r1 = run_kelly_example()
    r2 = run_avellaneda_example()

    log.info("Both examples completed successfully.")
    print("\nTo run individual tests:")
    print("  python -m pytest tests/ -v                  # all tests")
    print("  python -m pytest tests/unit/ -v             # unit tests only")
    print("  python -m pytest tests/integration/ -v      # integration tests only")
