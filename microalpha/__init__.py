"""
MICROALPHA — Professional Research-Grade Backtesting System
============================================================

A production-quality quantitative trading backtesting framework implementing:
- Tick-level market data replay (Databento MBP-10 format)
- L2 order book reconstruction with per-level delta computation
- Realistic execution simulation with latency, queue position, and FIFO fills
- Maker/taker fee model with regulatory fees
- Market impact modeling (square-root law)
- Kelly-constrained strategy sizing
- Avellaneda-Stoikov market-making
- Single-source-of-truth PnL reconstruction
- 70+ invariant checks for accounting correctness
- Comprehensive test suite

Usage:
    from microalpha.data.snapshot import MarketSnapshot
    from microalpha.engine.matching import MatchingEngine
    from microalpha.runner.backtest import BacktestRunner
    from microalpha.strategy.kelly import KellyConstrainedStrategy

See examples/run_example.py for a complete walkthrough.
"""

__version__ = "1.0.0"
__author__ = "MICROALPHA Quantitative Research Team"
