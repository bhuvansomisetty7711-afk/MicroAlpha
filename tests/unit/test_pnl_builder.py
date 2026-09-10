"""
tests/unit/test_pnl_builder.py
==============================
Unit tests for PnLBuilder: roundtrip, inventory flip, conservation invariant,
fee accumulation, and mark-to-market correctness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from microalpha.pnl.builder import PnLBuilder


def make_fill(time, side, qty, price, fee=0.0):
    return {"event": "limit_fill", "time": time, "side": side,
            "filled_qty": qty, "price": price, "fee": fee,
            "best_bid": price - 0.01, "best_ask": price + 0.01}


def make_mkt(time, side, qty, price, fee=0.0):
    return {"event": "market_order_executed", "time": time, "side": side,
            "filled_qty": qty, "avg_price": price, "fee": fee,
            "fills": [{"price": price, "qty": qty}],
            "best_bid": price - 0.01, "best_ask": price + 0.01}


def test_single_buy_conservation():
    """Invariant #21: total_pnl == cash + unrealized after a buy."""
    builder = PnLBuilder(initial_cash=100_000.0)
    logs = [make_fill(1.0, "buy", 100.0, 150.00)]
    df = builder.build(logs, assert_invariants=True)
    last = df.iloc[-1]

    # Cash decreased by notional
    expected_cash = 100_000.0 - 100.0 * 150.00
    assert abs(last["cash"] - expected_cash) < 0.01, f"Cash mismatch: {last['cash']} != {expected_cash}"

    # Position is +100
    assert abs(last["position"] - 100.0) < 0.001

    # Conservation
    assert abs(last["total_pnl"] - (last["cash"] + last["unrealized_pnl"])) < 0.01, \
        f"Conservation violated: total={last['total_pnl']}, cash={last['cash']}, unre={last['unrealized_pnl']}"


def test_round_trip_zero_pnl():
    """Buy and sell at same price → realized_pnl == 0 (ignoring fees)."""
    builder = PnLBuilder(initial_cash=100_000.0)
    logs = [
        make_fill(1.0, "buy",  100.0, 150.00),
        make_fill(2.0, "sell", 100.0, 150.00),
    ]
    df = builder.build(logs, assert_invariants=True)
    last = df.iloc[-1]

    assert abs(last["realized_pnl"] - 0.0) < 0.01, f"Expected 0 realized, got {last['realized_pnl']}"
    assert abs(last["position"] - 0.0) < 0.001, f"Expected flat position, got {last['position']}"
    assert abs(last["avg_cost"] - 0.0) < 0.001, f"avg_cost should be 0 when flat, got {last['avg_cost']}"


def test_round_trip_profit():
    """Buy at 150 then sell at 150.10 → realized PnL > 0."""
    builder = PnLBuilder(initial_cash=100_000.0)
    logs = [
        make_fill(1.0, "buy",  100.0, 150.00),
        make_fill(2.0, "sell", 100.0, 150.10),
    ]
    df = builder.build(logs, assert_invariants=True)
    last = df.iloc[-1]

    expected_realized = 100.0 * (150.10 - 150.00)  # = 10.0
    assert abs(last["realized_pnl"] - expected_realized) < 0.01, \
        f"Expected realized={expected_realized}, got {last['realized_pnl']}"


def test_inventory_flip():
    """Long → short flip correctly realizes PnL and opens new position. Invariant #34."""
    builder = PnLBuilder(initial_cash=100_000.0)
    logs = [
        make_fill(1.0, "buy",  100.0, 150.00),   # go long 100
        make_fill(2.0, "sell", 200.0, 150.05),   # sell 200: close 100 at profit, short 100
    ]
    df = builder.build(logs, assert_invariants=True)
    last = df.iloc[-1]

    # Position should be -100 (short 100 shares)
    assert abs(last["position"] - (-100.0)) < 0.001, f"Expected -100 position, got {last['position']}"
    # Realized PnL from closing the long: 100 * (150.05 - 150.00) = 5.0
    assert abs(last["realized_pnl"] - 5.0) < 0.01, f"Expected realized=5.0, got {last['realized_pnl']}"
    # avg_cost for the new short should be 150.05
    assert abs(last["avg_cost"] - 150.05) < 0.01, f"Expected avg_cost=150.05, got {last['avg_cost']}"


def test_fee_accumulation():
    """Fees accumulate correctly in cum_fees and reduce cash."""
    fee = 0.50  # 50 cents per fill
    builder = PnLBuilder(initial_cash=100_000.0)
    logs = [
        make_fill(1.0, "buy",  100.0, 150.00, fee=fee),
        make_fill(2.0, "sell", 100.0, 150.05, fee=fee),
    ]
    df = builder.build(logs, assert_invariants=True)
    last = df.iloc[-1]

    assert abs(last["cum_fees"] - 2 * fee) < 0.001, f"Expected cum_fees={2*fee}, got {last['cum_fees']}"


def test_conservation_multi_fills():
    """Conservation invariant holds across many fills."""
    import random
    rng = random.Random(42)
    builder = PnLBuilder(initial_cash=500_000.0)
    logs = []
    price = 100.0
    t = 1.0
    for _ in range(50):
        side = rng.choice(["buy", "sell"])
        qty = rng.randint(10, 200)
        price += rng.gauss(0, 0.05)
        price = max(50.0, price)
        logs.append(make_fill(t, side, qty, round(price, 4)))
        t += 0.01

    df = builder.build(logs, assert_invariants=True)

    for idx, row in df.iterrows():
        diff = abs(row["total_pnl"] - (row["cash"] + row["unrealized_pnl"]))
        assert diff < 0.001, f"Conservation violated at row {idx}: diff={diff}"


if __name__ == "__main__":
    test_single_buy_conservation()
    test_round_trip_zero_pnl()
    test_round_trip_profit()
    test_inventory_flip()
    test_fee_accumulation()
    test_conservation_multi_fills()
    print("All PnL builder tests passed!")
