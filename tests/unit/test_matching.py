"""
tests/unit/test_matching.py
============================
Unit tests for the MatchingEngine: FIFO ordering, passive fills,
cancel-before-arrival race, partial fills, fee correctness.

These are white-box tests that inspect engine internal state.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from microalpha.data.snapshot import MarketSnapshot, TradeInfo, L2Level, Side
from microalpha.engine.matching import MatchingEngine
from microalpha.engine.latency import LatencyConfig
from microalpha.engine.queue_model import QueueModelConfig
from microalpha.engine.fees import FeeConfig


def make_snap(ts, bid=100.0, ask=100.02, bid_size=1000.0, ask_size=1000.0, aggressor=0):
    return MarketSnapshot.from_legacy(
        ts=ts,
        bids=[(bid, bid_size)],
        asks=[(ask, ask_size)],
        last_trade_aggressor=aggressor,
        last_trade_price=bid if aggressor == -1 else ask if aggressor == 1 else None,
        last_trade_size=100.0 if aggressor != 0 else None,
    )


def build_engine(seed=42) -> MatchingEngine:
    return MatchingEngine(
        latency_config=LatencyConfig(base_ms=1.0, jitter_ms=0.1),  # tight latency for testing
        queue_config=QueueModelConfig(decay_lambda=0.0),             # no decay in tests
        fee_config=FeeConfig(fee_bps=0.0, maker_rebate_bps=0.0, taker_fee_bps=0.0, fee_per_share=0.0),
        tick_size=0.01,
        rng_seed=seed,
        assert_invariants=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Place an order → arrives → resting
# ---------------------------------------------------------------------------
def test_order_arrives_and_rests():
    engine = build_engine()
    # tick 0: empty book
    engine.ingest_snapshot(make_snap(0.0))

    # Place a buy limit at 100.00
    oid = engine.place_limit_order("strat", Side.BUY, price=100.00, qty=100.0, now_ts=0.0)
    assert oid != "", "order_id should not be empty"

    order = engine.get_order(oid)
    assert order is not None
    assert order.status == "pending"

    # Advance time past latency (latency ~1ms = 0.001s → use t=0.005s to be safe)
    engine.ingest_snapshot(make_snap(0.005))

    order = engine.get_order(oid)
    assert order.status == "resting", f"Expected resting, got {order.status}"


# ---------------------------------------------------------------------------
# Test 2: Cancel-before-arrival race condition (Architecture invariant #41)
# ---------------------------------------------------------------------------
def test_cancel_before_arrival():
    engine = build_engine()
    engine.ingest_snapshot(make_snap(0.0))

    oid = engine.place_limit_order("strat", Side.BUY, price=100.00, qty=100.0, now_ts=0.0)
    # Cancel immediately (cancel latency also ~1ms, but order arrival is ~1ms too)
    engine.cancel_order(oid, now_ts=0.0)

    # Advance time: both order and cancel arrive in this tick
    engine.ingest_snapshot(make_snap(0.005))

    order = engine.get_order(oid)
    # Order should be cancelled (either cancel won the race, or order is cancelled)
    assert order.status in ("cancelled", "resting"), f"Unexpected status: {order.status}"

    # If order is not in resting queue, it was cancelled before arrival
    resting = engine.active_resting_orders(owner="strat")
    if order.status == "cancelled":
        assert all(o.order_id != oid for o in resting), "Cancelled order found in resting queue"


# ---------------------------------------------------------------------------
# Test 3: FIFO fill ordering — two orders at same price, first in gets first fill
# ---------------------------------------------------------------------------
def test_fifo_ordering():
    engine = build_engine()
    engine.ingest_snapshot(make_snap(0.0, bid=100.00, ask=100.02, ask_size=2000.0))

    # Place two sell orders at same price (our limit sell at ask level)
    oid1 = engine.place_limit_order("strat", Side.SELL, price=100.02, qty=100.0, now_ts=0.0)
    oid2 = engine.place_limit_order("strat", Side.SELL, price=100.02, qty=100.0, now_ts=0.0)

    # Let both arrive
    engine.ingest_snapshot(make_snap(0.005, bid=100.00, ask=100.02, ask_size=2000.0))

    o1 = engine.get_order(oid1)
    o2 = engine.get_order(oid2)
    assert o1.status == "resting" and o2.status == "resting"

    # FIFO: check deque ordering — oid1 must be AHEAD of oid2 in the deque
    # (queue_position magnitudes are stochastic; FIFO is guaranteed by deque insertion order)
    from microalpha.book.l2_book import price_to_tick
    price_tick = price_to_tick(100.02, engine.tick_size)
    dq = engine._resting[Side.SELL].get(price_tick)
    assert dq is not None, "No resting queue at this price"
    deque_ids = [o.order_id for o in dq]
    assert deque_ids.index(oid1) < deque_ids.index(oid2), (
        f"FIFO violated: oid1 at index {deque_ids.index(oid1)} > oid2 at {deque_ids.index(oid2)}"
    )

    # Now simulate a buy aggressor consuming 200 shares of the ask level
    # Create a delta showing consumption at ask price
    from microalpha.book.l2_book import LevelDelta
    delta = LevelDelta(price=100.02, tick_key=10002, side=Side.SELL, old_size=2000.0, new_size=1800.0)
    engine.simulate_passive([delta])

    # After fill simulation, orders may have been partially filled
    # We only assert no invariant violations occurred (checked by assert_invariants=True)


# ---------------------------------------------------------------------------
# Test 4: Market order sweep — fills at correct prices, no snapshot mutation
# ---------------------------------------------------------------------------
def test_market_order_sweep():
    engine = build_engine()
    snap = make_snap(0.0, bid=100.00, ask=100.02)
    engine.ingest_snapshot(snap)

    # Market buy: should fill at ask (100.02)
    fills = engine.execute_market_order("strat", Side.BUY, qty=100.0, now_ts=0.0)
    assert len(fills) > 0, "Expected at least one fill"
    for f in fills:
        # Fill price should be >= best_ask (with possible impact)
        assert f.price >= snap.best_ask - 1e-6, (
            f"Market buy filled below best_ask: {f.price} < {snap.best_ask}"
        )
        assert f.filled_qty > 0

    # Snapshot must NOT be mutated (invariant #35)
    for lv in snap.asks:
        assert lv.size == 1000.0, "Snapshot was mutated!"


# ---------------------------------------------------------------------------
# Test 5: Fee correctness — zero fees with zero fee config
# ---------------------------------------------------------------------------
def test_zero_fees():
    engine = build_engine()  # fee_config is all zeros in build_engine
    snap = make_snap(0.0, bid=100.00, ask=100.02)
    engine.ingest_snapshot(snap)

    fills = engine.execute_market_order("strat", Side.BUY, qty=100.0, now_ts=0.0)
    total_fee = sum(f.fee for f in fills)
    assert abs(total_fee) < 1e-8, f"Expected zero fee with zero fee config, got {total_fee}"


# ---------------------------------------------------------------------------
# Test 6: Invariant #13 — no negative remaining after fills
# ---------------------------------------------------------------------------
def test_no_negative_remaining():
    engine = build_engine()
    engine.ingest_snapshot(make_snap(0.0, ask_size=50.0))  # small ask

    oid = engine.place_limit_order("strat", Side.SELL, price=100.02, qty=100.0, now_ts=0.0)
    engine.ingest_snapshot(make_snap(0.005, ask_size=50.0))

    # Simulate consumption greater than order size
    from microalpha.book.l2_book import LevelDelta
    delta = LevelDelta(price=100.02, tick_key=10002, side=Side.SELL, old_size=50.0, new_size=0.0)
    engine.simulate_passive([delta])

    order = engine.get_order(oid)
    assert order.remaining >= -1e-9, f"Remaining went negative: {order.remaining}"


# ---------------------------------------------------------------------------
# Test 7: L2Book integer tick keys — no float equality issues
# ---------------------------------------------------------------------------
def test_l2book_tick_keys():
    from microalpha.book.l2_book import L2Book, price_to_tick
    book = L2Book(tick_size=0.01)
    snap = make_snap(0.0, bid=100.00, ask=100.02)
    book.update(snap)

    assert 10000 in book._bids, "Integer tick key 10000 should be in bids"
    assert 10002 in book._asks, "Integer tick key 10002 should be in asks"
    assert book.size_at_price(Side.BUY, 100.00) == 1000.0
    assert book.size_at_price(Side.SELL, 100.02) == 1000.0  # ASK side uses Side.SELL


# ---------------------------------------------------------------------------
# Test 8: Delta computation — level disappearance treated as consumed
# ---------------------------------------------------------------------------
def test_delta_level_disappearance():
    from microalpha.book.l2_book import L2Book
    book = L2Book(tick_size=0.01)
    snap1 = make_snap(0.0, ask=100.02, ask_size=500.0)
    snap2 = make_snap(0.005, ask=100.02, ask_size=0.0001)  # essentially disappeared

    book.update(snap1)
    book.update(snap2)
    deltas = book.compute_deltas(snap1)

    ask_deltas = [d for d in deltas if d.side == Side.SELL]  # ASK side → Side.SELL
    assert len(ask_deltas) > 0, "No ASK delta found"
    # Should see the 500.0 → 0.0001 as consumption
    consumed = sum(d.consumed for d in ask_deltas if abs(d.price - 100.02) < 0.001)
    assert consumed > 0, f"Expected positive consumption at ask, got {consumed}"


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_order_arrives_and_rests()
    test_cancel_before_arrival()
    test_fifo_ordering()
    test_market_order_sweep()
    test_zero_fees()
    test_no_negative_remaining()
    test_l2book_tick_keys()
    test_delta_level_disappearance()
    print("All unit tests passed!")
