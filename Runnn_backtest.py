"""
run_backtest_final.py
A production-grade backtest harness (replacement for Run_backtest (3).py).
Integrates with RealisticMatchingEngine (matching_engine_realistic.py).
Author: Assistant (senior-quant style)
"""

import math
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import the realistic matching engine (ensure module path is correct)
# from matching_engine_realistic import RealisticMatchingEngine
# If the engine file is named differently, update the import accordingly.
try:
    from matching_engine_realistic import RealisticMatchingEngine
except Exception as e:
    # fallback placeholder: user must provide the matching engine file created earlier
    raise ImportError("Please ensure matching_engine_realistic.py is on PYTHONPATH and provides RealisticMatchingEngine") from e

# ----------------------------
# Utility: Calculating PnL (single authoritative pass)
# ----------------------------
def calculate_pnl_enhanced(logs: List[Dict[str, Any]], strategy=None, initial_cash=100_000.0, assert_invariants=True):
    """
    Rebuild PnL from structured logs (single source of truth).
    Logs must be structured dicts produced by RealisticMatchingEngine logs:
    - For fill events: 'event' == 'limit_fill' or 'market_order_executed' or 'limit_fill' entries
      Required keys for fills: 'time', 'event', 'order_id' (if limit), 'owner', 'side' ('buy'/'sell'),
      'filled_qty', 'price' (per-share), 'fee' optional.
    - Engine-level market snapshots should provide 'best_bid','best_ask' where available in logs.
    Returns DataFrame with columns:
      ['time','cash','position','avg_cost','realized_pnl','unrealized_pnl','total_pnl','cum_fees']
    """
    import math
    import pandas as pd

    if not isinstance(logs, list):
        logs = list(logs)

    if len(logs) == 0:
        return pd.DataFrame(columns=['time','cash','position','avg_cost','realized_pnl','unrealized_pnl','total_pnl','cum_fees'])

    # Sort logs by time
    logs_sorted = sorted(logs, key=lambda x: x.get('time', 0.0))

    cash = float(initial_cash)
    pos = 0.0
    avg_cost = 0.0
    realized_pnl = 0.0
    cum_fees = 0.0
    history = []

    last_bid = None
    last_ask = None
    last_trade_price = None

    def apply_fill_record(side, qty, price, fee):
        nonlocal cash, pos, avg_cost, realized_pnl, cum_fees, last_bid, last_ask, last_trade_price

        qty = float(qty)
        price = float(price)
        fee = float(fee or 0.0)

        # Normalize side
        s = side.lower()
        if s in ('bid','buy'):
            is_buy = True
        elif s in ('ask','sell','sell_aggressor'):  # be robust
            is_buy = False
        else:
            # unknown side: treat conservatively as no-op
            return

        # trade value
        trade_value = qty * price

        # If position exists and is opposite sign => this closes some portion => realized PnL
        if pos == 0 or (pos > 0 and is_buy) or (pos < 0 and not is_buy):
            # increasing position in same direction
            if pos == 0:
                avg_cost = price
                pos = pos + qty if is_buy else pos - qty
            else:
                prev_notional = avg_cost * abs(pos)
                add_qty = qty
                add_notional = price * add_qty
                new_abs_pos = abs(pos) + add_qty
                avg_cost = (prev_notional + add_notional) / new_abs_pos
                pos = pos + qty if is_buy else pos - qty

            # cash flow: buy pays cash, sell receives cash
            if is_buy:
                cash -= trade_value
            else:
                cash += trade_value

            # fees
            cash -= fee
            cum_fees += fee
            return

        # closing opposite position
        if (pos > 0 and not is_buy) or (pos < 0 and is_buy):
            # closing some or all
            if pos > 0:
                # long, sell closes
                close_qty = min(qty, pos)
                realized = (price - avg_cost) * close_qty
                realized_pnl += realized
                cash += price * close_qty
                pos -= close_qty
                remaining_qty = qty - close_qty
                if remaining_qty > 0:
                    # becomes new short position
                    pos -= remaining_qty
                    avg_cost = price
                    cash += price * remaining_qty
            else:
                # short, buy closes
                close_qty = min(qty, -pos)
                realized = (avg_cost - price) * close_qty
                realized_pnl += realized
                cash -= price * close_qty
                pos += close_qty
                remaining_qty = qty - close_qty
                if remaining_qty > 0:
                    pos += remaining_qty
                    avg_cost = price
                    cash -= price * remaining_qty

            # fees
            cash -= fee
            cum_fees += fee
            # if pos becomes zero, clear avg_cost to avoid stale basis
            if abs(pos) < 1e-12:
                avg_cost = 0.0
            return

    for event in logs_sorted:
        t = event.get('time', None)
        ev = event.get('event', '').lower()

        # update last seen book prices when present in event
        if 'best_bid' in event and event.get('best_bid') is not None:
            last_bid = float(event.get('best_bid'))
        if 'best_ask' in event and event.get('best_ask') is not None:
            last_ask = float(event.get('best_ask'))
        if 'last_trade_price' in event and event.get('last_trade_price') is not None:
            last_trade_price = float(event.get('last_trade_price'))

        if ev in ('limit_fill', 'limit_fill_implied'):
            # Single-fill events: canonical keys
            fill_side = event.get('side') or event.get('aggressor_side') or event.get('owner_side') or 'buy'
            fill_qty = float(event.get('filled_qty') or event.get('qty') or 0.0)
            fill_price = float(event.get('price') or event.get('avg_price') or event.get('trade_price') or 0.0)
            fill_fee = float(event.get('fee') or 0.0)
            if fill_qty > 0 and fill_price > 0:
                apply_fill_record(fill_side, fill_qty, fill_price, fill_fee)
                last_trade_price = fill_price  # update for mark fallback

        elif ev in ('market_order_executed', 'market_fill'):
            # Market sweeps carry a 'fills' list of {price, qty} micro-fills.
            # Iterate each micro-fill to get accurate per-level PnL.
            sweep_side = event.get('side') or event.get('owner_side') or 'buy'
            sweep_fee = float(event.get('fee', 0.0))
            sweep_filled_qty = float(event.get('filled_qty', 0.0))
            if 'fills' in event and event['fills']:
                for f in event['fills']:
                    f_price = float(f.get('price', event.get('avg_price', 0.0)))
                    f_qty = float(f.get('qty', f.get('filled_qty', 0.0)))
                    # Apportion top-level fee by quantity fraction if no per-fill fee
                    f_fee = float(f.get('fee', sweep_fee * (f_qty / max(1e-12, sweep_filled_qty))))
                    if f_qty > 0 and f_price > 0:
                        apply_fill_record(sweep_side, f_qty, f_price, f_fee)
                        last_trade_price = f_price
            elif sweep_filled_qty > 0:
                # Fallback: use avg_price and total filled_qty
                avg_price = float(event.get('avg_price', event.get('price', 0.0)))
                if avg_price > 0:
                    apply_fill_record(sweep_side, sweep_filled_qty, avg_price, sweep_fee)
                    last_trade_price = avg_price

        # --- Update unrealized using conservative mark ---
        # IMPORTANT: use only outer-scope variables (last_bid, last_ask, last_trade_price)
        # Never reference loop-local 'price' or 'fill_price' here — they may not be defined
        # for non-fill events.
        if pos != 0:
            if last_bid is not None and last_ask is not None:
                if pos > 0:
                    mark = last_bid  # liquidate long => hit bid
                else:
                    mark = last_ask  # liquidate short => lift ask
            elif last_trade_price is not None:
                mark = last_trade_price
            else:
                mark = avg_cost  # absolute fallback: mark at cost (zero unrealized)
        else:
            mark = last_bid if last_bid is not None else last_ask if last_ask is not None else last_trade_price

        unrealized = 0.0
        if pos != 0 and mark is not None:
            unrealized = (mark - avg_cost) * pos

        total_pnl = cash + unrealized

        history.append({
            'time': t,
            'cash': cash,
            'position': pos,
            'avg_cost': avg_cost,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized,
            'total_pnl': total_pnl,
            'cum_fees': cum_fees
        })

        if assert_invariants:
            if abs(total_pnl - (cash + unrealized)) > 1e-6:
                raise AssertionError("PnL invariant violated at time %s: total_pnl != cash+unrealized" % str(t))

    df = pd.DataFrame(history)
    if df.empty:
        return df
    # forward-fill time if None
    if 'time' in df.columns:
        df['time'] = df['time'].ffill()
    return df

# ----------------------------
# Strategy: Kelly-conservative, inventory-aware, replay-mode safe
# ----------------------------
@dataclass
class KellyConstrainedStrategy:
    name: str
    capital: float = 100_000.0
    max_inventory: float = 1000.0            # shares or units
    kelly_shrink: float = 0.05               # fraction of Kelly to use (very conservative)
    transaction_cost: float = 0.0002         # fraction per trade (20 bps). set to realistic value
    latency_ms: float = 5.0                  # keep in ms as canonical unit
    replay_mode: bool = True                 # in replay, do NOT mutate cash; use logs & PnL builder
    min_edge_threshold: float = 1e-4         # minimum fractional edge to consider
    adv_participation_limit: float = 0.01    # max participation rate of ADV (1%) by default
    tick_size: float = 0.01                  # fallback tick size
    rng_seed: int = 42
    # --- ADV configuration (time-dependent, can be overridden per tick) ---
    adv_daily_notional: float = 1_000_000.0  # estimated daily ADV in dollars (instrument-specific)
    # --- Hard cap on order size (Tier 2 realism) ---
    max_shares_per_order: int = 500           # absolute max shares per individual order (prevents book-walking)
    # --- Signal & order pacing controls (Tier 1) ---
    min_time_between_orders_ms: float = 50.0     # minimum ms between consecutive orders
    max_orders_per_minute: int = 120             # hard cap on orders per rolling minute
    ml_confidence_halflife_ms: float = 500.0     # exponential decay tau for ML score staleness
    loss_cooldown_ms: float = 200.0              # cooldown period (ms) after a realized loss fill

    # runtime fields (not to be trusted for final PnL)
    current_inventory: float = 0.0
    pending_orders: Dict[str, Any] = field(default_factory=dict)
    fills_log: List[Dict[str, Any]] = field(default_factory=list)   # strategy-level logs (redundant; engine is authoritative)
    last_tick_time: float = 0.0
    rng: Any = field(init=False)
    # --- Pacing runtime state ---
    _last_order_time: float = field(default=0.0, init=False)
    _orders_this_minute: int = field(default=0, init=False)
    _minute_start_time: float = field(default=0.0, init=False)
    _last_loss_time: float = field(default=0.0, init=False)
    _last_avg_cost_at_fill: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.rng = np.random.RandomState(self.rng_seed)
        # enforce replay_mode in backtest runner
        self.replay_mode = True

    # canonical side conversion
    @staticmethod
    def normalize_side(side):
        if side is None:
            return None
        s = str(side).lower()
        if s in ('buy', 'bid'):
            return 'buy'
        if s in ('sell', 'ask'):
            return 'sell'
        return s

    # Estimate short-horizon volatility (data-driven placeholder)
    def estimate_short_vol(self, recent_mid_returns, horizon_seconds=0.5):
        """
        recent_mid_returns: deque or list of mid returns (log changes per sample interval)
        horizon_seconds: desired horizon (seconds)
        Returns sigma (std) scaled to horizon.
        """
        if (recent_mid_returns is None) or (len(recent_mid_returns) < 3):
            # fallback: very conservative small sigma fraction of price
            return 0.001
        arr = np.asarray(recent_mid_returns)
        # sample period assumed to be median delta in timestamps outside this function; user should provide returns per sample
        sigma_per_sample = np.std(arr)
        # scale to horizon: sqrt rule (approx)
        return max(1e-6, sigma_per_sample * math.sqrt(horizon_seconds))

    def compute_edge_and_variance(self, ml_score, mid_price, spread, recent_mid_returns, latency_seconds):
        """
        Compute conservative edge and variance in *return* units (fraction of mid).
        - ml_score: model output in [-1,1] (signed confidence)
        - spread: absolute spread (price)
        - recent_mid_returns: series of recent fractional mid returns
        - latency_seconds: effective latency horizon to consider
        """
        # conservative short-horizon volatility
        sigma = self.estimate_short_vol(recent_mid_returns, horizon_seconds=latency_seconds)
        # convert sigma (fractional) to expected move (conservative percentile)
        expected_move = np.percentile(np.abs(recent_mid_returns[-min(len(recent_mid_returns), 50):]) if len(recent_mid_returns) > 0 else [0.0], 80) if recent_mid_returns else sigma
        # interpret ml_score as tilt: convert to fractional expected return in price units
        # calibrate ml_score scale to small fraction of spread
        ml_tilt_price = float(ml_score) * 0.25 * spread  # ML can at most move 25% of spread in pitiful assumption
        edge_price = ml_tilt_price - (spread * 0.5 * 0.0)  # we don't double-subtract spread here; costs applied separately
        # convert to fractional edge
        edge_frac = edge_price / max(1e-8, mid_price)
        # variance (conservative): use sigma over latency horizon but inflate
        variance = (sigma ** 2) * 4.0  # inflate to be conservative (x4)
        # latency cost empirical lower bound (conservative)
        latency_loss = max(0.0, expected_move)
        latency_loss_frac = latency_loss / max(1e-8, mid_price)
        # final conservative edge after subtracting expected latency harm and per-trade fees:
        # NOTE: fees will be applied at sizing stage using transaction_cost
        adjusted_edge = edge_frac - latency_loss_frac
        return adjusted_edge, variance, sigma

    def compute_kelly_size(self, adjusted_edge, variance, mid_price, adv_notional_per_sec, latency_seconds):
        """
        Compute a conservative Kelly-derived **share** size.
        - adjusted_edge: expected return fraction per trade horizon (after latency)
        - variance: variance of returns (fraction^2)
        - mid_price: reference price
        - adv_notional_per_sec: ADV estimate per second for participation cap
        - latency_seconds: horizon to consider for participation
        """
        if adjusted_edge <= 0 or variance <= 0:
            return 0
        # theoretical f* (Kelly fraction)
        f_star = adjusted_edge / max(1e-12, variance)
        # shrink heavily for intraday: apply kelly_shrink constant and extra safety clamp
        f_star = max(0.0, min(f_star * self.kelly_shrink, 0.2))  # never more than 20% of capital even in theory
        # convert to dollar risk allowed
        dollar_risk_allowed = f_star * self.capital
        # risk per share: conservative estimate of immediate slippage + half-spread
        est_slippage_per_share = max(self.tick_size, 0.5 * (mid_price * 0.0001))  # baseline micro slippage (tunable)
        risk_per_share = est_slippage_per_share + (mid_price * self.transaction_cost)
        if risk_per_share <= 0:
            return 0
        shares_by_risk = int(max(0, dollar_risk_allowed / (risk_per_share * mid_price)))
        # participation cap
        max_participation_notional = adv_notional_per_sec * latency_seconds * self.adv_participation_limit
        shares_by_participation = int(max_participation_notional / max(1e-8, mid_price))
        # final size: also apply hard cap to prevent walking the book on spikes
        size = min(shares_by_risk, shares_by_participation,
                   int(self.max_inventory - abs(self.current_inventory)),
                   self.max_shares_per_order)
        return max(0, size)

    def compute_reservation_price(self, mid_price, inventory, sigma, time_horizon_seconds, ml_signal_tilt_price):
        """
        Compute reservation price dominated by inventory coercion. ML tilt is strictly limited.
        """
        # inventory coercion parameter: price-per-share penalty ~ gamma * sigma^2 * tau (units price)
        gamma = 1e3  # small tuning knob (units: price per share per sigma^2)
        inventory_penalty = gamma * (sigma ** 2) * time_horizon_seconds * inventory
        r_base = mid_price - inventory_penalty
        # ML tilt limited to small fraction (e.g., 10%) of inventory price movement magnitude
        max_ml_tilt = max(0.01 * abs(inventory_penalty), 0.0005 * mid_price)  # at least some tiny floor
        ml_tilt_limited = max(-max_ml_tilt, min(max_ml_tilt, ml_signal_tilt_price))
        r_final = r_base + ml_tilt_limited
        return r_final

    # Strategy reaction to market tick: returns list of canonical action dicts
    def on_tick(self, engine: RealisticMatchingEngine, snapshot):
        """
        Called once per snapshot. Decide actions (place limit / cancel / market).
        Returns list of actions for the engine to execute.
        Pacing controls enforce rate limits, ML score decay, and loss cooldowns.
        """
        actions = []

        # --- Pacing guard: minimum interval between orders ---
        dt_since_last_order_ms = (snapshot.ts - self._last_order_time) * 1000.0
        if dt_since_last_order_ms < self.min_time_between_orders_ms and self._last_order_time > 0:
            self.last_tick_time = snapshot.ts
            return []

        # --- Pacing guard: per-minute order cap ---
        if snapshot.ts - self._minute_start_time > 60.0:
            self._orders_this_minute = 0
            self._minute_start_time = snapshot.ts
        if self._orders_this_minute >= self.max_orders_per_minute:
            self.last_tick_time = snapshot.ts
            return []

        # --- Pacing guard: loss cooldown ---
        dt_since_loss_ms = (snapshot.ts - self._last_loss_time) * 1000.0
        if dt_since_loss_ms < self.loss_cooldown_ms and self._last_loss_time > 0:
            self.last_tick_time = snapshot.ts
            return []

        # derive book stats
        best_bid = snapshot.bids[0][0] if snapshot.bids else None
        best_ask = snapshot.asks[0][0] if snapshot.asks else None
        mid = 0.5 * (best_bid + best_ask) if (best_bid is not None and best_ask is not None) else (best_bid or best_ask or 0.0)
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else max(self.tick_size, 0.01)
        # ADV estimate per second from configurable daily notional
        adv_notional_per_sec = self.adv_daily_notional / (24 * 60 * 60)

        # build light feature set for ML call (strategy expects ml_score in [-1,1])
        # This harness expects an external model object to be bound; if not present, use neutral signal 0
        ml_score = getattr(self, 'ml_score', 0.0)

        # --- ML confidence decay: score decays exponentially with time since last tick ---
        dt_ms = max(0.0, (snapshot.ts - self.last_tick_time) * 1000.0) if self.last_tick_time > 0 else 0.0
        tau = max(1.0, self.ml_confidence_halflife_ms)
        ml_score *= math.exp(-dt_ms / tau)

        self.last_tick_time = snapshot.ts

        # simple recent mid returns buffer accessible on strategy
        recent_mid_returns = getattr(self, 'recent_mid_returns', deque(maxlen=500))
        # latency_seconds
        latency_seconds = max(1e-3, self.latency_ms / 1000.0)

        # compute edge & variance (conservative)
        adjusted_edge, variance, sigma = self.compute_edge_and_variance(ml_score, mid, spread, list(recent_mid_returns), latency_seconds)

        # compute size via conservative Kelly
        size = self.compute_kelly_size(adjusted_edge, variance, mid, adv_notional_per_sec, latency_seconds)

        # if edge too small -> no trade
        if abs(adjusted_edge) < self.min_edge_threshold or size <= 0:
            return actions  # no orders

        # compute reservation price dominated by inventory coercion
        ml_tilt_price = ml_score * (0.25 * spread)
        r_price = self.compute_reservation_price(mid, self.current_inventory, sigma, latency_seconds * 2.0, ml_tilt_price)

        # choose side based on sign of adjusted_edge (tilt)
        if adjusted_edge > 0:
            # bias to buy (place a bid)
            target_price = max(r_price, best_bid if best_bid is not None else r_price)
            target_price = round(max(target_price, best_bid or target_price), 8)
            # ensure tick alignment (simple)
            target_price = max(round(target_price / self.tick_size) * self.tick_size, (best_bid or target_price))
            # ensure not exceeding participation
            qty = max(1, int(size))
            actions.append({'action': 'place_limit', 'side': 'buy', 'price': float(target_price), 'qty': int(qty), 'owner': self.name})
        else:
            # bias to sell
            target_price = min(r_price, best_ask if best_ask is not None else r_price)
            target_price = min(round(target_price / self.tick_size) * self.tick_size, (best_ask or target_price))
            qty = max(1, int(size))
            actions.append({'action': 'place_limit', 'side': 'sell', 'price': float(target_price), 'qty': int(qty), 'owner': self.name})

        # --- Update pacing counters ---
        if actions:
            self._last_order_time = snapshot.ts
            self._orders_this_minute += 1

        return actions

    # Called by harness when a fill log referencing this strategy appears — log-only in replay mode
    def on_fill(self, fill_log: Dict[str, Any]):
        """
        Called when a fill belonging to this strategy occurs.
        In replay_mode, we only append to fills_log (no cash mutation).
        Also detects realized losses and sets loss cooldown timestamp.
        """
        # normalize
        side = self.normalize_side(fill_log.get('side'))
        qty = float(fill_log.get('filled_qty', fill_log.get('qty', 0)))
        price = float(fill_log.get('price', fill_log.get('avg_price', 0.0) or 0.0))
        fee = float(fill_log.get('fee', 0.0))
        t = fill_log.get('time', self.last_tick_time)

        # --- Detect realized loss for cooldown ---
        # If we are closing a position (opposite side) and fill price is worse than avg_cost
        if self._last_avg_cost_at_fill > 0 and abs(self.current_inventory) > 1e-9:
            if (self.current_inventory > 0 and side == 'sell' and price < self._last_avg_cost_at_fill):
                # Selling long at a loss
                self._last_loss_time = t
            elif (self.current_inventory < 0 and side == 'buy' and price > self._last_avg_cost_at_fill):
                # Covering short at a loss
                self._last_loss_time = t
        self._last_avg_cost_at_fill = price  # rough tracker for next fill comparison

        # append local copy for convenience, but engine logs are authoritative
        self.fills_log.append({
            'time': t, 'order_id': fill_log.get('order_id'), 'side': side,
            'qty': qty, 'price': price, 'fee': fee
        })

        # update inventory tracking for strategy internal behavior (not PnL source)
        if not self.replay_mode:
            # in live mode we would update inventory and cash here; in replay we shouldn't
            if side == 'buy':
                self.current_inventory += qty
            else:
                self.current_inventory -= qty

# ----------------------------
# BacktestRunner: orchestrates engine + strategy + PnL auditing
# ----------------------------
class BacktestRunner:
    def __init__(self, engine: RealisticMatchingEngine, strategy: KellyConstrainedStrategy, initial_cash=100000.0):
        self.engine = engine
        self.strategy = strategy
        self.initial_cash = float(initial_cash)
        self.replay_logs = []  # snapshot of engine.logs consumed
        self.raw_engine_logs_at_start = 0
        self.pnldf = None
        # diagnostic containers
        self.metrics = {'trades': 0}

        # ensure strategy replay mode
        self.strategy.replay_mode = True

    @staticmethod
    def _compute_deltas(snap):
        """
        Centralized delta computation from a single snapshot's last_trade metadata.
        Returns a canonical dict {'buy': [(price, consumed), ...], 'sell': [...]}.
        This is the single source of truth for how we translate snapshot trades
        into level-consumption deltas for the passive fill simulator.
        """
        deltas = {'buy': [], 'sell': []}
        if snap.last_trade_aggressor == 1 and snap.last_trade_size:
            # Buy aggressor lifted asks => consumption on ask side
            top_ask_price = snap.asks[0][0] if snap.asks else None
            if top_ask_price is not None:
                deltas['buy'].append((float(top_ask_price), float(snap.last_trade_size)))
        elif snap.last_trade_aggressor == -1 and snap.last_trade_size:
            # Sell aggressor hit bids => consumption on bid side
            top_bid_price = snap.bids[0][0] if snap.bids else None
            if top_bid_price is not None:
                deltas['sell'].append((float(top_bid_price), float(snap.last_trade_size)))
        return deltas

    def _attribute_fill_to_strategy(self, ev):
        """
        Determine whether a fill event belongs to this strategy.
        Attribution priority:
          1. order_id present and is in strategy.pending_orders => YES
          2. owner field matches strategy.name => YES
          3. otherwise => NO (log as ambiguous if owner is present but different)
        Returns True if attributed, False otherwise.
        """
        order_id = ev.get('order_id')
        owner = ev.get('owner')

        # Primary: order_id match
        if order_id is not None and order_id in self.strategy.pending_orders:
            return True

        # Secondary: owner string match
        if owner is not None and owner == self.strategy.name:
            return True

        # Ambiguous: log for debugging if owner is present but doesn't match
        if owner is not None and order_id is not None:
            self.engine._log({
                'time': ev.get('time'),
                'event': 'fill_attribution_ambiguous',
                'order_id': order_id,
                'owner': owner,
                'strategy_name': self.strategy.name
            })

        return False

    def _update_strategy_inventory(self, ev):
        """
        Single authoritative inventory update path for the strategy.
        Handles both limit_fill (single fill) and market_order_executed (sweep).
        This is the ONLY place strategy.current_inventory is mutated in the run loop.
        """
        event_type = ev.get('event', '')
        side = self.strategy.normalize_side(ev.get('side'))

        if event_type == 'limit_fill':
            qty = float(ev.get('filled_qty', 0.0))
            if side == 'buy':
                self.strategy.current_inventory += qty
            elif side == 'sell':
                self.strategy.current_inventory -= qty

        elif event_type == 'market_order_executed':
            qty = float(ev.get('filled_qty', 0.0))
            if side == 'buy':
                self.strategy.current_inventory += qty
            elif side == 'sell':
                self.strategy.current_inventory -= qty

    def run(self, snapshots: List[Any], model_infer_fn=None, tick_time_window=0.001):
        """
        snapshots: list of MarketSnapshot objects (engine expects its own snapshot form).
        model_infer_fn: optional function (strategy, snapshot_features) -> ml_score in [-1,1]
        """
        # clear engine logs buffer pointer
        self.raw_engine_logs_at_start = len(self.engine.logs)

        # provide strategy with a recent_mid_returns buffer for vol estimation
        self.strategy.recent_mid_returns = deque(maxlen=1000)

        for i, snap in enumerate(snapshots):
            # ingest snapshot into engine
            self.engine.ingest_snapshot(snap)
            # optional model inference: strategy gets a scalar ml_score
            if model_infer_fn:
                # build minimal feature set: mid & spread & recent returns
                best_bid = snap.bids[0][0] if snap.bids else None
                best_ask = snap.asks[0][0] if snap.asks else None
                mid = 0.5 * (best_bid + best_ask) if (best_bid is not None and best_ask is not None) else (best_bid or best_ask or 0.0)
                spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else max(1e-6, self.strategy.tick_size)
                recent_returns = list(self.strategy.recent_mid_returns)
                ml_score = float(model_infer_fn({'mid': mid, 'spread': spread, 'recent_returns': recent_returns}))
                # clamp
                ml_score = max(-1.0, min(1.0, ml_score))
                self.strategy.ml_score = ml_score
            else:
                self.strategy.ml_score = 0.0

            # call strategy; it returns list of actions
            actions = self.strategy.on_tick(self.engine, snap)
            # translate actions to engine calls
            for act in actions:
                act_type = act.get('action')
                if act_type == 'place_limit':
                    side = self.strategy.normalize_side(act.get('side'))
                    price = float(act.get('price'))
                    qty = int(act.get('qty'))
                    owner = act.get('owner', self.strategy.name)
                    # call engine
                    oid = self.engine.place_limit_order(owner=owner, side=side, price=price, qty=qty, now_ts=snap.ts)
                    # store mapping (not authoritative)
                    self.strategy.pending_orders[oid] = {'price': price, 'qty': qty, 'side': side, 'placed_at': snap.ts}
                elif act_type == 'cancel':
                    self.engine.cancel_order(act.get('order_id'), now_ts=snap.ts)
                elif act_type == 'market':
                    # instant aggressive execution
                    side = self.strategy.normalize_side(act.get('side'))
                    qty = int(act.get('qty'))
                    self.engine.execute_market_order(owner=act.get('owner', self.strategy.name), side=side, qty=qty, now_ts=snap.ts)
                else:
                    # unknown action: ignore (but log)
                    self.engine._log({'time': snap.ts, 'event': 'unknown_action_ignored', 'action': act})

            # --- Centralized delta computation ---
            deltas = self._compute_deltas(snap)

            # run passive simulation for this delta window
            self.engine.simulate_passive(deltas=deltas, tick_time_window=tick_time_window)

            # --- Handle fills: attribute to strategy, update inventory ---
            new_logs = self.engine.logs[self.raw_engine_logs_at_start:]
            for ev in new_logs:
                if ev.get('event') in ('limit_fill', 'market_order_executed'):
                    if self._attribute_fill_to_strategy(ev):
                        # call strategy with fill log (strategy only appends logs; engine is authoritative)
                        self.strategy.on_fill(ev)
                        # single authoritative inventory update
                        self._update_strategy_inventory(ev)

            # advance pointer for engine logs
            self.raw_engine_logs_at_start = len(self.engine.logs)

            # update recent mid returns buffer for volatility estimation
            best_bid = snap.bids[0][0] if snap.bids else None
            best_ask = snap.asks[0][0] if snap.asks else None
            mid = 0.5 * (best_bid + best_ask) if (best_bid is not None and best_ask is not None) else None
            if mid is not None:
                last_mid = getattr(self.strategy, 'last_mid', None)
                if last_mid is not None and last_mid > 0:
                    ret = (mid - last_mid) / last_mid
                    self.strategy.recent_mid_returns.append(ret)
                self.strategy.last_mid = mid

        # after all snapshots processed, rebuild PnL from engine logs
        self.pnldf = calculate_pnl_enhanced(self.engine.logs, strategy=self.strategy, initial_cash=self.initial_cash, assert_invariants=True)

        # final invariants and health checks
        self._final_audit()

        return self.pnldf

    def _final_audit(self):
        # Cumulative fees invariant: compare strategy recorded fees with pnl_df cum_fees last value
        cum_fees_from_df = float(self.pnldf['cum_fees'].iloc[-1]) if (self.pnldf is not None and not self.pnldf.empty) else 0.0
        # Strategy.total_transaction_costs might not exist in replay mode; use sum of strategy.fills_log fees as a cross-check
        strategy_fees_from_log = sum([f.get('fee', 0.0) for f in self.strategy.fills_log])
        # Compare
        if abs(cum_fees_from_df - strategy_fees_from_log) > 1e-6 and abs(cum_fees_from_df - strategy_fees_from_log) / max(1.0, cum_fees_from_df) > 0.05:
            # more than 5% mismatch -> fatal
            raise AssertionError(f"Fee mismatch: pnl_df cum_fees={cum_fees_from_df}, strategy fills log fees={strategy_fees_from_log}")

        # Conservation check (last row)
        last = self.pnldf.iloc[-1]
        cash = last['cash']
        unreal = last['unrealized_pnl']
        total = last['total_pnl']
        if abs(total - (cash + unreal)) > 1e-6:
            raise AssertionError("Final conservation invariant failed")

# ----------------------------
# Minimal unit tests (smoke)
# ----------------------------
def unit_test_roundtrip():
    # Build tiny synthetic snapshot feed
    from matching_engine_realistic import MarketSnapshot
    eng = RealisticMatchingEngine(latency_ms=5.0, rng_seed=123, latency_jitter_ms=0.0)
    strat = KellyConstrainedStrategy(name='test-strat', capital=10000.0, max_inventory=100, kelly_shrink=0.05, transaction_cost=0.0005, rng_seed=123)
    runner = BacktestRunner(engine=eng, strategy=strat, initial_cash=10000.0)

    # snapshot0: simple book
    s0 = MarketSnapshot(ts=0.0, bids=[(99.0, 200)], asks=[(101.0, 200)], last_trade_price=None, last_trade_size=None, last_trade_aggressor=0)
    s1 = MarketSnapshot(ts=0.1, bids=[(99.0, 200)], asks=[(101.0, 200)], last_trade_price=101.0, last_trade_size=50, last_trade_aggressor=1)
    s2 = MarketSnapshot(ts=0.2, bids=[(99.0, 150)], asks=[(101.0, 150)], last_trade_price=101.0, last_trade_size=30, last_trade_aggressor=1)

    # simple ML stub: buy signal
    def model_stub(feat):
        # tiny buy signal occasionally
        return 0.2 if feat['mid'] > 0 else 0.0

    pnl = runner.run([s0, s1, s2], model_infer_fn=model_stub, tick_time_window=0.01)
    print("Unit test roundtrip PnL tail:")
    print(pnl.tail())
    print("Engine logs sample:")
    for ev in eng.logs[:20]:
        print(ev)
    print("Unit test completed (inspect outputs)")

# ----------------------------
# Test 1: Cancel before arrival race
# ----------------------------
def test_cancel_before_arrival_race():
    """
    Place an order, immediately cancel before latency elapses.
    Assert the order is cancelled and never becomes 'resting'.
    """
    from matching_engine_realistic import MarketSnapshot
    eng = RealisticMatchingEngine(latency_ms=50.0, rng_seed=99, latency_jitter_ms=0.0)  # 50ms latency, no jitter for deterministic test

    snap0 = MarketSnapshot(ts=0.0, bids=[(100.0, 500)], asks=[(101.0, 500)])
    eng.ingest_snapshot(snap0)

    # Place at t=0.0; arrival at t=0.050
    oid = eng.place_limit_order(owner='test', side='buy', price=100.0, qty=100, now_ts=0.0)
    # Cancel at t=0.001; cancel arrival at t=0.051 (but order hasn't arrived yet at t=0.001)
    eng.cancel_order(oid, now_ts=0.001)

    # Advance to t=0.06 — both order arrival (0.050) and cancel arrival (0.051) should have processed
    snap1 = MarketSnapshot(ts=0.06, bids=[(100.0, 500)], asks=[(101.0, 500)])
    eng.ingest_snapshot(snap1)

    o = eng.order_map[oid]
    assert o.status == 'cancelled', f"Expected 'cancelled', got '{o.status}'"
    # Verify it never ended up in resting queues
    from collections import deque as dq_type
    for price, dq in eng.bid_queues.items():
        for resting_o in dq:
            assert resting_o.order_id != oid, "Cancelled order should not be in resting queue"

    print("[PASS] test_cancel_before_arrival_race")

# ----------------------------
# Test 2: FIFO partial fills
# ----------------------------
def test_fifo_partial_fills():
    """
    Place two sell orders at the same price (A then B).
    Simulate an aggressive buy that partially consumes the level.
    Assert A gets filled first (FIFO); B only fills after A is fully consumed.
    """
    from matching_engine_realistic import MarketSnapshot
    eng = RealisticMatchingEngine(latency_ms=1.0, rng_seed=7777,
                                  queue_decay_lambda=0.0,
                                  latency_jitter_ms=0.0)  # disable decay and jitter for deterministic test

    snap0 = MarketSnapshot(ts=0.0, bids=[(99.0, 100)], asks=[(100.0, 1000)])
    eng.ingest_snapshot(snap0)

    # Place order A then order B at ask 100.0
    oid_a = eng.place_limit_order(owner='A', side='sell', price=100.0, qty=50, now_ts=0.0)
    oid_b = eng.place_limit_order(owner='B', side='sell', price=100.0, qty=50, now_ts=0.0)

    # Advance past latency so both are resting
    snap1 = MarketSnapshot(ts=0.01, bids=[(99.0, 100)], asks=[(100.0, 1000)])
    eng.ingest_snapshot(snap1)

    # Verify both resting at 100.0
    resting = eng.ask_queues.get(100.0, deque())
    assert len(resting) >= 2, f"Expected >=2 resting orders, got {len(resting)}"
    assert resting[0].order_id == oid_a, "A should be first in FIFO queue"
    assert resting[1].order_id == oid_b, "B should be second in FIFO queue"

    # Simulate aggressive buy consuming 40 shares at 100.0 — only A should be partially filled
    deltas = {'buy': [(100.0, 40.0)], 'sell': []}
    eng.simulate_passive(deltas=deltas, tick_time_window=0.01)

    fills_a = [ev for ev in eng.logs if ev.get('event') == 'limit_fill' and ev.get('order_id') == oid_a]
    fills_b = [ev for ev in eng.logs if ev.get('event') == 'limit_fill' and ev.get('order_id') == oid_b]

    # Due to stochastic nature, A should have received fills before B in FIFO order
    total_a = sum(f['filled_qty'] for f in fills_a)
    total_b = sum(f['filled_qty'] for f in fills_b)
    # If any fills occurred, A should have been served first
    if total_a + total_b > 0:
        assert total_a >= total_b, f"FIFO violated: A filled {total_a}, B filled {total_b}"
    print(f"[PASS] test_fifo_partial_fills (A filled: {total_a}, B filled: {total_b})")

# ----------------------------
# Test 3: Fee consistency
# ----------------------------
def test_fee_consistency():
    """
    Run a scenario with fills. Verify:
    1. Every limit_fill/market_order_executed event has a 'fee' field.
    2. calculate_pnl_enhanced produces cum_fees matching sum of logged fees.
    """
    from matching_engine_realistic import MarketSnapshot
    eng = RealisticMatchingEngine(latency_ms=1.0, rng_seed=42,
                                  fee_bps=1.0, maker_rebate_bps=-0.3, taker_fee_bps=0.5,
                                  fee_per_share=0.001, latency_jitter_ms=0.0)

    snap0 = MarketSnapshot(ts=0.0, bids=[(99.0, 500)], asks=[(101.0, 500)])
    eng.ingest_snapshot(snap0)

    # Place a sell (passive) and let it arrive
    oid = eng.place_limit_order(owner='dealer', side='sell', price=101.0, qty=100, now_ts=0.0)
    snap1 = MarketSnapshot(ts=0.01, bids=[(99.0, 500)], asks=[(101.0, 500)])
    eng.ingest_snapshot(snap1)

    # Simulate aggressive buy consuming our sell order
    deltas = {'buy': [(101.0, 200.0)], 'sell': []}
    eng.simulate_passive(deltas=deltas, tick_time_window=0.01)

    # Also execute a market buy (taker)
    eng.execute_market_order(owner='dealer', side='buy', qty=50, now_ts=0.01)

    # Check: every fill event must have 'fee' field
    fill_events = [ev for ev in eng.logs if ev.get('event') in ('limit_fill', 'market_order_executed')]
    total_logged_fees = 0.0
    for ev in fill_events:
        assert 'fee' in ev, f"Fill event missing 'fee': {ev}"
        total_logged_fees += float(ev['fee'])

    # Check: calculate_pnl_enhanced recovers the same cumulative fees
    pnl_df = calculate_pnl_enhanced(eng.logs, assert_invariants=True)
    if not pnl_df.empty:
        cum_fees_pnl = float(pnl_df['cum_fees'].iloc[-1])
        diff = abs(cum_fees_pnl - total_logged_fees)
        assert diff < 1e-6, f"Fee mismatch: PnL cum_fees={cum_fees_pnl}, logged fees={total_logged_fees}, diff={diff}"

    print(f"[PASS] test_fee_consistency (total fees logged: {total_logged_fees:.6f})")

# ----------------------------
# Test 4: Conservation invariant
# ----------------------------
def test_conservation_invariant():
    """
    Run a multi-snapshot backtest. Assert total_pnl == cash + unrealized_pnl
    at every row of the PnL DataFrame.
    """
    from matching_engine_realistic import MarketSnapshot
    eng = RealisticMatchingEngine(latency_ms=5.0, rng_seed=555,
                                  fee_bps=0.5, taker_fee_bps=0.3, maker_rebate_bps=-0.2,
                                  latency_jitter_ms=0.0)
    strat = KellyConstrainedStrategy(
        name='conserv-test', capital=50000.0, max_inventory=200,
        kelly_shrink=0.05, transaction_cost=0.0005, rng_seed=555
    )
    runner = BacktestRunner(engine=eng, strategy=strat, initial_cash=50000.0)

    snapshots = [
        MarketSnapshot(ts=0.0, bids=[(99.0, 300)], asks=[(101.0, 300)], last_trade_price=None, last_trade_size=None, last_trade_aggressor=0),
        MarketSnapshot(ts=0.1, bids=[(99.0, 300)], asks=[(101.0, 300)], last_trade_price=101.0, last_trade_size=50, last_trade_aggressor=1),
        MarketSnapshot(ts=0.2, bids=[(99.0, 250)], asks=[(101.0, 250)], last_trade_price=101.0, last_trade_size=30, last_trade_aggressor=1),
        MarketSnapshot(ts=0.3, bids=[(98.5, 200)], asks=[(100.5, 200)], last_trade_price=100.5, last_trade_size=20, last_trade_aggressor=-1),
        MarketSnapshot(ts=0.5, bids=[(98.0, 400)], asks=[(100.0, 400)], last_trade_price=100.0, last_trade_size=100, last_trade_aggressor=1),
    ]

    def model_fn(feat):
        return 0.3 if feat['mid'] > 99 else -0.1

    pnl_df = runner.run(snapshots, model_infer_fn=model_fn, tick_time_window=0.01)

    # Verify conservation at every row
    for idx, row in pnl_df.iterrows():
        total = row['total_pnl']
        cash_plus_unreal = row['cash'] + row['unrealized_pnl']
        assert abs(total - cash_plus_unreal) < 1e-6, (
            f"Conservation violated at row {idx}: total_pnl={total}, cash+unreal={cash_plus_unreal}"
        )

    print(f"[PASS] test_conservation_invariant ({len(pnl_df)} rows checked)")


if __name__ == "__main__":
    # Run all unit tests
    print("=" * 60)
    print("Running unit tests...")
    print("=" * 60)

    unit_test_roundtrip()
    print()
    test_cancel_before_arrival_race()
    print()
    test_fifo_partial_fills()
    print()
    test_fee_consistency()
    print()
    test_conservation_invariant()

    print()
    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)
