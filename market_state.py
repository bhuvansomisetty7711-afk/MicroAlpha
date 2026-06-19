#!/usr/bin/env python3
"""
market_state.py
================
Pulls live crypto market data from Binance PUBLIC endpoints (no API key needed)
and builds a feature set for market-state / regime analysis.

Assets : BTC, ETH, BNB, SOL, DOGE, LINK, AVAX  (all vs USDT)

Pipeline
--------
Stage 1 (Data Pull)  : 4H + 1H OHLCV, live bid/ask spread, perp funding rate,
                       taker buy/sell volume (from klines).
Stage 2 (Features)   : volatility, momentum, microstructure, positioning and
                       cross-sectional (relative-to-BTC) features. Log returns
                       are used throughout.

Output
------
  * A clean summary table printed to the console (one row per asset).
  * A timestamped CSV (features_YYYYMMDD_HHMM.csv) with the FULL feature set,
    so repeated runs build a historical snapshot dataset over time.

Run with:  python market_state.py   (re-runnable any time for a fresh snapshot)
"""

import sys
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ASSETS = ["BTC", "ETH", "BNB", "SOL", "DOGE", "LINK", "AVAX"]
QUOTE = "USDT"
SYMBOLS = {a: f"{a}{QUOTE}" for a in ASSETS}          # e.g. BTC -> BTCUSDT
BTC_SYMBOL = SYMBOLS["BTC"]

SPOT_BASE = "https://api.binance.com"
FUT_BASE = "https://fapi.binance.com"

KLINES_EP = f"{SPOT_BASE}/api/v3/klines"
BOOK_EP = f"{SPOT_BASE}/api/v3/ticker/bookTicker"
FUNDING_EP = f"{FUT_BASE}/fapi/v1/premiumIndex"

REQUEST_TIMEOUT = 10          # seconds per HTTP request
RATE_LIMIT_SLEEP = 0.25       # polite pause between calls to respect rate limits
MAX_RETRIES = 3               # transient-failure retries per request

# Binance kline columns (12 fields). Column index 9 = taker_buy_base.
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("market_state")


# --------------------------------------------------------------------------- #
# Low-level HTTP helper
# --------------------------------------------------------------------------- #
def _get(url, params=None):
    """GET with small retry/back-off. Returns parsed JSON or raises on failure."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:                       # noqa: BLE001 broad on purpose
            last_err = exc
            # Back off a touch on transient errors / rate limiting (HTTP 429/418).
            time.sleep(RATE_LIMIT_SLEEP * attempt)
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} tries: {last_err}")


# --------------------------------------------------------------------------- #
# Stage 1 — Data pull
# --------------------------------------------------------------------------- #
def fetch_klines(symbol, interval, limit):
    """
    Fetch OHLCV candles for `symbol`.

    Returns a tidy DataFrame with numeric OHLCV columns plus `taker_buy_base`
    and a derived `taker_buy_ratio` = taker_buy_base / volume per candle
    (the share of volume that lifted the offer — a buy/sell imbalance proxy).
    """
    raw = _get(KLINES_EP, {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(raw, columns=KLINE_COLS)

    num_cols = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Taker buy ratio per candle. Guard against zero-volume candles.
    df["taker_buy_ratio"] = np.where(
        df["volume"] > 0, df["taker_buy_base"] / df["volume"], np.nan
    )
    return df[["open_time", "close_time", "open", "high", "low", "close",
               "volume", "taker_buy_base", "taker_buy_ratio"]].reset_index(drop=True)


def fetch_spread(symbol):
    """
    Live best bid/ask from the order book top.

    Returns dict with bid, ask, mid, spread (absolute) and spread_bps
    (spread in basis points of the mid price — the relevant liquidity metric).
    """
    data = _get(BOOK_EP, {"symbol": symbol})
    bid = float(data["bidPrice"])
    ask = float(data["askPrice"])
    mid = (bid + ask) / 2.0
    spread = ask - bid
    spread_bps = (spread / mid) * 1e4 if mid > 0 else np.nan
    return {"bid": bid, "ask": ask, "mid": mid,
            "spread": spread, "spread_bps": spread_bps}


def fetch_funding(symbol):
    """
    Perp funding rate from the USDⓈ-M futures premium index.

    `lastFundingRate` is the most recent settled funding rate. Positive => longs
    pay shorts (crowded long); negative => shorts pay longs (crowded short).
    Returns NaN if the symbol has no perp listing.
    """
    try:
        data = _get(FUNDING_EP, {"symbol": symbol})
        return float(data["lastFundingRate"])
    except Exception as exc:                            # noqa: BLE001
        log.warning("Funding rate unavailable for %s (%s)", symbol, exc)
        return np.nan


def pull_asset(asset):
    """
    Pull every Stage-1 input for one asset. Returns a dict, or None on failure
    so the caller can log-and-skip without crashing the whole run.
    """
    symbol = SYMBOLS[asset]
    try:
        k4h = fetch_klines(symbol, "4h", 60)
        time.sleep(RATE_LIMIT_SLEEP)
        k1h = fetch_klines(symbol, "1h", 48)
        time.sleep(RATE_LIMIT_SLEEP)
        spread = fetch_spread(symbol)
        time.sleep(RATE_LIMIT_SLEEP)
        funding = fetch_funding(symbol)
        time.sleep(RATE_LIMIT_SLEEP)
        log.info("Pulled %s (%d x4H, %d x1H)", symbol, len(k4h), len(k1h))
        return {"asset": asset, "symbol": symbol,
                "k4h": k4h, "k1h": k1h, "spread": spread, "funding": funding}
    except Exception as exc:                            # noqa: BLE001
        log.error("FAILED to pull %s — skipping. (%s)", symbol, exc)
        return None


# --------------------------------------------------------------------------- #
# Stage 2 — Feature helpers
# --------------------------------------------------------------------------- #
def log_returns(prices):
    """Log returns of a price series (NaN for the first element)."""
    return np.log(prices / prices.shift(1))


def realized_vol(rets, window=14):
    """Std of log returns over a trailing window — point-in-time realized vol."""
    return rets.tail(window).std()


def compute_features(data, btc_ref):
    """
    Build the full feature dict for one asset.

    `data`    : output of pull_asset() for this asset.
    `btc_ref` : dict with BTC's 4H returns + 24-bar return, used for the
                cross-sectional (relative-strength / correlation) features.
    """
    k4h = data["k4h"].copy()
    k1h = data["k1h"].copy()
    spread = data["spread"]
    funding = data["funding"]

    close4 = k4h["close"]
    rets4 = log_returns(close4).dropna()               # 4H log returns
    price = float(close4.iloc[-1])

    f = {"asset": data["asset"], "symbol": data["symbol"], "price": price}

    # ---- Volatility -------------------------------------------------------- #
    # Realized vol: dispersion of recent returns (trailing 14 x 4H bars).
    rv = realized_vol(rets4, 14)
    f["realized_vol_14"] = rv
    # Annualized (6 x 4H bars/day * 365). Optional convenience figure.
    f["realized_vol_14_ann"] = rv * np.sqrt(6 * 365)

    # Vol-of-vol: std of a rolling 6-bar realized-vol series — how unstable
    # volatility itself is (regime turbulence).
    rolling_rv = rets4.rolling(6).std().dropna()
    f["vol_of_vol"] = rolling_rv.tail(20).std() if len(rolling_rv) else np.nan

    # ATR(14) and range-expansion: is the latest bar's range stretching vs norm?
    high, low, prev_close = k4h["high"], k4h["low"], close4.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    f["atr_14"] = true_range.tail(14).mean()
    cur_range = float((high - low).iloc[-1])
    mean_range = float((high - low).tail(14).mean())
    f["range_expansion"] = cur_range / mean_range if mean_range > 0 else np.nan

    # Vol regime: compare last 6-bar vol to the prior 6-bar vol.
    recent_vol = rets4.tail(6).std()
    prior_vol = rets4.iloc[-12:-6].std() if len(rets4) >= 12 else np.nan
    f["vol_regime"] = (
        "expanding" if pd.notna(prior_vol) and recent_vol > prior_vol
        else "contracting"
    )

    # ---- Momentum ---------------------------------------------------------- #
    # Cumulative log return over the last N bars (sum of log returns = log of
    # compounded return). Captures trend strength at multiple horizons.
    for n in (6, 12, 24):
        f[f"mom_{n}"] = rets4.tail(n).sum() if len(rets4) >= n else np.nan

    # Momentum persistence: lag-1 autocorrelation of the last 20 returns.
    # > 0 => trending (returns follow through); < 0 => mean-reverting/choppy.
    last20 = rets4.tail(20)
    f["mom_persistence"] = last20.autocorr(lag=1) if len(last20) >= 3 else np.nan

    # Distance from 20-bar MA as a vol-scaled z-score: how stretched is price
    # relative to its mean, normalized by realized vol so it's comparable.
    ma20 = close4.tail(20).mean()
    denom = rv * price
    f["ma20_zscore"] = (price - ma20) / denom if denom and denom > 0 else np.nan

    # ---- Microstructure ---------------------------------------------------- #
    # Volume z-score: is the latest bar's volume unusual vs the last 20 bars?
    vol = k4h["volume"]
    vmean, vstd = vol.tail(20).mean(), vol.tail(20).std()
    f["volume_zscore"] = (vol.iloc[-1] - vmean) / vstd if vstd and vstd > 0 else np.nan

    # Volume-price confirmation: correlation of last 10 bars' returns with
    # volume changes. Positive => moves are backed by volume (conviction).
    vol_chg = vol.diff()
    tail_ret = rets4.tail(10).reset_index(drop=True)
    tail_volchg = vol_chg.tail(10).reset_index(drop=True)
    if len(tail_ret) >= 3 and tail_ret.std() > 0 and tail_volchg.std() > 0:
        f["vol_price_confirm"] = float(np.corrcoef(tail_ret, tail_volchg)[0, 1])
    else:
        f["vol_price_confirm"] = np.nan

    # Live spread in bps (liquidity / transaction-cost proxy).
    f["spread_bps"] = spread["spread_bps"]
    f["bid"] = spread["bid"]
    f["ask"] = spread["ask"]

    # Taker buy ratio: aggressive-buy share of volume. Latest bar + 6-bar mean
    # smooth a buy/sell imbalance read (> 0.5 => buyers lifting offers).
    tbr = k4h["taker_buy_ratio"]
    f["taker_buy_ratio"] = float(tbr.iloc[-1])
    f["taker_buy_ratio_6avg"] = float(tbr.tail(6).mean())

    # ---- Positioning ------------------------------------------------------- #
    f["funding_rate"] = funding
    # Funding/price divergence: crowded positioning fighting the trend often
    # precedes squeezes. Flag price-up-but-funding-down (or the inverse).
    price_dir_6 = float(rets4.tail(6).sum())            # 6-bar price direction
    if pd.notna(funding):
        diverge = (price_dir_6 > 0 and funding < 0) or (price_dir_6 < 0 and funding > 0)
        f["funding_divergence"] = bool(diverge)
    else:
        f["funding_divergence"] = np.nan

    # ---- Relative (cross-sectional vs BTC) --------------------------------- #
    # 24H return = cumulative log return over last 6 x 4H bars (6*4h = 24h).
    ret_24h = rets4.tail(6).sum() if len(rets4) >= 6 else np.nan
    f["ret_24h"] = ret_24h
    f["rel_strength_vs_btc"] = ret_24h - btc_ref["ret_24h"]

    # Rolling 24-bar correlation of this asset's 4H returns to BTC's — how
    # tightly it co-moves with the market beta (1.0 for BTC itself).
    f["btc_corr_24"] = _aligned_corr(rets4, btc_ref["rets4"], window=24)

    # ---- Finer 1H state (where it helps) ----------------------------------- #
    rets1 = log_returns(k1h["close"]).dropna()
    f["realized_vol_1h_14"] = realized_vol(rets1, 14)   # short-horizon vol
    f["mom_1h_12"] = rets1.tail(12).sum() if len(rets1) >= 12 else np.nan  # ~12h trend

    f["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f


def _aligned_corr(a, b, window=24):
    """Correlation of the trailing `window` of two return series, index-aligned."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().tail(window)
    if len(joined) < 3 or joined["a"].std() == 0 or joined["b"].std() == 0:
        return np.nan
    return float(joined["a"].corr(joined["b"]))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
SUMMARY_COLS = [
    ("asset", "Asset"),
    ("realized_vol_14", "RVol14"),
    ("vol_regime", "VolRegime"),
    ("mom_24", "Mom24"),
    ("mom_persistence", "MomPersist"),
    ("ma20_zscore", "MAz"),
    ("volume_zscore", "VolZ"),
    ("spread_bps", "SpreadBps"),
    ("taker_buy_ratio", "TakerBuy"),
    ("funding_rate", "Funding"),
    ("rel_strength_vs_btc", "RelVsBTC"),
    ("btc_corr_24", "BTCcorr"),
]


def print_summary(df):
    """Pretty per-asset summary table to the console."""
    view = df[[c for c, _ in SUMMARY_COLS]].copy()
    view.columns = [h for _, h in SUMMARY_COLS]

    # Sensible rounding per column.
    round_map = {
        "RVol14": 4, "Mom24": 4, "MomPersist": 3, "MAz": 2, "VolZ": 2,
        "SpreadBps": 2, "TakerBuy": 3, "Funding": 6, "RelVsBTC": 4, "BTCcorr": 2,
    }
    for col, nd in round_map.items():
        view[col] = pd.to_numeric(view[col], errors="coerce").round(nd)

    print("\n" + "=" * 100)
    print(f"  MARKET-STATE SNAPSHOT — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("=" * 100)
    try:
        print(view.to_markdown(index=False))            # uses tabulate if present
    except Exception:
        print(view.to_string(index=False))
    print("=" * 100)


def write_csv(df):
    """Dump the full feature set to a timestamped CSV (local time of the run)."""
    fname = f"features_{datetime.now():%Y%m%d_%H%M}.csv"
    df.to_csv(fname, index=False)
    log.info("Full feature set written to %s (%d assets, %d cols)",
             fname, len(df), df.shape[1])
    return fname


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    log.info("Starting market-state pull for %d assets...", len(ASSETS))

    # Stage 1: pull everything, skipping assets that fail.
    pulled = {}
    for asset in ASSETS:
        res = pull_asset(asset)
        if res is not None:
            pulled[asset] = res

    if BTC_SYMBOL not in [d["symbol"] for d in pulled.values()] or "BTC" not in pulled:
        log.error("BTC data unavailable — relative features need BTC. Aborting.")
        sys.exit(1)

    # BTC reference series for cross-sectional features.
    btc_rets4 = log_returns(pulled["BTC"]["k4h"]["close"]).dropna()
    btc_ref = {
        "rets4": btc_rets4,
        "ret_24h": btc_rets4.tail(6).sum() if len(btc_rets4) >= 6 else np.nan,
    }

    # Stage 2: compute features per asset.
    rows = []
    for asset, data in pulled.items():
        try:
            rows.append(compute_features(data, btc_ref))
        except Exception as exc:                        # noqa: BLE001
            log.error("Feature computation failed for %s (%s)", asset, exc)

    if not rows:
        log.error("No features computed — nothing to output.")
        sys.exit(1)

    features = pd.DataFrame(rows)

    # Output: console table + timestamped CSV snapshot.
    print_summary(features)
    write_csv(features)


if __name__ == "__main__":
    main()
