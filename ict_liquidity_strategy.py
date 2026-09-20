"""
ICT / Liquidity-Sweep + FVG Strategy
=====================================
Implements the common "ICT concepts" style approach (liquidity sweep ->
market structure shift -> fair value gap retest entry) that TJR and many
similar educators teach publicly. These are widely-known price-action
concepts, not a proprietary formula — this is one reasonable, explicit
codification of them, not "the" official version (there is no single
official version; different educators vary the details).

Logic:
  LONG:
    1. Track a rolling liquidity pool: the lowest low over the last
       `liquidity_lookback` candles (excluding the current one) and the
       highest high over the same window.
    2. Liquidity sweep (sell-side): a candle's LOW dips below the recent
       low (grabbing resting stop-loss liquidity) but its CLOSE comes
       back above that level — a wick-based stop hunt, not a real
       breakdown.
    3. Market Structure Shift (MSS): within `mss_window` candles after
       the sweep, price must CLOSE above a nearby minor swing high
       (a pivot high formed since the sweep) — confirming short-term
       structure just flipped bullish.
    4. Fair Value Gap (FVG): scan the impulse leg from the sweep to the
       MSS candle for a 3-candle gap where candle[i-2].high < candle[i].low
       (a bullish imbalance / gap the market left behind).
    5. Entry: within `retest_window` candles after the MSS, wait for
       price to trade back into that FVG zone, then a confirmation
       candle (bullish close) triggers "long_entry".
    6. Stop: the low of the original sweep candle. Target: the opposite
       liquidity pool (the recent high at signal time), which functions
       as a natural take-profit level in this framework.

  SHORT is the mirror image (buy-side liquidity sweep, bearish MSS,
  bearish FVG, retest for short entry).

Reuses backtest(), load_csv(), load_yfinance(), and filter_session()
from breakout_retest_strategy.py so both strategies share the same
data-loading and backtesting plumbing.

Usage:
    python ict_liquidity_strategy.py --csv your_15m_data.csv
    python ict_liquidity_strategy.py --ticker MES=F --session-start 09:30 --session-end 16:00
"""

import argparse
import numpy as np
import pandas as pd

from breakout_retest_strategy import backtest, load_csv, load_yfinance, filter_session


# ----------------------------------------------------------------------
# Minor swing (pivot) detection — reused idea from the breakout strategy,
# but typically with a tighter left/right window since ICT-style structure
# shifts are read off small, local swing points rather than major ones.
# ----------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == window_high.max():
            pivot_high[i] = True
        if lows[i] == window_low.min():
            pivot_low[i] = True

    df = df.copy()
    df["pivot_high"] = pivot_high
    df["pivot_low"] = pivot_low
    return df


def find_fair_value_gap(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str):
    """
    Scans candles [start_idx, end_idx] for a 3-candle Fair Value Gap and
    returns the FIRST one found (chronologically), not the most recent.

    This matters: during a fast, continued impulse, later 3-candle windows
    can spuriously re-satisfy the same gap condition even though price has
    already traded back through that zone. Locking onto the first gap
    avoids drifting onto a stale/already-mitigated level as new candles
    keep arriving.

    direction: "bullish" (gap up, candle[i-2].high < candle[i].low) or
               "bearish" (gap down, candle[i-2].low > candle[i].high).
    Returns (fvg_low, fvg_high) of the first qualifying gap, or None.
    """
    for i in range(start_idx + 2, end_idx + 1):
        c0 = df.iloc[i - 2]
        c2 = df.iloc[i]
        if direction == "bullish" and c0["high"] < c2["low"]:
            return (c0["high"], c2["low"])  # (gap_low, gap_high)
        elif direction == "bearish" and c0["low"] > c2["high"]:
            return (c2["high"], c0["low"])  # (gap_low, gap_high)
    return None


# ----------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------
def generate_signals(
    df: pd.DataFrame,
    liquidity_lookback: int = 20,
    pivot_left: int = 3,
    pivot_right: int = 3,
    mss_window: int = 15,
    retest_window: int = 15,
) -> pd.DataFrame:
    df = find_pivots(df, pivot_left, pivot_right)
    n = len(df)
    signals = [None] * n

    # Rolling liquidity levels, computed from PRIOR candles only (shifted by 1)
    # so "recent high/low" never includes the current candle itself.
    roll_low = df["low"].rolling(liquidity_lookback).min().shift(1)
    roll_high = df["high"].rolling(liquidity_lookback).max().shift(1)

    # State machines: None -> "swept" (waiting for MSS) -> "shifted" (waiting for FVG retest)
    long_state = None
    long_sweep_idx = long_sweep_low = None
    long_mss_target = long_mss_idx = long_fvg = None
    long_liquidity_target = None

    short_state = None
    short_sweep_idx = short_sweep_high = None
    short_mss_target = short_mss_idx = short_fvg = None
    short_liquidity_target = None

    for i in range(n):
        row = df.iloc[i]
        rlow, rhigh = roll_low.iloc[i], roll_high.iloc[i]

        # ============== LONG side ==============
        if long_state is None and pd.notna(rlow):
            # Sweep sell-side liquidity: wick below recent low, close back above it
            if row["low"] < rlow and row["close"] > rlow:
                # The MSS target is the most recent swing high that existed
                # BEFORE this sweep — that's the structure level a bullish
                # reversal actually needs to reclaim. (Not a high made
                # during the impulse itself, which is usually the move's
                # own peak and gets re-tested, not broken, on the pullback.)
                search_start = max(0, i - liquidity_lookback * 3)
                pre_sweep = df.iloc[search_start:i]
                ph_mask = pre_sweep["pivot_high"]
                if ph_mask.any():
                    long_state = "swept"
                    long_sweep_idx = i
                    long_sweep_low = row["low"]
                    long_liquidity_target = rhigh
                    long_mss_target = pre_sweep.loc[ph_mask, "high"].iloc[-1]
                # else: no prior structure to break — skip, no valid setup yet

        elif long_state == "swept":
            bars_since = i - long_sweep_idx
            if bars_since > mss_window:
                long_state = None  # no MSS in time, setup expired
            elif row["close"] > long_mss_target:
                # Market Structure Shift confirmed. Move to "shifted" and
                # start watching for a retest into a FVG — the FVG may not
                # exist yet at this exact candle, so it's searched fresh
                # on each subsequent candle (using only data seen so far)
                # rather than fixed once at this moment.
                long_mss_idx = i
                long_state = "shifted"

        elif long_state == "shifted":
            bars_since = i - long_mss_idx
            if bars_since > retest_window:
                long_state = None
            else:
                # Recompute the FVG using everything from the sweep up to
                # (but not including) the current candle — causal, no lookahead.
                fvg = find_fair_value_gap(df, long_sweep_idx, i - 1, "bullish")
                if fvg is not None:
                    fvg_low, fvg_high = fvg
                    if row["low"] <= fvg_high and row["high"] >= fvg_low and row["close"] > row["open"]:
                        signals[i] = "long_entry"
                        long_state = None

        # ============== SHORT side ==============
        if short_state is None and pd.notna(rhigh):
            if row["high"] > rhigh and row["close"] < rhigh:
                search_start = max(0, i - liquidity_lookback * 3)
                pre_sweep = df.iloc[search_start:i]
                pl_mask = pre_sweep["pivot_low"]
                if pl_mask.any():
                    short_state = "swept"
                    short_sweep_idx = i
                    short_sweep_high = row["high"]
                    short_liquidity_target = rlow
                    short_mss_target = pre_sweep.loc[pl_mask, "low"].iloc[-1]

        elif short_state == "swept":
            bars_since = i - short_sweep_idx
            if bars_since > mss_window:
                short_state = None
            elif row["close"] < short_mss_target:
                short_mss_idx = i
                short_state = "shifted"

        elif short_state == "shifted":
            bars_since = i - short_mss_idx
            if bars_since > retest_window:
                short_state = None
            else:
                fvg = find_fair_value_gap(df, short_sweep_idx, i - 1, "bearish")
                if fvg is not None:
                    fvg_low, fvg_high = fvg
                    if row["low"] <= fvg_high and row["high"] >= fvg_low and row["close"] < row["open"]:
                        signals[i] = "short_entry"
                        short_state = None

    df["signal"] = signals
    return df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICT-concepts (liquidity sweep + MSS + FVG) signal generator")
    parser.add_argument("--csv", type=str, help="Path to a 15m OHLCV CSV file")
    parser.add_argument("--ticker", type=str, help="Ticker to fetch via yfinance (15m, last 60 days)")
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--pivot-left", type=int, default=3)
    parser.add_argument("--pivot-right", type=int, default=3)
    parser.add_argument("--mss-window", type=int, default=15, help="Max candles allowed between sweep and structure shift")
    parser.add_argument("--retest-window", type=int, default=15, help="Max candles allowed for price to retrace into the FVG")
    parser.add_argument("--rr-target", type=float, default=2.0)
    parser.add_argument("--session-start", type=str, default=None, help='e.g. "02:00" for London killzone')
    parser.add_argument("--session-end", type=str, default=None, help='e.g. "05:00" for London killzone')
    parser.add_argument("--session-timezone", type=str, default="America/New_York")
    args = parser.parse_args()

    if args.csv:
        data = load_csv(args.csv)
    elif args.ticker:
        data = load_yfinance(args.ticker)
    else:
        raise SystemExit("Provide either --csv path or --ticker symbol")

    data = filter_session(
        data,
        session_start=args.session_start,
        session_end=args.session_end,
        session_tz=args.session_timezone,
    )

    signals_df = generate_signals(
        data,
        liquidity_lookback=args.liquidity_lookback,
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        mss_window=args.mss_window,
        retest_window=args.retest_window,
    )

    entries = signals_df[signals_df["signal"].isin(["long_entry", "short_entry"])]
    print(f"\nFound {len(entries)} entry signals:\n")
    print(entries[["open", "high", "low", "close", "signal"]])

    results = backtest(signals_df, rr_target=args.rr_target)
    if len(results):
        print("\nBacktest results:")
        print(results)
        win_rate = (results["outcome"] == "target").mean()
        avg_r = results["pnl_r"].mean()
        print(f"\nTrades: {len(results)} | Win rate: {win_rate:.1%} | Avg R: {avg_r:.2f}")
    else:
        print("\nNo completed trades in this data range.")
