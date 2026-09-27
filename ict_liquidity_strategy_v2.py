"""
ICT / Liquidity-Sweep + FVG Strategy — v2 (fixed + improved)
==============================================================
This is a corrected and upgraded version of ict_liquidity_strategy.py,
based on the findings in the optimization study
(claude/ict-liquidity-sweep-optimization-study.md). Changes vs. v1:

  1. LOOKAHEAD BUG FIXED. v1's find_pivots() marks candle j as a pivot
     using data through j+pivot_right, but generate_signals() let the
     sweep-detection logic reference that pivot immediately, before it
     could actually have been confirmed. v2 only allows a pivot to be
     used once `pivot_right` bars have actually elapsed since it formed
     (see `_confirmed_pivot_search` below). This is not a tuning knob,
     it's a correctness fix -- v1's backtest numbers are optimistic and
     not reproducible live.

  2. LIQUIDITY-TARGET EXIT WIRED UP. v1 computed the opposite liquidity
     pool as "a natural take-profit level" but never actually used it --
     every trade exited at a blind fixed-R multiple. v2 adds a real
     partial-exit model ("hybrid" mode, the default): bank half the
     position at 1R, move the stop to breakeven on the remainder, and
     let the remainder run toward the actual liquidity pool (or the
     rr_target price, whichever is nearer, as a safety cap) instead of
     an arbitrary fixed multiple.

  3. TIME STOP ADDED. The optimization study found average holds of
     13-32 hours for the best NQ configs on 15m bars, with a long tail
     out to multiple days -- unresolved positions sitting open that long
     carry gap/news risk the backtest doesn't price in. v2 force-closes
     any trade still open after `time_stop_bars` bars, at that bar's
     close.

  4. STOP BUFFER ADDED. v1's stop is the raw sweep-candle wick -- a
     single-bar, noise-sensitive level. v2 can push the stop a small
     ATR fraction beyond the sweep extreme (`stop_buffer_atr_mult`) to
     reduce whipsaw stop-outs. Default 0 reproduces v1's raw-wick stop;
     set e.g. 0.15 to add 15% of the 14-bar ATR as a buffer.

  5. SESSION FILTER DEFAULTS TO 24H. The study found the killzone-only
     session filter in alerts.yml/live_alert_bot_ict.py collapses the
     tradable sample to single digits of trades over months -- nowhere
     near enough to validate. v2's CLI defaults --session-start/--end to
     None (unrestricted 24h), matching the only session config the study
     could actually validate. You can still pass a session window if you
     want one, but it is opt-in, not the default.

  6. SYMBOL-AGNOSTIC BY DESIGN, ES AND NQ BOTH SUPPORTED. Nothing in
     this file is instrument-specific -- pass --csv or --ticker for
     either ES/MES or NQ/NQ. IMPORTANT: the optimization study found
     this exact strategy has a real, broad edge on NQ and did NOT find
     one on ES with the v1 fixed-R exit. This file's fixes (liquidity
     target, time stop, stop buffer) were tested against both symbols
     specifically to see whether they change that -- see the
     accompanying backtest comparison. Read the results before assuming
     parity between the two symbols; "the code supports both" is not
     the same claim as "the edge exists equally on both."

Reuses load_csv-style helpers rewritten here to also accept the unix
"time" column format used by the uploaded TradingView-style CSVs
(CME_MINI_ES1_15.csv / CME_MINI_NQ1_15.csv), in addition to a
"timestamp" column, and to tolerate a missing volume column.

Usage:
    python ict_liquidity_strategy_v2.py --csv CME_MINI_NQ1_15_9eec6.csv \\
        --liquidity-lookback 30 --pivot-left 2 --pivot-right 2 \\
        --mss-window 20 --retest-window 15 --rr-target 2.5 \\
        --exit-mode hybrid --stop-buffer-atr 0.15 --time-stop-bars 200
"""

import argparse
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "time" in df.columns:
        # unix-seconds format used by the uploaded TradingView-style exports
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        raise SystemExit("CSV needs a 'timestamp' or 'time' column")
    df = df.sort_values("timestamp").set_index("timestamp")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def load_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(ticker, period="60d", interval="15m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]
    return data[["open", "high", "low", "close", "volume"]]


def filter_session(df: pd.DataFrame, session_start: str = None, session_end: str = None,
                    session_tz: str = "America/New_York") -> pd.DataFrame:
    """Optional time-of-day filter. Defaults to None/None = unrestricted 24h,
    per the optimization study's finding that killzone-only filtering
    starves this strategy of a validatable sample. Pass both start and end
    explicitly if you want a session window."""
    if not session_start or not session_end:
        return df
    idx = df.index
    idx_local = idx.tz_localize(session_tz) if idx.tz is None else idx.tz_convert(session_tz)
    start_t = pd.to_datetime(session_start).time()
    end_t = pd.to_datetime(session_end).time()
    times = pd.Series(idx_local.time, index=df.index)
    mask = (times >= start_t) & (times <= end_t)
    filtered = df.loc[mask]
    if filtered.empty:
        raise SystemExit(f"Session filter ({session_start}-{session_end} {session_tz}) removed all rows.")
    return filtered


# ----------------------------------------------------------------------
# Pivot detection
# ----------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        wh = highs[i - left: i + right + 1]
        wl = lows[i - left: i + right + 1]
        if highs[i] == wh.max():
            pivot_high[i] = True
        if lows[i] == wl.min():
            pivot_low[i] = True
    df = df.copy()
    df["pivot_high"] = pivot_high
    df["pivot_low"] = pivot_low
    return df


def find_fair_value_gap(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str):
    for i in range(start_idx + 2, end_idx + 1):
        c0 = df.iloc[i - 2]
        c2 = df.iloc[i]
        if direction == "bullish" and c0["high"] < c2["low"]:
            return (c0["high"], c2["low"])
        elif direction == "bearish" and c0["low"] > c2["high"]:
            return (c2["high"], c0["low"])
    return None


# ----------------------------------------------------------------------
# Signal generation -- FIX #1 (lookahead) applied here
# ----------------------------------------------------------------------
def generate_signals(
    df: pd.DataFrame,
    liquidity_lookback: int = 30,
    pivot_left: int = 2,
    pivot_right: int = 2,
    mss_window: int = 20,
    retest_window: int = 15,
) -> pd.DataFrame:
    df = find_pivots(df, pivot_left, pivot_right)
    n = len(df)
    signals = [None] * n
    stop_prices = [None] * n
    liquidity_targets = [None] * n

    roll_low = df["low"].rolling(liquidity_lookback).min().shift(1)
    roll_high = df["high"].rolling(liquidity_lookback).max().shift(1)

    ph = df["pivot_high"].values
    pl = df["pivot_low"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    long_state = None
    long_sweep_idx = long_sweep_low = long_mss_target = long_mss_idx = long_liq_target = None
    short_state = None
    short_sweep_idx = short_sweep_high = short_mss_target = short_mss_idx = short_liq_target = None

    for i in range(n):
        rlow, rhigh = roll_low.iloc[i], roll_high.iloc[i]
        row_low, row_high, row_close, row_open = lows[i], highs[i], closes[i], opens[i]

        # FIX #1: a pivot at index j is only usable once it is actually
        # confirmable, i.e. at bars i >= j + pivot_right. So when scanning
        # backward for "the most recent pivot", we only look at indices
        # up to i - 1 - pivot_right, not i - 1.
        max_confirmed_idx = i - 1 - pivot_right

        # ============== LONG ==============
        if long_state is None and pd.notna(rlow):
            if row_low < rlow and row_close > rlow:
                search_start = max(0, i - liquidity_lookback * 3)
                end_excl = max_confirmed_idx + 1
                found_idx = -1
                if end_excl > search_start:
                    seg = ph[search_start:end_excl]
                    nz = np.nonzero(seg)[0]
                    if len(nz):
                        found_idx = search_start + nz[-1]
                if found_idx >= 0:
                    long_state = "swept"
                    long_sweep_idx = i
                    long_sweep_low = row_low
                    long_liq_target = rhigh
                    long_mss_target = highs[found_idx]

        elif long_state == "swept":
            bars_since = i - long_sweep_idx
            if bars_since > mss_window:
                long_state = None
            elif row_close > long_mss_target:
                long_mss_idx = i
                long_state = "shifted"

        elif long_state == "shifted":
            bars_since = i - long_mss_idx
            if bars_since > retest_window:
                long_state = None
            else:
                fvg = find_fair_value_gap(df, long_sweep_idx, i - 1, "bullish")
                if fvg is not None:
                    fvg_low, fvg_high = fvg
                    if row_low <= fvg_high and row_high >= fvg_low and row_close > row_open:
                        signals[i] = "long_entry"
                        stop_prices[i] = long_sweep_low
                        liquidity_targets[i] = long_liq_target
                        long_state = None

        # ============== SHORT ==============
        if short_state is None and pd.notna(rhigh):
            if row_high > rhigh and row_close < rhigh:
                search_start = max(0, i - liquidity_lookback * 3)
                end_excl = max_confirmed_idx + 1
                found_idx = -1
                if end_excl > search_start:
                    seg = pl[search_start:end_excl]
                    nz = np.nonzero(seg)[0]
                    if len(nz):
                        found_idx = search_start + nz[-1]
                if found_idx >= 0:
                    short_state = "swept"
                    short_sweep_idx = i
                    short_sweep_high = row_high
                    short_liq_target = rlow
                    short_mss_target = lows[found_idx]

        elif short_state == "swept":
            bars_since = i - short_sweep_idx
            if bars_since > mss_window:
                short_state = None
            elif row_close < short_mss_target:
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
                    if row_low <= fvg_high and row_high >= fvg_low and row_close < row_open:
                        signals[i] = "short_entry"
                        stop_prices[i] = short_sweep_high
                        liquidity_targets[i] = short_liq_target
                        short_state = None

    df["signal"] = signals
    df["sweep_stop"] = stop_prices
    df["liquidity_target"] = liquidity_targets
    return df


# ----------------------------------------------------------------------
# Backtest -- FIX #2 (liquidity-target exit), #3 (time stop), #4 (stop buffer)
# ----------------------------------------------------------------------
def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def backtest(
    df: pd.DataFrame,
    rr_target: float = 2.5,
    exit_mode: str = "hybrid",       # "fixed_rr" (v1 behavior) | "liquidity" | "hybrid" (recommended)
    stop_buffer_atr_mult: float = 0.15,
    atr_window: int = 14,
    time_stop_bars: int = 200,
) -> pd.DataFrame:
    """
    exit_mode:
      "fixed_rr"  -- v1 behavior: stop = sweep extreme (+/- buffer), single
                     exit at rr_target * risk. Liquidity target unused.
      "liquidity" -- single exit at the opposite liquidity pool level
                     recorded at signal time (falls back to rr_target price
                     if the liquidity level is invalid, e.g. behind entry).
      "hybrid"    -- (recommended, matches the study's "wire up the
                     liquidity target" suggestion) bank half the position
                     at 1R, move stop to breakeven on the remainder, let
                     the remainder run toward the liquidity target capped
                     at rr_target (whichever is nearer), or exit at the
                     time stop if neither hits first.

    All modes respect `time_stop_bars`: any trade still open after that
    many bars is closed at that bar's close price.
    """
    atr = _atr(df, atr_window).values
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    ts = df.index
    n = len(df)
    trades = []

    for i in range(n):
        sig = df["signal"].iloc[i]
        if sig not in ("long_entry", "short_entry"):
            continue
        entry = c[i]
        sweep_stop = df["sweep_stop"].iloc[i]
        liq_target = df["liquidity_target"].iloc[i]
        if pd.isna(sweep_stop):
            continue
        buffer_amt = (atr[i] * stop_buffer_atr_mult) if not np.isnan(atr[i]) else 0.0
        direction = "long" if sig == "long_entry" else "short"

        if direction == "long":
            stop = sweep_stop - buffer_amt
            risk = entry - stop
        else:
            stop = sweep_stop + buffer_amt
            risk = stop - entry
        if risk <= 0 or np.isnan(risk):
            continue

        rr_price = entry + rr_target * risk if direction == "long" else entry - rr_target * risk

        if exit_mode == "fixed_rr":
            target1 = None
            target_final = rr_price
        elif exit_mode == "liquidity":
            if pd.notna(liq_target) and ((direction == "long" and liq_target > entry) or
                                          (direction == "short" and liq_target < entry)):
                target_final = liq_target
            else:
                target_final = rr_price
            target1 = None
        else:  # hybrid
            target1 = entry + risk if direction == "long" else entry - risk  # 1R partial
            if pd.notna(liq_target) and ((direction == "long" and liq_target > target1) or
                                          (direction == "short" and liq_target < target1)):
                target_final = liq_target
                # cap runaway targets at rr_target price so a distant liquidity
                # pool doesn't turn into an unbounded, never-hit target
                if direction == "long":
                    target_final = min(target_final, rr_price)
                else:
                    target_final = max(target_final, rr_price)
            else:
                target_final = rr_price

        max_j = min(n, i + 1 + time_stop_bars)
        outcome = None
        exit_price = None
        exit_j = None
        half_banked = False
        stop_cur = stop

        for j in range(i + 1, max_j):
            if direction == "long":
                hit_stop = l[j] <= stop_cur
                hit_t1 = (target1 is not None) and (not half_banked) and h[j] >= target1
                hit_tf = h[j] >= target_final
            else:
                hit_stop = h[j] >= stop_cur
                hit_t1 = (target1 is not None) and (not half_banked) and l[j] <= target1
                hit_tf = l[j] <= target_final

            if hit_stop and not half_banked:
                outcome, exit_price, exit_j = "stop", stop_cur, j
                break
            if exit_mode == "hybrid" and hit_t1 and not half_banked:
                half_banked = True
                stop_cur = entry  # move stop to breakeven on the remainder
                if hit_tf:  # same bar also reaches the final target
                    outcome, exit_price, exit_j = "target_partial_then_full", target_final, j
                    break
                continue
            if hit_stop and half_banked:
                outcome, exit_price, exit_j = "target_partial_then_be", entry, j
                break
            if hit_tf:
                outcome = "target_partial_then_full" if half_banked else "target"
                exit_price, exit_j = target_final, j
                break

        if outcome is None:
            # time stop: force-close at the last bar's close
            exit_j = max_j - 1 if max_j - 1 > i else None
            if exit_j is None:
                continue
            exit_price = c[exit_j]
            outcome = "time_stop"

        # ---- P&L in R ----
        if outcome == "stop":
            pnl_r = -1.0
        elif outcome == "target":
            pnl_r = rr_target
        elif outcome == "target_partial_then_full":
            second_half_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
            pnl_r = 0.5 * 1.0 + 0.5 * second_half_r
        elif outcome == "target_partial_then_be":
            pnl_r = 0.5 * 1.0 + 0.5 * 0.0
        else:  # time_stop
            raw_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
            pnl_r = (0.5 * 1.0 + 0.5 * raw_r) if half_banked else raw_r

        hold_min = (ts[exit_j] - ts[i]).total_seconds() / 60.0
        trades.append({
            "entry_time": ts[i], "exit_time": ts[exit_j], "direction": direction,
            "entry": entry, "stop": stop, "outcome": outcome, "pnl_r": pnl_r,
            "risk_points": risk, "hold_min": hold_min,
        })

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICT-concepts (liquidity sweep + MSS + FVG) v2 -- fixed + improved")
    parser.add_argument("--csv", type=str, help="Path to a 15m OHLCV CSV file")
    parser.add_argument("--ticker", type=str, help="Ticker to fetch via yfinance (15m, last 60 days)")
    parser.add_argument("--liquidity-lookback", type=int, default=30)
    parser.add_argument("--pivot-left", type=int, default=2)
    parser.add_argument("--pivot-right", type=int, default=2)
    parser.add_argument("--mss-window", type=int, default=20)
    parser.add_argument("--retest-window", type=int, default=15)
    parser.add_argument("--rr-target", type=float, default=2.5)
    parser.add_argument("--exit-mode", type=str, default="hybrid", choices=["fixed_rr", "liquidity", "hybrid"])
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15, help="Stop buffer as a fraction of ATR(14)")
    parser.add_argument("--time-stop-bars", type=int, default=200, help="Force-close after this many bars")
    parser.add_argument("--session-start", type=str, default=None, help="Optional, e.g. '02:00'. Default: unrestricted 24h")
    parser.add_argument("--session-end", type=str, default=None, help="Optional, e.g. '05:00'. Default: unrestricted 24h")
    parser.add_argument("--session-timezone", type=str, default="America/New_York")
    args = parser.parse_args()

    if args.csv:
        data = load_csv(args.csv)
    elif args.ticker:
        data = load_yfinance(args.ticker)
    else:
        raise SystemExit("Provide either --csv path or --ticker symbol")

    data = filter_session(data, args.session_start, args.session_end, args.session_timezone)

    signals_df = generate_signals(
        data, liquidity_lookback=args.liquidity_lookback, pivot_left=args.pivot_left,
        pivot_right=args.pivot_right, mss_window=args.mss_window, retest_window=args.retest_window,
    )

    entries = signals_df[signals_df["signal"].isin(["long_entry", "short_entry"])]
    print(f"\nFound {len(entries)} entry signals:\n")
    print(entries[["open", "high", "low", "close", "signal"]])

    results = backtest(
        signals_df, rr_target=args.rr_target, exit_mode=args.exit_mode,
        stop_buffer_atr_mult=args.stop_buffer_atr, time_stop_bars=args.time_stop_bars,
    )
    if len(results):
        print("\nBacktest results:")
        print(results)
        win_rate = (results["pnl_r"] > 0).mean()
        avg_r = results["pnl_r"].mean()
        print(f"\nTrades: {len(results)} | Win rate: {win_rate:.1%} | Avg R: {avg_r:.2f}")
    else:
        print("\nNo completed trades in this data range.")
