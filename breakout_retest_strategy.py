"""
Breakout + 50% Retest Strategy
===============================

Logic:
  LONG:
    1. Find swing highs/lows (pivots) on 15-min candles.
    2. Breakout: a candle CLOSES above the most recent swing high.
    3. Impulse leg = swing_low (before the swing high) -> swing_high.
    4. Retest level = swing_high - 0.5 * (swing_high - swing_low).
    5. Within `retest_window` candles, price must pull back into the
       retest zone (with tolerance) and then print a bullish
       confirmation candle (close > open, closing back above the zone).
    6. Invalidation: if price closes back below the original swing_high
       before confirming, the setup is cancelled.

  SHORT is the mirror image.

This is a signal generator + simple backtester, not a live trading bot.
Wire the `generate_signals()` output into your own execution layer
(broker API, webhook, alert bot, etc.) when you're ready.

Usage:
    python breakout_retest_strategy.py --csv your_15m_data.csv
    python breakout_retest_strategy.py --ticker AAPL   # uses yfinance, last 60 days of 15m bars

CSV format expected: columns [timestamp, open, high, low, close, volume]
"""

import argparse
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Pivot (swing high/low) detection
# ----------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    """
    Marks a candle as a pivot high/low if its high/low is the most
    extreme within `left` candles before and `right` candles after it.

    NOTE: a pivot at index i is only *confirmed* once you reach index
    i + right (you need the future candles to know it held). The signal
    loop below respects this lag so there's no lookahead bias.
    """
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


# ----------------------------------------------------------------------
# 2. Signal generation (state machine per direction)
# ----------------------------------------------------------------------
def generate_signals(
    df: pd.DataFrame,
    pivot_left: int = 5,
    pivot_right: int = 5,
    retest_window: int = 20,
    retest_tolerance: float = 0.10,  # +/- 10% of leg range around the 50% line
    volume_multiplier: float = None,  # e.g. 1.5 = breakout volume must be >=1.5x the rolling average. None disables the filter.
    volume_ma_window: int = 20,
) -> pd.DataFrame:
    df = find_pivots(df, pivot_left, pivot_right)
    n = len(df)

    # Rolling average volume, computed from PRIOR candles only (shifted by 1)
    # so the breakout candle is compared against history, not against itself.
    if volume_multiplier is not None:
        vol_ma = df["volume"].rolling(volume_ma_window).mean().shift(1)
    else:
        vol_ma = None

    signals = [None] * n  # "long_entry" / "short_entry" / "long_invalid" / "short_invalid"

    last_pivot_high = last_pivot_high_idx = None
    last_pivot_low = last_pivot_low_idx = None
    prior_pivot_low_before_high = None  # swing low that preceded the current swing high
    prior_pivot_high_before_low = None  # swing high that preceded the current swing low

    long_state = None   # None | "broken" (waiting for retest) | done
    short_state = None
    long_breakout_level = long_retest_level = long_broken_at = None
    short_breakout_level = short_retest_level = short_broken_at = None

    for i in range(n):
        row = df.iloc[i]

        # Reveal pivots once confirmed (lag by `pivot_right` bars, no lookahead)
        confirm_idx = i - pivot_right
        if confirm_idx >= 0:
            if df["pivot_high"].iloc[confirm_idx]:
                prior_pivot_low_before_high = last_pivot_low
                last_pivot_high = df["high"].iloc[confirm_idx]
                last_pivot_high_idx = confirm_idx
            if df["pivot_low"].iloc[confirm_idx]:
                prior_pivot_high_before_low = last_pivot_high
                last_pivot_low = df["low"].iloc[confirm_idx]
                last_pivot_low_idx = confirm_idx

        # ---------------- LONG side ----------------
        if long_state is None and last_pivot_high is not None and prior_pivot_low_before_high is not None:
            volume_ok = True
            if volume_multiplier is not None:
                avg_vol = vol_ma.iloc[i]
                volume_ok = pd.notna(avg_vol) and row["volume"] >= volume_multiplier * avg_vol

            if row["close"] > last_pivot_high and volume_ok:
                leg_range = last_pivot_high - prior_pivot_low_before_high
                if leg_range > 0:
                    long_breakout_level = last_pivot_high
                    long_retest_level = last_pivot_high - 0.5 * leg_range
                    long_broken_at = i
                    long_state = "broken"

        elif long_state == "broken":
            bars_since = i - long_broken_at
            tol = retest_tolerance * (long_breakout_level - long_retest_level)
            zone_hi, zone_lo = long_retest_level + tol, long_retest_level - tol

            # Invalidation: closes back below the broken level (failed breakout)
            if row["close"] < long_retest_level - 2 * tol:
                signals[i] = "long_invalid"
                long_state = None

            # Touch the retest zone
            elif row["low"] <= zone_hi and row["low"] >= zone_lo - tol:
                # Confirmation candle: bullish close, back above the zone
                if row["close"] > row["open"] and row["close"] >= long_retest_level:
                    signals[i] = "long_entry"
                    long_state = None  # reset, ready for next setup

            if bars_since > retest_window and long_state == "broken":
                long_state = None  # setup expired, no retest happened in time

        # ---------------- SHORT side ----------------
        if short_state is None and last_pivot_low is not None and prior_pivot_high_before_low is not None:
            volume_ok = True
            if volume_multiplier is not None:
                avg_vol = vol_ma.iloc[i]
                volume_ok = pd.notna(avg_vol) and row["volume"] >= volume_multiplier * avg_vol

            if row["close"] < last_pivot_low and volume_ok:
                leg_range = prior_pivot_high_before_low - last_pivot_low
                if leg_range > 0:
                    short_breakout_level = last_pivot_low
                    short_retest_level = last_pivot_low + 0.5 * leg_range
                    short_broken_at = i
                    short_state = "broken"

        elif short_state == "broken":
            bars_since = i - short_broken_at
            tol = retest_tolerance * (short_retest_level - short_breakout_level)
            zone_hi, zone_lo = short_retest_level + tol, short_retest_level - tol

            if row["close"] > short_retest_level + 2 * tol:
                signals[i] = "short_invalid"
                short_state = None

            elif row["high"] >= zone_lo and row["high"] <= zone_hi + tol:
                if row["close"] < row["open"] and row["close"] <= short_retest_level:
                    signals[i] = "short_entry"
                    short_state = None

            if bars_since > retest_window and short_state == "broken":
                short_state = None

    df["signal"] = signals
    return df


# ----------------------------------------------------------------------
# 3. Very simple backtest: enter on signal, exit on stop or measured-move target
# ----------------------------------------------------------------------
def backtest(df: pd.DataFrame, rr_target: float = 2.0) -> pd.DataFrame:
    """
    rr_target: reward-to-risk multiple for the take-profit.
    Stop = the retest extreme (low of the confirmation candle for longs,
    high of it for shorts). Very simplified — no slippage/fees modeled.
    """
    trades = []
    for i, row in df.iterrows():
        if row["signal"] not in ("long_entry", "short_entry"):
            continue

        entry = row["close"]
        if row["signal"] == "long_entry":
            stop = row["low"]
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + rr_target * risk
            direction = "long"
        else:
            stop = row["high"]
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - rr_target * risk
            direction = "short"

        # Walk forward to see which hits first: stop or target
        outcome, exit_price, exit_idx = "open", None, None
        idx_pos = df.index.get_loc(i)
        for j in range(idx_pos + 1, len(df)):
            bar = df.iloc[j]
            if direction == "long":
                hit_stop = bar["low"] <= stop
                hit_target = bar["high"] >= target
            else:
                hit_stop = bar["high"] >= stop
                hit_target = bar["low"] <= target

            if hit_stop and hit_target:
                outcome, exit_price = "stop", stop  # conservative: assume stop hit first
            elif hit_stop:
                outcome, exit_price = "stop", stop
            elif hit_target:
                outcome, exit_price = "target", target

            if outcome != "open":
                exit_idx = df.index[j]
                break

        pnl_r = None
        if outcome == "stop":
            pnl_r = -1.0
        elif outcome == "target":
            pnl_r = rr_target

        trades.append(
            {
                "entry_time": i,
                "direction": direction,
                "entry": entry,
                "stop": stop,
                "target": target,
                "outcome": outcome,
                "exit_time": exit_idx,
                "pnl_r": pnl_r,
            }
        )

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------
# 4. Data loading helpers
# ----------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def load_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    # yfinance only allows ~60 days of history at 15m resolution
    data = yf.download(ticker, period="60d", interval="15m", progress=False)

    # Newer yfinance versions return MultiIndex columns (e.g. ('Open', 'AAPL'))
    # even for a single ticker. Flatten to plain column names either way.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]

    return data[["open", "high", "low", "close", "volume"]]


def filter_session(
    df: pd.DataFrame,
    session_start: str = None,
    session_end: str = None,
    session_tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Restricts the data to a daily time-of-day window (e.g. US regular
    trading hours, 09:30-16:00 Eastern). Useful for near-24h markets
    like futures (ES=F, MES=F, NQ=F, MNQ=F) where overnight/thin-liquidity
    candles can produce noisy, low-quality pivots if left in.

    session_start / session_end: "HH:MM" strings, e.g. "09:30" and "16:00".
    If either is None, no filtering is applied (returns df unchanged).

    NOTE: this simplification treats the filtered rows as if they were
    still contiguous — the pivot-detection window (`pivot_left`/`pivot_right`)
    will span across the overnight gap between one day's close and the next
    day's open. For most uses this is fine, but be aware the leg measured
    right at the start/end of a session may straddle two different days.
    """
    if not session_start or not session_end:
        return df

    idx = df.index
    if idx.tz is None:
        # yfinance intraday data is sometimes tz-naive (already in
        # exchange-local time) and sometimes tz-aware (UTC), depending on
        # version/ticker. If naive, assume it's already in session_tz.
        idx_local = idx.tz_localize(session_tz)
    else:
        idx_local = idx.tz_convert(session_tz)

    start_t = pd.to_datetime(session_start).time()
    end_t = pd.to_datetime(session_end).time()
    times = pd.Series(idx_local.time, index=df.index)
    mask = (times >= start_t) & (times <= end_t)

    filtered = df.loc[mask]
    if filtered.empty:
        raise SystemExit(
            f"Session filter ({session_start}-{session_end} {session_tz}) "
            f"removed all rows — check the timezone and that your data's "
            f"timestamps are what you expect."
        )
    return filtered


# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breakout + 50% retest signal generator")
    parser.add_argument("--csv", type=str, help="Path to a 15m OHLCV CSV file")
    parser.add_argument("--ticker", type=str, help="Ticker to fetch via yfinance (15m, last 60 days)")
    parser.add_argument("--pivot-left", type=int, default=5)
    parser.add_argument("--pivot-right", type=int, default=5)
    parser.add_argument("--retest-window", type=int, default=20)
    parser.add_argument("--retest-tolerance", type=float, default=0.10)
    parser.add_argument("--rr-target", type=float, default=2.0)
    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=None,
        help="Require breakout volume >= this multiple of the rolling average (e.g. 1.5). Omit to disable the filter.",
    )
    parser.add_argument("--volume-ma-window", type=int, default=20)
    parser.add_argument(
        "--session-start",
        type=str,
        default=None,
        help='Only consider candles from this time onward, e.g. "09:30". Useful for futures. Omit to disable.',
    )
    parser.add_argument(
        "--session-end",
        type=str,
        default=None,
        help='Only consider candles up to this time, e.g. "16:00". Must be set together with --session-start.',
    )
    parser.add_argument(
        "--session-timezone",
        type=str,
        default="America/New_York",
        help="Timezone the session-start/session-end times are interpreted in (default: US Eastern).",
    )
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
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        retest_window=args.retest_window,
        retest_tolerance=args.retest_tolerance,
        volume_multiplier=args.volume_multiplier,
        volume_ma_window=args.volume_ma_window,
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
