"""
Live Alert Bot — ICT Liquidity-Sweep Strategy
================================================
Same polling design as live_alert_bot.py, but wired to the ICT-concepts
strategy (liquidity sweep -> market structure shift -> FVG retest) in
ict_liquidity_strategy.py instead of the breakout-retest strategy.

Requires ict_liquidity_strategy.py AND breakout_retest_strategy.py in the
same folder (the ICT script reuses the breakout script's data loading and
backtest helpers).

Usage:
  python live_alert_bot_ict.py --tickers MES=F,MNQ=F --once --dry-run

  python live_alert_bot_ict.py --tickers MES=F,MNQ=F \
      --telegram-token YOUR_BOT_TOKEN --telegram-chat-id YOUR_CHAT_ID \
      --session-start 02:00 --session-end 05:00
"""

import argparse
import json
import os
import time

import pandas as pd

from ict_liquidity_strategy import generate_signals
from breakout_retest_strategy import load_yfinance, filter_session


STATE_FILE_DEFAULT = "ict_alert_state.json"


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def format_alert(ticker: str, timestamp, signal: str, row: pd.Series) -> str:
    direction = "LONG" if "long" in signal else "SHORT"
    return (
        f"[ICT ENTRY] {ticker} {direction}\n"
        f"Time: {timestamp}\n"
        f"Close: {row['close']:.4f} | High: {row['high']:.4f} | Low: {row['low']:.4f}\n"
        f"Signal: {signal}"
    )


def send_telegram(token: str, chat_id: str, message: str) -> None:
    import requests

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
    resp.raise_for_status()


def send_discord(webhook_url: str, message: str) -> None:
    import requests

    resp = requests.post(webhook_url, json={"content": message}, timeout=10)
    resp.raise_for_status()


def dispatch_alert(message: str, args) -> None:
    if args.dry_run:
        print("---- DRY RUN ALERT ----")
        print(message)
        print("-----------------------")
        return
    if args.telegram_token and args.telegram_chat_id:
        send_telegram(args.telegram_token, args.telegram_chat_id, message)
    if args.discord_webhook:
        send_discord(args.discord_webhook, message)
    if not (args.telegram_token or args.discord_webhook):
        print(message)


def check_ticker(ticker: str, state: dict, args) -> None:
    try:
        data = load_yfinance(ticker)
    except Exception as e:
        print(f"[{ticker}] failed to fetch data: {e}")
        return

    data = filter_session(
        data,
        session_start=args.session_start,
        session_end=args.session_end,
        session_tz=args.session_timezone,
    )

    min_needed = args.liquidity_lookback + args.mss_window + args.retest_window + 10
    if data.empty or len(data) < min_needed:
        print(f"[{ticker}] not enough data yet, skipping")
        return

    signals_df = generate_signals(
        data,
        liquidity_lookback=args.liquidity_lookback,
        pivot_left=args.pivot_left,
        pivot_right=args.pivot_right,
        mss_window=args.mss_window,
        retest_window=args.retest_window,
    )

    last_row = signals_df.iloc[-1]
    last_ts = str(signals_df.index[-1])
    signal = last_row["signal"]

    last_alerted = state.get(ticker)

    if pd.notna(signal) and last_alerted != last_ts:
        message = format_alert(ticker, last_ts, signal, last_row)
        dispatch_alert(message, args)
        state[ticker] = last_ts
    else:
        print(f"[{ticker}] no new signal at {last_ts} (latest signal: {signal})")


def run_once(tickers, args) -> None:
    state = load_state(args.state_file)
    for ticker in tickers:
        check_ticker(ticker, state, args)
    save_state(args.state_file, state)


def run_loop(tickers, args) -> None:
    print(f"Starting loop, checking every {args.interval_minutes} minutes. Ctrl+C to stop.")
    while True:
        run_once(tickers, args)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ICT liquidity-sweep alert bot")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g. MES=F,MNQ=F")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--state-file", type=str, default=STATE_FILE_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--telegram-token", type=str, default=None)
    parser.add_argument("--telegram-chat-id", type=str, default=None)
    parser.add_argument("--discord-webhook", type=str, default=None)

    # Strategy params (mirror ict_liquidity_strategy.py)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--pivot-left", type=int, default=3)
    parser.add_argument("--pivot-right", type=int, default=3)
    parser.add_argument("--mss-window", type=int, default=15)
    parser.add_argument("--retest-window", type=int, default=15)

    # Session/killzone filter
    parser.add_argument("--session-start", type=str, default=None, help='e.g. "02:00" for London killzone')
    parser.add_argument("--session-end", type=str, default=None, help='e.g. "05:00" for London killzone')
    parser.add_argument("--session-timezone", type=str, default="America/New_York")

    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if args.once:
        run_once(tickers, args)
    else:
        run_loop(tickers, args)
