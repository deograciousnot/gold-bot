"""
Step 3: backtest a London-open breakout on gold (XAUUSD).

THE STRATEGY (in plain words)
  - Gold is usually calmer during the Asian session, then wakes up when London opens.
  - Asian range: the highest and lowest price between 01:00 and 10:00 server time
    (server time is GMT+2/+3, so London opens at 10:00 server time all year).
  - From 10:00 to 14:00 we wait for price to break out of that range:
      break above the high -> buy,  break below the low -> sell.
    Only the first breakout of the day is traded (one trade per day).
  - Stop loss:   the other side of the Asian range.
  - Take profit: 2x the stop distance (each win is worth about 2 losses).
  - If neither is hit, the trade is closed at 23:00 server time (no overnight holds).
  - Risk: 1% of the account per trade, same as the trend backtest.

HONESTY RULES (same as the trend backtest)
  - Candle prices are bid prices; buys pay the spread to get in, sells pay it to get out.
  - If a stop loss and take profit are touched in the same candle, we assume the loss.
  - If one candle breaks BOTH sides of the range, we count it as a loss.
  - Results are split into an older "build" period and a recent "unseen" period.

Usage:  python backtest_breakout.py          (15-minute candles, ~2 years)
        python backtest_breakout.py H1       (1-hour candles, ~4 years)
"""

import sys
from pathlib import Path

import pandas as pd

from backtest_trend import CONTRACT_SIZE, POINT, SPLIT, START_BALANCE, lot_size, report

TIMEFRAME = sys.argv[1] if len(sys.argv) > 1 else "M15"
DATA_FILE = Path(__file__).parent / "data" / f"XAUUSD_{TIMEFRAME}.csv"

# --- Settings you can experiment with (server-time hours) ---
RANGE_START, RANGE_END = 1, 10    # Asian range: 01:00 up to 10:00
TRADE_START, TRADE_END = 10, 14   # look for a breakout between 10:00 and 14:00
CLOSE_HOUR = 23                   # close any open trade at 23:00
TP_MULTIPLE = 2.0                 # take profit = 2x the stop distance


def load_data():
    if not DATA_FILE.exists():
        sys.exit(f"Missing {DATA_FILE}. Run download_data.py first.")
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])
    df["date"] = df.time.dt.date
    df["hour"] = df.time.dt.hour
    return df


def trade_day(day, balance):
    """Simulate one trading day. Returns a trade dict, 'skipped', or None (no trade)."""
    asia = day[(day.hour >= RANGE_START) & (day.hour < RANGE_END)]
    if len(asia) < 4:
        return None  # not enough data (holiday or short day)
    range_high, range_low = asia.high.max(), asia.low.min()

    session = day[day.hour >= TRADE_START]
    pos = None
    for bar in session.itertuples():
        spread = bar.spread * POINT

        if pos is None:
            if bar.hour >= TRADE_END:
                return None  # no breakout in the window: no trade today
            broke_up = bar.high + spread >= range_high  # buy stop fills at the ask
            broke_down = bar.low <= range_low           # sell stop fills at the bid
            if not (broke_up or broke_down):
                continue
            side = 1 if broke_up else -1
            # If price already jumped past the level at the candle's open, we fill at that worse price
            entry = max(range_high, bar.open + spread) if side == 1 else min(range_low, bar.open)
            stop = range_low if side == 1 else range_high
            stop_distance = abs(entry - stop) + spread
            lots = lot_size(balance, stop_distance)
            if lots == 0:
                return "skipped"
            pos = {"side": side, "entry_time": bar.time, "entry": entry, "lots": lots,
                   "sl": stop, "tp": entry + side * TP_MULTIPLE * abs(entry - stop),
                   "range": range_high - range_low}
            if broke_up and broke_down:
                # Whipsaw candle: both sides broken, assume the worst
                return close(pos, bar.time, pos["sl"], balance, win=False)

        # Manage the open trade (including the candle it opened on)
        if pos["side"] == 1:
            hit_sl, hit_tp = bar.low <= pos["sl"], bar.high >= pos["tp"]
        else:
            hit_sl, hit_tp = bar.high + spread >= pos["sl"], bar.low + spread <= pos["tp"]
        if hit_sl:
            return close(pos, bar.time, pos["sl"], balance, win=False)
        if hit_tp:
            return close(pos, bar.time, pos["tp"], balance, win=True)
        if bar.hour >= CLOSE_HOUR:
            exit_price = bar.close if pos["side"] == 1 else bar.close + spread
            return close(pos, bar.time, exit_price, balance, win=None)

    if pos:  # day's data ended before 23:00 (early close): exit at the last price
        last = session.iloc[-1]
        exit_price = last.close if pos["side"] == 1 else last.close + last.spread * POINT
        return close(pos, last.time, exit_price, balance, win=None)
    return None


def close(pos, time, exit_price, balance, win):
    pnl = (exit_price - pos["entry"]) * pos["side"] * pos["lots"] * CONTRACT_SIZE
    if win is None:  # closed at end of day: it's a win if it made money
        win = pnl > 0
    return {**pos, "exit_time": time, "exit": exit_price, "pnl": pnl,
            "balance": balance + pnl, "win": win}


def run(df):
    balance, trades, skipped = START_BALANCE, [], 0
    for _, day in df.groupby("date"):
        result = trade_day(day, balance)
        if result == "skipped":
            skipped += 1
        elif result:
            trades.append(result)
            balance = result["balance"]
    return pd.DataFrame(trades), skipped


if __name__ == "__main__":
    df = load_data()
    trades, skipped = run(df)
    days = df.date.nunique()
    print(f"Data: {TIMEFRAME} candles, {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d} ({days} trading days)")
    print(f"Gold moved from ${df.close.iloc[0]:,.0f} to ${df.close.iloc[-1]:,.0f} over this period")
    if len(trades):
        print(f"Traded on {len(trades)} of {days} days; average Asian range ${trades.range.mean():.2f}")
    if skipped:
        print(f"Skipped {skipped} days: the account was too small for even 0.01 lot at 1% risk")

    report("WHOLE PERIOD", trades, START_BALANCE)

    split_time = df.time.iloc[int(len(df) * SPLIT)]
    build = trades[trades.entry_time < split_time]
    unseen = trades[trades.entry_time >= split_time]
    report("BUILD period (older 70%)", build, START_BALANCE)
    report("UNSEEN period (recent 30%)", unseen, build.balance.iloc[-1] if len(build) else START_BALANCE)

    out = Path(__file__).parent / "data" / f"trades_breakout_{TIMEFRAME}.csv"
    trades.to_csv(out, index=False)
    print(f"\nEvery trade saved to {out.name}")
