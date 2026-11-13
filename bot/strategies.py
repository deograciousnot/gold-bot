"""
Strategies plug in here. A strategy is a function that looks at closed candles and returns
a Signal (buy or sell, with stop loss and take profit) or None (do nothing).

To add one: write a function with the same shape and add it to STRATEGIES at the bottom.
Then set STRATEGY in config.py to its name.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    side: int        # +1 buy, -1 sell
    stop: float      # stop-loss price
    target: float    # take-profit price
    reason: str      # written to the log, so you always know why it traded


def atr(df, period=14):
    prev_close = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev_close).abs(), (df.low - prev_close).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(period).mean()


def example_trend(df):
    """
    EXAMPLE ONLY - this is the trend-follower from backtest_trend.py, which roughly broke even
    over 20 years. It's here to exercise the bot's plumbing on the demo, not to make money.

    Buy when the 20 EMA crosses above the 50 EMA while price is above the 200 EMA
    (and the mirror image for sells). Stop 2x ATR away, target 4x ATR.
    """
    fast = df.close.ewm(span=20, adjust=False).mean()
    slow = df.close.ewm(span=50, adjust=False).mean()
    trend = df.close.ewm(span=200, adjust=False).mean()
    a = atr(df).iloc[-1]
    price = df.close.iloc[-1]

    crossed_up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
    crossed_down = fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
    if crossed_up and price > trend.iloc[-1]:
        return Signal(+1, price - 2 * a, price + 4 * a, f"20/50 EMA crossed up in an uptrend, ATR {a:.2f}")
    if crossed_down and price < trend.iloc[-1]:
        return Signal(-1, price + 2 * a, price - 4 * a, f"20/50 EMA crossed down in a downtrend, ATR {a:.2f}")
    return None


def inside_day(df):
    """
    Inside-day breakout (run on 1-hour candles). The only strategy that survived the tournament:
    profit factor 1.36 on 2005-2023 and 1.74 on the sealed 2024-2026 data, with sells profitable too.

    If YESTERDAY was an "inside day" (its high and low both inside the day before), then the first
    hourly close TODAY beyond yesterday's high is a buy, below yesterday's low a sell.
    Stop at the other side of yesterday's range; target 2x the risk; closed after 24 hours.
    Same rules as "V2 Inside-day breakout" in tournament.py.
    """
    days = df.groupby(df.time.dt.normalize()).agg(high=("high", "max"), low=("low", "min"))
    if len(days) < 3:
        return None
    today = df.time.iloc[-1].normalize()
    if days.index[-1] != today:          # the last closed candle must belong to today
        return None
    y, before = days.iloc[-2], days.iloc[-3]
    if not (y.high < before.high and y.low > before.low):
        return None                      # yesterday wasn't an inside day

    todays = df[df.time.dt.normalize() == today]
    beyond = (todays.close > y.high) | (todays.close < y.low)
    if beyond.sum() != 1 or not beyond.iloc[-1]:
        return None                      # only the FIRST breakout of the day, on the candle that just closed

    price = df.close.iloc[-1]
    if price > y.high:
        return Signal(+1, y.low, price + 2 * (price - y.low),
                      f"inside-day breakout up: closed {price:.2f} above yesterday's high {y.high:.2f}")
    return Signal(-1, y.high, price - 2 * (y.high - price),
                  f"inside-day breakout down: closed {price:.2f} below yesterday's low {y.low:.2f}")


def never_trade(df):
    """Does nothing. Useful for checking connections and safety rules without any signals."""
    return None


STRATEGIES = {"inside_day": inside_day, "example_trend": example_trend, "never_trade": never_trade}
