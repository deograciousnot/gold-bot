"""
Backtest of a "Smart Money Concepts" (SMC / ICT) strategy on gold, version 1.

SMC is usually traded by eye. A computer can only test exact rules, so this is ONE
mechanical reading of the standard definitions. The rules below were fixed BEFORE
looking at any results, and should not be tweaked to make the numbers look better.

THE RULES (top-down: daily -> 4 hour -> 15 minute)
  1. Bias. Market structure on the DAILY and the 4-HOUR chart, by "break of structure":
     using swing points (a candle whose high/low is the extreme of the 2 candles either side),
     the trend turns bullish when a candle CLOSES above the last swing high, and stays
     bullish until a close below the last swing low ("change of character").
     Only buy when daily AND 4H are bullish; only sell when both are bearish.
  2. Liquidity sweep. For buys: a 15-minute candle trades BELOW the previous day's low or
     the Asian session low (where stop losses cluster) and closes back ABOVE it.
     Mirror image for sells, using the previous day's / Asian session high.

  Version history: v1 required "higher high AND higher low" for the bias and used only the
  previous day's levels. It found 3 trades in 2 years: too strict to be SMC as traded.
  v1.1 switched to the standard break-of-structure bias and added the Asian range, before
  any v1.1 results were seen.
  3. Displacement + market structure shift. Within 2 hours of the sweep, a strong
     15-minute candle (body bigger than the 15-min ATR) closes above the last swing high
     (for buys), leaving a FAIR VALUE GAP: the gap between the high of the candle before
     it and the low of the candle after it.
  4. Entry. A limit order at the middle of that FVG ("consequent encroachment"),
     valid for 3 hours.
  5. Stop loss just beyond the sweep's extreme. Take profit at 2x the risk.
     Anything still open at 23:00 server time is closed (no overnight holds).
  6. Only enter between 09:00 and 19:00 server time (London + New York sessions).
  Risk 1% per trade, same position sizing as the other backtests.

HONESTY RULES (same as before)
  - Higher timeframes only use candles that had already CLOSED at that moment.
  - Swing points only count once they are confirmed (2 candles later).
  - Spread paid on every trade; buys fill at the ask, sells close at the ask.
  - If stop and target are both touched in one candle, assume the stop.

Usage:  python backtest_smc.py                        (HFM 15-min data, Sep 2024-Oct 2026: development)
        python backtest_smc.py M15_long 2024-01-01    (Dukascopy 15-min, 2005-2023: the unseen exam)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_trend import CONTRACT_SIZE, POINT, START_BALANCE, lot_size, report

DATASET = sys.argv[1] if len(sys.argv) > 1 else "M15"
END = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. 2024-01-01: keep the development years out
DATA_FILE = Path(__file__).parent / "data" / f"XAUUSD_{DATASET}.csv"

SWING = 2                 # candles either side that make a swing point
SWEEP_WINDOW = 8          # 15-min candles (2 hours) to see displacement after a sweep
ORDER_LIFETIME = 12       # 15-min candles (3 hours) the limit order stays valid
RR = 2.0                  # take profit = 2x the risk
ENTRY_HOURS = range(9, 19)
CLOSE_HOUR = 23


def swings(df):
    """Confirmed swing highs/lows: (confirm_time, price) lists. A swing at candle j is
    only known once candle j+SWING has closed."""
    w = 2 * SWING + 1
    is_high = df.high == df.high.rolling(w, center=True).max()
    is_low = df.low == df.low.rolling(w, center=True).min()
    confirm = df.close_time.shift(-SWING)
    highs = pd.DataFrame({"t": confirm[is_high], "p": df.high[is_high]}).dropna()
    lows = pd.DataFrame({"t": confirm[is_low], "p": df.low[is_low]}).dropna()
    return highs.reset_index(drop=True), lows.reset_index(drop=True)


def structure_bias(df):
    """Trend by break of structure: turns bullish (+1) when a candle CLOSES above the last
    confirmed swing high, bearish (-1) when it closes below the last confirmed swing low,
    and stays that way until the opposite break. Known at each candle's close."""
    highs, lows = swings(df)
    hi = pd.merge_asof(df[["close_time"]], highs, left_on="close_time", right_on="t",
                       direction="backward", allow_exact_matches=False).p.to_numpy()
    lo = pd.merge_asof(df[["close_time"]], lows, left_on="close_time", right_on="t",
                       direction="backward", allow_exact_matches=False).p.to_numpy()
    close, state, out = df.close.to_numpy(), 0, []
    for k in range(len(df)):
        if not np.isnan(hi[k]) and close[k] > hi[k]:
            state = 1
        elif not np.isnan(lo[k]) and close[k] < lo[k]:
            state = -1
        out.append(state)
    return pd.DataFrame({"time": df.close_time, "bias": out})


def resample(m15, rule):
    out = m15.resample(rule, on="time", label="left", closed="left").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna().reset_index()
    out["close_time"] = out.time + pd.Timedelta(rule)
    return out


def load():
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])[["time", "open", "high", "low", "close", "spread"]]
    if END:
        df = df[df.time < END]
    df = df.sort_values("time").reset_index(drop=True)
    df["close_time"] = df.time + pd.Timedelta("15min")

    # Higher-timeframe bias, attached to each 15-min candle using only CLOSED information
    for name, rule in (("d1", "1D"), ("h4", "4h")):
        bias = structure_bias(resample(df, rule)).rename(columns={"bias": f"bias_{name}", "time": "known"})
        df = pd.merge_asof(df, bias, left_on="time", right_on="known", direction="backward").drop(columns="known")
    df[["bias_d1", "bias_h4"]] = df[["bias_d1", "bias_h4"]].fillna(0)

    # Previous day's high and low (the liquidity pools)
    day = df.time.dt.normalize()
    daily = df.groupby(day).agg(dh=("high", "max"), dl=("low", "min"))
    prev = daily.shift(1)
    df["pdh"], df["pdl"] = day.map(prev.dh), day.map(prev.dl)
    # Asian session high/low (01:00-09:00 server time), only used after 09:00 when it's complete
    asia = df[(df.time.dt.hour >= 1) & (df.time.dt.hour < 9)].groupby(day).agg(ah=("high", "max"), al=("low", "min"))
    df["asia_h"], df["asia_l"] = day.map(asia.ah), day.map(asia.al)

    # 15-min ATR and confirmed 15-min swing levels (for the structure shift)
    prev_close = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    highs, lows = swings(df)
    df = pd.merge_asof(df, highs.rename(columns={"t": "known", "p": "swing_high"}), left_on="time",
                       right_on="known", direction="backward").drop(columns="known")
    df = pd.merge_asof(df, lows.rename(columns={"t": "known", "p": "swing_low"}), left_on="time",
                       right_on="known", direction="backward").drop(columns="known")
    return df.dropna(subset=["pdh", "pdl", "atr"]).reset_index(drop=True)


def run(df):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    spread = df.spread.to_numpy() * POINT
    atr, pdh, pdl = df.atr.to_numpy(), df.pdh.to_numpy(), df.pdl.to_numpy()
    ah, al = df.asia_h.to_numpy(), df.asia_l.to_numpy()
    sw_hi, sw_lo = df.swing_high.to_numpy(), df.swing_low.to_numpy()
    b_d1, b_h4 = df.bias_d1.to_numpy(), df.bias_h4.to_numpy()
    times, hours = df.time, df.time.dt.hour.to_numpy()
    dates = df.time.dt.date.to_numpy()

    balance, trades, skipped = START_BALANCE, [], 0
    sweep = None    # {"side", "i", "extreme", "level_before"} waiting for displacement
    order = None    # pending limit order
    pos = None
    stats = {"sweeps": 0, "fvgs": 0, "fills": 0}

    for i in range(2, len(df)):
        bias = b_d1[i] if b_d1[i] == b_h4[i] else 0

        # --- 1) manage an open position
        if pos:
            if pos["side"] == 1:
                hit_sl, hit_tp = l[i] <= pos["sl"], h[i] >= pos["tp"]
            else:
                hit_sl, hit_tp = h[i] + spread[i] >= pos["sl"], l[i] + spread[i] <= pos["tp"]
            exit_price = None
            if hit_sl:
                exit_price, win = pos["sl"], False
            elif hit_tp:
                exit_price, win = pos["tp"], True
            elif hours[i] >= CLOSE_HOUR or dates[i] != pos["date"]:
                exit_price = o[i] if pos["side"] == 1 else o[i] + spread[i]
                win = (exit_price - pos["entry"]) * pos["side"] > 0
            if exit_price is not None:
                pnl = (exit_price - pos["entry"]) * pos["side"] * pos["lots"] * CONTRACT_SIZE
                balance += pnl
                trades.append({**pos, "exit_time": times[i], "exit": exit_price, "pnl": pnl,
                               "balance": balance, "win": win})
                pos = None
            continue

        # --- 2) a pending limit order: filled, expired, or invalid?
        if order:
            if i > order["expires"] or dates[i] != order["date"]:
                order = None
            elif order["side"] == 1 and l[i] + spread[i] <= order["entry"]:
                pos, order = order, None
            elif order["side"] == -1 and h[i] >= order["entry"]:
                pos, order = order, None
            if pos:
                stats["fills"] += 1
                pos["entry_time"] = times[i]
                # same-candle stop check (pessimistic)
                if (pos["side"] == 1 and l[i] <= pos["sl"]) or (pos["side"] == -1 and h[i] + spread[i] >= pos["sl"]):
                    pnl = (pos["sl"] - pos["entry"]) * pos["side"] * pos["lots"] * CONTRACT_SIZE
                    balance += pnl
                    trades.append({**pos, "exit_time": times[i], "exit": pos["sl"], "pnl": pnl,
                                   "balance": balance, "win": False})
                    pos = None
            continue

        if bias == 0 or hours[i] not in ENTRY_HOURS:
            sweep = None if hours[i] not in ENTRY_HOURS else sweep
            continue

        # --- 3) waiting for displacement after a sweep
        if sweep and sweep["side"] == bias:
            if i - sweep["i"] > SWEEP_WINDOW:
                sweep = None
            else:
                side = sweep["side"]
                sweep["extreme"] = min(sweep["extreme"], l[i]) if side == 1 else max(sweep["extreme"], h[i])
                body = abs(c[i - 1] - o[i - 1])
                # FVG formed by candles i-2, i-1 (displacement), i
                if side == 1:
                    gap = (h[i - 2], l[i]) if l[i] > h[i - 2] else None
                    shifted = c[i - 1] > sweep["level_before"]
                else:
                    gap = (h[i], l[i - 2]) if l[i - 2] > h[i] else None
                    shifted = c[i - 1] < sweep["level_before"]
                if gap and shifted and body > atr[i - 1]:
                    stats["fvgs"] += 1
                    entry = (gap[0] + gap[1]) / 2
                    stop = sweep["extreme"] - side * spread[i]
                    risk = abs(entry - stop)
                    lots = lot_size(balance, risk + spread[i])
                    if lots == 0 or risk <= 0:
                        skipped += 1
                    else:
                        order = {"side": side, "entry": entry, "sl": stop, "tp": entry + side * RR * risk,
                                 "lots": lots, "expires": i + ORDER_LIFETIME, "date": dates[i],
                                 "setup_time": times[i]}
                    sweep = None
                continue

        # --- 4) look for a liquidity sweep in the bias direction
        swept_low = any(l[i] < lvl < c[i] for lvl in (pdl[i], al[i]) if not np.isnan(lvl))
        swept_high = any(h[i] > lvl > c[i] for lvl in (pdh[i], ah[i]) if not np.isnan(lvl))
        if bias == 1 and swept_low and not np.isnan(sw_hi[i]):
            sweep = {"side": 1, "i": i, "extreme": l[i], "level_before": sw_hi[i]}
            stats["sweeps"] += 1
        elif bias == -1 and swept_high and not np.isnan(sw_lo[i]):
            sweep = {"side": -1, "i": i, "extreme": h[i], "level_before": sw_lo[i]}
            stats["sweeps"] += 1

    return pd.DataFrame(trades), skipped, stats


if __name__ == "__main__":
    df = load()
    trades, skipped, stats = run(df)
    days = df.time.dt.date.nunique()
    agree = (df.bias_d1 == df.bias_h4) & (df.bias_d1 != 0)
    print(f"Data: {DATASET}, {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d} ({days} trading days)")
    print(f"Gold moved from ${df.close.iloc[0]:,.0f} to ${df.close.iloc[-1]:,.0f}")
    print(f"Daily and 4H bias agreed {agree.mean():.0%} of the time "
          f"(bullish {((df.bias_d1 == 1) & agree).mean():.0%}, bearish {((df.bias_d1 == -1) & agree).mean():.0%})")
    print(f"Setups: {stats['sweeps']} sweeps -> {stats['fvgs']} displacement FVGs -> {stats['fills']} filled orders"
          + (f" ({skipped} skipped: too small to size)" if skipped else ""))
    report("WHOLE PERIOD", trades, START_BALANCE)
    if len(trades) > 1:
        mid = df.time.iloc[len(df) // 2]
        first, second = trades[trades.entry_time < mid], trades[trades.entry_time >= mid]
        report("FIRST HALF", first, START_BALANCE)
        report("SECOND HALF", second, first.balance.iloc[-1] if len(first) else START_BALANCE)
        r = trades.pnl / (trades.balance - trades.pnl)  # each trade's result as a % of the balance
        print(f"\nAverage result per trade: {r.mean():+.2%} of the account "
              f"(confidence score t = {r.mean() / (r.std() / np.sqrt(len(r))):+.2f})")
    out = Path(__file__).parent / "data" / f"trades_smc_{DATASET}.csv"
    trades.to_csv(out, index=False)
    print(f"Every trade saved to {out.name}")
