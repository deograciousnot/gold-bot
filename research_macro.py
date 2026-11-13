"""
Macro research: do changes in US real interest rates and the dollar PREDICT gold's next week?

It's well known that gold moves opposite to real rates and the dollar on the SAME day.
That's useless for trading - by the time you see it, the move has happened.
The question here is whether the direction of the drivers over the past few weeks
tells us anything about gold's NEXT week.

Every signal only uses data that was already published before the trade:
  - real yields (published daily, next business day): lagged 2 business days
  - dollar index (published weekly, with a delay):     lagged 7 business days
Trades are weekly: decide at Friday's close, hold until next Friday's close.
Costs: spread on each change of position + overnight financing ("swap") while holding.

Usage:  python research_macro.py explore    (2006-2018: look for signals)
        python research_macro.py validate   (2019-2023: test the LOCKED signals once)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
PERIODS = {"explore": ("2006-03-01", "2019-01-01"),
           "validate": ("2019-01-01", "2024-01-01")}
SPREAD_PCT = 0.0002         # ~0.02% of price per position change (pessimistic)
FINANCING_PER_YEAR = 0.04   # assumed swap cost while holding, either direction (estimate)

# Locked after reading the explore results - do not change after running validate.
# Explore result: no signal reached t > 2 (best was 1.35). The top one is locked anyway,
# to show whether it holds up - expected to be noise.
LOCKED = ["both agree 120d (long/flat)"]


def load():
    gold = pd.read_csv(DATA / "XAUUSD_H1_long.csv", parse_dates=["time"])
    daily = gold.groupby(gold.time.dt.normalize()).close.last()
    daily = daily[daily.index.dayofweek < 5]

    macro = pd.read_csv(DATA / "macro_daily.csv", index_col="date", parse_dates=True)
    macro = macro.reindex(daily.index).ffill()
    macro["real_yield"] = macro.real_yield.shift(2)   # only what was already published
    macro["dollar"] = macro.dollar.shift(7)
    return daily, macro


def signals(daily, macro):
    """Each signal: +1 = long gold, -1 = short, 0 = stay out."""
    s = {}
    for L in (20, 60, 120):
        ry = -np.sign(macro.real_yield.diff(L))   # real yields falling -> long gold
        usd = -np.sign(np.log(macro.dollar).diff(L))  # dollar falling -> long gold
        s[f"real yield {L}d (long/short)"] = ry
        s[f"real yield {L}d (long/flat)"] = ry.clip(lower=0)
        s[f"dollar {L}d (long/short)"] = usd
        s[f"dollar {L}d (long/flat)"] = usd.clip(lower=0)
        both = ((ry + usd) / 2).where(ry == usd, 0)       # trade only when both agree
        s[f"both agree {L}d (long/short)"] = both
        s[f"both agree {L}d (long/flat)"] = both.clip(lower=0)
    s["buy and hold (benchmark)"] = pd.Series(1.0, index=daily.index)
    return s


def weekly_results(daily, position):
    fridays = daily.index[daily.index.dayofweek == 4]
    price = daily.reindex(fridays)
    pos = position.reindex(fridays).fillna(0)
    ret = np.log(price.shift(-1) / price)                 # next week's gold move
    gross = pos * ret
    cost = pos.diff().abs().fillna(pos.abs()) * SPREAD_PCT + pos.abs() * FINANCING_PER_YEAR / 52
    return (gross - cost).dropna()


def summarize(name, r):
    t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
    by_year = r.groupby(r.index.year).sum()
    equity = r.cumsum()
    dd = (np.exp(equity) / np.exp(equity).cummax() - 1).min()
    in_market = (r != 0).mean()
    return {"signal": name, "total %": r.sum() * 100, "per year %": r.mean() * 52 * 100, "t": t,
            "worst drawdown %": dd * 100, "years +": f"{(by_year > 0).sum()}/{len(by_year)}",
            "in market": f"{in_market:.0%}"}


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "explore"
    start, end = PERIODS[period]
    daily, macro = load()
    sigs = signals(daily, macro)
    if period != "explore":
        sigs = {k: v for k, v in sigs.items() if k in LOCKED or "benchmark" in k}
        if len(sigs) < 2:
            sys.exit("No signals locked yet - run explore first and fill in LOCKED.")

    rows = []
    for name, pos in sigs.items():
        r = weekly_results(daily, pos)
        r = r[(r.index >= start) & (r.index < end)]
        rows.append(summarize(name, r))
    table = pd.DataFrame(rows).set_index("signal").round(2)
    print(f"=== {period.upper()} {start[:4]}-{int(end[:4]) - 1}: weekly trades, after spread and "
          f"{FINANCING_PER_YEAR:.0%}/yr financing ===\n")
    print(table.to_string())
