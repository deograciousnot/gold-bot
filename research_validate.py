"""
Research step 2: test the candidate rules found in research_explore.py on years
they were NOT discovered in, after paying the spread on every trade.

The rules below were LOCKED after exploring 2005-2018 only. Do not edit them
after looking at the validate/test results - that would turn this back into
curve-fitting.

  A. Asian drift:   buy at 02:00 server time, sell at 09:00.
     Why it might be real: gold has historically risen during Asian hours and
     fallen during London/New York hours (Asian physical demand, Western selling).
     Hours 00:00-01:00 are skipped: their big "moves" are mostly the spread
     widening at the daily rollover - an artifact of bid-only data, not tradeable.
  B. London fix:    sell at 12:00, buy back at 13:00, and again 16:00 -> 17:00.
     Why: price has tended to dip in the hours around the London gold price
     auctions ("fixes"), which are well documented in academic research.
  C. Friday:        buy at Thursday's close, sell at Friday's close.
     Why: weakest rationale of the three (possibly pre-weekend safe-haven buying),
     but it was positive in 13 of 14 explored years.

Usage:  python research_validate.py validate   (2019-2023)
        python research_validate.py test       (2024-2026, run ONCE at the very end)
        python research_validate.py explore    (2005-2018, to see them after costs)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "XAUUSD_H1_long.csv"
PERIODS = {"explore": ("2005-01-01", "2019-01-01"),
           "validate": ("2019-01-01", "2024-01-01"),
           "test": ("2024-01-01", "2030-01-01")}
POINT = 0.01


def hold(df, enter_hour, exit_hour, side):
    """One trade per day: enter at the open of enter_hour, exit at the open of exit_hour.
    Returns each trade's net % result after paying the spread once."""
    day = df.time.dt.date
    entry = df[df.hour == enter_hour].set_index(day[df.hour == enter_hour])
    exit_ = df[df.hour == exit_hour].set_index(day[df.hour == exit_hour])
    both = entry.join(exit_, lsuffix="_in", rsuffix="_out", how="inner")
    gross = side * np.log(both.open_out / both.open_in)
    cost = both.spread_in * POINT / both.open_in  # spread paid once per round trip
    return pd.Series((gross - cost).values, index=pd.to_datetime(both.index))


def friday(df):
    closes = df.groupby(df.time.dt.date).close.last()
    closes.index = pd.to_datetime(closes.index)
    spread = df.groupby(df.time.dt.date).spread.last().values * POINT
    ret = np.log(closes / closes.shift(1)) - spread / closes
    return ret[ret.index.dayofweek == 4].dropna()


def report(name, r):
    if r.empty:
        print(f"{name}: no trades")
        return
    t = r.mean() / (r.std() / np.sqrt(len(r)))
    by_year = r.groupby(r.index.year).sum()
    print(f"{name}")
    print(f"  trades {len(r):5}   win rate {(r > 0).mean():.0%}   avg per trade {r.mean() * 100:+.4f}%   "
          f"t = {t:+.2f}   total {r.sum() * 100:+.1f}%")
    print("  by year: " + "  ".join(f"{y}: {v * 100:+.1f}%" for y, v in by_year.items()))


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "validate"
    start, end = PERIODS[period]
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])
    df = df[(df.time >= start) & (df.time < end)].reset_index(drop=True)
    df["hour"] = df.time.dt.hour
    print(f"=== {period.upper()}: {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d}, after spread costs ===\n")

    report("A. Asian drift (long 02:00 -> 09:00)", hold(df, 2, 9, +1))
    fix = pd.concat([hold(df, 12, 13, -1), hold(df, 16, 17, -1)]).sort_index()
    report("B. London fix (short 12-13 and 16-17)", fix)
    report("C. Friday (long Thu close -> Fri close)", friday(df))
    report("   Buy-and-hold gold, for comparison (per day)",
           np.log(df.groupby(df.time.dt.date).close.last()).diff().dropna()
           .set_axis(pd.to_datetime(df.groupby(df.time.dt.date).close.last().index[1:])))
