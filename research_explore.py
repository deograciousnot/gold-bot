"""
Research step 1: look for repeating patterns in gold, using ONLY 2005-2018.

The later years (2019-2026) are deliberately hidden from this script so they can
be used afterwards as an honest test. Nothing here trades or touches MT5.

How to read the output:
  - "avg" is the average % move. Small numbers matter: 0.01% per hour is a lot.
  - "t" is a confidence score. Roughly: |t| below 2 = could easily be luck,
    above 3 = probably real. Because we test many ideas at once, we want |t| > 3.
  - "years +" is how many of the 14 years the effect pointed the same way.
    A real effect should show up in most years, not just one lucky stretch.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "XAUUSD_H1_long.csv"
EXPLORE_END = "2019-01-01"   # everything from here on stays hidden


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan


def summarize(groups, returns, years):
    """Average return, t-score and year-by-year consistency for each group."""
    rows = []
    for key, r in returns.groupby(groups):
        by_year = r.groupby(years[r.index]).mean()
        sign = np.sign(r.mean())
        rows.append({"group": key, "count": len(r), "avg %": r.mean() * 100, "t": tstat(r),
                     "years +": f"{int((np.sign(by_year) == sign).sum())}/{len(by_year)}"})
    return pd.DataFrame(rows).set_index("group").round({"avg %": 4, "t": 2})


df = pd.read_csv(DATA_FILE, parse_dates=["time"])
df = df[df.time < EXPLORE_END].reset_index(drop=True)
df["ret"] = np.log(df.close).diff()          # % move of each hour (as a fraction)
df["hour"] = df.time.dt.hour
df["year"] = df.time.dt.year
print(f"Exploring {len(df):,} hourly candles, {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d}\n")

# ---------------------------------------------------------------------------
print("=== 1. Does gold tend to rise or fall at certain hours? (server time) ===")
print(summarize(df.hour, df.ret, df.year).to_string(), "\n")

# ---------------------------------------------------------------------------
daily = df.groupby(df.time.dt.date).agg(open=("open", "first"), high=("high", "max"),
                                        low=("low", "min"), close=("close", "last"))
daily.index = pd.to_datetime(daily.index)
daily = daily[daily.index.dayofweek < 5]  # drop the short Sunday-evening sessions
daily["ret"] = np.log(daily.close).diff()
daily["year"] = daily.index.year

print("=== 2. Day of the week (0 = Monday) ===")
print(summarize(daily.index.dayofweek, daily.ret, daily.year).to_string(), "\n")

# ---------------------------------------------------------------------------
print("=== 3. After a big day, does gold continue or reverse the next day? ===")
daily["next"] = daily.ret.shift(-1)
vol = daily.ret.rolling(20).std()
daily["move"] = pd.cut(daily.ret / vol, [-np.inf, -2, -1, 1, 2, np.inf],
                       labels=["big drop", "drop", "normal", "rise", "big rise"])
print(summarize(daily.move, daily.next, daily.year).to_string(), "\n")

# ---------------------------------------------------------------------------
print("=== 4. Trend over weeks/months: does the past direction predict the next 5 days? ===")
daily["next5"] = np.log(daily.close.shift(-5) / daily.close)
for lookback in (20, 60, 120, 250):
    past = np.sign(np.log(daily.close / daily.close.shift(lookback)))
    signed = daily.next5 * past  # positive = the trend continued
    sample = signed.iloc[::5]    # non-overlapping weeks, so the t-score is honest
    by_year = sample.groupby(daily.year[sample.index]).mean()
    print(f"  past {lookback:3} days: avg {sample.mean() * 100:+.3f}% per week in the trend's direction, "
          f"t = {tstat(sample):.2f}, years + {int((by_year > 0).sum())}/{len(by_year)}")
print()

# ---------------------------------------------------------------------------
print("=== 5. Calm vs wild markets: are trends more reliable in some volatility regimes? ===")
past60 = np.sign(np.log(daily.close / daily.close.shift(60)))
signed = (daily.next5 * past60).iloc[::5]
regime = pd.qcut(vol, 3, labels=["calm", "medium", "wild"])[signed.index]
print(summarize(regime, signed, daily.year[signed.index]).to_string(), "\n")

# ---------------------------------------------------------------------------
print("=== 6. Monday gaps: after the weekend jump, does price keep going or fill the gap? ===")
mondays = daily[daily.index.dayofweek == 0].copy()
fridays = daily.close.shift(1)[mondays.index]
gap = np.log(mondays.open / fridays)
rest_of_monday = np.log(mondays.close / mondays.open)
signed = (rest_of_monday * np.sign(gap))[gap.abs() > gap.abs().median()]  # only real gaps
print(f"  after a gap, rest of Monday in the gap's direction: avg {signed.mean() * 100:+.3f}%, "
      f"t = {tstat(signed):.2f}, n = {len(signed)}")
