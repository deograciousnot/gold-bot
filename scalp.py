"""
Scalping tournament: 7 classic 1-minute gold scalps, each run TWICE:
with real spreads, and with zero costs (to show how much the spread eats).

Same honest rules as tournament.py: signals on a closed candle, entry at the next candle's open,
stop assumed first if stop and target are hit in the same candle. Develop on 2005-2018, judge on
2019-2023, 2024+ stays sealed. With ~33 strategies now tested in total, the bar is t >= 3.3.

The 1-minute data is rebuilt from the cached Dukascopy files once, then saved as data/m1.npz.
Usage (on the server):  python scalp.py
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import tournament as T
from download_dukascopy import RAW_DIR, assumed_spread, parse_day, to_server_time

DATA = Path(__file__).parent / "data"
CACHE = DATA / "m1.npz"
T_BAR = 3.3


def load_m1():
    if CACHE.exists():
        z = np.load(CACHE)
        df = pd.DataFrame({k: z[k] for k in ("open", "high", "low", "close")})
        df["time"] = pd.to_datetime(z["time"])
    else:
        rows, day = [], date(2005, 1, 1)
        while day < date(2024, 1, 1):
            f = RAW_DIR / f"{day:%Y%m%d}.bi5"
            if f.exists():
                rows.extend(parse_day(day, f.read_bytes()) or [])
            day += timedelta(days=1)
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
        df["time"] = to_server_time(df.time)
        df = df.sort_values("time").reset_index(drop=True)
        np.savez(CACHE, time=df.time.values.astype("datetime64[s]"), **{k: df[k].to_numpy() for k in ("open", "high", "low", "close")})
    df["spread"] = (df.close.map(assumed_spread) / T.POINT).round()
    return df


def strategies(m):
    S = {}
    c, h, l = m.close, m.high, m.low
    a = T.atr(m, 14).to_numpy()
    hrs = m.time.dt.hour.to_numpy()
    mins = (m.time.dt.hour * 60 + m.time.dt.minute).to_numpy()
    active = (hrs >= 10) & (hrs < 19)                      # London + New York hours (server time)

    # SC1 momentum burst: a candle 3x the usual size, closing in its top/bottom fifth
    rng = (h - l).to_numpy()
    pos = ((c - l) / (h - l).replace(0, np.nan)).to_numpy()
    burst = np.where(active & (rng > 3 * a) & (pos > 0.8), 1, np.where(active & (rng > 3 * a) & (pos < 0.2), -1, 0))
    S["SC1 Momentum burst"] = dict(df=m, side=burst, sl_level=np.where(burst > 0, l, h), tp_r=1, max_hold=15)

    # SC2 30-minute range breakout during active hours
    hi30, lo30 = h.shift(1).rolling(30).max(), l.shift(1).rolling(30).min()
    lo10, hi10 = l.rolling(10).min(), h.rolling(10).max()
    brk = np.where(active & (c > hi30), 1, np.where(active & (c < lo30), -1, 0))
    S["SC2 30-min range breakout"] = dict(df=m, side=brk, sl_level=np.where(brk > 0, lo10, hi10), tp_r=1.5, max_hold=30)

    # SC3 RSI(3) dip-buy in an uptrend / spike-sell in a downtrend
    r3, e200 = T.rsi(c, 3), T.ema(c, 200)
    dip = np.where(active & (r3 < 5) & (c > e200), 1, np.where(active & (r3 > 95) & (c < e200), -1, 0))
    S["SC3 RSI(3) dip-buy"] = dict(df=m, side=dip, sl_dist=2 * a, exit_long=(r3 > 50).to_numpy(), exit_short=(r3 < 50).to_numpy(), max_hold=10)

    # SC4 / SC5: first break of the pre-open range in the first 30 minutes of London / New York
    day = m.time.dt.normalize()

    def open_scalp(range_from, range_to, trade_to):
        mask = (mins >= range_from) & (mins < range_to)
        g = m[mask].groupby(day[mask])
        rh, rl = day.map(g.high.max()).to_numpy(), day.map(g.low.min()).to_numpy()
        win = (mins >= range_to) & (mins < trade_to)
        side = np.where(win & (c > rh), 1, np.where(win & (c < rl), -1, 0))
        side = np.where(pd.Series(side != 0).groupby(day).cumsum().to_numpy() == 1, side, 0)
        return dict(df=m, side=side, sl_level=np.where(side > 0, l.rolling(5).min(), h.rolling(5).max()), tp_r=1, max_hold=30)

    S["SC4 London open scalp"] = open_scalp(8 * 60, 10 * 60, 10 * 60 + 30)
    S["SC5 New York open scalp"] = open_scalp(16 * 60, 16 * 60 + 30, 17 * 60)

    # SC6 stretch fade: far from the 60-minute average -> bet on the snap back
    mean60 = c.rolling(60).mean()
    stretch = ((c - mean60) / pd.Series(a)).to_numpy()
    fade = np.where(active & (stretch < -2.5), 1, np.where(active & (stretch > 2.5), -1, 0))
    S["SC6 Stretch fade"] = dict(df=m, side=fade, sl_dist=1.5 * a, exit_long=(c > mean60).to_numpy(), exit_short=(c < mean60).to_numpy(), max_hold=30)

    # SC7 EMA pullback: stacked EMAs, a dip to the 21 EMA, then a close back above the 9 EMA
    e9, e21, e50 = T.ema(c, 9), T.ema(c, 21), T.ema(c, 50)
    up, dn = (e9 > e21) & (e21 > e50), (e9 < e21) & (e21 < e50)
    touched_up = (l <= e21).rolling(3).max() > 0
    touched_dn = (h >= e21).rolling(3).max() > 0
    pb = np.where(active & up & touched_up & (c > e9) & (c.shift(1) <= e9.shift(1)), 1,
                  np.where(active & dn & touched_dn & (c < e9) & (c.shift(1) >= e9.shift(1)), -1, 0))
    S["SC7 EMA pullback"] = dict(df=m, side=pb, sl_level=np.where(pb > 0, np.minimum(e50, l.rolling(5).min()), np.maximum(e50, h.rolling(5).max())), tp_r=1.5, max_hold=30)
    return S


if __name__ == "__main__":
    t0 = time.time()
    m = load_m1()
    print(f"1-minute data: {len(m):,} candles, {m.time.iloc[0]:%Y-%m-%d} to {m.time.iloc[-1]:%Y-%m-%d} ({time.time() - t0:.0f}s)")
    free = m.copy()
    free["spread"] = 0
    rows = []
    for name, spec in strategies(m).items():
        s0 = time.time()
        out = {}
        for label, frame in (("real", m), ("free", free)):
            trades = T.simulate(frame, **{k: v for k, v in spec.items() if k != "df"})
            out[label] = trades
        real, free_t = out["real"], out["free"]
        a, g = T.score(real), T.score(free_t)
        d, j = T.score(real[real.time < pd.Timestamp("2019-01-01")]), T.score(real[real.time >= pd.Timestamp("2019-01-01")])
        passed = a["t"] >= T_BAR and d["ret"] > 0 and j["ret"] > 0
        rows.append({"scalp": name, "trades": a["n"], "per year": a["n"] / 19, "win%": a["win"] * 100,
                     "avg R (no costs)": g["avgR"], "avg R (real)": a["avgR"], "t (real)": a["t"],
                     "2005-18 %": d["ret"] * 100, "2019-23 %": j["ret"] * 100, "verdict": "PASS" if passed else "-"})
        print(f"  {name:28} {a['n']:6} trades  no-cost {g['avgR']:+.3f}R  real {a['avgR']:+.3f}R  t={a['t']:+.2f}  ({time.time() - s0:.0f}s)", flush=True)
    board = pd.DataFrame(rows).sort_values("t (real)", ascending=False)
    pd.set_option("display.width", 220)
    print(f"\n=== SCALPING LEADERBOARD (bar: t >= {T_BAR} AND profitable in both periods) ===")
    print(board.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    board.to_csv(DATA / "scalp_results.csv", index=False)
    print(f"\n{sum(board.verdict == 'PASS')} of {len(board)} scalps passed. ({time.time() - t0:.0f}s)")
