"""
The strategy tournament: 25 gold strategies (and blends), all run through ONE honest simulator.

Rules of the tournament (fixed before any results were seen):
  - Develop on 2005-2018, judge on 2019-2023. 2024-2026 stays sealed for a final exam.
  - Every trade has a stop loss, pays the real spread, and pays overnight financing (4%/yr).
  - Signals are calculated on a closed candle and acted on at the NEXT candle's open.
  - If a stop and a target are hit in the same candle, the stop is assumed.
  - With 25+ strategies tested, luck alone will make some look good. A strategy only counts as
    real if its confidence score (t) is at least 3.2 over the whole period AND it made money in
    BOTH the development and the judging years.

Results are measured in "R": 1R = the amount risked on a trade (entry to stop). +2R = won twice
the risk. "Equity" compounds 1% risk per trade.

Usage (on the server, where the data lives):  python tournament.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
EXPLORE_END, VALIDATE_END = "2019-01-01", "2024-01-01"   # 2024+ is sealed
POINT = 0.01
FINANCING_PER_YEAR = 0.04
T_BAR = 3.2


# ============================== data and indicators ==============================

def load():
    m15 = pd.read_csv(DATA / "XAUUSD_M15_long.csv", parse_dates=["time"])
    m15 = m15[m15.time < VALIDATE_END].reset_index(drop=True)

    def resample(rule):
        out = m15.resample(rule, on="time", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), spread=("spread", "mean")).dropna().reset_index()
        return out

    h1 = resample("1h")
    d1 = resample("1D")
    d1 = d1[d1.time.dt.dayofweek < 5].reset_index(drop=True)
    events = pd.read_csv(DATA / "events.csv", parse_dates=["time"])
    macro = pd.read_csv(DATA / "macro_daily.csv", index_col="date", parse_dates=True)
    return m15, h1, d1, events, macro


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(s, n):
    d = s.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def daily_to(df, d1, series):
    """Attach a daily value to intraday bars, using only the PREVIOUS (completed) day."""
    s = pd.Series(series.values, index=d1.time.dt.normalize()).shift(1)
    return df.time.dt.normalize().map(s)


# ============================== the simulator ==============================

def simulate(df, side, sl_dist=None, sl_level=None, tp_r=None, tp_dist=None, exit_long=None, exit_short=None,
             max_hold=10_000, flat_hour=None):
    """One position at a time. side[i] = +1/-1 signal on bar i's close -> enter at bar i+1's open.
    Stop: sl_level[i] (a price) or sl_dist[i] (distance from entry). Target: tp_r (multiple of risk)
    or tp_dist[i]. Optional exits on signals, after max_hold bars, or at flat_hour (server time)."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy() * POINT
    hours = df.time.dt.hour.to_numpy()
    times = df.time.to_numpy()
    side = np.nan_to_num(np.asarray(side, float))
    n = len(df)
    trades = []
    next_free = 0                           # first bar where a new trade may start
    for i in np.flatnonzero(side[:-1]):     # jump straight from one signal to the next
        if i < next_free:
            continue
        s = side[i]
        if flat_hour is not None and hours[i + 1] >= flat_hour:
            continue
        e = i + 1
        entry = o[e] + spr[e] if s > 0 else o[e]
        if sl_level is not None and not np.isnan(sl_level[i]):
            stop = sl_level[i]
        elif sl_dist is not None and not np.isnan(sl_dist[i]):
            stop = entry - s * sl_dist[i]
        else:
            continue
        risk = (entry - stop) * s
        if not risk > 0:
            continue
        if tp_r is not None:
            target = entry + s * tp_r * risk
        elif tp_dist is not None and not np.isnan(tp_dist[i]):
            target = entry + s * tp_dist[i]
        else:
            target = np.inf * s
        exit_px, j = None, e
        while j < n:
            sp = spr[j]
            # stop / target within the candle (stop first if both)
            if s > 0:
                if l[j] <= stop: exit_px = stop; break
                if h[j] >= target: exit_px = target; break
            else:
                if h[j] + sp >= stop: exit_px = stop; break
                if l[j] + sp <= target: exit_px = target; break
            # exits at the close of this candle
            done = (j - e + 1 >= max_hold
                    or (flat_hour is not None and (hours[j] >= flat_hour or (j + 1 < n and hours[j + 1] < hours[j])))
                    or (s > 0 and exit_long is not None and exit_long[j])
                    or (s < 0 and exit_short is not None and exit_short[j]))
            if done:
                exit_px = c[j] if s > 0 else c[j] + sp
                break
            j += 1
        if exit_px is None:
            break
        days = (times[min(j, n - 1)] - times[e]) / np.timedelta64(1, "D")
        financing = entry * FINANCING_PER_YEAR * max(days, 0) / 365
        r = ((exit_px - entry) * s - financing) / risk
        trades.append((times[e], s, r))
        next_free = j + 1
    return pd.DataFrame(trades, columns=["time", "side", "r"])


# ============================== the strategies ==============================

def strategies(m15, h1, d1, events, macro):
    S = {}
    a1 = atr(h1).to_numpy()
    c = h1.close
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    up_t, dn_t = c > e200, c < e200

    def cross(a, b): return (a > b) & (a.shift(1) <= b.shift(1)), (a < b) & (a.shift(1) >= b.shift(1))

    # ---------- trend ----------
    cu, cd = cross(e20, e50)
    S["T1 EMA 20/50 cross"] = lambda: simulate(h1, np.where(cu & up_t, 1, np.where(cd & dn_t, -1, 0)), sl_dist=2 * a1, tp_dist=4 * a1, max_hold=200)

    hi55, lo55 = h1.high.shift(1).rolling(55).max(), h1.low.shift(1).rolling(55).min()
    hi20, lo20 = h1.high.shift(1).rolling(20).max(), h1.low.shift(1).rolling(20).min()
    turtle = np.where(c > hi55, 1, np.where(c < lo55, -1, 0))
    S["T2 Turtle (Donchian 55)"] = lambda: simulate(h1, turtle, sl_dist=2 * a1, exit_long=(c < lo20).to_numpy(), exit_short=(c > hi20).to_numpy(), max_hold=500)
    S["T3 Donchian 20 breakout"] = lambda: simulate(h1, np.where(c > hi20, 1, np.where(c < lo20, -1, 0)), sl_dist=2 * a1, tp_dist=4 * a1, max_hold=100)

    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    mu, md = cross(macd, sig)
    S["T4 MACD cross"] = lambda: simulate(h1, np.where(mu & up_t, 1, np.where(md & dn_t, -1, 0)), sl_dist=2 * a1, tp_dist=4 * a1, max_hold=150)

    dmom = np.sign(d1.close / d1.close.shift(60) - 1)
    flips = (dmom != dmom.shift(1)) & dmom.notna()
    ad = atr(d1).to_numpy()
    S["T5 Daily 60-day momentum"] = lambda: simulate(d1, np.where(flips, dmom, 0), sl_dist=3 * ad, exit_long=(dmom < 0).to_numpy(), exit_short=(dmom > 0).to_numpy(), max_hold=250)

    # ---------- mean reversion ----------
    r2 = rsi(c, 2); sma5 = c.rolling(5).mean()
    rsi_side = np.where((r2 < 10) & up_t, 1, np.where((r2 > 90) & dn_t, -1, 0))
    S["M1 RSI(2) extremes"] = lambda: simulate(h1, rsi_side, sl_dist=2 * a1, exit_long=(c > sma5).to_numpy(), exit_short=(c < sma5).to_numpy(), max_hold=24)

    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    lowb, upb = mid - 2 * sd, mid + 2 * sd
    bb_side = np.where((c.shift(1) < lowb.shift(1)) & (c > lowb), 1, np.where((c.shift(1) > upb.shift(1)) & (c < upb), -1, 0))
    S["M2 Bollinger snap-back"] = lambda: simulate(h1, bb_side, sl_dist=1.5 * a1, exit_long=(c > mid).to_numpy(), exit_short=(c < mid).to_numpy(), max_hold=30)

    z = (c - c.rolling(50).mean()) / c.rolling(50).std()
    S["M3 Z-score stretch"] = lambda: simulate(h1, np.where(z < -2.5, 1, np.where(z > 2.5, -1, 0)), sl_dist=2.5 * a1, exit_long=(z > 0).to_numpy(), exit_short=(z < 0).to_numpy(), max_hold=48)

    # ---------- intraday session setups (15-minute) ----------
    hrs = m15.time.dt.hour; mins = m15.time.dt.minute; day = m15.time.dt.normalize()
    a15 = atr(m15).to_numpy()

    def day_range(h_from, h_to):
        mask = (hrs >= h_from) & (hrs < h_to)
        g = m15[mask].groupby(day[mask])
        return day.map(g.high.max()), day.map(g.low.min())

    def range_breakout(h_from, h_to, trade_from, trade_to, flat, skip_days=None):
        rh, rl = day_range(h_from, h_to)
        inwin = (hrs >= trade_from) & (hrs < trade_to)
        side = np.where(inwin & (m15.close > rh), 1, np.where(inwin & (m15.close < rl), -1, 0))
        # only the first breakout of each day
        first = pd.Series(side != 0).groupby(day).cumsum().to_numpy() == 1
        side = np.where(first, side, 0)
        if skip_days is not None:
            side = np.where(day.isin(skip_days), 0, side)
        level = np.where(side > 0, rl, rh)
        return simulate(m15, side, sl_level=level, tp_r=2, flat_hour=flat)

    news_days = set(events.time.dt.normalize())
    S["S1 London breakout"] = lambda: range_breakout(1, 10, 10, 14, 23)
    S["S2 New York open breakout"] = lambda: range_breakout(15, 16, 16, 20, 23)
    S["S3 Morning range breakout"] = lambda: range_breakout(1, 3, 3, 9, 12)

    ah, al = day_range(1, 10)
    lw = (hrs >= 10) & (hrs < 13)
    fade = np.where(lw & (m15.high > ah) & (m15.close < ah), -1, np.where(lw & (m15.low < al) & (m15.close > al), 1, 0))
    fade = np.where(pd.Series(fade != 0).groupby(day).cumsum().to_numpy() == 1, fade, 0)
    fade_stop = np.where(fade < 0, m15.high + a15 * 0.2, m15.low - a15 * 0.2)
    S["M4 Asian-range fade"] = lambda: simulate(m15, fade, sl_level=fade_stop, tp_dist=np.abs(m15.close - (ah + al) / 2).to_numpy(), flat_hour=23)

    # ---------- volatility ----------
    width = (upb - lowb) / mid
    squeezed = (width <= width.rolling(120).quantile(0.1)).rolling(5).max() > 0
    S["V1 Bollinger squeeze breakout"] = lambda: simulate(h1, np.where(squeezed & (c > upb), 1, np.where(squeezed & (c < lowb), -1, 0)), sl_dist=2 * a1, tp_dist=4 * a1, max_hold=72)

    inside = (d1.high < d1.high.shift(1)) & (d1.low > d1.low.shift(1))
    yh, yl = daily_to(h1, d1, d1.high.where(inside)), daily_to(h1, d1, d1.low.where(inside))
    ib = np.where(c > yh, 1, np.where(c < yl, -1, 0))
    ib = np.where(pd.Series(ib != 0).groupby(h1.time.dt.normalize()).cumsum().to_numpy() == 1, ib, 0)
    S["V2 Inside-day breakout"] = lambda: simulate(h1, ib, sl_level=np.where(ib > 0, yl, yh), tp_r=2, max_hold=24)

    # ---------- calendar ----------
    thu_close = (h1.time.dt.dayofweek == 3) & (h1.time.dt.hour == 22)
    S["C1 Friday long"] = lambda: simulate(h1, np.where(thu_close, 1, 0), sl_dist=4 * a1, max_hold=24)
    month = d1.time.dt.to_period("M")
    left = d1.groupby(month).cumcount(ascending=False)          # trading days left in the month
    S["C2 Turn-of-month long"] = lambda: simulate(d1, np.where(left == 2, 1, 0), sl_dist=3 * ad, max_hold=5)

    # ---------- scheduled news (15-minute) ----------
    def event_bars(kind, minutes_after=0):
        times = set(events[events.event == kind].time + pd.Timedelta(minutes=minutes_after))
        return m15.time.isin(times).to_numpy()

    rng = (m15.high - m15.low).to_numpy()
    body = np.sign(m15.close - m15.open).to_numpy()
    nfp_bar = event_bars("NFP")                    # the 15:30-15:45 candle is the release candle
    big = rng > 1.5 * a15
    S["N1 NFP momentum"] = lambda: simulate(m15, np.where(nfp_bar & big, body, 0), sl_level=np.where(body > 0, m15.low, m15.high), tp_r=2, flat_hour=23)
    move_start = pd.Series(m15.open).shift(1).to_numpy()  # open of the release candle, seen at the 2nd candle
    nfp_2nd = event_bars("NFP", 15)
    swing = m15.close.to_numpy() - move_start
    hi2 = np.maximum(m15.high, m15.high.shift(1)).to_numpy(); lo2 = np.minimum(m15.low, m15.low.shift(1)).to_numpy()
    S["N2 NFP fade"] = lambda: simulate(m15, np.where(nfp_2nd & (np.abs(swing) > 1.5 * a15), -np.sign(swing), 0), sl_level=np.where(swing > 0, hi2 + 0.2 * a15, lo2 - 0.2 * a15), tp_r=1.5, flat_hour=23)
    fomc_bar = event_bars("FOMC")
    S["N3 FOMC drift"] = lambda: simulate(m15, np.where(fomc_bar & (rng > a15), body, 0), sl_level=np.where(body > 0, m15.low, m15.high), tp_r=2, max_hold=52)
    fomc_days = events[events.event == "FOMC"].time.dt.normalize()
    eve = set(fomc_days - pd.Timedelta(days=1))
    pre = (h1.time.dt.normalize().isin(eve) & (h1.time.dt.hour == 21)).to_numpy()
    S["N4 Pre-Fed drift"] = lambda: simulate(h1, np.where(pre, 1, 0), sl_dist=3 * a1, max_hold=22)

    # ---------- blends ----------
    d_up = daily_to(h1, d1, d1.close > ema(d1.close, 50)).fillna(False).astype(bool)
    blend1 = np.where((r2 < 10) & d_up, 1, np.where((r2 > 90) & ~d_up, -1, 0))
    S["B1 RSI(2) with the daily trend"] = lambda: simulate(h1, blend1, sl_dist=2 * a1, exit_long=(c > sma5).to_numpy(), exit_short=(c < sma5).to_numpy(), max_hold=24)

    vote = np.sign(e20 - e50) + np.sign(c - (hi55 + lo55) / 2) + np.sign(macd - sig)
    v_on = np.where((vote == 3) & (vote.shift(1) < 3), 1, np.where((vote == -3) & (vote.shift(1) > -3), -1, 0))
    S["B2 Trend vote (3 must agree)"] = lambda: simulate(h1, v_on, sl_dist=3 * a1, exit_long=(vote <= 0).to_numpy(), exit_short=(vote >= 0).to_numpy(), max_hold=400)

    S["B3 London breakout, no news days"] = lambda: range_breakout(1, 10, 10, 14, 23, skip_days=news_days)

    vol_rank = pd.Series(a1).rolling(500).rank(pct=True).to_numpy()
    donch = np.where(c > hi20, 1, np.where(c < lo20, -1, 0))
    regime = np.where(vol_rank < 0.5, rsi_side, donch)
    S["B4 Calm: revert / wild: trend"] = lambda: simulate(h1, regime, sl_dist=2 * a1, tp_dist=np.where(vol_rank < 0.5, 1.5, 4) * a1, max_hold=48)

    ry = macro.real_yield.reindex(pd.to_datetime(d1.time.dt.normalize())).ffill().shift(2)   # published with a lag
    falling = pd.Series((ry.diff(60) < 0).to_numpy(), index=d1.time.dt.normalize())
    ry_fall = h1.time.dt.normalize().map(falling.shift(1)).fillna(False).astype(bool)
    turtle_ry = np.where((turtle > 0) & ry_fall, 1, np.where((turtle < 0) & ~ry_fall, -1, 0))
    S["B5 Turtle + real-yield filter"] = lambda: simulate(h1, turtle_ry, sl_dist=2 * a1, exit_long=(c < lo20).to_numpy(), exit_short=(c > hi20).to_numpy(), max_hold=500)
    return S


# ============================== scoring ==============================

def score(t):
    if len(t) < 2:
        return dict(n=len(t), win=np.nan, pf=np.nan, avgR=np.nan, t=np.nan, ret=np.nan, dd=np.nan)
    r = t.r.to_numpy()
    eq = np.cumprod(1 + 0.01 * r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), win=(r > 0).mean(), pf=gains / losses if losses else np.inf, avgR=r.mean(),
                t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), ret=eq[-1] - 1, dd=dd)


if __name__ == "__main__":
    t0 = time.time()
    m15, h1, d1, events, macro = load()
    print(f"Data: {m15.time.iloc[0]:%Y-%m-%d} to {m15.time.iloc[-1]:%Y-%m-%d} · {len(m15):,} 15-min candles, {len(h1):,} hourly, {len(d1):,} daily")
    rows = []
    for name, run in strategies(m15, h1, d1, events, macro).items():
        s0 = time.time()
        trades = run()
        trades.to_csv(DATA / f"tourn_{name.split()[0]}.csv", index=False)
        dev, judge = trades[trades.time < pd.Timestamp(EXPLORE_END)], trades[trades.time >= pd.Timestamp(EXPLORE_END)]
        a, d, j = score(trades), score(dev), score(judge)
        passed = (a["t"] >= T_BAR) and d["ret"] > 0 and j["ret"] > 0
        rows.append({"strategy": name, "trades": a["n"], "win%": a["win"] * 100, "PF": a["pf"], "avg R": a["avgR"],
                     "t": a["t"], "2005-18 %": d["ret"] * 100, "2019-23 %": j["ret"] * 100,
                     "worst DD %": a["dd"] * 100, "verdict": "PASS" if passed else "-"})
        print(f"  {name:34} {a['n']:5} trades  t={a['t']:+.2f}  ({time.time() - s0:.0f}s)", flush=True)
    rows.append({"strategy": "SMC v1.1 (exam, earlier)", "trades": 315, "win%": 33, "PF": 0.70, "avg R": np.nan, "t": -2.57,
                 "2005-18 %": np.nan, "2019-23 %": np.nan, "worst DD %": -39.4, "verdict": "-"})
    board = pd.DataFrame(rows).sort_values("t", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\n=== LEADERBOARD (bar to pass: t >= {T_BAR} AND profitable in both periods) ===")
    print(board.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    board.to_csv(DATA / "tournament_results.csv", index=False)
    print(f"\n{sum(board.verdict == 'PASS')} of {len(board)} strategies passed. ({time.time() - t0:.0f}s)")
