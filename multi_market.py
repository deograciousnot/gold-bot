"""
Does the inside-day breakout work beyond gold?

Downloads 20 years of HOURLY candles for other markets from Dukascopy (one file per month, so
much faster than 1-minute data), then runs the exact same rules as "V2 Inside-day breakout" in
tournament.py through the same simulator. The strategy was designed on gold only, so for every
other market the whole history is unseen data.

Spreads are typical retail spreads (a little pessimistic), converted to each market's units.

Usage (on the server):  python multi_market.py            (download what's missing, then test)
                        python multi_market.py test       (test only, using what's downloaded)
"""

import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import tournament as T

DATA = Path(__file__).parent / "data"
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/BID_candles_hour_1.bi5"
RECORD = struct.Struct(">5if")

# symbol: (name, typical retail spread in price units, plausible price range for scale detection)
MARKETS = {
    "XAGUSD": ("Silver", 0.03, (3, 120)),
    "EURUSD": ("EUR/USD", 0.00012, (0.8, 1.7)),
    "GBPUSD": ("GBP/USD", 0.00015, (1.0, 2.2)),
    "USDJPY": ("USD/JPY", 0.015, (70, 200)),
    "LIGHTCMDUSD": ("US oil (WTI)", 0.04, (15, 200)),     # median price (oil briefly went negative in 2020)
    "USA500IDXUSD": ("S&P 500", 0.6, (500, 9000)),
    "USATECHIDXUSD": ("Nasdaq 100", 1.5, (1000, 30000)),
}


def fetch_month(sym, month):
    folder = DATA / "dukascopy" / sym
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"H{month:%Y%m}.bi5"
    if cache.exists():
        return cache.read_bytes()
    url = URL.format(sym=sym, y=month.year, m=month.month - 1)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = b""
                break
            time.sleep(10 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(10 * (attempt + 1))
    else:
        return None
    if (month.year, month.month) != (date.today().year, date.today().month):   # current month still growing
        tmp = cache.with_suffix(".part"); tmp.write_bytes(data); tmp.replace(cache)
    return data


def months(first=2005):
    m, out = date(first, 1, 1), []
    while m <= date.today():
        out.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def download(sym):
    ms = months()
    with ThreadPoolExecutor(max_workers=1) as pool:      # gentler: throttling caused the gaps
        got = list(pool.map(lambda m: fetch_month(sym, m), ms))
    failed = sum(g is None for g in got)
    return {m: g for m, g in zip(ms, got) if g}, failed


def decode(sym, files):
    """Raw integers -> prices. The divisor differs per market, so pick the one that gives plausible prices."""
    rows = []
    for month, data in files.items():
        try:
            raw = lzma.decompress(data)
        except lzma.LZMAError:
            continue
        start = datetime(month.year, month.month, 1, tzinfo=timezone.utc)
        for secs, o, c, lo, hi, vol in RECORD.iter_unpack(raw):
            if vol > 0:
                rows.append((start + timedelta(seconds=secs), o, hi, lo, c))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    lo_ok, hi_ok = MARKETS[sym][2]
    med = df.close.median()
    scale = next(s for s in (100000, 10000, 1000, 100, 10, 1) if lo_ok <= med / s <= hi_ok)
    df[["open", "high", "low", "close"]] /= scale
    df["time"] = df.time.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)  # server time
    df["spread"] = MARKETS[sym][1] / T.POINT        # in the simulator's units (x 0.01 = price)
    return df.sort_values("time").reset_index(drop=True), scale


def inside_day_trades(h1):
    """Exactly the V2 rules from tournament.py."""
    days = h1.groupby(h1.time.dt.normalize()).agg(high=("high", "max"), low=("low", "min"))
    days = days[days.index.dayofweek < 5]
    inside = (days.high < days.high.shift(1)) & (days.low > days.low.shift(1))
    norm = h1.time.dt.normalize()
    yh, yl = norm.map(days.high.where(inside).shift(1)), norm.map(days.low.where(inside).shift(1))
    c = h1.close
    ib = np.where(c > yh, 1, np.where(c < yl, -1, 0))
    ib = np.where(pd.Series(ib != 0).groupby(norm).cumsum().to_numpy() == 1, ib, 0)
    return T.simulate(h1, ib, sl_level=np.where(ib > 0, yl, yh), tp_r=2, max_hold=24)


if __name__ == "__main__":
    test_only = len(sys.argv) > 1 and sys.argv[1] == "test"
    rows, all_trades = [], []
    for sym, (name, spread, _) in MARKETS.items():
        t0 = time.time()
        if test_only:
            folder = DATA / "dukascopy" / sym
            files = {datetime.strptime(f.stem[1:], "%Y%m").date(): f.read_bytes() for f in sorted(folder.glob("H*.bi5"))} if folder.exists() else {}
            failed = 0
        else:
            files, failed = download(sym)
        if not files:
            print(f"{name}: no data"); continue
        h1, scale = decode(sym, files)
        h1 = h1.drop_duplicates("time")
        print(f"{name:13} {len(files)} months ({failed} failed) · {h1.time.iloc[0]:%Y-%m} to {h1.time.iloc[-1]:%Y-%m} · "
              f"price {h1.close.iloc[0]:.5g} -> {h1.close.iloc[-1]:.5g} (divisor {scale}) · {time.time() - t0:.0f}s", flush=True)
        t = inside_day_trades(h1)
        t["market"] = name
        all_trades.append(t)
        a = T.score(t)
        parts = {lbl: T.score(t[(t.time >= s) & (t.time < e)]) for lbl, s, e in
                 (("2005-18", "2000", "2019"), ("2019-23", "2019", "2024"), ("2024-26", "2024", "2030"))}
        buys, sells = T.score(t[t.side > 0]), T.score(t[t.side < 0])
        rows.append({"market": name, "trades": a["n"], "per year": a["n"] / max(1, (t.time.iloc[-1] - t.time.iloc[0]).days / 365),
                     "win%": a["win"] * 100, "PF": a["pf"], "avg R": a["avgR"], "t": a["t"],
                     "2005-18 R": parts["2005-18"]["avgR"], "2019-23 R": parts["2019-23"]["avgR"], "2024-26 R": parts["2024-26"]["avgR"],
                     "buys R": buys["avgR"], "sells R": sells["avgR"], "worst DD %": a["dd"] * 100})
    board = pd.DataFrame(rows).sort_values("t", ascending=False)
    pd.set_option("display.width", 240)
    print("\n=== INSIDE-DAY BREAKOUT ACROSS MARKETS (avg R per trade; t = confidence) ===")
    print(board.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    board.to_csv(DATA / "multi_market_results.csv", index=False)
    if all_trades:
        allt = pd.concat(all_trades).sort_values("time")
        s = T.score(allt)
        print(f"\nAll markets together: {s['n']} trades · avg {s['avgR']:+.3f}R · PF {s['pf']:.2f} · t={s['t']:+.2f} · worst DD {s['dd']*100:.1f}% (1% risk per trade)")
        allt.to_csv(DATA / "multi_market_trades.csv", index=False)
