"""
Draw the 20-year backtest equity curves (trend vs breakout vs just holding gold) as an SVG,
for the portfolio. Plain SVG text, so no plotting library is needed.

Usage: python make_equity_chart.py <output.svg>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 720, 300, 52, 20, 20, 34

trend = pd.read_csv(DATA / "trades_trend_H1_long.csv", parse_dates=["exit_time"])
breakout = pd.read_csv(DATA / "trades_breakout_H1_long.csv", parse_dates=["exit_time"])
gold = pd.read_csv(DATA / "XAUUSD_H1_long.csv", parse_dates=["time"])
gold = gold.groupby(gold.time.dt.to_period("W")).close.last()

series = {
    "Holding gold": (pd.Series(10_000 * gold.values / gold.values[0], index=gold.index.to_timestamp()), "#d8b25a"),
    "Trend-follower": (trend.set_index("exit_time").balance, "#6aa9ff"),
    "London breakout": (breakout.set_index("exit_time").balance, "#ff7a6b"),
}

t0, t1 = gold.index[0].to_timestamp(), gold.index[-1].to_timestamp()
lo, hi = np.log(2_000), np.log(120_000)   # log scale so the gold line doesn't flatten the others


def x(t):
    return PAD_L + (t - t0) / (t1 - t0) * (W - PAD_L - PAD_R)


def y(v):
    return PAD_T + (hi - np.log(v)) / (hi - lo) * (H - PAD_T - PAD_B)


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
         'aria-label="20-year backtest: holding gold vs trend-follower vs London breakout" '
         'font-family="system-ui, sans-serif" font-size="11">']
for v in (5_000, 10_000, 20_000, 50_000, 100_000):
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="currentColor" '
                 f'stroke-opacity="0.12"/><text x="{PAD_L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" '
                 f'fill="currentColor" fill-opacity="0.6">${v // 1000}k</text>')
for year in range(2006, t1.year + 1, 4):
    t = pd.Timestamp(f"{year}-01-01")
    parts.append(f'<text x="{x(t):.1f}" y="{H - 12}" text-anchor="middle" fill="currentColor" '
                 f'fill-opacity="0.6">{year}</text>')
for name, (s, color) in series.items():
    s = s[~s.index.duplicated()].resample("W").last().dropna()
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in s.items())
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                 'stroke-linejoin="round"/>')
    parts.append(f'<text x="{x(s.index[-1]) - 4:.1f}" y="{y(s.iloc[-1]) - 6:.1f}" text-anchor="end" '
                 f'fill="{color}" font-weight="600">{name}</text>')
parts.append("</svg>")

Path(sys.argv[1]).write_text("\n".join(parts), encoding="utf-8")
print("wrote", sys.argv[1])
