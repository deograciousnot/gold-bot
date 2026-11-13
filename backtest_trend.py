"""
Step 2b: backtest a simple trend-following strategy on gold (XAUUSD, 1-hour candles).

THE STRATEGY (in plain words)
  - Trend filter: only buy when price is above the 200-candle average (uptrend),
    only sell when it's below (downtrend).
  - Entry: a fast average (20) crossing a slower one (50) in the trend's direction.
  - Stop loss:   2 x ATR away  (ATR = how much gold typically moves per candle)
  - Take profit: 4 x ATR away  (so each win is worth about 2 losses)
  - Risk: each trade risks 1% of the account. Trade size is worked out from that.

HONESTY RULES BUILT IN
  - Trades enter at the NEXT candle's open, so the strategy never "sees the future".
  - The real spread from MT5's data is paid on every trade.
  - If the stop loss and take profit are both touched in the same candle,
    we assume the LOSS happened first (pessimistic).
  - Results are split into an older "build" period and a recent "unseen" period.

This is a simulation on past data only. It never connects to MT5 or trades.
"""

import sys
from pathlib import Path

import pandas as pd

# python backtest_trend.py          -> HFM's 1-hour data (~4 years)
# python backtest_trend.py H1_long  -> Dukascopy 1-hour data (~20 years)
DATASET = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "H1"
DATA_FILE = Path(__file__).parent / "data" / f"XAUUSD_{DATASET}.csv"

# --- Settings you can experiment with ---
START_BALANCE = 10_000      # dollars
RISK_PER_TRADE = 0.01       # 1% of balance per trade
FAST, SLOW, TREND = 20, 50, 200
ATR_PERIOD = 14
SL_ATR, TP_ATR = 2.0, 4.0
CONTRACT_SIZE = 100         # ounces per 1.00 lot on HFM
MIN_LOT, LOT_STEP = 0.01, 0.01
POINT = 0.01                # MT5 spread is given in points of $0.01
SPLIT = 0.7                 # first 70% of history = "build", last 30% = "unseen"


def load_data():
    if not DATA_FILE.exists():
        sys.exit(f"Missing {DATA_FILE}. Run download_data.py first.")
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])

    df["ema_fast"] = df.close.ewm(span=FAST, adjust=False).mean()
    df["ema_slow"] = df.close.ewm(span=SLOW, adjust=False).mean()
    df["ema_trend"] = df.close.ewm(span=TREND, adjust=False).mean()

    # ATR: average of each candle's true range (high-low, including gaps from the previous close)
    prev_close = df.close.shift(1)
    true_range = pd.concat([df.high - df.low,
                            (df.high - prev_close).abs(),
                            (df.low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()

    # Signal on the candle that just CLOSED; we act on the next one
    cross_up = (df.ema_fast > df.ema_slow) & (df.ema_fast.shift(1) <= df.ema_slow.shift(1))
    cross_down = (df.ema_fast < df.ema_slow) & (df.ema_fast.shift(1) >= df.ema_slow.shift(1))
    df["signal"] = 0
    df.loc[cross_up & (df.close > df.ema_trend), "signal"] = 1     # buy
    df.loc[cross_down & (df.close < df.ema_trend), "signal"] = -1  # sell
    return df.iloc[TREND:].reset_index(drop=True)


def lot_size(balance, stop_distance):
    """How many lots so that hitting the stop loses RISK_PER_TRADE of the balance."""
    risk_dollars = balance * RISK_PER_TRADE
    lots = risk_dollars / (stop_distance * CONTRACT_SIZE)
    lots = int(lots / LOT_STEP) * LOT_STEP  # round DOWN to the broker's step
    return round(lots, 2) if lots >= MIN_LOT else 0.0


def run(df):
    balance = START_BALANCE
    trades, skipped = [], 0
    pos = None  # the open trade, if any

    for i in range(1, len(df)):
        bar = df.iloc[i]
        spread = bar.spread * POINT  # candle prices are bid prices; ask = bid + spread

        # 1) Open a new trade at this candle's open if the PREVIOUS candle gave a signal
        signal = df.signal.iloc[i - 1]
        if pos is None and signal != 0:
            atr = df.atr.iloc[i - 1]
            # Buys fill at the ask; sells fill at the bid
            entry = bar.open + spread if signal == 1 else bar.open
            stop_distance = SL_ATR * atr + spread  # spread counts toward the risk
            lots = lot_size(balance, stop_distance)
            if lots == 0:
                skipped += 1
            else:
                pos = {"side": signal, "entry_time": bar.time, "entry": entry, "lots": lots,
                       "sl": entry - signal * SL_ATR * atr,
                       "tp": entry + signal * TP_ATR * atr}

        # 2) Manage the open trade (including the candle it opened on):
        #    did this candle hit the stop or the target?
        if pos:
            if pos["side"] == 1:   # a buy closes by selling at the bid
                hit_sl, hit_tp = bar.low <= pos["sl"], bar.high >= pos["tp"]
            else:                  # a sell closes by buying at the ask
                hit_sl, hit_tp = bar.high + spread >= pos["sl"], bar.low + spread <= pos["tp"]
            if hit_sl or hit_tp:
                exit_price = pos["sl"] if hit_sl else pos["tp"]  # loss first if both
                pnl = (exit_price - pos["entry"]) * pos["side"] * pos["lots"] * CONTRACT_SIZE
                balance += pnl
                trades.append({**pos, "exit_time": bar.time, "exit": exit_price,
                               "pnl": pnl, "balance": balance, "win": hit_tp and not hit_sl})
                pos = None

    return pd.DataFrame(trades), skipped


def report(name, t, start_balance):
    if t.empty:
        print(f"\n--- {name}: no trades ---")
        return
    wins, losses = t[t.win], t[~t.win]
    equity = pd.concat([pd.Series([start_balance]), t.balance])
    drawdown = (equity / equity.cummax() - 1).min()
    end = t.balance.iloc[-1]
    gross_win, gross_loss = wins.pnl.sum(), -losses.pnl.sum()
    # Longest run of losses in a row
    streak = longest = 0
    for w in t.win:
        streak = 0 if w else streak + 1
        longest = max(longest, streak)

    print(f"\n--- {name} ({t.entry_time.iloc[0]:%Y-%m-%d} to {t.exit_time.iloc[-1]:%Y-%m-%d}) ---")
    print(f"Trades:            {len(t)}  (buys {sum(t.side == 1)}, sells {sum(t.side == -1)})")
    print(f"Win rate:          {len(wins) / len(t):.0%}")
    print(f"Profit:            ${end - start_balance:,.0f}  ({end / start_balance - 1:+.1%})")
    print(f"Profit factor:     {gross_win / gross_loss:.2f}" if gross_loss else "Profit factor: n/a")
    print(f"Worst drawdown:    {drawdown:.1%}")
    print(f"Longest losing run: {longest} trades")
    for side, label in ((1, "Buys"), (-1, "Sells")):
        s = t[t.side == side]
        if len(s):
            print(f"  {label:5} -> {len(s)} trades, win rate {s.win.mean():.0%}, P/L ${s.pnl.sum():,.0f}")


if __name__ == "__main__":
    df = load_data()
    trades, skipped = run(df)
    print(f"Data: {len(df):,} one-hour candles, {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d}")
    print(f"Gold moved from ${df.close.iloc[0]:,.0f} to ${df.close.iloc[-1]:,.0f} over this period")
    if skipped:
        print(f"Skipped {skipped} signals: the account was too small for even 0.01 lot at 1% risk")

    report("WHOLE PERIOD", trades, START_BALANCE)

    split_time = df.time.iloc[int(len(df) * SPLIT)]
    build = trades[trades.entry_time < split_time]
    unseen = trades[trades.entry_time >= split_time]
    report("BUILD period (older 70%)", build, START_BALANCE)
    unseen_start = build.balance.iloc[-1] if len(build) else START_BALANCE
    report("UNSEEN period (recent 30%)", unseen, unseen_start)

    out = Path(__file__).parent / "data" / f"trades_trend_{DATASET}.csv"
    trades.to_csv(out, index=False)
    print(f"\nEvery trade saved to {out.name}")
