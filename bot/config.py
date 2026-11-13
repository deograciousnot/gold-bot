"""
Bot settings. Everything you'd want to tweak lives here.

The demo-only lock is NOT a setting - it's hard-coded in safety.py on purpose,
so it can't be switched off by accident.
"""

SYMBOL = "XAUUSD"
TIMEFRAME = "H1"             # candle size the strategy looks at: M5, M15, H1...
STRATEGY = "inside_day"      # which strategy in strategies.py to run (the tournament survivor)

# --- Dry run: True = only log what it WOULD do, never send orders ---
DRY_RUN = False

# --- Risk limits ---
RISK_PER_TRADE = 0.01        # risk 1% of equity per trade (distance to stop loss)
MAX_LOTS = 0.10              # hard cap per trade, whatever the maths says
DAILY_LOSS_LIMIT = 0.03      # stop trading for the day after losing 3% of the day's starting equity
MAX_OPEN_POSITIONS = 1       # bot positions open at the same time
MAX_TRADES_PER_DAY = 3
MAX_HOLD_HOURS = 24          # close a bot trade that has been open this long (matches the backtest)

# --- When not to trade ---
MAX_SPREAD = 0.60            # dollars; skip if the spread is wider (news, rollover, thin market)
NO_TRADE_HOURS = [23, 0, 1]  # server hours around the daily rollover, when spreads balloon
CLOSE_BEFORE_WEEKEND = True  # close bot positions on Friday from FRIDAY_CLOSE_HOUR
FRIDAY_CLOSE_HOUR = 22       # server time

# --- Identification and housekeeping ---
MAGIC = 260_826              # tag on every order, so the bot only ever touches its own trades
COMMENT = "gold-bot"
DEVIATION_POINTS = 50        # max slippage accepted when sending an order
POLL_SECONDS = 10            # how often the bot checks for a new candle
KILL_SWITCH_FILE = "STOP"    # create a file with this name in gold-bot/ to stop the bot and close its trades
