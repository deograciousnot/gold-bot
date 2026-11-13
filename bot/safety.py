"""
Safety rules. Every check returns (ok, reason) so the log always says WHY the bot did nothing.

These run before every trade. If any fails, no order is sent.
"""

from pathlib import Path

from . import config

DEMO = 0  # MT5's code for a demo account (ACCOUNT_TRADE_MODE_DEMO)


class RealAccountError(RuntimeError):
    """Raised when the bot is pointed at anything other than a demo account."""


def require_demo(account):
    """Hard lock: refuse to run at all on a real (or contest) account.
    Deliberately not configurable - to trade real money, a human has to change this code."""
    if account.trade_mode != DEMO:
        raise RealAccountError(
            f"Account {account.login} on {account.server} is NOT a demo account. "
            "The bot refuses to run. Switch MT5 to your demo account.")


def kill_switch_on(base_dir):
    return (Path(base_dir) / config.KILL_SWITCH_FILE).exists()


def daily_loss_hit(equity, day_start_equity):
    if day_start_equity <= 0:
        return False
    return equity <= day_start_equity * (1 - config.DAILY_LOSS_LIMIT)


def can_open_trade(*, server_time, spread, open_positions, trades_today, equity, day_start_equity):
    """All the reasons not to open a new trade right now."""
    if daily_loss_hit(equity, day_start_equity):
        return False, f"daily loss limit hit ({config.DAILY_LOSS_LIMIT:.0%}): no new trades until tomorrow"
    if open_positions >= config.MAX_OPEN_POSITIONS:
        return False, f"already {open_positions} open position(s) (max {config.MAX_OPEN_POSITIONS})"
    if trades_today >= config.MAX_TRADES_PER_DAY:
        return False, f"already {trades_today} trades today (max {config.MAX_TRADES_PER_DAY})"
    if server_time.hour in config.NO_TRADE_HOURS:
        return False, f"{server_time:%H:%M} is in the no-trade hours around the daily rollover"
    if config.CLOSE_BEFORE_WEEKEND and server_time.weekday() == 4 and server_time.hour >= config.FRIDAY_CLOSE_HOUR:
        return False, "too close to the weekend"
    if server_time.weekday() >= 5:
        return False, "weekend"
    if spread > config.MAX_SPREAD:
        return False, f"spread ${spread:.2f} is wider than the ${config.MAX_SPREAD:.2f} limit"
    return True, "ok"


def too_old(open_time, server_time):
    """True if a position has been open longer than MAX_HOLD_HOURS (both times in server time)."""
    return (server_time - open_time).total_seconds() >= config.MAX_HOLD_HOURS * 3600


def must_close_all(*, server_time, equity, day_start_equity, base_dir):
    """Reasons to close every bot position immediately."""
    if kill_switch_on(base_dir):
        return True, "kill switch file found"
    if daily_loss_hit(equity, day_start_equity):
        return True, "daily loss limit hit"
    if config.CLOSE_BEFORE_WEEKEND and server_time.weekday() == 4 and server_time.hour >= config.FRIDAY_CLOSE_HOUR:
        return True, "closing before the weekend"
    return False, ""
