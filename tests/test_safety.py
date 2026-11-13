"""
Tests for the safety rules and position sizing. They don't need MT5 running.

    python -m unittest discover tests
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from bot import config, safety
from bot.risk import compute_lots

TUESDAY_NOON = datetime(2026, 10, 27, 12, 0)
OK_STATE = dict(server_time=TUESDAY_NOON, spread=0.30, open_positions=0, trades_today=0,
                equity=100_000, day_start_equity=100_000)


class DemoLock(unittest.TestCase):
    def test_demo_account_allowed(self):
        safety.require_demo(SimpleNamespace(trade_mode=0, login=1, server="Demo"))

    def test_real_account_refused(self):
        with self.assertRaises(safety.RealAccountError):
            safety.require_demo(SimpleNamespace(trade_mode=2, login=1, server="Live"))

    def test_contest_account_refused(self):
        with self.assertRaises(safety.RealAccountError):
            safety.require_demo(SimpleNamespace(trade_mode=1, login=1, server="Contest"))


class OpenTradeRules(unittest.TestCase):
    def check(self, **changes):
        return safety.can_open_trade(**{**OK_STATE, **changes})

    def test_normal_conditions_allowed(self):
        self.assertEqual(self.check(), (True, "ok"))

    def test_daily_loss_limit(self):
        lost = 100_000 * (1 - config.DAILY_LOSS_LIMIT)
        self.assertFalse(self.check(equity=lost)[0])
        self.assertTrue(self.check(equity=lost + 1)[0])

    def test_max_open_positions(self):
        self.assertFalse(self.check(open_positions=config.MAX_OPEN_POSITIONS)[0])

    def test_max_trades_per_day(self):
        self.assertFalse(self.check(trades_today=config.MAX_TRADES_PER_DAY)[0])

    def test_rollover_hours(self):
        for hour in config.NO_TRADE_HOURS:
            self.assertFalse(self.check(server_time=TUESDAY_NOON.replace(hour=hour))[0])

    def test_wide_spread(self):
        self.assertFalse(self.check(spread=config.MAX_SPREAD + 0.01)[0])

    def test_friday_evening_and_weekend(self):
        friday_late = datetime(2026, 10, 30, config.FRIDAY_CLOSE_HOUR, 0)
        saturday = datetime(2026, 10, 31, 12, 0)
        self.assertFalse(self.check(server_time=friday_late)[0])
        self.assertFalse(self.check(server_time=saturday)[0])


class CloseAllRules(unittest.TestCase):
    def test_kill_switch_file(self):
        with tempfile.TemporaryDirectory() as d:
            args = dict(server_time=TUESDAY_NOON, equity=100_000, day_start_equity=100_000, base_dir=d)
            self.assertFalse(safety.must_close_all(**args)[0])
            (Path(d) / config.KILL_SWITCH_FILE).touch()
            self.assertTrue(safety.must_close_all(**args)[0])

    def test_daily_loss_closes_everything(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(safety.must_close_all(server_time=TUESDAY_NOON, equity=90_000,
                                                  day_start_equity=100_000, base_dir=d)[0])


class Sizing(unittest.TestCase):
    BROKER = dict(volume_step=0.01, volume_min=0.01, volume_max=100, cap=10)

    def test_one_percent_risk(self):
        # $10,000 equity, 1% = $100 risk; a $10 stop on 1 lot (100 oz) loses $1,000 -> 0.10 lots
        self.assertEqual(compute_lots(equity=10_000, risk_fraction=0.01, loss_per_lot=1_000, **self.BROKER), 0.10)

    def test_rounds_down_never_up(self):
        # exact answer 0.0999... must round down to 0.09, not up to 0.10
        self.assertEqual(compute_lots(equity=9_999, risk_fraction=0.01, loss_per_lot=1_000, **self.BROKER), 0.09)

    def test_cap_applies(self):
        self.assertEqual(compute_lots(equity=10_000_000, risk_fraction=0.01, loss_per_lot=1_000,
                                      **{**self.BROKER, "cap": 0.10}), 0.10)

    def test_too_small_account_returns_zero(self):
        self.assertEqual(compute_lots(equity=500, risk_fraction=0.01, loss_per_lot=1_000, **self.BROKER), 0.0)

    def test_bad_inputs_return_zero(self):
        self.assertEqual(compute_lots(equity=10_000, risk_fraction=0.01, loss_per_lot=0, **self.BROKER), 0.0)


if __name__ == "__main__":
    unittest.main()
