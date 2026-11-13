"""
Tests for the inside-day breakout strategy and the 24-hour time limit. No MT5 needed.

    python -m unittest discover tests
"""

import unittest
from datetime import datetime, timedelta

import pandas as pd

from bot import config, safety
from bot.strategies import inside_day


def candles(days):
    """Build hourly candles from a list of days, each a list of (high, low, close) per hour."""
    rows, start = [], datetime(2026, 11, 2, 1)     # a Monday, 01:00 server time
    for d, hours in enumerate(days):
        for h, (hi, lo, cl) in enumerate(hours):
            rows.append((start + timedelta(days=d, hours=h), cl, hi, lo, cl))
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


BIG = [(110, 90, 100)] * 4          # day 1: range 90-110
INSIDE = [(105, 95, 100)] * 4       # day 2: inside day (95-105)
WIDE = [(112, 94, 100)] * 4         # day 2 that is NOT an inside day


class InsideDay(unittest.TestCase):
    def test_breakout_up_on_first_close_above(self):
        s = inside_day(candles([BIG, INSIDE, [(103, 99, 101), (107, 100, 106)]]))
        self.assertIsNotNone(s)
        self.assertEqual(s.side, 1)
        self.assertEqual(s.stop, 95)                 # other side of the inside day
        self.assertEqual(s.target, 106 + 2 * (106 - 95))

    def test_breakout_down(self):
        s = inside_day(candles([BIG, INSIDE, [(101, 97, 98), (99, 92, 93)]]))
        self.assertEqual(s.side, -1)
        self.assertEqual(s.stop, 105)

    def test_no_signal_if_yesterday_not_inside(self):
        self.assertIsNone(inside_day(candles([BIG, WIDE, [(113, 99, 112)]])))

    def test_no_signal_while_still_inside(self):
        self.assertIsNone(inside_day(candles([BIG, INSIDE, [(104, 96, 100)]])))

    def test_only_first_breakout_of_the_day(self):
        # first breakout at hour 2, a second close beyond at hour 3 must not signal again
        self.assertIsNone(inside_day(candles([BIG, INSIDE, [(103, 99, 101), (107, 100, 106), (108, 104, 107)]])))


class TimeLimit(unittest.TestCase):
    def test_too_old(self):
        t0 = datetime(2026, 11, 3, 10)
        self.assertFalse(safety.too_old(t0, t0 + timedelta(hours=config.MAX_HOLD_HOURS - 1)))
        self.assertTrue(safety.too_old(t0, t0 + timedelta(hours=config.MAX_HOLD_HOURS)))


if __name__ == "__main__":
    unittest.main()
