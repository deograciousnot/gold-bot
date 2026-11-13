"""
Build a list of scheduled US news events, for the news strategies in tournament.py.

  NFP (jobs report): by the Bureau of Labor Statistics' rule, released on the third Friday after
       the week (Sunday-Saturday) that contains the 12th of the month, at 8:30 New York time.
       A few dates moved because of holidays or government shutdowns; those few are simply
       off by a week here, which can only make the news strategies look worse, not better.
  FOMC: Federal Reserve rate decisions, scraped from the Fed's own published meeting calendars.
        Statements come out at 14:00 New York time (14:15 before 2013).

Output: data/events.csv with columns  time (server time, GMT+2/+3), event
"""

import html as htmllib
import re
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data" / "events.csv"
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}


def nfp_dates(first=2005, last=2026):
    out = []
    for y in range(first, last + 1):
        for m in range(1, 13):
            twelfth = date(y, m, 12)
            week_end = twelfth + timedelta(days=(5 - twelfth.weekday()) % 7)   # Saturday ending that week
            first_friday_after = week_end + timedelta(days=6)                    # the Friday after
            out.append(first_friday_after + timedelta(weeks=2))                  # third Friday after the week
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf8", "replace")


def fomc_dates(first=2005, last=2026):
    """Decision day = the last day of each scheduled meeting ("January 30-31" -> Jan 31)."""
    days = set()
    pages = [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in range(first, 2021)]
    pages.append("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    for url in pages:
        try:
            html = fetch(url)
        except Exception as e:
            print("could not fetch", url, e)
            continue
        text = re.sub(r"<[^>]+>", " ", html)
        text = htmllib.unescape(text).replace("–", "-").replace("—", "-")   # en/em dashes -> "-"
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*-\s*", "-", text)
        if "fomccalendars" in url:
            # Recent page: sections "2024 FOMC Meetings" then rows like "January 30-31"
            for ym in re.finditer(r"(\d{4}) FOMC Meetings(.*?)(?=\d{4} FOMC Meetings|$)", text):
                year, body = int(ym.group(1)), ym.group(2)
                # Meetings are two-day ranges ("January 25-26", or "Apr/May 30-1" across months);
                # single dates on this page are minutes releases, so they're skipped.
                for mm in re.finditer(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:/(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))?\s+(\d{1,2})-(\d{1,2})\*?", body):
                    m1 = mm.group(2) or mm.group(1)              # the decision happens in the second month
                    month = next(v for k, v in MONTHS.items() if k.startswith(m1.lower()[:3]))
                    if first <= year <= last:
                        try: days.add(date(year, month, int(mm.group(4))))
                        except ValueError: pass
        else:
            year = int(re.search(r"fomchistorical(\d{4})", url).group(1))
            M = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            # "March 20-21 Meeting", "March 21 Meeting", or across months "Jan/Feb 31-1 Meeting"
            for mm in re.finditer(M + r"(?:/" + M + r")?\s+(\d{1,2})(?:-(\d{1,2}))?\s+(?:FOMC\s+)?Meeting\b", text):
                name = (mm.group(2) or mm.group(1)).lower()[:3]   # the decision happens in the second month
                month = next(v for k, v in MONTHS.items() if k.startswith(name))
                day = int(mm.group(4) or mm.group(3))
                try: days.add(date(year, month, day))
                except ValueError: pass
    return sorted(days)


def to_server_time(d, hour, minute):
    """New York local time -> HFM server time (New York + 7 hours)."""
    return datetime(d.year, d.month, d.day, hour, minute) + timedelta(hours=7)


if __name__ == "__main__":
    rows = [(to_server_time(d, 8, 30), "NFP") for d in nfp_dates()]
    fomc = fomc_dates()
    rows += [(to_server_time(d, 14, 0 if d.year >= 2013 else 15), "FOMC") for d in fomc]
    df = pd.DataFrame(rows, columns=["time", "event"]).sort_values("time")
    df.to_csv(OUT, index=False)
    per_year = df[df.event == "FOMC"].groupby(df.time.dt.year).size()
    print(f"NFP: {sum(df.event == 'NFP')} dates · FOMC: {len(fomc)} decision days")
    print("FOMC per year:", dict(per_year))
    print(df.tail(4).to_string(index=False))
