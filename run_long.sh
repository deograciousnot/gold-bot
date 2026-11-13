#!/bin/bash
# Server job: download 20 years of 1-minute gold data (slow: Dukascopy throttles),
# then run the precise 15-minute backtests. Safe to restart: finished days are cached.
#
# Start:  screen -dmS gold ~/gold-bot/run_long.sh
# Watch:  screen -r gold    (leave with Ctrl+A then D)
cd ~/gold-bot
mkdir -p logs
PY=~/.venvs/gold-bot/bin/python

for i in $(seq 1 30); do
  $PY download_dukascopy.py >> logs/download_minute.log 2>&1
  tail -5 logs/download_minute.log | grep -q "failed" || break
  sleep 60
done
echo "MINUTE_DONE $(date)" >> logs/status.txt

$PY backtest_trend.py M15_long > logs/result_trend_M15_long.txt 2>&1
$PY backtest_breakout.py M15_long > logs/result_breakout_M15_long.txt 2>&1
# SMC exam: 2005-2023 only (2024+ was used to develop it)
$PY backtest_smc.py M15_long 2024-01-01 > logs/result_smc_exam.txt 2>&1
echo "BACKTESTS_DONE $(date)" >> logs/status.txt
