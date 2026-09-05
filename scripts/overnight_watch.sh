#!/bin/bash
# Arm: nohup caffeinate -s ./scripts/overnight_watch.sh > /tmp/overnight_watch.out 2>&1 &
# Waits until START (HH:MM today), launches overnight.sh, then a headless Claude watchdog.
set -u
START="${START:-20:00}"
cd "$(dirname "$0")/.."
now=$(date +%s); at=$(date -j -f "%H:%M" "$START" +%s)
[ "$at" -gt "$now" ] && sleep $(( at - now ))
echo "$(date) launching overnight.sh"
rm -rf /tmp/hitter-overnight
nohup ./scripts/overnight.sh > /tmp/overnight.out 2>&1 &
sleep 60
echo "$(date) starting watchdog"
claude -p "$(cat docs/overnight-watch-prompt.md)" \
  --model "${WATCH_MODEL:-opus}" \
  --allowedTools "Bash,Read,Edit,Write,Grep,Glob,Agent" \
  --max-turns 400 \
  > /tmp/overnight_claude.out 2>&1
echo "$(date) watchdog exited $?"
