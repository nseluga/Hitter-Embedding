#!/bin/bash
# Run the rest of Phase D.5's compute unattended: wait out the d10 sweep, then hand off to
# the step-5 scoring driver. Both halves already resume from their own state -- the sweep from
# results/phase_d/sweep_log.csv, the driver from the per-run diagnostics JSON -- so this can be
# killed and relaunched at any point without losing finished work.
#
# JOBS defaults to 1. The pool size is the open question: the driver's own header argues for 3,
# but the five baseline runs that actually ran at 3 came in at 3.4-5.2h each against a 2.3h
# single-run measurement, because three torch processes do not fit in 8GB and the machine spends
# its cycles paging. Start at 1, measure the fidelity re-read, then relaunch with a bigger JOBS
# if the clean single-run rate says a pool wins. Relaunching is free.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-d10}
export JOBS=${JOBS:-1}
export THREADS=${THREADS:-4}   # the 4 performance cores; the efficiency cores would set the pace

echo "=== d5_finish start $(date '+%F %H:%M') | JOBS=$JOBS THREADS=$THREADS ==="

# The sweep is launched separately and may already be done. Poll rather than `wait`, since this
# script is not its parent. The [p] keeps the pattern from matching this script's own pgrep.
while pgrep -f "src.model.swee[p] --stage $STAGE" > /dev/null 2>&1; do sleep 60; done
echo "=== sweep clear $(date '+%F %H:%M') ==="
awk -F, -v s="$STAGE" '$1==s && $4=="ok"' results/phase_d/sweep_log.csv | wc -l \
  | xargs echo "trained runs in ledger:"

STAGE="$STAGE" bash scripts/d5_step5_score.sh
echo "=== d5_finish done $(date '+%F %H:%M') ==="
