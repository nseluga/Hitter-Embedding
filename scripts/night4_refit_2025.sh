#!/bin/zsh
# Night 4: overnight.sh stage 4 ONLY on the clean-screen winner -- the irreversible refit
# (fresh 2015-2024 tensor build + 5 seeds). Split from stages 5-6 (night4b) because train.py
# has no --resume and overnight.sh only cuts at stage boundaries: a hard kill mid-stage
# wastes the whole stage, so a 10h/night budget has to land ON a boundary, not mid-run.
# Stage 4 does NOT spend 2025 -- that's night4b. Approved 2026-09-15 after night 3's gate
# passed cleanly (reference 1.02328 vs gate 1.023268 +/- 0.00013, miss 0.00001).
#
# All FINAL_* out-dirs are repointed to *_clean variants so this does NOT overwrite the
# canonical arm's already-spent 2025 exhibit (results/model_evaluation_final etc, dated
# 2026-09-05) -- caught and fixed in overnight.sh before scheduling this (FINAL_EVAL_OUT,
# FINAL_PROC_OUT, FINAL_VIZ_OUT are now overridable, same pattern as night 3's SCHEDULE fix).
#
# GATE_OVERRIDE=1: stage 3 already passed (night 3), but in a different $LOG dir, so its
# sentinel isn't where FROM=4 looks for one.
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night4.log
OVERNIGHT_LOG=/tmp/hitter-night4-overnight
MAX_RETRIES=3

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

branch=$(git branch --show-current)
if [[ "$branch" != "$REPO_BRANCH" ]]; then
  log "ABORT: on branch '$branch', expected '$REPO_BRANCH' -- not running"
  exit 1
fi

if [[ ! -f /tmp/hitter-night3-overnight/gate_passed ]]; then
  log "ABORT: /tmp/hitter-night3-overnight/gate_passed missing -- night 3's gate never passed"
  exit 1
fi

retry() {
  local desc=$1; shift
  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    log "$desc: attempt $attempt/$MAX_RETRIES -- $*"
    if "$@" >>"$LOG" 2>&1; then
      log "$desc: succeeded"
      return 0
    fi
    log "$desc: failed (exit $?)"
    attempt=$((attempt + 1))
    sleep 30
  done
  log "$desc: gave up after $MAX_RETRIES attempts"
  return 1
}

log "=== night 4 start (refit only, stage 4) ==="

retry "refit" env \
  FROM=4 TO=4 \
  ARM=clean_clean_dim64 \
  DATA=data/processed/phase_d5_clean \
  TRAIN_FLAGS="--embedding-optimizer sgd --embedding-lr 1 --embedding-dim 64" \
  BUILD_FLAGS="--career-pitchers" \
  GATE_CENTER=1.023268 GATE_TOL=0.00013 \
  SCHEDULE=results/model_v1/replay_schedule_clean_dim64.json \
  FINAL_ARM=clean_dim64_final \
  FINAL_DATA=data/processed/phase_d5_clean_final \
  FINAL_STATS=results/model_visualization_final_clean/hitter_stats.csv \
  FINAL_EVAL_OUT=results/model_evaluation_final_clean \
  FINAL_PROC_OUT=results/process_calibration_final_clean \
  FINAL_VIZ_OUT=results/model_visualization_final_clean \
  GATE_OVERRIDE=1 \
  LOG="$OVERNIGHT_LOG" \
  scripts/overnight.sh

log "=== night 4 done ==="
