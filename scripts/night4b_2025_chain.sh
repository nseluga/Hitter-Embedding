#!/bin/zsh
# Night 4b: overnight.sh stages 5-6 on the clean-screen winner -- the 2025 queries + chain.
# THIS IS WHERE THE SEALED 2025 SEASON IS SPENT. Run only after night4_refit_2025.sh (stage 4)
# has completed and results/checkpoints/clean_dim64_final_s{0..4}.pt exist.
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night4b.log
OVERNIGHT_LOG=/tmp/hitter-night4-overnight
MAX_RETRIES=3

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

branch=$(git branch --show-current)
if [[ "$branch" != "$REPO_BRANCH" ]]; then
  log "ABORT: on branch '$branch', expected '$REPO_BRANCH' -- not running"
  exit 1
fi

if [[ ! -f results/checkpoints/clean_dim64_final_s4.pt ]]; then
  log "ABORT: results/checkpoints/clean_dim64_final_s4.pt missing -- stage 4 (the refit) never finished"
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

log "=== night 4b start (2025 queries + chain -- spends 2025) ==="

retry "2025 chain" env \
  FROM=5 \
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

log "=== night 4b done ==="
