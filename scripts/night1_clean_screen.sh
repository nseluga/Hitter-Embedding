#!/bin/zsh
# Night 1: build phase_d5_clean, then run the clean screen (1 seed x 6 arms) for up to
# BUDGET_HOURS. Self-healing: retries a failed step a few times before giving up so it
# survives a transient crash unattended. Logs everything to $LOG for the morning /brief.
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night1.log
BUDGET_HOURS=${BUDGET_HOURS:-9.5}   # leaves slack inside the 10h window for the build step
MAX_RETRIES=3

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

branch=$(git branch --show-current)
if [[ "$branch" != "$REPO_BRANCH" ]]; then
  log "ABORT: on branch '$branch', expected '$REPO_BRANCH' -- not running"
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

start=$SECONDS
log "=== night 1 start ==="

if [[ -f data/processed/phase_d5_clean/manifest.json ]]; then
  log "phase_d5_clean already built, skipping build step"
else
  retry "build phase_d5_clean" \
    .venv/bin/python -m src.data.model_dataset --out-dir data/processed/phase_d5_clean \
      --n-bins 24 --career-pitchers
  if [[ ! -f data/processed/phase_d5_clean/manifest.json ]]; then
    log "FATAL: build never produced a manifest, not starting the screen"
    exit 1
  fi
fi

elapsed_h=$(( (SECONDS - start) / 3600.0 ))
remaining_h=$(( BUDGET_HOURS - elapsed_h ))
if (( remaining_h < 0.5 )); then
  remaining_h=0.5
fi
log "build took ${elapsed_h}h, giving the screen ${remaining_h}h"

# sweep.py already resumes from the ledger on its own, so a retry here is safe: a rerun
# just picks up whatever runs are still pending instead of redoing finished ones.
retry "clean screen" \
  .venv/bin/python -m src.model.sweep --stage clean --seeds 1 --hours "$remaining_h"

log "=== night 1 done ==="
