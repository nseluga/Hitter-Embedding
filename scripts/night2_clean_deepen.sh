#!/bin/zsh
# Night 2: deepen the clean-screen winners (clean_dropout, clean_dim64) plus clean_base
# to 5 seeds each. No build step -- phase_d5_clean already exists from night 1.
# Self-healing: retries a failed step a few times before giving up. Logs to $LOG.
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night2.log
BUDGET_HOURS=${BUDGET_HOURS:-9.5}
MAX_RETRIES=3
CONFIGS=(clean_base clean_dropout clean_dim64)

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

branch=$(git branch --show-current)
if [[ "$branch" != "$REPO_BRANCH" ]]; then
  log "ABORT: on branch '$branch', expected '$REPO_BRANCH' -- not running"
  exit 1
fi

if [[ ! -f data/processed/phase_d5_clean/manifest.json ]]; then
  log "ABORT: data/processed/phase_d5_clean/manifest.json missing -- night 1 build never ran"
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

log "=== night 2 start ==="

# sweep.py resumes from the ledger on its own, so a retry here is safe: a rerun just
# picks up whatever seeds are still pending instead of redoing finished ones.
retry "clean deepen" \
  .venv/bin/python -m src.model.sweep --stage clean --seeds 5 --hours "$BUDGET_HOURS" \
    --configs "${CONFIGS[@]}"

log "=== night 2 done ==="
