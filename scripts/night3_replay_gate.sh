#!/bin/zsh
# Night 3: the clean-screen winner's replay gate ONLY (overnight.sh stage 3), via TO=3.
# Deliberately stops BEFORE stage 4 -- the refit is irreversible and stage 6 spends the
# sealed 2025 test season, so those wait for a human look at the gate result first.
#
# Winner: clean_dim64 (mean reference 1.023268 over 5 seeds, beats clean_dropout 1.023574
# and clean_base 1.023782 -- results/model_v1/sweep_log.csv, night 2). GATE_CENTER/GATE_TOL
# are that arm's own 5-seed mean and sample SD (statistics.stdev), same convention as the
# 2026-09-04 registration for embedding_sgd_sgd_lr1 (docs/decision-log.md:1611).
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night3.log
OVERNIGHT_LOG=/tmp/hitter-night3-overnight
MAX_RETRIES=3

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

log "=== night 3 start (replay gate only, stage 3, stops before the refit) ==="

retry "replay gate" env \
  FROM=3 TO=3 \
  ARM=clean_clean_dim64 \
  DATA=data/processed/phase_d5_clean \
  TRAIN_FLAGS="--embedding-optimizer sgd --embedding-lr 1 --embedding-dim 64" \
  GATE_CENTER=1.023268 GATE_TOL=0.00013 \
  SCHEDULE=results/model_v1/replay_schedule_clean_dim64.json \
  LOG="$OVERNIGHT_LOG" \
  scripts/overnight.sh

log "=== night 3 done -- check $OVERNIGHT_LOG/gate_passed before running stage 4 (the refit) ==="
