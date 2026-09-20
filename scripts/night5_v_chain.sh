#!/bin/zsh
# Night 5: V chain (expanded_tuning.md step 9) on clean_dim64, the parameterizable pieces.
# Reads only <=2024 data; 2025 stays untouched. ALL outputs go to results/v_chain_clean/ --
# nothing canonical is overwritten. ONE-SHOT: unloads its own launchd job when done.
#
# Steps 5-9 need the 2024 clean predictions (step 5) -> platoon frame (step 6). Not included:
# paper_figures tuned_well + level_query_strata (need clean seed-stability / dimension-usage /
# exposure-loadings / level_query / surfaces artifacts that no script here produces).
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night5.log
OUT=results/v_chain_clean
BUILD=data/processed/phase_d5_clean
ARM=clean_clean_dim64
PY=.venv/bin/python
LABEL=com.nate.hitter-night5

log() { print -r -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
step() {
  local desc=$1; shift
  log "$desc: start -- $*"
  local try
  for try in 1 2; do  # one retry: a transient OOM/IO failure should not sink a multi-hour chain
    if "$@" >>"$LOG" 2>&1; then log "$desc: ok (try $try)"; return 0; fi
    log "$desc: failed (try $try)"
  done
  log "$desc: FAILED after 2 tries"; FAILED=$((FAILED + 1))
}

FAILED=0
branch=$(git branch --show-current)
if [[ "$branch" != "$REPO_BRANCH" ]]; then
  log "ABORT: on branch '$branch', expected '$REPO_BRANCH'"
  launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null
  exit 1
fi
mkdir -p "$OUT"
log "=== night 5 start (V chain, clean_dim64, outputs -> $OUT) ==="

step "hitter stats (train 2015-2023 build)" $PY -m src.analysis.model_visualization_stats \
  --manifest $BUILD/manifest.json --out-dir $OUT

step "norm check (5 seeds)" $PY -m src.analysis.embedding_norm_check \
  --stage clean --config clean_dim64 --seeds 0 1 2 3 4 \
  --stats-csv $OUT/hitter_stats.csv --out-dir $OUT

step "embedding structure" $PY -m src.analysis.embedding_structure \
  --arm $ARM --hitter-stats $OUT/hitter_stats.csv --out-dir $OUT/embedding_structure

step "level query gradient (b), held-out 2024" $PY -m src.analysis.model_v1_level \
  --data-dir $BUILD --eval-season 2024 --skip-hbp --gradient-b-arm $ARM \
  --seeds 0 1 2 3 4 --out-dir $OUT --out-name model_v1_level_attribution_clean.json

PRED=results/model_v1/model_v1_predictions_$ARM.csv
[[ -f $PRED ]] && log "2024 clean predictions already on disk, skipping query" || \
step "2024 query, clean arm (5 seeds)" $PY -m src.model.query --arm $ARM --data-dir $BUILD \
  --hitter-stats $OUT/hitter_stats.csv --seeds 0 1 2 3 4

step "eval -> platoon frame (2024)" $PY -m src.analysis.model_evaluation_eval --arm $ARM \
  --data-dir $BUILD --predictions $PRED --out-dir $OUT/model_evaluation
FRAME=$OUT/model_evaluation/platoon_frame.csv

step "platoon sign agreement" $PY -m src.analysis.platoon_sign_agreement \
  --frame $FRAME --out-dir $OUT/paper_figures

step "axis screen" env PYTHONPATH=. $PY results/embedding_structure/axis_screen.py \
  --arm $ARM --hitter-stats $OUT/hitter_stats.csv --out-dir $OUT/embedding_structure

for FIG in reliability_curve claim1_2025_strata calibration_2025; do
  step "paper figure $FIG (clean)" $PY -m src.analysis.paper_figures --only $FIG \
    --eval-dir results/model_evaluation_final_clean --out-dir $OUT/paper_figures
done

# the long one (~4 h at dim 64); last so everything above lands first
PV="$PY -m src.analysis.model_visualization_platoon --data-dir $BUILD --arm $ARM \
  --platoon-frame $FRAME --hitter-stats $OUT/hitter_stats.csv --n-dims 64"
step "platoon gradient (finite differences)" ${=PV} gradient --out-dir $OUT/platoon --resume
step "platoon analyse (spread gate)" ${=PV} analyse --raw $OUT/platoon/platoon_gradient_raw.csv \
  --out-dir $OUT/platoon

log "=== night 5 done, $FAILED step(s) failed ==="
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null
rm -f ~/Library/LaunchAgents/$LABEL.plist
