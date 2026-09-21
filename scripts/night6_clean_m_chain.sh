#!/bin/zsh
# Night 6: the M chain (measurement ceiling) plus the V artifacts night 5 could not produce,
# all on clean_dim64. SCORING ONLY -- no training, no 2025. Reads <=2024 data.
# Every path and arm is passed as an explicit flag, so a later default flip in src/ cannot
# perturb a run already in flight. Outputs go to results/v_chain_clean/ and
# results/measurement_ceiling_clean/ -- nothing canonical is overwritten.
# ONE-SHOT: unloads its own launchd job when done.
#
# Not included: paper_figures tuned_well + level_query_strata. Those two read module
# constants with no CLI override, so they run after the Step 2(b) default flips land.
set -u
cd ~/hitter-embedding

REPO_BRANCH=clean-refit
LOG=/tmp/hitter-night6.log
OUT=results/v_chain_clean
MC=results/measurement_ceiling_clean
BUILD=data/processed/phase_d5_clean
ARM=clean_clean_dim64
PY=.venv/bin/python
LABEL=com.nate.hitter-night6

PRED=results/model_v1/model_v1_predictions_$ARM.csv
FRAME=$OUT/model_evaluation/platoon_frame.csv
STATS=$OUT/hitter_stats.csv

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
for f in $PRED $FRAME $STATS $BUILD/manifest.json; do
  [[ -f $f ]] || { log "ABORT: missing prerequisite $f"; exit 1; }
done
mkdir -p "$OUT" "$MC" "$OUT/process_calibration" "$OUT/paper_figures"
log "=== night 6 start (M chain, clean_dim64, outputs -> $OUT and $MC) ==="

# --- A: per-seed 2024 queries. The long pole (~25 min x 5). Everything in D waits on it.
for SEED in 0 1 2 3 4; do
  SP=results/model_v1/model_v1_predictions_${ARM}_s${SEED}.csv
  if [[ -f $SP ]]; then
    log "A seed $SEED: already on disk, skipping"
  else
    step "A: 2024 query, seed $SEED" $PY -m src.model.query --arm $ARM --data-dir $BUILD \
      --hitter-stats $STATS --seeds $SEED --label ${ARM}_s${SEED} \
      --out-dir results/model_v1
  fi
done

# --- B: E.15 measurement ceiling on the observed platoon differential (clean 2024 frame).
step "B: platoon ceiling (2024)" $PY -m src.analysis.model_evaluation_platoon_ceiling \
  --frame $FRAME --eval-season 2024 --out-dir $OUT/model_evaluation

# --- C: F.5 claim-1 metric with handedness pooled away.
step "C: pooled process calibration" $PY -m src.analysis.process_calibration_pooled \
  --arm $ARM --seeds 0 1 2 3 4 --eval-season 2024 --data-dir $BUILD \
  --model-predictions $PRED --out-dir $OUT/process_calibration

# --- D: Phase M. Needs A+B+C. BUILD_STAMP was flipped to clean_dim64 before this launch,
#        so every row of the exhibit stamps the build it was actually scored on.
#        --gbm-full-cache is reused: the GBM differential baseline is model-independent.
step "D: measurement ceiling report" $PY -m src.analysis.measurement_ceiling_report \
  --eval-season 2024 \
  --platoon-frame $FRAME \
  --model-predictions $PRED \
  --seed-predictions "results/model_v1/model_v1_predictions_${ARM}_s{seed}.csv" \
  --seeds 0 1 2 3 4 \
  --manifest $BUILD/manifest.json \
  --pooled-scores $OUT/process_calibration/pooled_scores.csv \
  --ceiling-json $OUT/model_evaluation/ceiling.json \
  --gbm-full-cache results/measurement_ceiling/differential_gbm_full_predictions.csv \
  --out-dir $MC

# --- E: V.1/V.2/V.6/V.7 embedding artifacts. MUST precede G into the SAME out-dir, or G
#        falls back to a PCA basis computed from its own checkpoint instead of reusing E's.
step "E: embedding views (dimension usage, exposure loadings, coords)" \
  $PY -m src.analysis.model_visualization_embeddings --arm $ARM \
  --checkpoint-dir results/checkpoints --out-dir $OUT

# --- F: seed stability against the corrected null (shuffle before fitting the rotation).
step "F: seed stability, corrected null" $PY -m src.analysis.seed_stability_corrected_null \
  --arm $ARM --checkpoint-dir results/checkpoints \
  --out $OUT/seed_stability_corrected_null.json

# --- G: V.3/V.4 head fingerprints and the level query, then its held-out 2024 version.
step "G1: head fingerprints + level query" $PY -m src.analysis.model_visualization_heads \
  --checkpoint results/checkpoints/${ARM}_s0.pt --data-dir $BUILD \
  --hitter-stats $STATS --level-query-predictions $PRED --out-dir $OUT

step "G2: level query, held out on 2024 wOBA" $PY -m src.analysis.level_query_heldout \
  --level-query-csv $OUT/level_query.csv --supersedes $OUT/level_query.json \
  --out-dir $OUT/paper_figures

# --- H: D.5 level-bias attribution with gradient (a), which needs A's baseline predictions.
step "H: level attribution + gradients" $PY -m src.analysis.model_v1_level \
  --data-dir $BUILD --eval-season 2024 --skip-hbp \
  --predictions $PRED --gradient-b-arm $ARM --seeds 0 1 2 3 4 \
  --checkpoint-dir results/checkpoints \
  --out-dir results/model_v1 --out-name model_v1_level_attribution_clean_gradients.json
# --out-dir is results/model_v1 ON PURPOSE. model_v1_level.py:539 looks for the per-seed
# prediction CSVs INSIDE its own --out-dir, so night 5's run with --out-dir results/v_chain_clean
# skipped all five seeds and wrote `"gradient_b": {}` while still exiting 0.

log "=== night 6 done, $FAILED step(s) failed ==="
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null
rm -f ~/Library/LaunchAgents/$LABEL.plist
