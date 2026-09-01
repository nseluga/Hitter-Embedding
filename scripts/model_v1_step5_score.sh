#!/bin/bash
# Step 5 -- claim-1 for every d10 arm. Thirteen composition runs: eight ensembles (the
# comparable, shippable number) plus five per-seed baseline runs (the claim-1 noise floor).
#
# Why a script and not thirteen commands. One `predict` pass over ~1,074 hitters measured
# 8,344s on the shipped arm, so serially this is ~30h -- well past the plan's ~8-10h estimate,
# which only holds with a pool. Three concurrent processes on 4 performance cores at 2 torch
# threads each brings it to roughly 10-12h of wall clock. Four would oversubscribe: the
# efficiency cores run this workload at a fraction of the rate and would set the pace.
#
# Resume keys on the output file. The marker is the DIAGNOSTICS JSON, not the predictions
# CSV: the CSV lands right after `predict` and the composition block runs for another ~90s after
# that, so a run killed in between leaves a CSV that looks complete and a run with no composition
# in it. The JSON is written last, so its presence is the only honest "this finished".
#
# Every claim-1 run carries BOTH Step 3 fixes -- the take-count offsets and the unmeasured-
# category split. They landed, so they are the composition now; scoring one arm with them and
# another without would make the ablation contrast architecture plus composition.
#
# The one exception is the fourteenth run, `<stage>_baseline_fidelity`, which is step 4's
# league-fidelity re-read rather than a claim-1 score. It runs offsets-OFF because step 0's
# pre-rebuild read was offsets-off, and the delta against step 0 is supposed to be the REBUILD's
# contribution to level fidelity -- leaving the offsets on would hand the rebuild credit for
# them.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data/processed/phase_d5}
OUT_DIR=${OUT_DIR:-results/phase_d}
STAGE=${STAGE:-d10}
JOBS=${JOBS:-3}
LEDGER=${LEDGER:-results/phase_d/sweep_log.csv}
SEEDS_REQUIRED=${SEEDS_REQUIRED:-5}
THREADS=${THREADS:-2}
ARMS=${ARMS:-"baseline dim16 dim64 bilinear meanweight invfreq nospray block"}

# `block` trains on the no-block build and must be SCORED on it too -- its context tower has
# 35 input columns against the others' 46, so loading it against the main manifest fails on
# shape rather than silently mis-scoring. That is the only arm whose data dir differs.
data_dir_for() {
  case "$1" in
    block) echo "${DATA_DIR}_noblock" ;;
    *)     echo "$DATA_DIR" ;;
  esac
}

# An arm is scoreable the moment ITS OWN five seeds are trained, not when the whole sweep ends.
# The sweep's queue is arm-major, so waiting per arm pipelines scoring behind training instead of
# stacking ~14 h of composition after ~23 h of retraining. Training measured under one core of
# the four performance cores, so the overlap costs little. `wait` not `skip`: an arm whose seeds
# are still coming is not an arm to leave unscored.
wait_for_arm() {  # wait_for_arm <arm>
  local arm=$1 have
  while :; do
    have=$(awk -F, -v a="${arm#${STAGE}_}" -v s="$STAGE" \
             '$1==s && $2==a && $4=="ok"' "$LEDGER" | wc -l | tr -d ' ')
    [ "$have" -ge "$SEEDS_REQUIRED" ] && return 0
    if ! pgrep -f "src.model.sweep --stage $STAGE" > /dev/null; then
      echo "give up on $arm: $have/$SEEDS_REQUIRED seeds trained and the sweep is not running"
      return 1
    fi
    sleep 120
  done
}

run() {  # run <label> <arm> <data-dir> <extra...>
  local label=$1 arm=$2 data=$3; shift 3
  if [ -f "$OUT_DIR/d5_diagnostics_${label}.json" ]; then
    echo "skip $label (already finished)"; return 0
  fi
  wait_for_arm "$arm" || return 0
  echo "start $label"
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  .venv/bin/python -u -m src.model.query \
    --arm "$arm" --label "$label" --data-dir "$data" \
    --out-dir "$OUT_DIR" "$@" \
    > "$OUT_DIR/step5_${label}.log" 2>&1 \
    && echo "done  $label" || echo "FAIL  $label (see $OUT_DIR/step5_${label}.log)"
}
export -f run wait_for_arm
export OUT_DIR THREADS STAGE LEDGER SEEDS_REQUIRED

# The queue is written out first so the pool has no ordering logic in it. Step 4's fidelity
# re-read leads, then baseline's five per-seed runs, then the eight ensembles: everything at the
# front reads the baseline arm, which is what gates whether the other seven are worth scoring.
queue=$(mktemp)
trap 'rm -f "$queue"' EXIT
printf '%s\n' "${STAGE}_baseline_fidelity|${STAGE}_baseline|$DATA_DIR" >> "$queue"
for seed in 0 1 2 3 4; do
  printf '%s\n' \
    "${STAGE}_baseline_s${seed}|${STAGE}_baseline|$DATA_DIR|-t|--seeds|$seed" >> "$queue"
done
for arm in $ARMS; do
  printf '%s\n' "${STAGE}_${arm}|${STAGE}_${arm}|$(data_dir_for "$arm")|-t" >> "$queue"
done

echo "$(wc -l < "$queue") runs queued, ${JOBS} at a time, ${THREADS} threads each"
# `|` is the field separator because no label, arm, or path here contains one, and it survives
# the trip through xargs -I{} intact where whitespace would not.
# `-t` in a queue line stands in for --take-count-offsets, so the one offsets-off run is a
# missing token rather than a second code path.
xargs -P "$JOBS" -I{} bash -c \
  'IFS="|" read -ra a <<< "{}"; run "${a[@]/#-t/--take-count-offsets}"' < "$queue"

echo "--- queue drained; runs without a diagnostics JSON did not finish ---"
for arm in $ARMS; do
  [ -f "$OUT_DIR/d5_diagnostics_${STAGE}_${arm}.json" ] || echo "missing: ${STAGE}_${arm}"
done
