#!/bin/zsh
# The overnight pipeline (docs/handoff-overnight-run.md). SERIAL BY CONSTRUCTION: this box
# has 8 GB and last night two concurrent heavy jobs put both into uninterruptible sleep with
# 10 GB of swap (docs/run-notes-eval-chain-pass.md). Nothing here backgrounds anything.
#
#   ./scripts/overnight.sh          run every stage
#   DRY=1 ./scripts/overnight.sh    print the commands and evaluate no gate
#   FROM=4 ./scripts/overnight.sh   resume at stage 4 (stages are numbered below)
#   GATE_OVERRIDE=1 ...             let stage 4 start without stage 3's sentinel
#
# STAGE 4 IS THE ONE THAT CANNOT BE UNDONE: it builds the tensors and models that stage 6
# scores on 2025, and 2025 is the sealed test season. Stage 3's gate is what stands between
# the two, so it exits non-zero rather than warning. FROM=4 skips it, so the gate leaves a
# sentinel in $LOG and stage 4 refuses to run without one.
set -e
cd ~/hitter-embedding

P=.venv/bin/python
LOG=/tmp/hitter-overnight
FROM=${FROM:-1}

ARM=${ARM:-embedding_sgd_sgd_lr1}
DATA=${DATA:-data/processed/phase_d5}
STATS=${STATS:-results/model_visualization/hitter_stats.csv}

FINAL_ARM=${FINAL_ARM:-embedding_sgd_sgd_lr1_final}
FINAL_DATA=${FINAL_DATA:-data/processed/phase_d5_final}
FINAL_STATS=${FINAL_STATS:-results/model_visualization_final/hitter_stats.csv}
FINAL_SEASON=2025

# the arm's own recipe, shared by the stage-3 replay and the stage-4 refit; BUILD_FLAGS only
# applies to stage 4's tensor build (e.g. the clean winner's --career-pitchers).
TRAIN_FLAGS=${TRAIN_FLAGS:-"--embedding-optimizer sgd --embedding-lr 1"}
BUILD_FLAGS=${BUILD_FLAGS:-""}
# The 2025 chain writes to ITS OWN directories. Every module below writes arm-less, season-less
# filenames (calibration.csv, pooled_scores.csv, ...) straight into the committed 2024 results,
# so pointing stage 6 at the default out-dirs would replace the 2024 exhibit with 2025 numbers
# in place, on the same night stage 1 rebuilt it, recoverable only from git.
FINAL_EVAL_OUT=results/model_evaluation_final
FINAL_PROC_OUT=results/process_calibration_final

# the replay check reproduces the frozen-split arm from a fixed budget instead of from early
# stopping. Its `reference` has to land inside the five seeds' own spread or the replay is
# not reproducing them: 1.02386 is their mean, 0.00009 is one SD of it (2026-09-04 log).
if [[ "$ARM" != embedding_sgd_sgd_lr1 && -z "${GATE_CENTER:-}" ]]; then
  echo "ARM=$ARM but GATE_CENTER is the 2024 embedding_sgd_sgd_lr1 number; re-derive it from" \
       "the arm's five seeds (mean) and pass GATE_CENTER= and GATE_TOL= (one SD)" >&2
  exit 1
fi
GATE_CENTER=${GATE_CENTER:-1.02386}
GATE_TOL=${GATE_TOL:-0.00009}

# budget and cuts are DERIVED (stage 3), never typed: src/model/replay_schedule.py reads them
# out of the five runs' own logs, which is the only place they were ever recorded.
SCHEDULE=results/model_v1/replay_schedule.json

mkdir -p $LOG

# fd 3 is the console. Every `run` redirects its own stdout+stderr into a stage log, so the
# progress line has to escape that redirection or the transcript ends up inside the log it is
# announcing -- and under DRY the echoed command is the whole output.
exec 3>&1

say() { echo "$@" >&3; }

run() {
  say "--- $(date +%H:%M:%S) $*"
  if [[ -n "$DRY" ]]; then return 0; fi
  caffeinate -i "$@"
}

stage() {  # stage <n> <name>; returns 1 when the stage is being skipped by FROM
  if (( $1 < FROM )); then say "=== $(date +%H:%M) stage $1 ($2) SKIPPED (FROM=$FROM)"; return 1; fi
  say "=== $(date +%H:%M) stage $1: $2"
  return 0
}

# --- 1. the 2024 chain, cold-start prior ON ------------------------------------------
# Every query below takes the prior by default now; --hitter-stats must name the panel built
# on the SAME tensor build as --data-dir, or the low stratum is defined on the wrong era.
if stage 1 "2024 chain, prior on"; then
  Q="$P -m src.model.query --arm $ARM --data-dir $DATA --hitter-stats $STATS"
  run ${=Q} --seeds 0 1 2 3 4 > $LOG/01_query_ensemble.log 2>&1
  for N in 0 1 2 3 4; do
    run ${=Q} --seeds $N --label ${ARM}_s$N > $LOG/01_query_s$N.log 2>&1
  done

  for M in model_evaluation_eval model_evaluation_swing model_evaluation_price_draw \
           model_evaluation_take_mass model_evaluation_min_pa_sweep model_evaluation_bip_value \
           model_evaluation_probe_coverage model_evaluation_platoon_ceiling \
           process_calibration_heads process_calibration_process process_calibration_pooled \
           measurement_ceiling_report model_visualization_heads model_visualization_disagreement; do
    run $P -m src.analysis.$M >> $LOG/01_chain.log 2>&1
  done

  run $P -m src.analysis.model_v1_ablation_report \
      --predictions results/model_v1/model_v1_predictions_${ARM}.csv \
      --label min_pa_sweep_${ARM} --out-dir results/model_evaluation \
      --manifest $DATA/manifest.json >> $LOG/01_post.log 2>&1
  for NB in 06_claim_evaluation 08_measurement; do
    run .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 notebooks/$NB.ipynb >> $LOG/01_post.log 2>&1
  done
  run $P -m src.analysis.cold_start_prior_eval >> $LOG/01_post.log 2>&1

  # post2: these three read what the notebooks and the M report just rewrote
  run $P -m src.analysis.model_evaluation_platoon_ceiling >> $LOG/01_post2.log 2>&1
  run $P -m src.analysis.measurement_ceiling_report >> $LOG/01_post2.log 2>&1
  run .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 notebooks/08_measurement.ipynb >> $LOG/01_post2.log 2>&1
fi

# --- 2. no-decay ablation, one seed, plus its norm check -----------------------------
# Isolates the mechanism behind the SGD arm's +0.438 exposure-vs-norm slope: with the decay
# off the table and everything else identical, a slope that survives is the optimizer's.
if stage 2 "embedding_sgd_nodecay s0 + norm check"; then
  run $P -m src.model.sweep --stage embedding_sgd_nodecay --seeds 1 > $LOG/02_nodecay.log 2>&1
  run $P -m src.analysis.embedding_norm_check --stage embedding_sgd_nodecay \
      --config sgd_lr1_nodecay --seeds 0 --stats-csv $STATS \
      --out-dir results/model_visualization > $LOG/02_norm_check.log 2>&1
fi

# --- 3. replay check on the FROZEN split, and the gate -------------------------------
# One seed, same frozen split, same data, but stopped by the budget instead of by validation.
# Validation is off as a CONTROL signal; `reference` is still scored once at the end, because
# a check whose number is never printed cannot gate anything.
if stage 3 "replay check + gate"; then
  run $P -m src.model.replay_schedule --arm $ARM --out $SCHEDULE > $LOG/03_schedule.log 2>&1
  if [[ -n "$DRY" ]]; then
    BUDGET=16537; CUTS="6300 8662 11025 13633"
  else
    BUDGET=$($P -c "import json;print(json.load(open('$SCHEDULE'))['step_budget'])")
    CUTS=$($P -c "import json;print(' '.join(map(str,json.load(open('$SCHEDULE'))['lr_cut_steps'])))")
  fi
  say "    budget $BUDGET, cuts $CUTS"

  run $P -m src.model.train --split --data-dir $DATA \
      ${=TRAIN_FLAGS} --seed 0 \
      --step-budget $BUDGET --lr-cut-steps ${=CUTS} \
      --run-name replay_check > $LOG/03_replay.log 2>&1

  if [[ -z "$DRY" ]]; then
    REF=$(grep -o 'reference [0-9.]*' $LOG/03_replay.log | tail -1 | awk '{print $2}')
    say "    replay reference: $REF (gate $GATE_CENTER +/- $GATE_TOL)"
    $P -c "
import sys
ref, center, tol = float('$REF'), $GATE_CENTER, $GATE_TOL
print(f'gate: |{ref:.5f} - {center:.5f}| = {abs(ref-center):.5f} vs {tol:.5f}')
sys.exit(0 if abs(ref - center) <= tol else 1)
" || { echo "GATE FAILED: the replay does not reproduce the frozen-split arm. STOPPING BEFORE THE REFIT."; exit 1; }
    say "    GATE PASSED"
    echo "$REF $(date)" > $LOG/gate_passed
  fi
fi

# --- 4. the refit: new tensor build through 2024, then five seeds ---------------------
# The build is NOT optional. The hitter vocabulary, the quality-bin edges and the context
# standardisation are all fitted on train seasons only, so training 2015-2024 against the
# phase_d5 tensors would route every 2024 debutant to the padding row and decode contact on
# bins fitted without 2024. Same pitch table, same n_bins; only the fitted quantities move.
if stage 4 "refit build + 5 seeds (2015-2024)"; then
  # FROM=4 skips stage 3, so the gate leaves a sentinel behind and stage 4 refuses to start
  # without one. GATE_OVERRIDE=1 is the deliberate way past it.
  if [[ -z "$DRY" && -z "$GATE_OVERRIDE" && ! -f $LOG/gate_passed ]]; then
    echo "no $LOG/gate_passed: stage 3's gate has not passed in this log dir. Run stage 3, or"
    echo "set GATE_OVERRIDE=1 if it passed in an earlier one. Refusing to start the refit."
    exit 1
  fi
  run $P -m src.data.model_dataset --out-dir $FINAL_DATA --n-bins 24 \
      --train-seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
      ${=BUILD_FLAGS} > $LOG/04_build.log 2>&1
  if [[ -z "$DRY" ]]; then
    BUDGET=$($P -c "import json;print(json.load(open('$SCHEDULE'))['step_budget'])")
    CUTS=$($P -c "import json;print(' '.join(map(str,json.load(open('$SCHEDULE'))['lr_cut_steps'])))")
  else
    BUDGET=16537; CUTS="6300 8662 11025 13633"
  fi
  for N in 0 1 2 3 4; do
    run $P -m src.model.train --split --data-dir $FINAL_DATA \
        --split-config src/config/split_config_final_run.json \
        ${=TRAIN_FLAGS} --seed $N \
        --step-budget $BUDGET --lr-cut-steps ${=CUTS} \
        --run-name $FINAL_ARM > $LOG/04_refit_s$N.log 2>&1
  done
  # the prior's population has to be rebuilt too: it is defined by the BUILD's train seasons
  run $P -m src.analysis.model_visualization_stats --manifest $FINAL_DATA/manifest.json \
      --out-dir results/model_visualization_final > $LOG/04_stats.log 2>&1
fi

# --- 5. the 2025 queries. THIS IS WHERE THE TEST SEASON IS SPENT ---------------------
if stage 5 "2025 ensemble + per-seed queries"; then
  Q="$P -m src.model.query --arm $FINAL_ARM --data-dir $FINAL_DATA --hitter-stats $FINAL_STATS
     --eval-season $FINAL_SEASON --final-run"
  run ${=Q} --seeds 0 1 2 3 4 > $LOG/05_query_ensemble.log 2>&1
  for N in 0 1 2 3 4; do
    run ${=Q} --seeds $N --label ${FINAL_ARM}_s$N > $LOG/05_query_s$N.log 2>&1
  done
fi

# --- 6. the 2025 chain ---------------------------------------------------------------
# Only the modules that take BOTH --eval-season and --final-run are here. The three that do
# not (model_evaluation_platoon_ceiling, model_visualization_heads, model_visualization_
# disagreement) cannot be pointed at a test season without editing them, and editing analysis
# code on the night 2025 is spent is not a trade worth making -- they stay 2024 artifacts.
# measurement_ceiling_report takes both flags but is out too, for the same reason one level up:
# its --platoon-frame and --ceiling-json inputs are written by model_evaluation_platoon_ceiling,
# which cannot run on 2025, so a "2025" ceiling report would be a 2024 ceiling under a 2025 name.
if stage 6 "2025 chain"; then
  mkdir -p $FINAL_EVAL_OUT $FINAL_PROC_OUT
  # eval runs FIRST: bip_value and price_draw both read the report it writes.
  # resample (arm-independent, refit on the final train window) writes the audit price_draw
  # reads; take_mass reads price_draw's summary -- so: eval, swing, resample, price_draw, take_mass.
  for M in model_evaluation_eval model_evaluation_swing; do
    run $P -m src.analysis.$M --arm $FINAL_ARM --data-dir $FINAL_DATA \
        --eval-season $FINAL_SEASON --final-run --out-dir $FINAL_EVAL_OUT >> $LOG/06_chain.log 2>&1
  done
  for M in model_evaluation_resample model_evaluation_price_draw; do
    run $P -m src.analysis.$M --data-dir $FINAL_DATA \
        --eval-season $FINAL_SEASON --final-run --out-dir $FINAL_EVAL_OUT >> $LOG/06_chain.log 2>&1
  done
  # probe_coverage (E.14) is out: it asserts against --final-run by design (retrospective only).
  run $P -m src.analysis.model_evaluation_take_mass --arm $FINAL_ARM --data-dir $FINAL_DATA \
      --eval-season $FINAL_SEASON --final-run --out-dir $FINAL_EVAL_OUT >> $LOG/06_chain.log 2>&1
  # bip_value (E.13) is out: it hard-pins the 2024 E.3 constants (E3_VALUE_*_BIP) and asserts
  # the report matches them, so it cannot run on a 2025 report without a code change (Tier 2).
  for M in process_calibration_heads process_calibration_process; do
    run $P -m src.analysis.$M --arm $FINAL_ARM --data-dir $FINAL_DATA \
        --eval-season $FINAL_SEASON --final-run --out-dir $FINAL_PROC_OUT >> $LOG/06_chain.log 2>&1
  done
  # pooled's --model-predictions defaults to the 2024 ARM by name, not to --arm
  run $P -m src.analysis.process_calibration_pooled --arm $FINAL_ARM --data-dir $FINAL_DATA \
      --eval-season $FINAL_SEASON --final-run --out-dir $FINAL_PROC_OUT \
      --model-predictions results/model_v1/model_v1_predictions_${FINAL_ARM}.csv \
      >> $LOG/06_chain.log 2>&1
  run $P -m src.analysis.model_evaluation_min_pa_sweep \
      --predictions results/model_v1/model_v1_predictions_${FINAL_ARM}.csv \
      --label min_pa_sweep_${FINAL_ARM} --out-dir $FINAL_EVAL_OUT \
      --eval-season $FINAL_SEASON --final-run >> $LOG/06_chain.log 2>&1
  run $P -m src.analysis.model_v1_ablation_report \
      --predictions results/model_v1/model_v1_predictions_${FINAL_ARM}.csv \
      --label min_pa_sweep_${FINAL_ARM} --out-dir $FINAL_EVAL_OUT \
      --manifest $FINAL_DATA/manifest.json --eval-season $FINAL_SEASON --final-run \
      >> $LOG/06_chain.log 2>&1
fi

say "=== $(date +%H:%M) overnight done; logs in $LOG"
