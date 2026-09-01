"""
Phase D.5 — put the model's wOBA predictions on the Phase C ladder and score them.

This is the first point in the project where a Phase D number is comparable to anything.
Everything before it — 39 training runs, an ablation table, a promote-only screen — is
measured in held-out log loss per scored row, which says which model predicts PITCHES
better and nothing at all about whether it projects HITTERS better than empirical Bayes.

Two rules this module exists to enforce, both of which fail silently otherwise:

1. THE LADDER IS RE-SCORED, NOT REMEMBERED. Every Phase C rung is recomputed on the same
   eval frame in the same process, so Phase D is never compared against numbers produced
   under a different filter, weighting, or metric vintage. `baseline_ladder_report` owns those baselines
   and is imported rather than reimplemented.

2. THE PAIRED COMPARISON IS THE CLAIM, NOT THE TWO ABSOLUTE NUMBERS. Both models are
   scored against the same noisy answer key, so only the paired difference with batter
   clustering can resolve them (2026-07-29). `_paired_setup` additionally refuses to run
   at all unless both models scored exactly the same groups, which is what makes the
   coverage contract in the D.5 spec load-bearing rather than tidy.

The gate the numbers are read against (2026-07-30, 2026-08-01): RMSE against C.3-full,
ordering against BOTH C.2 and C.3-full on the denominator-weighted rank correlation, each
by a paired bootstrap whose 95% interval excludes zero. A low-stratum ordering null is the
expected outcome on this frame and is not evidence against Phase D.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import baseline_ladder_report, claim1_eval as evaluation, baseline_ladder_gbm as gbm

OUT_DIR = Path("results/model_v1")
DEFAULT_PREDICTIONS = OUT_DIR / "model_v1_predictions_presplit_baseline.csv"
MODEL_NAME = "model_v1"
# the two rungs the pre-registered gate names; everything else on the ladder is context
GATE_OPPONENTS = ("gbm_full", "eb_bivariate")


def load_predictions(path):
    """The D.5 prediction table, with the claim-1 contract checked before it is scored."""
    frame = pd.read_csv(path)
    missing = sorted(set(evaluation.KEY + ["pred_woba"]) - set(frame.columns))
    assert not missing, f"D.5 predictions are missing {missing}"
    assert not frame.duplicated(evaluation.KEY).any(), "duplicate (batter, season, hand)"
    return frame[evaluation.KEY + ["pred_woba"]]


def compare(frames, name=MODEL_NAME, seed=0):
    """
    Paired RMSE and paired weighted-rank differences against each gate opponent.
    Sign conventions differ between the two and are recorded per row rather than assumed:
    NEGATIVE rmse_difference favours `name`, POSITIVE rank_difference favours `name`.
    """
    tables = []
    for opponent in GATE_OPPONENTS:
        # both helpers already return one row per stratum and do their own slicing, so
        # this must not slice as well — passing a pre-sliced frame would resample inside
        # a stratum and silently narrow every interval
        rmse = evaluation.paired_rmse_difference(frames[name], frames[opponent], seed=seed)
        rank = evaluation.paired_rank_difference(frames[name], frames[opponent], seed=seed)
        merged = rmse.merge(rank, on=["stratum", "n_hitters", "n_batters"],
                            suffixes=("_rmse", "_rank"))
        merged.insert(0, "opponent", opponent)
        # NEGATIVE rmse_difference favours Phase D; POSITIVE rank_difference does. The
        # gate is the INTERVAL excluding zero, not the point estimate's sign.
        merged["rmse_favours_model_v1"] = merged["ci_high_rmse"] < 0
        merged["rank_favours_model_v1"] = merged["ci_low_rank"] > 0
        tables.append(merged)
    return pd.concat(tables, ignore_index=True)


def second_target_table(frames):
    """
    Every rung and arm scored against BOTH answer keys on one eval frame (D5-R8).

    The point is the GAP between the two columns, not either alone. Error visible against xwOBA
    too is "wrong about batted-ball quality"; error that appears only against realized wOBA is
    "could not have known" -- fielder placement and sequencing, which the composition has no
    channel to express. A model can lose on the primary target and be fine on the second, and
    that is a different diagnosis from losing on both.

    Realized wOBA stays PRIMARY: `TARGETS` is ordered so the primary is scored first, and no
    verdict anywhere in this module reads the xwOBA rows.
    """
    tables = []
    for name, frame in frames.items():
        for target in evaluation.TARGETS:
            table = evaluation.score(frame, target=target)
            table.insert(0, "model", name)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def debiased_diagnostic(frames, name, seed=0):
    """
    Re-run the paired RMSE comparison with the PA-weighted mean excess removed from `name`.

    Reported because it changes how the null READS: debiased, Phase D and C.3-full are tied
    rather than second-and-first, so the significant loss lives in the level and in the high
    stratum, not in the ordering. D5-R15 measured it and no reporting line carried it.

    A DIAGNOSTIC, never a fix. The 2026-08-08 knob entry forbids subtracting a computed bias
    off the level, and the excess is monotone in exposure (+0.01179 / +0.01785 / +0.01941), so
    there is no single level to remove. The constant shift below is deliberately the thing the
    real bias is not, which is why its result is a reading and not a correction.

    Returns (mean_excess, table).
    """
    frame = frames[name].copy()
    weights = frame["denominator"].to_numpy(dtype=float)
    excess = float(np.average(frame["pred_woba"] - frame["woba"], weights=weights))
    frame["pred_woba"] = frame["pred_woba"] - excess

    table = evaluation.paired_rmse_difference(frame, frames["gbm_full"], seed=seed)
    table.insert(0, "opponent", "gbm_full")
    return excess, table


def power_restatement(comparisons, opponent="gbm_full", power=0.80, alpha=0.05):
    """
    D5-R18(4): restate a rank-gap null as what it is -- underpowered -- with the factor.

    "Null" and "underpowered" are different claims about the same interval, and only one of them
    is what this frame can support. The standard error comes from the bootstrap interval already
    computed rather than from a formula, so it inherits the batter clustering; the multiplier is
    the ratio of the z needed for `power` to the z observed, squared, which is the usual
    sample-size scaling because the SE falls as 1/sqrt(n).

    Reported for every stratum, including the ones that already resolve. A stratum whose interval
    excludes zero gets a multiplier below 1, which is the honest reading of "already sufficient"
    and is not evidence for anything on its own.
    """
    from scipy.stats import norm

    z_needed = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    rows = comparisons[comparisons["opponent"] == opponent]
    out = []
    for _, row in rows.iterrows():
        se = (row["ci_high_rank"] - row["ci_low_rank"]) / 2 / norm.ppf(1 - alpha / 2)
        z = row["rank_difference"] / se if se else float("nan")
        out.append({"stratum": row["stratum"], "n_batters": int(row["n_batters"]),
                    "rank_difference": row["rank_difference"], "se_rank": se, "z": z,
                    "batters_multiplier": (z_needed / z) ** 2 if z else float("nan"),
                    "batters_needed": int(round(row["n_batters"] * (z_needed / z) ** 2)) if z
                    else None})
    return pd.DataFrame(out)


def trained_row_spread(frame, vocabulary):
    """
    D5-R18(1): the predicted-spread diagnostic with cold-start rows separated out, not pooled.

    A hitter outside the training vocabulary is routed to the reserved row, so every one of them
    predicts from the SAME embedding and their spread measures context variation alone. Pooling
    them into the low-exposure stratum drags that stratum's spread toward a constant and produces
    the shrinkage story the pooled read tells. On trained rows the sign reverses: low-exposure
    hitters spread WIDER than regulars, which is anti-shrinkage and the sharper finding.
    """
    frame = frame.copy()
    frame["trained"] = frame["batter"].isin(vocabulary)
    rows = []
    for stratum in evaluation.STRATUM_NAMES:
        group = frame[frame["stratum"] == stratum]
        if not len(group):
            continue
        trained, cold = group[group["trained"]], group[~group["trained"]]
        rows.append({"stratum": stratum, "n": len(group),
                     "sd_pooled": group["pred_woba"].std(),
                     "n_cold_start": len(cold),
                     "cold_start_share": len(cold) / len(group),
                     "sd_cold_start": cold["pred_woba"].std() if len(cold) > 1 else float("nan"),
                     "distinct_cold_values": int(cold["pred_woba"].nunique()),
                     "n_trained": len(trained),
                     "sd_trained": trained["pred_woba"].std() if len(trained) > 1
                     else float("nan")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Score Phase D.5 against the Phase C ladder.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--label", default=MODEL_NAME)
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--manifest", default="data/processed/phase_d/manifest.json",
                        help="the build the scored arm trained on; supplies the hitter "
                             "vocabulary D5-R18(1) needs to separate cold-start rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-run", action="store_true",
                        help="the ONLY way the frozen test season is ever scored")
    args = parser.parse_args()

    evaluation.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = baseline_ladder_report.load_targets(args.eval_targets)
    params, _, _ = baseline_ladder_report.fit_and_describe(pa_df, args.eval_season, args.seed)
    process_seasons = gbm.season_process(
        pd.read_parquet(args.pitch_events, columns=baseline_ladder_report.PITCH_COLUMNS))

    predictions, _ = baseline_ladder_report.build_predictions(pa_df, args.eval_season, params,
                                                process_seasons, seed=args.seed)
    predictions[args.label] = load_predictions(args.predictions)
    frames, scored, _ = baseline_ladder_report.score_all(pa_df, predictions, args.eval_season)

    comparisons = compare(frames, name=args.label, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / f"model_v1_claim1_scores_{args.label}.csv", index=False)
    comparisons.to_csv(out_dir / f"model_v1_claim1_paired_{args.label}.csv", index=False)

    print("claim-1 scores (all models, one eval frame)")
    print(scored.to_string(index=False, float_format="%.4f"))

    # D5-R8: the second target and the achievable floor. The floor compares the two ANSWER KEYS
    # to each other, so it is a property of the eval frame and not of any model -- read off one
    # frame, and identical on every other by construction.
    both = second_target_table(frames)
    floor = evaluation.achievable_floor(frames[args.label])
    both.to_csv(out_dir / f"model_v1_both_targets_{args.label}.csv", index=False)
    floor.to_csv(out_dir / f"model_v1_achievable_floor_{args.label}.csv", index=False)
    print("\nscored against both answer keys (realized wOBA is primary; no gate reads xwoba)")
    print(both.to_string(index=False, float_format="%.4f"))
    print("\nachievable floor -- RMSE between the two answer keys, a reference line, "
          "never subtracted")
    print(floor.to_string(index=False, float_format="%.5f"))
    print(f"\npaired comparisons for {args.label}")
    print("  rmse_difference NEGATIVE favours Phase D; rank_difference POSITIVE favours it")
    print(comparisons.to_string(index=False, float_format="%.4f"))

    # D5-R18(3): both gates read exactly one hard-coded stratum row, and reading exactly one is
    # the defect -- the ordering claim is specifically about the low-exposure stratum, which the
    # 2026-07-30 entry states as "in the stratum claimed". Every stratum now gets a verdict, and
    # the DECISIVE one is pre-registered as `low`. Swapping which single stratum is hard-coded
    # would have left a report that still cannot say where a claim holds and where it fails.
    DECISIVE_STRATUM = "low"
    verdicts = {}
    for stratum in sorted(comparisons["stratum"].unique()):
        gate = comparisons[comparisons["stratum"] == stratum]
        against_gbm = gate[gate["opponent"] == "gbm_full"]["rmse_favours_model_v1"]
        verdicts[stratum] = {
            "rmse_gate_vs_gbm_full": bool(against_gbm.all()) if len(against_gbm) else None,
            "ordering_gate_vs_both": bool(gate["rank_favours_model_v1"].all()),
            "n_comparisons": int(len(gate)),
        }
    assert DECISIVE_STRATUM in verdicts, \
        f"the decisive stratum {DECISIVE_STRATUM!r} produced no comparison rows"

    print(f"\nper-stratum gate verdicts (decisive stratum: {DECISIVE_STRATUM})")
    for stratum, verdict in verdicts.items():
        mark = "  <- decisive" if stratum == DECISIVE_STRATUM else ""
        print(f"  {stratum:>6s}  RMSE vs C.3-full: "
              f"{'PASS' if verdict['rmse_gate_vs_gbm_full'] else 'not met':>7s}   "
              f"ordering vs BOTH: "
              f"{'PASS' if verdict['ordering_gate_vs_both'] else 'not met':>7s}{mark}")

    # D5-R18(4) and (1): both restate existing numbers, so they are computed here rather than
    # quoted anywhere. (4) turns each stratum's rank null into a power statement; (1) splits the
    # spread diagnostic on training-vocabulary membership, where its sign reverses.
    power = power_restatement(comparisons)
    power.to_csv(out_dir / f"model_v1_power_restatement_{args.label}.csv", index=False)
    print("\nD5-R18(4): rank-gap power against gbm_full -- a null with a factor attached")
    print(power.to_string(index=False, float_format="%.5f"))

    spread = None
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        vocabulary = {int(key) for key in
                      json.loads(manifest_path.read_text())["vocabulary"]}
        spread = trained_row_spread(frames[args.label], vocabulary)
        spread.to_csv(out_dir / f"model_v1_trained_spread_{args.label}.csv", index=False)
        print("\nD5-R18(1): predicted spread, cold-start rows separated rather than pooled")
        print(spread.to_string(index=False, float_format="%.4f"))
    else:
        print(f"\nD5-R18(1) skipped: no manifest at {manifest_path}")

    excess, debiased = debiased_diagnostic(frames, args.label, seed=args.seed)
    debiased.to_csv(out_dir / f"model_v1_claim1_debiased_{args.label}.csv", index=False)
    print(f"\ndebiased diagnostic (mean excess {excess:+.5f} removed) -- NOT a fix, see the "
          f"2026-08-08 knob entry")
    print(debiased.to_string(index=False, float_format="%.5f"))

    decisive = verdicts[DECISIVE_STRATUM]
    rmse_pass, rank_pass = decisive["rmse_gate_vs_gbm_full"], decisive["ordering_gate_vs_both"]
    print(f"\nRMSE gate vs C.3-full ({DECISIVE_STRATUM}): {'PASS' if rmse_pass else 'not met'}")
    print(f"ordering gate vs BOTH C.2 and C.3-full ({DECISIVE_STRATUM}): "
          f"{'PASS' if rank_pass else 'not met'}")
    (out_dir / f"model_v1_claim1_verdict_{args.label}.json").write_text(json.dumps(
        {"decisive_stratum": DECISIVE_STRATUM,
         "rmse_gate_vs_gbm_full": rmse_pass, "ordering_gate_vs_both": rank_pass,
         "by_stratum": verdicts,
         # the debiased reading rides in the verdict file so it cannot be quoted without the
         # excess that produced it
         "debiased_mean_excess_removed": excess,
         "debiased_rmse_difference": debiased.set_index("stratum")[
             ["rmse_difference", "ci_low", "ci_high"]].to_dict("index"),
         # a null in the verdict file that has a power factor beside it cannot be read as
         # "measured and absent", which is the misreading D5-R18(4) is about
         "rank_power": power.set_index("stratum")[
             ["se_rank", "z", "batters_multiplier", "batters_needed"]].to_dict("index"),
         "trained_row_spread": (spread.set_index("stratum")[
             ["sd_pooled", "sd_trained", "cold_start_share"]].to_dict("index")
             if spread is not None else None),
         "eval_season": args.eval_season, "label": args.label}, indent=2))


if __name__ == "__main__":
    main()
