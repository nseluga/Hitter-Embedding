"""
Phase C report — every Phase C number in one committed, seeded command.

Follows the b1_report.py pattern and exists for the same reason: until now Phase C
figures lived in transcripts, which is not a reproducible record (§6 — numbers are
seeded, config-driven, with the config committed). Writes results/phase_c/.

What it produces:
  1. the fitted C.2 prior per batter type, with a bootstrap CI on rho
  2. the implied split-regression constant against The Book's 2200 / 1000
  3. claim-1 scores for every Phase C baseline on the same eval frame
  4. the paired bootstraps that are the actual head-to-head claims — C.2 vs C.1 in
     both variants (raw and bucketed), C.3 vs C.2, and C.3's process ablation — on
     BOTH frozen metrics: RMSE per comparison, and rank correlation in one table
     (`c_rank_paired.csv`)
  5. the diagnostics that keep those claims honest: a label-shuffle null, per-stratum
     prediction bias, an oracle level-corrected bound, exposure-talent correlation,
     and a decode-one-hitter table for eyeballing against the source
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import c1_trailing as c1
from src.analysis import c2_bivariate_eb as c2
from src.analysis import c3_gbm as c3
from src.analysis import claim1_eval as evaluation

# the only columns c3.season_process reads; the labeled table is 340MB and the
# rest of it is context the hitter-level baselines have no use for
PITCH_COLUMNS = ["batter", "season", "p_throws", "pitch_type", "strikes",
                 "swing", "contact", "ev", "la", "spray"]

DEFAULT_EVAL_SEASON = 2024
OUT_DIR = Path("results/phase_c")

# The Book p. 157 (via Tango, insidethebook.com 2009), converted to the rho it is
# equivalent to via n*_d = n*_side / (2(1 - rho)) with n*_side = 226 (B.1, hitters
# only). Approximate — assumes tau_L = tau_R. A REFERENCE ROW, never fitted.
#
# Scope, decided 2026-07-29: this is a PLAUSIBILITY CHECK on our own fitted prior,
# not a scored incumbent. We do not have a copy of The Book, so its functional form
# cannot be reproduced faithfully and a head-to-head against it would be a comparison
# with a reconstruction, not with the source. The check that remains is in parameter
# space — our rho and implied split constant against the published ones, printed below
# and committed in c2_prior_parameters.csv — which answers "are we in the right
# ballpark" without claiming to have run their model. The paired bootstrap that used
# to score this row was removed for that reason.
BOOK_IMPLIED_RHO = {"R": 0.949, "L": 0.887, "S": 0.949}

# no-information reference: predict the side-specific league average for everyone.
# Its deconvolved error is the full between-hitter talent spread, so skill scores
# against it read as the share of talent variance a model captures.
NO_INFO_BUCKETS = ((0, 0.0),)


def load_targets(path):
    return pd.read_parquet(path)


def fit_and_describe(pa_df, eval_season, seed):
    """Fit the C.2 prior and return (params, parameter table, rho CIs)."""
    params = c2.fit(pa_df, eval_season)
    rho_ci = c2.bootstrap_rho(pa_df, eval_season, seed=seed)

    rows = []
    for batter_type, record in params.items():
        n_star_split, tau2_split = c2.implied_split_constant(record)
        point, low, high, at_bound = rho_ci.get(batter_type, (record["rho"], np.nan, np.nan, np.nan))
        for side_index, hand in enumerate(c2.SIDES):
            tau2 = record["tau2"][side_index]
            rows.append({
                "batter_type": batter_type,
                "vs_hand": hand,
                "n_hitters": record["n_hitters"],
                "mu": record["mu"][side_index],
                "tau": np.sqrt(tau2),
                "tau2": tau2,
                "sigma2_within_pa": record["sigma2"][side_index],
                "n_star_rate": record["sigma2"][side_index] / tau2 if tau2 > 0 else np.inf,
                "rho": point,
                "rho_ci_low": low,
                "rho_ci_high": high,
                "rho_at_bound": record["rho_at_bound"],
                "rho_bootstrap_share_at_bound": at_bound,
                "n_rho_hitters": record["n_rho_hitters"],
                "rho_unclipped": record["rho_unclipped"],
                "tau2_split_derived": tau2_split,
                "n_star_split_implied": n_star_split,
                "book_split_constant": c2.BOOK_SPLIT_CONSTANT.get(batter_type, np.nan),
                "book_implied_rho": BOOK_IMPLIED_RHO.get(batter_type, np.nan),
            })
    return params, pd.DataFrame(rows), rho_ci


def build_predictions(pa_df, eval_season, params, process_seasons, seed=0):
    """Every Phase C baseline, plus the two reference rows, on one eval frame."""
    predictions = {
        "c2_bivariate": c2.predict(pa_df, eval_season, params=params),
        "c2_book_rho_reference": c2.predict(pa_df, eval_season, params=params,
                                            rho_override=BOOK_IMPLIED_RHO),
        "c1_raw": c1.predict(pa_df, eval_season, variant="raw"),
        "c1_bucketed": c1.predict(pa_df, eval_season, variant="bucketed"),
        "no_info_league_average": c1.predict(pa_df, eval_season, variant="bucketed",
                                             buckets=NO_INFO_BUCKETS),
    }
    # C.3 in both feature sets: "outcome" sees exactly what C.1/C.2 saw, "full"
    # adds process and its context slices. The pair is the ablation.
    models = {}
    for feature_set in ("outcome", "full"):
        prediction, model, n_rounds = c3.predict(pa_df, process_seasons, eval_season,
                                                 feature_set=feature_set, seed=seed,
                                                 return_model=True)
        predictions[f"c3_gbm_{feature_set}"] = prediction
        models[feature_set] = (model, n_rounds)
    return predictions, models


def score_all(pa_df, predictions, eval_season):
    """Claim-1 metrics for every model, plus skill scores against the no-info reference."""
    frames, scores = {}, []
    for name, prediction in predictions.items():
        frame, coverage = evaluation.build_eval_frame(pa_df, prediction, eval_season)
        frames[name] = frame
        table = evaluation.score(frame)
        table.insert(0, "model", name)
        scores.append(table)
    scored = pd.concat(scores, ignore_index=True)

    reference = scored[scored["model"] == "no_info_league_average"].set_index("stratum")["model_rmse"]
    scored["skill_vs_no_info"] = [
        evaluation.skill_score(row["model_rmse"], reference[row["stratum"]])
        if reference[row["stratum"]] > 0 else np.nan
        for _, row in scored.iterrows()
    ]
    return frames, scored, coverage


def shuffle_null_table(pa_df, process_seasons, reference_frame, eval_season, seed,
                       real_differences):
    """
    How large a margin over C.2 does the C.3 fitting procedure produce with NO
    hitter information? Trains the same model on shuffled labels several times
    and scores each against the same reference, giving the real margin a null to
    be judged against rather than an eyeballed "that looks big".
    """
    nulls = c3.shuffle_null(pa_df, process_seasons, eval_season)
    differences = []
    for null_seed, prediction in nulls.items():
        frame, _ = evaluation.build_eval_frame(pa_df, prediction, eval_season)
        table = evaluation.paired_rmse_difference(frame, reference_frame, n_boot=200, seed=seed)
        table["null_seed"] = null_seed
        differences.append(table)
    stacked = pd.concat(differences, ignore_index=True)
    real = real_differences.set_index("stratum")["rmse_difference"]
    stacked["beats_real"] = [row["rmse_difference"] <= real.get(row["stratum"], -np.inf)
                             for _, row in stacked.iterrows()]

    return (stacked.groupby("stratum", observed=True)
            .agg(n_seeds=("rmse_difference", "size"),
                 mean_difference=("rmse_difference", "mean"),
                 min_difference=("rmse_difference", "min"),
                 max_difference=("rmse_difference", "max"),
                 n_null_fits_beating_real=("beats_real", "sum"))
            .reset_index())


def min_eval_pa_sensitivity(pa_df, predictions, eval_season, seed,
                            thresholds=evaluation.MIN_EVAL_PA_SENSITIVITY):
    """
    Does the headline survive the eval filter, or is it an artifact of it?

    MIN_EVAL_PA censors on eval-season playing time — decided after the projection and
    partly by the hitter's own performance — and it bites hardest in the low stratum
    the thesis is graded on. A threshold chosen once and never varied is an untested
    assumption sitting upstream of every Phase C number, so the headline C.3-vs-C.2
    comparison is re-scored across a range of it and the result is committed.

    Cheap by construction: predictions do not depend on the threshold, so nothing is
    refit — only the eval frame is rebuilt. Reports, per threshold and stratum, the
    RMSE margin with its CI, the ordering margin with its CI, both models' level bias,
    and the oracle share (how much of the RMSE margin is level rather than ordering),
    which is the quantity most exposed to the filter.
    """
    rows = []
    for threshold in thresholds:
        frames = {name: evaluation.build_eval_frame(pa_df, prediction, eval_season,
                                                    min_eval_pa=threshold)[0]
                  for name, prediction in predictions.items()}
        c3, c2 = frames["c3_gbm_full"], frames["c2_bivariate"]
        rmse = evaluation.paired_rmse_difference(c3, c2, seed=seed).set_index("stratum")
        rank = evaluation.paired_rank_difference(c3, c2, seed=seed).set_index("stratum")
        oracle = evaluation.paired_rmse_difference(oracle_debias(c3), oracle_debias(c2),
                                                  seed=seed).set_index("stratum")
        bias = prediction_bias({k: frames[k] for k in ("c3_gbm_full", "c2_bivariate")})
        bias = bias.set_index(["model", "stratum"])["bias"]

        for stratum in rmse.index:
            reported, debiased = rmse.loc[stratum, "rmse_difference"], oracle.loc[stratum, "rmse_difference"]
            rows.append({
                "min_eval_pa": threshold,
                "stratum": stratum,
                "n_hitters": int(rmse.loc[stratum, "n_hitters"]),
                "rmse_difference": reported,
                "rmse_ci_low": rmse.loc[stratum, "ci_low"],
                "rmse_ci_high": rmse.loc[stratum, "ci_high"],
                "rmse_favours_c3": rmse.loc[stratum, "favours_a_share"],
                "rank_difference": rank.loc[stratum, "rank_difference"],
                "rank_ci_low": rank.loc[stratum, "ci_low"],
                "rank_ci_high": rank.loc[stratum, "ci_high"],
                "bias_c2": bias.get(("c2_bivariate", stratum), np.nan),
                "bias_c3": bias.get(("c3_gbm_full", stratum), np.nan),
                "oracle_share_level": 1 - debiased / reported if reported else np.nan,
            })
    return pd.DataFrame(rows)


def rank_paired_table(frames, comparisons, seed):
    """
    The ordering head-to-heads, each with an interval.

    Every RMSE claim in this report goes through a paired bootstrap; until now the
    rank claims did not, and they are the noisier of the two metrics. Quoting a pair
    of bare rank correlations as evidence that two models order hitters differently
    holds the ordering metric to a lower bar than the calibration metric — which
    matters most in the low stratum, where the thesis is graded and where the gap
    between any two models sits inside its own resampling interval.
    """
    tables = []
    for label, (name_a, name_b) in comparisons.items():
        table = evaluation.paired_rank_difference(frames[name_a], frames[name_b], seed=seed)
        table.insert(0, "comparison", label)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def oracle_debias(frame):
    """
    Recenter a model's predictions to zero PA-weighted bias within each stratum,
    USING THE HELD-OUT ACTUALS. This is an oracle: it peeks at the answer key, so
    it can never be a baseline and is never scored as one. Its only job is to
    separate a model's level error from its ordering, so the C.2/C.3 comparison
    can say how much of the margin is each.
    """
    out = frame.copy()
    for stratum in evaluation.STRATUM_NAMES:
        mask = out["stratum"] == stratum
        if not mask.any():
            continue
        weight = out.loc[mask, "denominator"].astype(float)
        bias = np.average(out.loc[mask, "pred_woba"] - out.loc[mask, "woba"], weights=weight)
        out.loc[mask, "pred_woba"] -= bias
    return out


def prediction_bias(frames):
    """
    PA-weighted mean prediction minus mean actual, per stratum. The level error
    every Phase C baseline carries: they shrink toward LEAGUE average, but
    low-exposure hitters are genuinely below it (managers bench bad hitters), so
    a model that can express an exposure-conditional level gains RMSE without
    ordering hitters any better. Reading the skill scores without this column
    invites crediting a level correction as projection skill.
    """
    rows = []
    for name, frame in frames.items():
        for stratum in evaluation.STRATUM_NAMES:
            part = frame[frame["stratum"] == stratum]
            if not len(part):
                continue
            weight = part["denominator"].astype(float)
            rows.append({
                "model": name, "stratum": stratum,
                "bias": float(np.average(part["pred_woba"] - part["woba"], weights=weight)),
                "mean_pred": float(np.average(part["pred_woba"], weights=weight)),
                "mean_actual": float(np.average(part["woba"], weights=weight)),
            })
    return pd.DataFrame(rows)


def exposure_talent_correlation(frame):
    """
    Brown (2008) remark 3b: empirical Bayes assumes exchangeability, but better hitters
    get more PA, and in the platoon case managers ASSIGN weak-side exposure (§5.5
    deployment bias). Correlation between prior exposure and observed talent quantifies
    how far the assumption is from holding. Reported, not corrected for.
    """
    rows = []
    for hand in c2.SIDES:
        part = frame[frame["p_throws"] == hand]
        rows.append({
            "vs_hand": hand,
            "n_groups": len(part),
            "corr_prior_pa_vs_observed_woba": float(part["prior_pa"].corr(part["woba"], method="spearman")),
        })
    return pd.DataFrame(rows)


def decode_sample(frames, pa_df, eval_season, n_each=2, seed=0):
    """
    Decode-one-batch analog: real hitters at each exposure level with their inputs and
    every model's prediction side by side, for eyeballing against the source. Catches
    join and alignment bugs where every column is individually well formed but
    describes the wrong hitter.
    """
    base = frames["c2_bivariate"][evaluation.KEY + ["prior_pa", "stratum", "pa", "woba"]].copy()
    for name, frame in frames.items():
        base = base.merge(frame[evaluation.KEY + ["pred_woba"]].rename(columns={"pred_woba": name}),
                          on=evaluation.KEY, how="left")

    window = c2._fitting_window(pa_df, eval_season, c2.TRAILING_SEASONS)
    pairs = c2.to_pairs(c2.side_observations(window))
    base = base.merge(pairs, on="batter", how="left")

    rng = np.random.default_rng(seed)
    picks = [base[base["prior_pa"] == 0]]
    for stratum in evaluation.STRATUM_NAMES:
        part = base[base["stratum"] == stratum]
        if len(part):
            picks.append(part.iloc[rng.choice(len(part), min(n_each, len(part)), replace=False)])
    return pd.concat(picks).head(8)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase C baseline report.")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-season", type=int, default=DEFAULT_EVAL_SEASON)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    # the frozen test season is scored ONCE, for the final reported result;
    # this flag is the deliberate act of doing so, never a convenience
    parser.add_argument("--final-run", action="store_true",
                        help="permit scoring the frozen TEST season")
    args = parser.parse_args()

    # never score against the frozen test season outside a final run
    evaluation.assert_not_test_season(args.eval_season, final_run=args.final_run)
    pa_df = load_targets(args.eval_targets)

    params, parameter_table, rho_ci = fit_and_describe(pa_df, args.eval_season, args.seed)
    print("fitted C.2 prior")
    print(parameter_table.drop(columns=["tau2", "rho_unclipped"]).to_string(index=False, float_format="%.5f"))

    print("\nrho vs The Book's implied value")
    for batter_type, (point, low, high, at_bound) in rho_ci.items():
        book = BOOK_IMPLIED_RHO.get(batter_type, np.nan)
        verdict = "NOT IDENTIFIED (at bound)" if params[batter_type]["rho_at_bound"] else (
            "consistent with The Book" if low <= book <= high else "DIFFERS from The Book")
        print(f"  type {batter_type}: rho {point:.3f} 95% CI [{low:.3f}, {high:.3f}] "
              f"vs Book-implied {book:.3f} -> {verdict}  [{at_bound:.0%} of resamples at bound]")

    process_seasons = c3.season_process(
        pd.read_parquet(args.pitch_events, columns=PITCH_COLUMNS))
    predictions, c3_models = build_predictions(pa_df, args.eval_season, params,
                                               process_seasons, seed=args.seed)
    for feature_set, (_, n_rounds) in c3_models.items():
        print(f"C.3 [{feature_set}]: {n_rounds} boosting rounds "
              f"(early-stopped on target season {args.eval_season - 1})")

    frames, scored, coverage = score_all(pa_df, predictions, args.eval_season)
    print("\nclaim-1 scores")
    print(scored.to_string(index=False, float_format="%.4f"))

    print(f"\ncoverage: {json.dumps(coverage)}")

    paired = evaluation.paired_rmse_difference(frames["c2_bivariate"], frames["c1_bucketed"],
                                               seed=args.seed)
    print("\npaired bootstrap: C.2 bivariate minus C.1 bucketed (negative favours C.2)")
    print(paired.to_string(index=False, float_format="%.5f"))

    # C.1 raw is the same trailing record with no shrinkage at all, so this pair prices
    # regularization itself rather than the quality of it — the step C.1 bucketed splits in half
    paired_raw = evaluation.paired_rmse_difference(frames["c2_bivariate"], frames["c1_raw"],
                                                   seed=args.seed)
    print("\npaired bootstrap: C.2 bivariate minus C.1 raw (negative favours C.2)")
    print(paired_raw.to_string(index=False, float_format="%.5f"))

    # the two head-to-heads C.3 exists to settle: does a GBM beat the EB incumbent,
    # and does giving it process features (rather than the same inputs C.2 had) help
    paired_c3 = evaluation.paired_rmse_difference(frames["c3_gbm_full"],
                                                  frames["c2_bivariate"], seed=args.seed)
    print("\npaired bootstrap: C.3 full minus C.2 bivariate (negative favours C.3)")
    print(paired_c3.to_string(index=False, float_format="%.5f"))

    paired_c3_ablation = evaluation.paired_rmse_difference(frames["c3_gbm_full"],
                                                           frames["c3_gbm_outcome"], seed=args.seed)
    print("\npaired bootstrap: C.3 full minus C.3 outcome-only (negative favours process features)")
    print(paired_c3_ablation.to_string(index=False, float_format="%.5f"))

    shuffle = shuffle_null_table(pa_df, process_seasons, frames["c2_bivariate"],
                                 args.eval_season, args.seed, paired_c3)
    print("\nlabel-shuffle null: same fitting procedure, hitter->target link destroyed")
    print(shuffle.to_string(index=False, float_format="%.5f"))
    real_low = paired_c3[paired_c3["stratum"] == "low"]["rmse_difference"].iat[0]
    null_low = shuffle[shuffle["stratum"] == "low"]
    print(f"  C.3 full's low-stratum margin {real_low:+.5f} vs a null of "
          f"{null_low['mean_difference'].iat[0]:+.5f} "
          f"[{null_low['min_difference'].iat[0]:+.5f}, {null_low['max_difference'].iat[0]:+.5f}]"
          f" -> {int(null_low['n_null_fits_beating_real'].iat[0])} of "
          f"{int(null_low['n_seeds'].iat[0])} null fits match it")

    # how much of C.3's margin is the exposure-conditional level C.2 cannot express?
    # (decision log 2026-07-28 — the confound is reported, not corrected)
    debiased = evaluation.paired_rmse_difference(oracle_debias(frames["c3_gbm_full"]),
                                                 oracle_debias(frames["c2_bivariate"]),
                                                 seed=args.seed)
    print("\nORACLE bound (peeks at actuals, never a baseline): C.3 full minus C.2, "
          "both level-corrected")
    print(debiased.to_string(index=False, float_format="%.5f"))
    reported_low = paired_c3[paired_c3["stratum"] == "low"]["rmse_difference"].iat[0]
    debiased_low = debiased[debiased["stratum"] == "low"]["rmse_difference"].iat[0]
    print(f"  low stratum: {1 - debiased_low / reported_low:.0%} of C.3's reported margin "
          "is the level correction, not ordering")

    # the same head-to-heads on the ORDERING metric, which Layer 2 actually consumes
    rank_paired = rank_paired_table(frames, {
        "c3_full_vs_c2": ("c3_gbm_full", "c2_bivariate"),
        "c3_full_vs_c3_outcome": ("c3_gbm_full", "c3_gbm_outcome"),
        "c2_vs_c1_bucketed": ("c2_bivariate", "c1_bucketed"),
        "c2_vs_c1_raw": ("c2_bivariate", "c1_raw"),
    }, args.seed)
    print("\npaired bootstrap on RANK CORRELATION (positive favours A; batters resampled)")
    print(rank_paired.to_string(index=False, float_format="%.4f"))
    headline = rank_paired[(rank_paired["comparison"] == "c3_full_vs_c2")
                           & (rank_paired["stratum"] == "low")].iloc[0]
    print(f"  low stratum ordering: C.3 full {headline['rank_a']:.3f} vs C.2 "
          f"{headline['rank_b']:.3f}, difference {headline['rank_difference']:+.3f} "
          f"CI [{headline['ci_low']:+.3f}, {headline['ci_high']:+.3f}] -> "
          + ("C.3 orders better" if headline["ci_low"] > 0 else
             "C.2 orders better" if headline["ci_high"] < 0 else
             "NEITHER model demonstrably orders better"))

    sensitivity = min_eval_pa_sensitivity(pa_df, predictions, args.eval_season, args.seed)
    print("\nMIN_EVAL_PA sensitivity: C.3 full vs C.2 across the eval filter")
    print(sensitivity.to_string(index=False, float_format="%.4f"))
    low = sensitivity[sensitivity["stratum"] == "low"]
    span = low["rmse_difference"].max() - low["rmse_difference"].min()
    print(f"  low-stratum RMSE margin ranges {low['rmse_difference'].min():+.5f} to "
          f"{low['rmse_difference'].max():+.5f} (span {span:.5f}) across "
          f"min_eval_pa {list(low['min_eval_pa'])}; "
          + ("sign is stable" if (low["rmse_difference"] < 0).all() else "SIGN FLIPS — the margin is a filter artifact"))

    bias = prediction_bias(frames)
    print("\nprediction bias by stratum (mean predicted minus mean actual, PA-weighted)")
    print(bias.pivot(index="model", columns="stratum", values="bias").to_string(float_format="%.4f"))

    importance = c3.feature_importance(c3_models["full"][0], "full")
    print("\nC.3 gain importance (in-sample, interpretation only — the ablation is the evidence)")
    print(importance.head(12).to_string(index=False, float_format="%.4f"))

    exposure = exposure_talent_correlation(frames["c2_bivariate"])
    print("\nexposure-talent correlation (EB exchangeability check, reported not corrected)")
    print(exposure.to_string(index=False, float_format="%.4f"))

    decoded = decode_sample(frames, pa_df, args.eval_season, seed=args.seed)
    print("\ndecode check: real hitters, inputs and every model's prediction")
    print(decoded.to_string(index=False, float_format="%.4f"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parameter_table.to_csv(out_dir / "c2_prior_parameters.csv", index=False)
    scored.to_csv(out_dir / "c_claim1_scores.csv", index=False)
    paired.to_csv(out_dir / "c2_vs_c1_bucketed_paired.csv", index=False)
    paired_raw.to_csv(out_dir / "c2_vs_c1_raw_paired.csv", index=False)
    exposure.to_csv(out_dir / "c2_exposure_talent_correlation.csv", index=False)
    decoded.to_csv(out_dir / "c_decode_sample.csv", index=False)
    paired_c3.to_csv(out_dir / "c3_vs_c2_paired.csv", index=False)
    paired_c3_ablation.to_csv(out_dir / "c3_process_ablation_paired.csv", index=False)
    rank_paired.to_csv(out_dir / "c_rank_paired.csv", index=False)
    sensitivity.to_csv(out_dir / "c_min_eval_pa_sensitivity.csv", index=False)
    importance.to_csv(out_dir / "c3_feature_importance.csv", index=False)
    shuffle.to_csv(out_dir / "c3_shuffle_null.csv", index=False)
    bias.to_csv(out_dir / "c_prediction_bias.csv", index=False)
    debiased.to_csv(out_dir / "c3_vs_c2_oracle_debiased.csv", index=False)
    (out_dir / "c_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"\nwrote 14 files to {out_dir}")


if __name__ == "__main__":
    main()
