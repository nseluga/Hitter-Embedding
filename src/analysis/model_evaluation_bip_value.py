"""
Phase E.13 — why is modelled E[wOBA | ball in play] 0.37864 against 0.36353 observed?
(docs/phase-e-spec.md §12.3)

E.3 put 74.5% of the +0.013876 wOBA level bias in the VALUE channel, and the balls-in-play
term carries all of it: `value_modelled.bip` 0.37864234 against `value_observed.bip`
0.36352638, a gap of +0.01511. That modelled number is recovered as a RESIDUAL inside
`model_evaluation_eval.level_closure` -- it is whatever the wOBA level implies once bb/hbp/k are paid their
fixed weights -- so it absorbs every error the four absorbing rates cannot express. This
module asks which of two seams in the composition owns it. HARD CAP AT TWO checks: whatever
they leave is written down as unexplained, not pursued.

READ-ONLY with respect to the model. No checkpoint is loaded, no forward pass runs, nothing
trains, `query.py` is not edited, and 2025 is never read. Everything below is counted
frequencies off the same tensors and the same labeled parquet Phase D.5 already built.

CHECK A -- the measured-share seam (`query.py:486`).
    expected_points = measured_share * quality + (1 - measured_share) * unmeasured_value
`quality` is E[points | in play AND Statcast measured it], because `V` is fit only on balls
carrying all three quality bins. Both `measured_share` and `unmeasured_value` are TRAIN-SEASON
LEAGUE CONSTANTS (`query_tables._unmeasured_outcomes`), applied unchanged to 2024. If either
constant is wrong for 2024, the seam emits a level offset with no mispredicted pitch behind it.

CHECK B -- era drift in a frozen V.
The pre-registered Check B was Jensen's inequality on a convex payoff. It is REFUTED
STRUCTURALLY, before any number was computed, and that refutation is recorded rather than
quietly replaced: `V` is a LOOKUP TABLE of category probabilities per (ev, la, spray) bin
(`query_tables.py:286-305`) and the composition forms sum_b p(b) * points(b), which is LINEAR
in the bin distribution. Jensen's inequality has no purchase on a linear functional -- an
over-dispersed bin distribution moves this sum by exactly the amount the mass moved, with no
convexity premium. The substitute run in its place is FORK-OPENED, not pre-registered: `V`'s
CONDITIONALS are fit on 2015-2023 and applied to 2024, so the question is whether
P(outcome | bin) itself drifted. That part transports into the composition as error, because
the model supplies its own bin distribution but inherits V's conditionals. The bin MIX is
reported alongside as context only -- the model predicts its own mix, so mix drift tells us
what the model SHOULD have shifted, not what it got wrong.

Both drifts are priced under the SAME 2024 wOBA weights so weight drift cannot contaminate
either one. Both V fits use the SAME bin edges -- the tensors' bin indices, fit on train
seasons only (`model_dataset.fit_bin_edges`) -- which is asserted, not assumed: refitting
edges on 2024 would redefine the bins and make the comparison meaningless.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval, model_evaluation_eval
from src.data import eval_targets
from src.data.model_dataset import MASKED
from src.model import query_tables as qt

# the E.3 numbers this module explains, quoted so a silent change upstream trips an assertion
# (embedding_sgd_sgd_lr1 chain, 2026-09-03; rebuild_baseline was 0.37864233821410337)
E3_VALUE_MODELLED_BIP = 0.3757228903779065
E3_VALUE_OBSERVED_BIP = 0.3635263829493724

# the two train-season league constants the composition applies to 2024
# (results/model_v1/model_v1_diagnostics_<arm>.json)
TRAIN_MEASURED_SHARE = 0.9231315908347482
TRAIN_UNMEASURED_POINTS = 0.1414959852073254


# ------------------------------------------------------------------ pure arithmetic


def seam_attribution(value_modelled_bip, s_train, u_train, s_obs, u_obs):
    """
    Check A's arithmetic, kept free of data so it can be unit-tested.

    The composition prices a ball in play as

        m = s_train * q + (1 - s_train) * u_train

    with q the model's E[points | measured]. q is not observable directly, so it is BACKED OUT
    of the modelled value: q = (m - (1 - s_train) * u_train) / s_train. Under 2024's own
    constants the same q would have been priced

        m' = s_obs * q + (1 - s_obs) * u_obs

    and m - m' is the part of the gap the seam owns. Split Oaxaca-style, in the same order
    `level_closure` splits its own gap:

        m - m' = (s_train - s_obs) * (q - u_train)   <- the measured-share constant being wrong
               + (1 - s_obs)      * (u_train - u_obs) <- the unmeasured-value constant being wrong

    Positive means the seam pushes the modelled value ABOVE truth, the sign of the +0.01511 gap.
    """
    assert 0.0 < s_train <= 1.0 and 0.0 < s_obs <= 1.0, "a measured share outside (0, 1]"
    quality = (value_modelled_bip - (1.0 - s_train) * u_train) / s_train
    share_term = (s_train - s_obs) * (quality - u_train)
    unmeasured_term = (1.0 - s_obs) * (u_train - u_obs)
    repriced = s_obs * quality + (1.0 - s_obs) * u_obs
    total = value_modelled_bip - repriced
    assert abs((share_term + unmeasured_term) - total) < 1e-12, \
        f"the seam split does not close: {share_term + unmeasured_term} against {total}"
    return {
        "implied_quality": float(quality),
        "repriced_under_2024_constants": float(repriced),
        "share_term": float(share_term),
        "unmeasured_term": float(unmeasured_term),
        "total": float(total),
    }


def drift_terms(points_train, points_obs, mass_train, mass_obs):
    """
    Check B's arithmetic, likewise data-free.

    points_*: per-bin E[wOBA points | bin], both under 2024 weights.
    mass_*:   per-bin P(bin), each summing to 1.

    CONDITIONAL drift, aggregated over the TRAINING bin marginal because that is the
    distribution V's conditionals were fit against and the one that says how much error a
    frozen conditional carries per ball:

        sum_b m_train(b) * (points_train(b) - points_obs(b))

    MIX drift, priced with the frozen train conditionals so only the mass moves:

        sum_b (m_train(b) - m_obs(b)) * points_train(b)

    Both signed model-minus-truth, so positive is comparable against the +0.01511 gap.
    """
    for mass in (mass_train, mass_obs):
        assert abs(float(mass.sum()) - 1.0) < 1e-9, "a bin marginal does not sum to 1"
    assert points_train.shape == points_obs.shape == mass_train.shape == mass_obs.shape, \
        "the four per-bin arrays disagree in shape"
    return {
        "conditional_drift": float((mass_train * (points_train - points_obs)).sum()),
        "mix_drift": float(((mass_train - mass_obs) * points_train).sum()),
        "value_train_conditionals_train_mix": float((mass_train * points_train).sum()),
        "value_obs_conditionals_train_mix": float((mass_train * points_obs).sum()),
        "value_train_conditionals_obs_mix": float((mass_obs * points_train).sum()),
        "value_obs_conditionals_obs_mix": float((mass_obs * points_obs).sum()),
    }


# ------------------------------------------------------------------ Check A, on data


def matched_groups(predictions, pa_df, manifest, eval_season):
    """
    The (batter, season, p_throws) groups that survive E.3's join, with their denominators.

    Replicated from `model_evaluation_eval.level_closure` rather than approximated: Check A has to be read
    against a number computed on that exact population, and any hitter present on one side
    and absent on the other would move the comparison by more than the effect being measured.
    """
    trained = model_evaluation_eval.trained_batters(manifest)
    actual = eval_targets.aggregate(eval_targets.drop_pitcher_batters(pa_df),
                                    by=tuple(claim1_eval.KEY))
    actual = actual[actual["season"] == eval_season]
    observed = model_evaluation_eval.per_group_absorbing(pa_df, [eval_season])

    frame = (predictions.merge(actual[claim1_eval.KEY + ["woba", "denominator"]],
                               on=claim1_eval.KEY, how="inner")
             .merge(observed.drop(columns=["denominator"]), on=["batter", "p_throws"],
                    how="inner"))
    frame = frame[frame["batter"].isin(trained)]
    assert len(frame), "no trained hitter survived the level-closure join"
    assert not frame.duplicated(subset=claim1_eval.KEY).any(), \
        "the level-closure join produced duplicate groups; every weight below would be wrong"
    return frame[claim1_eval.KEY + ["denominator"]].copy()


def observed_seam(frame, bins, pa_df, groups, eval_season, weights):
    """
    The 2024 truth behind the two train constants, on E.3's matched population.

    PA-denominator weighting is what E.3 uses, and a group's denominator IS its count of
    denominator plate appearances, so pooling over the matched population's PAs at pitch level
    reproduces that weighting exactly rather than approximating it -- no per-group average is
    needed and none is taken.

    Measured/unmeasured is defined exactly as `query_tables._unmeasured_outcomes` defines it:
    an in-play pitch is MEASURED when all three quality bins are present, and the split is
    counted over pitches whose plate appearance joins to a wOBA category.
    """
    matched = eval_targets.drop_pitcher_batters(pa_df)
    matched = matched[(matched["season"].to_numpy() == eval_season)
                      & matched["in_denominator"].to_numpy()]
    matched = matched.merge(groups[["batter", "p_throws"]].drop_duplicates(),
                            on=["batter", "p_throws"], how="inner")
    assert len(matched), "the matched population is empty at PA level"
    assert not matched.duplicated(subset=qt.JOIN_KEYS).any(), \
        "a plate-appearance key appears twice in the matched population"

    outcome = matched.set_index(qt.JOIN_KEYS)["woba_category"]
    scored = ((bins["ev"] != MASKED) & (bins["la"] != MASKED) & (bins["spray"] != MASKED))
    in_play = ((frame["description"].to_numpy() == qt.IN_PLAY_DESCRIPTION)
               & (frame["season"].to_numpy() == eval_season))

    counts = {}
    for name, mask in (("measured", in_play & scored), ("unmeasured", in_play & ~scored)):
        rows = np.flatnonzero(mask)
        klass = outcome.reindex(pd.MultiIndex.from_frame(frame.iloc[rows][qt.JOIN_KEYS]))
        klass = klass.map(qt.BIP_CATEGORY_TO_CLASS)
        klass = klass.to_numpy()[klass.notna().to_numpy()].astype("int64")
        counts[name] = np.bincount(klass, minlength=len(qt.BIP_CLASSES)).astype("float64")

    n_measured, n_unmeasured = counts["measured"].sum(), counts["unmeasured"].sum()
    assert n_measured > 0 and n_unmeasured > 0, \
        "one side of the measured split is empty on the matched population"
    values = qt.woba_points_table(np.eye(len(qt.BIP_CLASSES)), weights)
    return {
        "n_measured": int(n_measured), "n_unmeasured": int(n_unmeasured),
        "measured_share": float(n_measured / (n_measured + n_unmeasured)),
        "unmeasured_points": float((counts["unmeasured"] / n_unmeasured) @ values),
        "measured_points": float((counts["measured"] / n_measured) @ values),
        "categories_unmeasured": (counts["unmeasured"] / n_unmeasured).tolist(),
    }


# ------------------------------------------------------------------ Check B, on data


def fit_two_v_tables(frame, bins, pa_df, manifest, eval_season):
    """
    `V` fit twice off the SAME bin indices: once on train seasons, once on 2024.

    The bin indices come from the built tensors, whose edges `model_dataset.fit_bin_edges`
    fit on train seasons only, so the 2024 fit cannot silently redefine its own bins. That is
    the load-bearing condition of the whole comparison and it is asserted below rather than
    trusted -- equal-mass edges refit on 2024 would put a different set of balls in every cell
    and the per-bin difference would be measuring the binning, not the era.

    The train fit is also the identity check on the whole build: it must reproduce the two
    league constants the D.10 composition actually applied, to the digit. It does not for
    every tensor build on disk -- `data/processed/phase_d` (bin edges of 2026-08-01) gives
    0.96371 and `data/processed/phase_d5` (bin edges of 2026-08-12) gives 0.92313 -- so the
    wrong `--data-dir` would silently price Check A against constants no run ever used.

    Returns {name: (table, cell_counts, n_joined)} with the tables as CATEGORY probabilities,
    exactly the shape `query.predict` collapses under the eval season's weights.
    """
    train_seasons = manifest["train_seasons"]
    assert eval_season not in train_seasons, \
        f"{eval_season} is inside the train seasons; there would be no train/eval contrast"
    assert set(manifest["quality_bin_edges"]) == {"ev", "la", "spray"}, \
        "the manifest does not carry the three quality bin edge sets"
    n_bins = manifest["n_quality_bins"]

    season = frame["season"].to_numpy()
    masks = {"train": np.isin(season, train_seasons), "obs": season == eval_season}
    out = {}
    for name, mask in masks.items():
        # `bins` is the SAME dict object on both calls -- one set of train-fit bin indices,
        # never refit per side
        table, n_joined, unmeasured, counts = qt.fit_outcome_table(frame, bins, mask,
                                                                   pa_df, n_bins)
        assert n_joined > 0, f"no {name} ball in play joined to a plate-appearance outcome"
        out[name] = (table, counts.astype("float64"), int(n_joined), unmeasured)
    assert abs(out["train"][3]["measured_share"] - TRAIN_MEASURED_SHARE) < 1e-12, \
        (f"this build gives a train measured share of {out['train'][3]['measured_share']}, "
         f"not the {TRAIN_MEASURED_SHARE} the D.10 run applied -- wrong --data-dir")
    assert out["train"][0].shape == out["obs"][0].shape == \
        (n_bins, n_bins, n_bins, len(qt.BIP_CLASSES)), "V came back the wrong shape"
    return out


# ------------------------------------------------------------------ driver


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.13 — who owns the balls-in-play value gap.")
    parser.add_argument("--arm", default="embedding_sgd_sgd_lr1")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--predictions", default=None)
    # the D.10 build, not `model_v1`: the two differ in quality bin edges and only this one
    # reproduces the measured-share constant the D.10 predictions were composed with
    parser.add_argument("--data-dir", default="data/processed/phase_d5")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--e-report", default="results/model_evaluation/model_evaluation_report.json")
    parser.add_argument("--out-dir", default="results/model_evaluation")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    season = np.asarray(np.load(data_dir / "season.npy", mmap_mode="r"))
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, season)
    bins = {name: np.asarray(np.load(data_dir / f"{name}.npy")) for name in
            ("ev", "la", "spray")}
    for name, array in bins.items():
        assert len(array) == len(frame), f"{name}.npy is {len(array)} rows against {len(frame)}"
    pa_df = pd.read_parquet(args.eval_targets)
    predictions = pd.read_csv(args.predictions
                              or f"results/model_v1/model_v1_predictions_{args.arm}.csv")
    weights = eval_targets.load_weights()[str(args.eval_season)]

    # the gap comes from the E.3 artefact on disk, not from a constant retyped here -- but the
    # constants are asserted against it so a re-run of E.3 that moves them cannot go unnoticed
    e3 = json.loads(Path(args.e_report).read_text())["e3_level_closure"]
    for label, on_disk, quoted in (("modelled", e3["value_modelled"]["bip"], E3_VALUE_MODELLED_BIP),
                                   ("observed", e3["value_observed"]["bip"], E3_VALUE_OBSERVED_BIP)):
        assert abs(on_disk - quoted) < 1e-9, \
            f"E.3's {label} bip value moved: {on_disk} against the quoted {quoted}"
    gap = e3["value_modelled"]["bip"] - e3["value_observed"]["bip"]

    # Check B's fits come first because the train fit is the guard on Check A's constants
    print("E.13 Check B — era drift in a frozen V (substitute; Jensen refuted structurally)")
    fits = fit_two_v_tables(frame, bins, pa_df, manifest, args.eval_season)
    values = qt.woba_points_table(np.eye(len(qt.BIP_CLASSES)), weights)
    assert abs(np.asarray(fits["train"][3]["categories"]) @ values
               - TRAIN_UNMEASURED_POINTS) < 1e-12, \
        "this build does not reproduce the train unmeasured-value constant the D.10 run applied"
    points = {name: fits[name][0] @ values for name in fits}
    mass = {name: fits[name][1] / fits[name][1].sum() for name in fits}
    drift = drift_terms(points["train"], points["obs"], mass["train"], mass["obs"])

    print("E.13 Check A — the measured-share seam")
    groups = matched_groups(predictions, pa_df, manifest, args.eval_season)
    assert abs(float(groups["denominator"].sum()) - e3["n_pa"]) < 1e-6, \
        "the replicated matched population does not carry E.3's plate-appearance count"
    obs_seam = observed_seam(frame, bins, pa_df, groups, args.eval_season, weights)
    attribution = seam_attribution(e3["value_modelled"]["bip"], TRAIN_MEASURED_SHARE,
                                   TRAIN_UNMEASURED_POINTS, obs_seam["measured_share"],
                                   obs_seam["unmeasured_points"])

    # per-(ev, la) contributions, spray summed out: 13,824 joint cells is not a readable table,
    # and the marginal is the level at which V's own backoff pool is defined
    n_bins = manifest["n_quality_bins"]
    contribution = mass["train"] * (points["train"] - points["obs"])
    rows = []
    for ev in range(n_bins):
        for la in range(n_bins):
            rows.append({"ev_bin": ev, "la_bin": la,
                         "train_mass": float(mass["train"][ev, la].sum()),
                         "obs_mass": float(mass["obs"][ev, la].sum()),
                         "conditional_contribution": float(contribution[ev, la].sum())})
    pd.DataFrame(rows).to_csv(out_dir / "bip_value_bip_value_by_bin.csv", index=False)

    explained = attribution["total"] + drift["conditional_drift"]
    residual = gap - explained
    report = {
        "arm": args.arm, "eval_season": args.eval_season, "data_dir": args.data_dir,
        "build_note": "the D.10 composition's train constants reproduce ONLY from "
                      "data/processed/phase_d5 (measured_share 0.92313); "
                      "data/processed/phase_d, whose quality bin edges differ, gives 0.96371",
        "gap_being_explained": {
            "value_modelled_bip": e3["value_modelled"]["bip"],
            "value_observed_bip": e3["value_observed"]["bip"],
            "gap": gap,
            "note": "value_modelled.bip is a RESIDUAL in model_evaluation_eval.level_closure, so it absorbs "
                    "every error not expressible in the bb/hbp/k rates",
            "n_groups": e3["n_groups"], "n_pa": e3["n_pa"],
        },
        "check_a_measured_share_seam": {
            "train_constants": {"measured_share": TRAIN_MEASURED_SHARE,
                                "unmeasured_points": TRAIN_UNMEASURED_POINTS},
            "observed_2024_matched_population": obs_seam,
            "absolute_differences": {
                "measured_share": obs_seam["measured_share"] - TRAIN_MEASURED_SHARE,
                "unmeasured_points": obs_seam["unmeasured_points"] - TRAIN_UNMEASURED_POINTS,
            },
            "attribution": attribution,
            "share_of_gap": attribution["total"] / gap,
        },
        "jensen_refuted_structurally": True,
        "jensen_refutation_reason":
            "V is a lookup table of category probabilities per (ev, la, spray) bin and the "
            "composition forms sum_b p(b) * points(b), a LINEAR functional of the bin "
            "distribution; Jensen's inequality requires a convex map of the quantity being "
            "averaged, so over-dispersion buys no premium here.",
        "check_b_era_drift": {
            "status": "fork-opened, not pre-registered — substitute for the refuted Jensen test",
            "priced_under_weights": f"{args.eval_season} wOBA weights, both sides",
            "n_bip_train": fits["train"][2], "n_bip_obs": fits["obs"][2],
            **drift,
            "conditional_share_of_gap": drift["conditional_drift"] / gap,
            "mix_note": "mix drift does NOT enter the composition — the model predicts its own "
                        "bin distribution; it says what the model should have shifted",
        },
        "residual_unexplained": residual,
        "residual_share_of_gap": residual / gap,
        "expectations": {
            "check_a_preregistered": {
                "statement": "a minority of the gap, order 0.002-0.004",
                "observed": attribution["total"],
                "held": bool(0.002 <= attribution["total"] <= 0.004),
            },
            "check_b_preregistered_jensen": {
                "statement": "the larger share, order 0.008-0.012 of the gap",
                "observed": None,
                "held": False,
                "why": "not testable — refuted structurally before running (see above)",
            },
            "stop_rule": {
                "statement": "if the two checks leave more than a third of the gap "
                             "unexplained, the residual is unowned and Phase E closes on it",
                "residual_share": residual / gap,
                "triggered": bool(abs(residual) > gap / 3.0),
            },
        },
        "assumptions": [
            "Check A treats the model's implied E[points | measured] as correct and prices "
            "only the two league constants; any error inside `quality` itself lands in the "
            "residual, not in Check A.",
            "The matched population is reproduced at PA level by (batter, p_throws) group "
            "membership; group denominator equals that group's denominator PA count, so "
            "pooling PAs reproduces E.3's denominator weighting exactly.",
            "Check B's 2024 V is fit with the same shrinkage estimator as the train V, so a "
            "2024 cell with little data is pulled toward the 2024 (ev, la) marginal. That "
            "attenuates per-cell conditional drift toward its own-season aggregate; the "
            "aggregate itself is close to preserved, but the reported conditional drift is a "
            "mildly conservative estimate rather than an unbiased one.",
            "Bins are the tensors' train-fit equal-mass indices for both fits; 2024 balls "
            "outside the train support are clipped into edge cells by `assign_bins`.",
            "Two tensor builds sit on disk with different quality bin edges; this module "
            "uses phase_d5 because it is the one that reproduces the D.10 run's league "
            "constants exactly. Whether E.3 itself was computed against the same build is "
            "not re-verified here beyond the matched population's PA count matching.",
            "Balls in play whose PA does not join a wOBA category (SH, INT, unjoined rows) "
            "drop from both V fits, as they do in the shipped `fit_outcome_table`.",
            "The two checks are treated as additive when summed against the gap; they touch "
            "different terms (the seam's constants against V's conditionals) but the split is "
            "not orthogonalised, so the residual carries any interaction between them.",
        ],
    }
    path = out_dir / "bip_value_bip_value.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
