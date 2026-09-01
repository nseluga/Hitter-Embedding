"""
Phase E.14 — the §5.1 probe, and the ensemble interval coverage owed since 2026-08-08
(docs/phase-e-spec.md §12.4).

Two questions, one module, because both read the same frozen D.10 baseline artefacts.

READ-ONLY WITH RESPECT TO THE MODEL (spec §10). Nothing here trains, fine-tunes, or
scores a new season. The checkpoints are opened with `torch.load` and only the
`embedding.weight` tensor is read; no forward pass is run. 2024 is the eval season and
2025 is never touched.

--------------------------------------------------------------------- PART 1, the probe

WHY. Architecture §1.4 pre-registered a failure mode: the model learns hitter main
effects plus context main effects and NO interaction, in which case every conditional
query returns the league-average context penalty and platoon skill washes out entirely.
The probe was specified as the early detector of that mode and never ran. It runs here
as a RETROSPECTIVE DIAGNOSTIC and is logged as a demotion: it can no longer gate
anything, only explain E.5.

WHY A LINEAR DECODE. The question is whether platoon information is PRESENT in the
frozen hitter representation, independent of whether the composition machinery
(`query.predict`) manages to express it. A linear read-out is the conservative test:
if a linear probe finds the signal it is unambiguously there, and a linear probe is what
the trunk's first layer can itself apply to the embedding.

WHY THE POSTERIOR SPLIT AND NOT THE OBSERVED SPLIT. A hitter's raw observed split is
noise-dominated at single-season exposure — C.2 measures `n_star_split_implied` at 320.65
(LHB) / 562.98 (RHB) PA against realized weak-side exposures far below both. Decoding
against the raw split would mostly measure the embedding's correlation with sampling
noise, which is zero by construction, and the probe would return "no platoon knowledge"
for a model that had perfect platoon knowledge. The C.2 empirical-Bayes POSTERIOR MEAN
split is the best available estimate of the hitter's true split, so it is the target.

WHY PER-SEED DECODE, AVERAGED. The five seeds are five independent inits. Embedding
coordinates are not aligned across seeds — nothing in the loss ties seed 2's axis 7 to
seed 3's axis 7 — so averaging the raw embedding tensors across seeds would average
unrelated bases and destroy the signal. The decode is therefore run once per seed and
the resulting CORRELATIONS are averaged. This module asserts that no raw embedding is
averaged across seeds.

WHY COLD-START HITTERS ARE EXCLUDED. Row 0 of the embedding table is one shared reserved
vector for every hitter absent from the training vocabulary (`query.py:390`,
`query.py:695` `vocabulary.get(int(b), 0)`). Including them would put many distinct
hitters on one identical feature row, which a ridge can only map to one constant — it
would inject a large block of pure-noise residual and understate the probe.

WHY A NULL. A ridge with 32 features fitted on a few hundred hitters reaches a non-zero
out-of-fold correlation by chance. The same pipeline is therefore re-run against a
permuted target. Without that number the probe's correlations have no scale and the
step is uninterpretable.

------------------------------------------------------------------ PART 2, the coverage

WHY. Architecture §1.3 defines the prediction as the mean of the five per-seed
conditionals and the uncertainty as the seed spread. The 2026-08-08 decision-log entry
closed the RPS screen on its promotion clause alone and recorded that the reliability
machinery is owed exactly once, to §5.3's ensemble calibration check in the low-exposure
strata. That debt is discharged here.

THIS IS NOT E.4. E.4 measured the regression SLOPE of observed on predicted (0.529 low,
0.664 medium, 0.998 high, 0.841 pooled) — whether the SPREAD of point predictions is
right. Coverage asks whether the stated INTERVALS are honest. Different property, passes
or fails independently, and the two are never conflated here.

WHY COVERAGE IS REPORTED TWICE. The observed wOBA is itself a mean over a finite number
of plate appearances, so it carries its own sampling noise (`claim1_eval.sampling_noise`).
An interval that is asked to cover a NOISY realization must be widened by that noise, or
the ensemble is condemned for the answer key's variance. Reported both ways:
  (a) raw     — mean +/- z * seed_sd, against observed. Honest about the ensemble alone.
  (b) widened — mean +/- z * sqrt(seed_sd^2 + noise_var), against observed. The FAIR test.
(b) is the number to read; (a) is kept because their gap is exactly how much of the
interval width the target's own noise has to supply.

STANDING CAVEAT (architecture §243). 42.7% of the low-exposure stratum sits on the shared
cold-start row, so a low-stratum coverage number is substantially a statement about the
context tower and not about the hitter embedding. Low-stratum coverage is therefore also
reported split by cold-start against trained.

CONSEQUENCE IS BOUNDED IN ADVANCE. Architecture §2.1 line 124: "if calibration fails,
that is a reported limitation, not a rebuild." This module proposes no fix.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.analysis import baseline_ladder_bivariate_eb, claim1_eval
from src.analysis.baseline_ladder_trailing import TRAILING_SEASONS, trailing_window
from src.data.eval_targets import drop_pitcher_batters

DEFAULT_OUT_DIR = "results/model_evaluation"
DEFAULT_ARM = "rebuild_baseline"
SEEDS = (0, 1, 2, 3, 4)

# Two-sided normal quantiles for the nominal levels §12.4 names. Stated explicitly
# rather than computed inline so the report can be checked against them by eye.
Z_BY_LEVEL = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.95: 1.959963984540054}

# Ridge penalties searched by RidgeCV's efficient leave-one-out route INSIDE each
# training fold. The grid is wide because the right penalty is unknown a priori and a
# probe that under-regularizes reports its own overfitting as signal.
RIDGE_ALPHAS = np.logspace(-3, 5, 33)

N_FOLDS = 5
N_BOOT = 2000


# ------------------------------------------------------------------ shared loading

def load_seed_embeddings(checkpoint_dir, arm, seeds=SEEDS):
    """
    The frozen hitter embedding matrix from each seed's checkpoint, as a dict
    {seed: array of shape (n_rows, dim)}.

    Reads the tensor and nothing else — no model is constructed and no forward pass is
    run, so there is no eval-mode hazard here. Row 0 is the reserved cold-start row and
    is returned as-is; callers exclude it by never mapping a hitter to it.
    """
    out = {}
    for seed in seeds:
        path = Path(checkpoint_dir) / f"{arm}_s{seed}.pt"
        assert path.exists(), f"missing checkpoint {path}"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        weight = saved["model"]["embedding.weight"]
        assert isinstance(weight, torch.Tensor) and weight.ndim == 2, \
            f"embedding.weight in {path} is not a 2-D tensor"
        out[seed] = weight.detach().numpy().astype("float64")
    dims = {value.shape for value in out.values()}
    assert len(dims) == 1, f"seeds disagree on embedding shape: {dims}"
    return out


def batter_stands(pa_df, eval_season, n_seasons=TRAILING_SEASONS):
    """
    Which side of the plate each hitter bats from, as C.2 types them: L / R / S.

    Same routine C.2 itself uses (`eb_bivariate_eb.batter_types`, minority-share rule),
    read off the trailing window so the probe groups hitters exactly the way the target
    it decodes was grouped. Hitters with no trailing-window PA (debuts) are typed off the
    eval season, which is the same static-roster-attribute exception `eb.predict`
    documents — handedness is never an outcome.
    """
    hitters_only = drop_pitcher_batters(pa_df)
    window = trailing_window(hitters_only, eval_season, n_seasons)
    window = window[window["in_denominator"]]
    types = eb_bivariate_eb.batter_types(window)

    active = hitters_only[hitters_only["season"] == eval_season]
    debut = eb_bivariate_eb.batter_types(active[active["in_denominator"]])
    debut = debut.rename(columns={"batter_type": "debut_type"})
    merged = debut.merge(types, on="batter", how="outer")
    merged["batter_type"] = merged["batter_type"].fillna(merged["debut_type"])
    assert merged["batter_type"].notna().all(), "a hitter could not be typed L/R/S"
    return merged[["batter", "batter_type"]]


# ------------------------------------------------------------------ E.14 part 1: probe

def eb_posterior_split(pa_df, eval_season):
    """
    The decode TARGET: each hitter's C.2 empirical-Bayes posterior mean platoon split,
    defined as posterior E[wOBA | vs LHP] - posterior E[wOBA | vs RHP].

    Source: `src/analysis/eb_bivariate_eb.py::predict(pa_df, eval_season)`, column
    `pred_woba`, pivoted on `p_throws`. That function is C.2's own scorer — the same call
    whose output `results/baseline_ladder/baseline_ladder_claim1_scores.csv` grades as `eb_bivariate`. Its
    hyper-parameters are the ones recorded in `results/baseline_ladder/eb_prior_parameters.csv`
    (rho 0.652 LHB / 0.719 RHB, `tau2_split_derived` 0.00073228 L / 0.00049177 R); the
    per-hitter posterior itself has no committed file, so it is recomputed here from the
    frozen PA table rather than approximated.

    Returns one row per batter with `true_split`.
    """
    predictions = eb_bivariate_eb.predict(pa_df, eval_season)
    assert set(predictions["p_throws"]) <= {"L", "R"}, "unexpected pitcher hand in C.2 output"
    wide = predictions.pivot(index="batter", columns="p_throws", values="pred_woba")
    # a hitter projected against only one hand has no split to decode
    wide = wide.dropna(subset=["L", "R"])
    out = pd.DataFrame({"batter": wide.index.astype(int).to_numpy(),
                        "true_split": (wide["L"] - wide["R"]).to_numpy()})
    assert np.isfinite(out["true_split"]).all(), "C.2 posterior split is non-finite"
    return out.reset_index(drop=True)


def probe_frame(pa_df, manifest, eval_season):
    """
    Join the decode target to the embedding ROW INDEX of each hitter, dropping cold start.

    `manifest["vocabulary"]` maps batter id -> embedding row for every TRAINED hitter;
    rows run 1..n_hitters and row 0 is reserved (`reserved_hitter_index`). A hitter absent
    from the vocabulary is scored on row 0 at query time, so "in the vocabulary" is exactly
    "has his own vector" and is the inclusion rule.
    """
    vocabulary = {int(batter): int(row) for batter, row in manifest["vocabulary"].items()}
    assert 0 not in set(vocabulary.values()), \
        "row 0 is reserved for cold start and must not appear in the vocabulary"

    target = eb_posterior_split(pa_df, eval_season)
    target = target.merge(batter_stands(pa_df, eval_season), on="batter", how="left")
    assert target["batter_type"].notna().all(), "stand lookup lost a hitter"

    n_all = len(target)
    target["row"] = target["batter"].map(vocabulary)
    trained = target[target["row"].notna()].copy()
    trained["row"] = trained["row"].astype(int)
    assert (trained["row"] > 0).all(), "a trained hitter mapped to the reserved row"
    assert not trained["batter"].duplicated().any(), "duplicate batter in the probe frame"
    return trained.reset_index(drop=True), {"hitters_with_a_eb_split": int(n_all),
                                            "cold_start_excluded": int(n_all - len(trained)),
                                            "hitters_probed": int(len(trained))}


def out_of_fold_decode(features, target, groups, n_folds=N_FOLDS, alphas=RIDGE_ALPHAS):
    """
    Out-of-fold ridge predictions of `target` from `features`, cross-validated by batter.

    GroupKFold on batter id: one row per batter here, so the grouping is degenerate by
    construction, but it is kept because it is the constraint that MATTERS — it makes the
    guarantee explicit and stays correct if a caller ever passes a per-(batter, hand) frame.
    Standardization and the penalty search both live inside the fold, so no fold's fit sees
    its own held-out rows.
    """
    features, target = np.asarray(features, dtype="float64"), np.asarray(target, dtype="float64")
    assert features.ndim == 2 and target.ndim == 1, "features must be 2-D, target 1-D"
    assert len(features) == len(target) == len(groups), \
        f"shape mismatch: {features.shape} {target.shape} {len(groups)}"
    assert len(features) >= n_folds * 2, "too few hitters to cross-validate"

    predicted = np.full(len(target), np.nan)
    for train_index, test_index in GroupKFold(n_splits=n_folds).split(features, target, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        model.fit(features[train_index], target[train_index])
        predicted[test_index] = model.predict(features[test_index])
    assert np.isfinite(predicted).all(), "a hitter received no out-of-fold prediction"
    return predicted


def _correlations(actual, predicted):
    """Pearson and Spearman, returning NaN rather than raising on a degenerate slice."""
    if len(actual) < 3 or np.std(predicted) == 0 or np.std(actual) == 0:
        return float("nan"), float("nan")
    return (float(stats.pearsonr(actual, predicted)[0]),
            float(stats.spearmanr(actual, predicted)[0]))


def probe_group(frame, embeddings, seeds=SEEDS, n_boot=N_BOOT, seed=0, ci=(2.5, 97.5),
                shuffle=False):
    """
    Run the decode on one group of hitters and return the seed-averaged correlations.

    `shuffle=True` permutes the target and changes nothing else — that is the NULL, the
    correlation this exact pipeline reaches at this sample size and dimensionality with
    no signal present.

    The bootstrap resamples HITTERS (which is the cluster: one row per hitter), recomputes
    both correlations on the already-out-of-fold predictions of every seed, and averages
    across seeds inside each draw, so the interval is on the seed-averaged statistic that
    is reported.
    """
    rng = np.random.default_rng(seed)
    target = frame["true_split"].to_numpy(dtype="float64")
    if shuffle:
        target = rng.permutation(target)
    rows = frame["row"].to_numpy()
    groups = frame["batter"].to_numpy()

    per_seed = {}
    for s in seeds:
        matrix = embeddings[s]
        features = matrix[rows]
        assert features.shape == (len(frame), matrix.shape[1]), \
            f"feature shape {features.shape} does not match {(len(frame), matrix.shape[1])}"
        per_seed[s] = out_of_fold_decode(features, target, groups)
    assert len(per_seed) == len(seeds), "a seed produced no decode"

    stacked = np.vstack([per_seed[s] for s in seeds])
    point = np.array([_correlations(target, stacked[i]) for i in range(len(seeds))])

    draws = np.empty((n_boot, 2))
    n = len(frame)
    for b in range(n_boot):
        index = rng.integers(0, n, n)
        picked = np.array([_correlations(target[index], stacked[i][index])
                           for i in range(len(seeds))])
        draws[b] = np.nanmean(picked, axis=0)

    low_p, high_p = np.nanpercentile(draws[:, 0], ci)
    low_s, high_s = np.nanpercentile(draws[:, 1], ci)
    return {
        "n_hitters": int(n),
        "n_seeds": len(seeds),
        "pearson": float(np.nanmean(point[:, 0])),
        "pearson_ci_low": float(low_p),
        "pearson_ci_high": float(high_p),
        "pearson_per_seed": [float(v) for v in point[:, 0]],
        "spearman": float(np.nanmean(point[:, 1])),
        "spearman_ci_low": float(low_s),
        "spearman_ci_high": float(high_s),
        "spearman_per_seed": [float(v) for v in point[:, 1]],
        "target_sd": float(np.std(target, ddof=1)),
    }


def run_probe(pa_df, manifest, embeddings, eval_season, n_boot=N_BOOT, seed=0):
    """
    The full probe: real and null decode, per stand and pooled, plus the expectation check.

    The pre-registered expectation (spec §12.4) is that LHB decode meaningfully and RHB
    decode near zero, tracking E.5's 54%-against-18% recovered-spread asymmetry. "Decodes"
    is read here as: the seed-averaged Pearson CI excludes the null's upper CI bound, which
    is the only reading that survives the finite-sample ceiling the null measures.
    """
    frame, counts = probe_frame(pa_df, manifest, eval_season)

    # This is the assertion the module's method rests on: correlations are averaged across
    # seeds, raw embeddings never are, because the seeds' coordinate systems are unrelated.
    assert isinstance(embeddings, dict) and len(embeddings) == len(SEEDS), \
        "the probe decodes per seed; it must be handed one embedding matrix per seed"

    groups = {"pooled": frame}
    for stand in ("L", "R", "S"):
        part = frame[frame["batter_type"] == stand]
        if len(part) >= N_FOLDS * 2:
            groups[stand] = part

    results = {}
    for name, part in groups.items():
        real = probe_group(part, embeddings, n_boot=n_boot, seed=seed, shuffle=False)
        null = probe_group(part, embeddings, n_boot=n_boot, seed=seed + 1, shuffle=True)
        beats = bool(real["pearson_ci_low"] > null["pearson_ci_high"])
        results[name] = {"real": real, "null": null, "decodes_above_null": beats}

    lhb = results.get("L", {}).get("decodes_above_null")
    rhb = results.get("R", {}).get("decodes_above_null")
    if lhb is None or rhb is None:
        held, note = None, "a stand group was too small to decode"
    elif lhb and not rhb:
        held, note = True, "LHB decode above null, RHB do not — the pre-registered pattern"
    elif lhb and rhb:
        held, note = False, ("BOTH stands decode above null: platoon information is PRESENT "
                             "in the representation, so E.5's shortfall is a failure of the "
                             "composition, not of the embedding — the more interesting result")
    elif not lhb and not rhb:
        held, note = False, ("NEITHER stand decodes above null: consistent with the §1.4 "
                             "no-interaction failure mode at the representation level")
    else:
        held, note = False, "RHB decode above null and LHB do not — the expectation inverted"

    return {
        "target": {
            "source_module": "src/analysis/eb_bivariate_eb.py::predict",
            "column": "pred_woba, pivoted on p_throws, split = L - R",
            "quantity": "C.2 empirical-Bayes POSTERIOR MEAN split (not the raw observed split)",
            "hyper_parameters_committed_at": "results/baseline_ladder/eb_prior_parameters.csv",
        },
        "features": {
            "source": "results/checkpoints/rebuild_baseline_s{0..4}.pt, key model.embedding.weight",
            "vocabulary": "data/processed/phase_d5/manifest.json::vocabulary",
            "seed_handling": "per-seed decode, correlations averaged; raw embeddings NEVER averaged",
            "embedding_dim": int(next(iter(embeddings.values())).shape[1]),
        },
        "exclusions": counts,
        "cross_validation": f"GroupKFold({N_FOLDS}) on batter id, RidgeCV inside each fold",
        "by_stand": results,
        "expectation": {
            "pre_registered": ("LHB decodes meaningfully, RHB near zero, tracking E.5's "
                               "54% vs 18% recovered spread"),
            "held": held,
            "note": note,
        },
    }


# --------------------------------------------------------------- E.14 part 2: coverage

def ensemble_frame(pa_df, seed_predictions, manifest, eval_season):
    """
    The eval frame with the ensemble mean and the seed spread attached.

    Built by `claim1_eval.build_eval_frame` on the ENSEMBLE MEAN prediction, so the
    min-PA filter, the strata, and the target sampling noise are the project's own and are
    not reimplemented here. The per-seed standard deviation is a separate join keyed on
    (batter, season, p_throws), asserted complete.

    `cold_start` marks a group whose batter has no vocabulary row and is therefore scored
    on the shared reserved embedding (architecture §243's 42.7% caveat).
    """
    key = claim1_eval.KEY
    assert len(seed_predictions) >= 2, "a seed spread needs at least two seeds"
    stacked = None
    for seed, table in sorted(seed_predictions.items()):
        assert set(key + ["pred_woba"]).issubset(table.columns), f"seed {seed} is missing columns"
        part = table[key + ["pred_woba"]].rename(columns={"pred_woba": f"pred_s{seed}"})
        stacked = part if stacked is None else stacked.merge(part, on=key, how="inner")
    seed_columns = [column for column in stacked.columns if column.startswith("pred_s")]
    assert len(seed_columns) == len(seed_predictions), "the seed join dropped a seed"
    for seed, table in seed_predictions.items():
        assert len(stacked) == len(table), \
            f"seed {seed} does not cover the same rows as the others ({len(table)} vs {len(stacked)})"

    values = stacked[seed_columns].to_numpy(dtype="float64")
    assert np.isfinite(values).all(), "a per-seed prediction is non-finite"
    stacked["pred_woba"] = values.mean(axis=1)
    # ddof=1: the five seeds are a SAMPLE of the init distribution, not the population
    stacked["seed_sd"] = values.std(axis=1, ddof=1)

    frame, coverage = claim1_eval.build_eval_frame(pa_df, stacked[key + ["pred_woba"]],
                                                   eval_season)
    before = len(frame)
    frame = frame.merge(stacked[key + ["seed_sd"]], on=key, how="left")
    assert len(frame) == before, "the seed-sd join changed the row count"
    assert frame["seed_sd"].notna().all(), "a scored group has no seed spread"
    assert frame["noise_var"].notna().all(), "a scored group has no target sampling noise"

    vocabulary = {int(batter) for batter in manifest["vocabulary"]}
    frame["cold_start"] = ~frame["batter"].astype(int).isin(vocabulary)
    return frame, coverage


def wilson_interval(successes, n, level=0.95):
    """
    Wilson score interval on a binomial rate. Preferred over the Wald form because the
    empirical coverage rates here sit near 0 and 1, where Wald intervals leave the unit
    interval and would report a coverage bound above 100%.
    """
    assert 0 <= successes <= n and n > 0, f"invalid binomial input {successes}/{n}"
    z = stats.norm.ppf(0.5 + level / 2)
    phat = successes / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def covered(frame, z, widen):
    """
    Boolean cover indicator for the interval mean +/- z * width.

    widen=False -> width is the seed spread alone (report (a)).
    widen=True  -> width is sqrt(seed_sd^2 + noise_var), the seed spread convolved with the
                   TARGET's own sampling noise (report (b), the fair test), because the
                   observed wOBA the interval must cover is itself a finite-PA mean.
    """
    sd = frame["seed_sd"].to_numpy(dtype="float64")
    if widen:
        sd = np.sqrt(sd ** 2 + frame["noise_var"].to_numpy(dtype="float64"))
    return np.abs(frame["woba"].to_numpy(dtype="float64")
                  - frame["pred_woba"].to_numpy(dtype="float64")) <= z * sd


def coverage_rows(frame, label, levels=Z_BY_LEVEL):
    """One reliability-diagram row per (nominal level, interval kind) for one slice."""
    rows = []
    for nominal, z in sorted(levels.items()):
        for kind, widen in (("seed_only", False), ("seed_plus_target_noise", True)):
            hit = covered(frame, z, widen)
            n = int(len(hit))
            successes = int(hit.sum())
            low, high = wilson_interval(successes, n) if n else (float("nan"), float("nan"))
            rows.append({
                "slice": label,
                "n_groups": n,
                "nominal": nominal,
                "z": z,
                "interval": kind,
                "empirical": float(successes / n) if n else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "gap": float(successes / n - nominal) if n else float("nan"),
                "mean_half_width": float(np.mean(
                    z * (np.sqrt(frame["seed_sd"] ** 2 + frame["noise_var"]) if widen
                         else frame["seed_sd"]))) if n else float("nan"),
            })
    return rows


def run_coverage(frame):
    """
    The coverage table: every stratum, pooled, and the low stratum split by cold start.

    Returns (reliability DataFrame, verdict dict). The pre-registered expectation is that
    intervals are TOO NARROW in the low stratum, because a five-seed spread measures
    disagreement between seeds and not the shared shrinkage all five inherit from the same
    empirical-Bayes prior. It is judged on the FAIR interval (b): under-coverage on the raw
    interval alone would partly be the answer key's noise, which is not the ensemble's fault.
    """
    rows = []
    for name in claim1_eval.STRATUM_NAMES:
        part = frame[frame["stratum"] == name]
        if len(part):
            rows += coverage_rows(part, name)
    rows += coverage_rows(frame, "all")

    low = frame[frame["stratum"] == "low"]
    for flag, label in ((True, "low_cold_start"), (False, "low_trained")):
        part = low[low["cold_start"] == flag]
        if len(part):
            rows += coverage_rows(part, label)
    table = pd.DataFrame(rows)

    fair = table[(table["slice"] == "low") & (table["interval"] == "seed_plus_target_noise")]
    assert len(fair) == len(Z_BY_LEVEL), "the low stratum lost a nominal level"
    too_narrow = bool((fair["ci_high"] < fair["nominal"]).all())
    verdict = {
        "pre_registered": ("intervals are too narrow in the low-exposure stratum, because a "
                           "5-seed spread measures seed disagreement, not the shared shrinkage "
                           "all five inherit from the same empirical-Bayes prior"),
        "held": too_narrow,
        "judged_on": "the fair interval (seed spread convolved with target sampling noise)",
        "low_stratum_fair": fair[["nominal", "empirical", "ci_low", "ci_high"]]
                            .to_dict(orient="records"),
        "cold_start_share_of_low": float(low["cold_start"].mean()) if len(low) else float("nan"),
        "note": ("architecture §243: a large share of the low stratum sits on the shared "
                 "cold-start row, so the low-stratum number is substantially a statement "
                 "about the context tower, not about the hitter embedding"),
        "not_a_fix": ("architecture §2.1 line 124 — if calibration fails that is a reported "
                      "limitation, not a rebuild. No remedy is proposed here."),
    }
    return table, verdict


# ---------------------------------------------------------------------------- driver

ASSUMPTIONS = [
    "The C.2 posterior mean split is treated as the hitter's TRUE split. It is an estimate, "
    "shrunk toward the league split, so it is smoother than the truth; the probe therefore "
    "measures agreement with a shrunk target and is conservative about idiosyncratic hitters "
    "and generous about the league-average component.",
    "A LINEAR read-out is taken as the test of whether platoon information is present. A "
    "non-linear probe could find structure a ridge cannot, so a null probe bounds LINEARLY "
    "DECODABLE information only.",
    "Seeds are treated as exchangeable draws whose embedding coordinate systems are NOT "
    "aligned, so correlations are averaged and raw embeddings are never averaged.",
    "The seed standard deviation is computed with ddof=1 over five seeds: the ensemble is a "
    "sample of the init distribution. With n=5 that estimate is itself noisy.",
    "The interval is Gaussian: mean +/- z * sd with normal quantiles. Five seeds cannot "
    "identify a tail shape, so this is an assumption, not a measurement.",
    "The target's sampling noise is taken as independent of the ensemble spread when the "
    "two are convolved in quadrature. They share no data-generating channel, so this is "
    "reasonable, but it is not verified here.",
    "claim1_eval.sampling_noise estimates each group's within-group wOBA variance from the "
    "eval season itself, so for a group with few scorable PA the noise estimate is noisy.",
    "Cold-start hitters are excluded from the probe entirely, so the probe speaks only about "
    "hitters who own a trained embedding row.",
    "The POOLED probe row is the weakest of the four to interpret: LHB and RHB splits have "
    "opposite signs, so pooled target sd (0.0263) is roughly double either within-stand sd "
    "(0.0127 L / 0.0093 R) and a pooled decode can score by recovering STAND alone. The "
    "per-stand rows are the ones that speak to platoon SKILL; pooled is reported for "
    "completeness, not as the headline.",
]


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.14 — the §5.1 probe and the ensemble interval coverage.")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d5")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # the guard that keeps 2025 out of every path but --final-run
    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    assert not args.final_run, "E.14 is a retrospective diagnostic and never scores the test season"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    pa_df = pd.read_parquet(args.eval_targets)
    seed_predictions = {
        seed: pd.read_csv(f"results/model_v1/model_v1_predictions_{args.arm}_s{seed}.csv")
        for seed in SEEDS
    }
    embeddings = load_seed_embeddings(args.checkpoint_dir, args.arm)

    print("E.14 part 1 — the §5.1 probe (retrospective diagnostic)")
    probe = run_probe(pa_df, manifest, embeddings, args.eval_season,
                      n_boot=args.n_boot, seed=args.seed)

    print("E.14 part 2 — ensemble interval coverage")
    frame, eval_coverage = ensemble_frame(pa_df, seed_predictions, manifest, args.eval_season)
    table, verdict = run_coverage(frame)
    table.to_csv(out_dir / "coverage.csv", index=False)

    coverage_report = {
        "arm": args.arm,
        "eval_season": args.eval_season,
        "sources": {
            "per_seed_predictions": [f"results/model_v1/model_v1_predictions_{args.arm}_s{s}.csv"
                                     for s in SEEDS],
            "prediction_column": "pred_woba",
            "eval_targets": args.eval_targets,
            "eval_frame_builder": "src/analysis/claim1_eval.py::build_eval_frame "
                                  f"(MIN_EVAL_PA={claim1_eval.MIN_EVAL_PA}, "
                                  f"strata={list(claim1_eval.STRATUM_NAMES)})",
            "target_noise": "src/analysis/claim1_eval.py::sampling_noise -> noise_var",
            "manifest": str(Path(args.data_dir) / "manifest.json"),
        },
        "z_by_level": {str(level): z for level, z in sorted(Z_BY_LEVEL.items())},
        "interval_kinds": {
            "seed_only": "(a) mean +/- z*seed_sd against observed wOBA — the ensemble alone",
            "seed_plus_target_noise": "(b) mean +/- z*sqrt(seed_sd^2 + noise_var) — the FAIR test",
        },
        "eval_frame_coverage": eval_coverage,
        "n_scored_groups": int(len(frame)),
        "mean_seed_sd": float(frame["seed_sd"].mean()),
        "mean_target_noise_sd": float(np.sqrt(frame["noise_var"]).mean()),
        "reliability": table.to_dict(orient="records"),
        "expectation": verdict,
        "not_calibration": ("E.4 measured the SLOPE of observed on predicted (0.529 low / 0.664 medium / "
                   "0.998 high / 0.841 pooled). This step measures INTERVAL HONESTY. They are "
                   "different properties and can pass or fail independently."),
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage_report, indent=2, default=float))

    probe_report = {
        "arm": args.arm,
        "eval_season": args.eval_season,
        "status": "retrospective diagnostic — demoted from the pre-registered early detector; "
                  "it explains E.5 and gates nothing",
        "probe": probe,
        "assumptions": ASSUMPTIONS,
    }
    (out_dir / "coverage_probe.json").write_text(json.dumps(probe_report, indent=2, default=float))

    coverage_report["assumptions"] = ASSUMPTIONS
    (out_dir / "coverage.json").write_text(json.dumps(coverage_report, indent=2, default=float))

    print(f"wrote {out_dir / 'coverage_probe.json'}, {out_dir / 'coverage.json'}, "
          f"{out_dir / 'coverage.csv'}")


if __name__ == "__main__":
    main()
