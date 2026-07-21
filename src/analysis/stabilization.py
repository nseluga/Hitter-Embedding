"""
Split-half reliability / stabilization analysis (Phase B step 1).

Measures how much of a metric's between-hitter variance is signal vs. noise at a
given sample size, i.e. how many observations until the metric "stabilizes". This
ranks candidate metrics by signal-per-PA, the currency of the small-sample thesis:
the headline question is whether process metrics (whiff rate, exit velocity)
stabilize faster than the outcome we project (side-specific wOBA).

Method (classical test theory). For sample size n, each qualifying hitter's n
observations are split into two random halves of n/2. The Pearson correlation r
of the half-metrics across hitters is the reliability of an n/2-sized measurement;
Spearman-Brown projects it up to n:  reliability(n) = 2r / (1 + r). Algebraically
this equals signal / (signal + noise/n), the reliability of an n-observation
estimate. The stabilization point is the n where reliability crosses 0.5 (half the
variance is signal) -- the widely-used baseball convention (Carleton / Baseball
Prospectus reliability work; unverified, to source before it is cited in the paper).

The metric is supplied as a function of a slice of observations, so it covers both
simple means (exit velocity) and ratios (wOBA = sum(points)/sum(denominator)). The
caller pre-filters to the metric's natural observation unit (swings for whiff, PAs
vs. a given pitcher hand for side-specific wOBA) and to train seasons.
"""

import numpy as np
import pandas as pd


def spearman_brown(r):
    """Project a split-half correlation to full-length reliability: 2r / (1 + r)."""
    return 2.0 * r / (1.0 + r)


def mean_metric(column):
    """Metric fn for a simple average, e.g. exit velocity or a 0/1 rate."""
    return lambda observations: observations[column].mean()


def ratio_metric(numerator, denominator):
    """Metric fn for a ratio like wOBA = sum(points) / sum(denominator); nan if denom is 0."""
    def compute(observations):
        denom = observations[denominator].sum()
        return observations[numerator].sum() / denom if denom > 0 else np.nan
    return compute


def _pearson(a, b):
    """Pearson correlation over finite pairs; nan if under 2 usable pairs or no spread."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def reliability_at(groups, metric_fn, n, rng, n_resamples=1, split="random"):
    """
    Spearman-Brown corrected split-half reliability of an n-observation estimate.

    groups: dict of hitter_id -> that hitter's observations (index reset).
    Splits n sampled rows into halves of n//2, computes the metric on each half,
    and correlates the halves across hitters. Returns (reliability, n_qualifying).

    split="random" permutes each hitter's rows before halving -- the two halves are
    drawn from the same circumstances, so this measures within-sample consistency
    (an optimistic upper bound). split="sequential" takes the first n rows in their
    existing (chronological) order and splits early-half vs. late-half; the halves
    are separated in time, so this measures across-circumstance reliability -- the
    projection-relevant number (Carleton, "Reliably Stable"). n_resamples>1 only
    affects "random"; "sequential" is deterministic so it runs a single split.
    """
    half = n // 2
    qualifying = [g for g in groups.values() if len(g) >= n]
    if half < 1 or len(qualifying) < 2:
        return np.nan, len(qualifying)

    resamples = 1 if split == "sequential" else n_resamples
    reliabilities = []
    for _ in range(resamples):
        half_a, half_b = [], []
        for group in qualifying:
            order = np.arange(n) if split == "sequential" else rng.permutation(len(group))[:n]
            half_a.append(metric_fn(group.iloc[order[:half]]))
            half_b.append(metric_fn(group.iloc[order[half:2 * half]]))
        r = _pearson(half_a, half_b)
        if np.isfinite(r):
            reliabilities.append(spearman_brown(r))

    reliability = float(np.mean(reliabilities)) if reliabilities else np.nan
    return reliability, len(qualifying)


def stabilization_curve(df, group_col, metric_fn, n_grid, seed=0, n_resamples=1, split="random"):
    """
    Reliability vs. sample size for one metric. Returns a DataFrame with columns
    n, reliability, n_groups. n_groups is the qualifying-hitter count at each n and
    falls as n grows (survivorship) -- report it wherever the curve is reported.
    split is passed through to reliability_at ("random" or "sequential").
    """
    rng = np.random.default_rng(seed)
    groups = {gid: sub.reset_index(drop=True) for gid, sub in df.groupby(group_col)}
    rows = []
    for n in n_grid:
        reliability, n_groups = reliability_at(groups, metric_fn, n, rng, n_resamples, split)
        rows.append({"n": n, "reliability": reliability, "n_groups": n_groups})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance-components estimator (Fix B): one signal/noise decomposition over ALL
# hitters, instead of a split-half at each n that only uses hitters with >= n
# observations. This removes the survivorship bias at large n, yields an analytic
# reliability(n) curve and stabilization point, and supports a bootstrap CI --
# the split-half method gives none of those. Method: one-way random-effects ANOVA
# (method of moments) with unequal group sizes, the classical route to Cronbach's
# alpha / KR-21 (FanGraphs, "A New Way to Look at Sample Size"). Handles only
# mean-type metrics: feed a 0/1 or continuous per-observation value column. A ratio
# like wOBA is a mean of woba_points over its denominator PAs, so restrict to
# denominator rows and pass woba_points.
# ---------------------------------------------------------------------------


def hitter_stats(df, group_col, value_col):
    """
    Per-hitter sufficient statistics for the variance decomposition: count, mean,
    and within-hitter sum of squares. Returns a DataFrame (one row per hitter) that
    is cheap to bootstrap-resample without re-touching the raw observations.
    """
    grouped = df.groupby(group_col)[value_col]
    stats = grouped.agg(n="count", mean="mean", var=lambda x: x.var(ddof=0))
    stats["ss"] = stats["var"] * stats["n"]  # sum of squares = pop-variance * n
    return stats[["n", "mean", "ss"]].reset_index(drop=True)


def variance_components(stats):
    """
    Estimate (signal_var, noise_var) from per-hitter stats via one-way ANOVA.
    signal_var is between-hitter true-talent variance; noise_var is per-observation
    within-hitter variance. signal_var is clamped at 0 (a negative moment estimate
    means the signal is indistinguishable from zero). nan if under 2 usable hitters.
    """
    n_i = stats["n"].to_numpy(dtype=float)
    mean_i = stats["mean"].to_numpy(dtype=float)
    ss_i = stats["ss"].to_numpy(dtype=float)
    k = len(n_i)
    total = n_i.sum()
    if k < 2 or total <= k:  # need >1 hitter and some within-hitter df
        return np.nan, np.nan

    grand_mean = float((n_i * mean_i).sum() / total)
    ssw = float(ss_i.sum())
    ssb = float((n_i * (mean_i - grand_mean) ** 2).sum())
    msw = ssw / (total - k)                      # estimates noise_var
    msb = ssb / (k - 1)                          # estimates noise_var + n0 * signal_var
    n0 = (total - (n_i ** 2).sum() / total) / (k - 1)
    signal_var = max((msb - msw) / n0, 0.0)
    return signal_var, msw


def reliability_vc(signal_var, noise_var, n):
    """Analytic reliability of an n-observation mean: signal / (signal + noise/n)."""
    if not np.isfinite(signal_var) or signal_var <= 0:
        return 0.0
    return signal_var / (signal_var + noise_var / n)


def stabilization_point_vc(signal_var, noise_var, threshold=0.5):
    """
    Analytic sample size where reliability_vc crosses `threshold`:
    n* = (noise/signal) * (threshold / (1 - threshold)). inf if there is no signal.
    At threshold 0.5 this is noise/signal -- the regression-to-the-mean ballast, the
    league-average PA-equivalent you would add when shrinking (equal weight point).
    """
    if not np.isfinite(signal_var) or signal_var <= 0:
        return np.inf
    return (noise_var / signal_var) * (threshold / (1.0 - threshold))


def stabilization_ci(stats, threshold=0.5, n_boot=500, seed=0, ci=(2.5, 97.5)):
    """
    Bootstrap percentile CI for the variance-components stabilization point, resampling
    hitters (the independent unit) with replacement. Returns (point, lo, hi); the point
    is the estimate on the full sample. Bootstrap draws that yield no signal (n*=inf)
    are kept as inf so the upper bound honestly reflects "may never stabilize".
    """
    signal_var, noise_var = variance_components(stats)
    point = stabilization_point_vc(signal_var, noise_var, threshold)
    rng = np.random.default_rng(seed)
    k = len(stats)
    values = stats.to_numpy()
    draws = []
    for _ in range(n_boot):
        sample = pd.DataFrame(values[rng.integers(0, k, k)], columns=stats.columns)
        s, no = variance_components(sample)
        draws.append(stabilization_point_vc(s, no, threshold))
    lo, hi = np.percentile(draws, ci)  # percentile of a set with inf stays finite unless >ci% are inf
    return point, float(lo), float(hi)


def stabilization_point(curve, threshold=0.5):
    """
    Sample size where reliability first crosses `threshold`, linearly interpolated
    between the bracketing grid points. nan if the curve never reaches threshold.
    """
    usable = curve.dropna(subset=["reliability"]).sort_values("n").reset_index(drop=True)
    previous = None
    for _, row in usable.iterrows():
        if row["reliability"] >= threshold:
            if previous is None:
                return float(row["n"])
            span_r = row["reliability"] - previous["reliability"]
            if span_r <= 0:
                return float(row["n"])
            frac = (threshold - previous["reliability"]) / span_r
            return float(previous["n"] + frac * (row["n"] - previous["n"]))
        previous = row
    return np.nan


def _fmt(x, unit):
    if not np.isfinite(x):
        return "not reached"
    return f"{x:.0f} {unit}"


def _report(label, df, value_col, unit, seed, resamples):
    """
    Print the full panel for one metric: the variance-components stabilization point
    with a bootstrap CI at both the 0.5 (equal-weight-with-prior) and 0.7 (strict
    "reliable") thresholds, plus split-half cross-checks -- random (optimistic) and
    sequential (across-time, projection-relevant). df must already be filtered.
    """
    grid = [10, 25, 50, 100, 200, 400, 800]
    stats = hitter_stats(df, "batter", value_col)
    n_hitters = len(stats)
    print(f"\n{label}  [{n_hitters} hitters]")
    for threshold in (0.5, 0.7):
        point, lo, hi = stabilization_ci(stats, threshold=threshold, seed=seed)
        print(f"  variance-components n* (r={threshold}): {_fmt(point, unit)}  95% CI [{_fmt(lo, unit)}, {_fmt(hi, unit)}]")
    metric = mean_metric(value_col)
    rand = stabilization_point(stabilization_curve(df, "batter", metric, grid, seed=seed, n_resamples=resamples), 0.5)
    seq = stabilization_point(stabilization_curve(df, "batter", metric, grid, seed=seed, split="sequential"), 0.5)
    print(f"  split-half n* (r=0.5): random {_fmt(rand, unit)} (optimistic) | sequential {_fmt(seq, unit)} (across-time)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase B stabilization panel: process vs. outcome metrics on train seasons.")
    parser.add_argument("--labeled", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--max-train-season", type=int, default=2023)
    parser.add_argument("--resamples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = lambda d: d[d["season"] <= args.max_train_season]

    # process metrics, one per model head, each in its natural observation unit
    labeled = pd.read_parquet(args.labeled, columns=["batter", "season", "p_throws", "swing", "contact", "ev", "la", "spray"])
    labeled = train(labeled)
    swings = labeled[labeled["swing"] == 1].copy()
    swings["whiff"] = (swings["contact"] == 0).astype(float)
    in_play = labeled[labeled["ev"].notna()]
    sprayed = labeled[labeled["spray"].notna()]

    _report("swing rate (per pitch) -> swing head", labeled, "swing", "pitches", args.seed, args.resamples)
    _report("whiff rate (per swing) -> contact head", swings, "whiff", "swings", args.seed, args.resamples)
    _report("exit velocity (per ball-in-play) -> quality head", in_play, "ev", "BBIP", args.seed, args.resamples)
    _report("launch angle (per ball-in-play) -> quality head", in_play, "la", "BBIP", args.seed, args.resamples)
    _report("spray / hit direction (per ball-in-play) -> quality head", sprayed, "spray", "BBIP", args.seed, args.resamples)

    # Fix C(a): whiff and EV sliced by pitcher hand, so process is measured on the
    # same side-specific slice as the outcome -- the matched comparison.
    for hand, hand_label in [("L", "vs LHP"), ("R", "vs RHP")]:
        _report(f"whiff rate {hand_label} (matched slice)", swings[swings["p_throws"] == hand], "whiff", "swings", args.seed, args.resamples)
        _report(f"exit velocity {hand_label} (matched slice)", in_play[in_play["p_throws"] == hand], "ev", "BBIP", args.seed, args.resamples)

    # outcome metric: side-specific wOBA, each pitcher hand (denominator PAs only)
    pa = train(pd.read_parquet(args.eval_targets, columns=["batter", "season", "p_throws", "woba_points", "in_denominator"]))
    pa = pa[pa["in_denominator"] == 1]
    for hand, hand_label in [("L", "vs LHP"), ("R", "vs RHP")]:
        _report(f"wOBA {hand_label} (per PA) -> outcome", pa[pa["p_throws"] == hand], "woba_points", "PA", args.seed, args.resamples)


if __name__ == "__main__":
    main()
