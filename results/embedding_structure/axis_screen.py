"""
Axis screen: candidate per-hitter scalars for an embedding figure, scored by
how well the FROZEN hitter embedding predicts each candidate's tercile,
held out (5-fold LDA CV).

Two kinds of candidate:
- marginal: per-hitter average of something the model has a head for
  (swing, contact, ev, la, spray heads in src/model/v1.py). Baselines.
- contrast: a difference between two context slices for the SAME hitter.
  No head predicts these directly -- the model must compose them from the
  hitter x context interaction. These are the interesting candidates.

Reads a frozen checkpoint and existing results only. No retraining, no
model/pipeline changes. Season filtered to <=2024 throughout (no 2025 data).

Run: PYTHONPATH=. .venv/bin/python results/embedding_structure/axis_screen.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

from src.analysis import embedding_structure as es

SEED = 7
N_SPLITS = 5
MIN_N = 300
SEASON_MAX = 2024
PITCH_EVENTS_PATH = "data/processed/pitch_events_labeled.parquet"
OUT_DIR = Path("results/embedding_structure")

# Same fastball/breaking Statcast pitch_type groups as
# src/analysis/baseline_ladder_gbm.PITCH_GROUPS, copied here for the same
# reason embedding_alt_views.py copies them: avoid that module's xgboost
# import. Statcast zone codes 1-9 are in-zone, 11-14 are out-of-zone
# (chase region) -- same convention hitter_stats.csv's chase_rate /
# zone_swing_rate already use.
PITCH_GROUPS = {
    "fastball": {"FF", "SI", "FC", "FA"},
    "breaking": {"SL", "CU", "KC", "ST", "SV", "CS", "KN", "SC", "EP"},
}

MARGINAL_COLUMNS = [
    "swing_rate", "contact_rate", "chase_rate", "zone_swing_rate",
    "ev_p90", "la_mean", "pull_rate", "woba_level",
]

MIN_SWINGS_PER_SLICE = 20   # rate contrasts over swings (whiff, count/zone splits)
MIN_PITCHES_PER_SLICE = 20  # rate contrasts over all pitches (swing-rate splits)
MIN_BATTED_BALLS_PER_SLICE = 15  # ev/la contrasts (batted-ball only)
MIN_SWINGS_FOR_SLOPE = 50   # velo_whiff_slope needs enough spread in release_speed


# --------------------------------------------------------------------- pitch-level loading

def load_pitch_events(batters):
    cols = ["batter", "pitch_type", "swing", "contact", "ev", "la", "balls",
            "strikes", "stand", "p_throws", "plate_z", "zone", "release_speed",
            "season"]
    df = pd.read_parquet(PITCH_EVENTS_PATH, columns=cols)
    df = df[df["season"] <= SEASON_MAX]
    df = df[df["batter"].isin(batters)]
    group = pd.Series(np.nan, index=df.index, dtype=object)
    for name, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = name
    df = df.assign(pitch_group=group)
    return df


def slice_diff_by_batter(df, mask_a, mask_b, value_col, min_n):
    """Per-batter mean(value_col) under mask_a minus under mask_b, restricted
    to batters with >= min_n non-null observations in BOTH slices."""
    sub = df.loc[mask_a | mask_b, ["batter", value_col]].copy()
    sub = sub.dropna(subset=[value_col])
    sub["grp"] = np.where(mask_a.loc[sub.index], "a", "b")
    counts = sub.groupby(["batter", "grp"]).size().unstack("grp", fill_value=0)
    means = sub.groupby(["batter", "grp"])[value_col].mean().unstack("grp")
    valid = (counts.get("a", 0) >= min_n) & (counts.get("b", 0) >= min_n)
    diff = (means["a"] - means["b"])[valid]
    return diff


def whiff_col(df):
    return (df["contact"] == 0).astype(float)


# --------------------------------------------------------------------- contrast builders

def build_contrasts(hitters):
    batters = set(hitters["batter"])
    df = load_pitch_events(batters)
    swings = df[df["swing"] == 1].copy()
    swings["whiff"] = whiff_col(swings)

    out = {}

    # 1. whiff_brk_minus_fb -- reuse the existing worked example.
    from src.analysis.embedding_alt_views import whiff_brk_minus_fb_by_batter
    out["whiff_brk_minus_fb"] = whiff_brk_minus_fb_by_batter(PITCH_EVENTS_PATH)
    # apply the same min-swings-per-group threshold as everything else here
    grp = swings.dropna(subset=["pitch_group"])
    counts = grp.groupby(["batter", "pitch_group"]).size().unstack(fill_value=0)
    valid = (counts.get("fastball", 0) >= MIN_SWINGS_PER_SLICE) & \
            (counts.get("breaking", 0) >= MIN_SWINGS_PER_SLICE)
    out["whiff_brk_minus_fb"] = out["whiff_brk_minus_fb"][
        out["whiff_brk_minus_fb"].index.isin(valid[valid].index)]

    # 2. selectivity = zone_swing_rate - chase_rate (already-computed marginals)
    out["selectivity"] = (hitters.set_index("batter")["zone_swing_rate"]
                           - hitters.set_index("batter")["chase_rate"]).dropna()

    # 3. two_strike_gear = swing_rate(2 strikes) - swing_rate(<2 strikes)
    mask_a = df["strikes"] >= 2
    mask_b = df["strikes"] < 2
    out["two_strike_gear"] = slice_diff_by_batter(df, mask_a, mask_b, "swing",
                                                    MIN_PITCHES_PER_SLICE)

    # 4. count_leverage_swing = swing_rate(behind) - swing_rate(ahead)
    mask_behind = df["strikes"] > df["balls"]
    mask_ahead = df["balls"] > df["strikes"]
    out["count_leverage_swing"] = slice_diff_by_batter(
        df, mask_behind, mask_ahead, "swing", MIN_PITCHES_PER_SLICE)

    # 5/6. ev_brk_minus_fb, la_brk_minus_fb -- batted balls only (ev/la notna)
    mask_fb = swings["pitch_group"] == "fastball"
    mask_brk = swings["pitch_group"] == "breaking"
    out["ev_brk_minus_fb"] = slice_diff_by_batter(
        swings, mask_brk, mask_fb, "ev", MIN_BATTED_BALLS_PER_SLICE)
    out["la_brk_minus_fb"] = slice_diff_by_batter(
        swings, mask_brk, mask_fb, "la", MIN_BATTED_BALLS_PER_SLICE)

    # 7/8. platoon_whiff, platoon_ev -- opposite-handed pitcher minus same-handed
    mask_opp = swings["stand"] != swings["p_throws"]
    mask_same = swings["stand"] == swings["p_throws"]
    out["platoon_whiff"] = slice_diff_by_batter(
        swings, mask_opp, mask_same, "whiff", MIN_SWINGS_PER_SLICE)
    out["platoon_ev"] = slice_diff_by_batter(
        swings, mask_opp, mask_same, "ev", MIN_BATTED_BALLS_PER_SLICE)

    # 9. zone_height_swing = swing_rate(upper third) - swing_rate(lower third),
    # plate_z terciles computed over ALL pitches (not per-batter).
    z_tercile = pd.qcut(df["plate_z"], 3, labels=False, duplicates="drop")
    mask_upper = z_tercile == 2
    mask_lower = z_tercile == 0
    out["zone_height_swing"] = slice_diff_by_batter(
        df, mask_upper, mask_lower, "swing", MIN_PITCHES_PER_SLICE)

    # 10. velo_whiff_slope = per-hitter OLS slope of whiff (0/1) on
    # release_speed, over swings only.
    sw = swings.dropna(subset=["release_speed"])
    def ols_slope(g):
        x = g["release_speed"].to_numpy()
        y = g["whiff"].to_numpy()
        if len(x) < MIN_SWINGS_FOR_SLOPE or np.var(x) == 0:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])
    out["velo_whiff_slope"] = sw.groupby("batter").apply(ols_slope, include_groups=False).dropna()

    # 11. EXTRA (defensible addition): two_strike_chase = chase_rate(2 strikes)
    # - chase_rate(<2 strikes). Distinct from #3: #3 is overall swing rate by
    # count leverage; this isolates the DISCIPLINE component (swings specifically
    # on out-of-zone pitches) rather than the mixed in+out-of-zone swing rate.
    # It's a genuine count x zone interaction the model has no head for -- the
    # swing head sees count/zone as input context but chase_rate itself is
    # never a supervised target, so this is composed, not read off a head.
    out_of_zone = df["zone"] >= 11
    chase_df = df[out_of_zone].assign(chase=df.loc[out_of_zone, "swing"])
    out["two_strike_chase"] = slice_diff_by_batter(
        chase_df, chase_df["strikes"] >= 2, chase_df["strikes"] < 2, "chase",
        MIN_SWINGS_PER_SLICE)

    return out


# --------------------------------------------------------------------- scoring

def lda_cv(normalized, tercile, seed=SEED, n_splits=N_SPLITS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in kf.split(normalized):
        model = LinearDiscriminantAnalysis()
        model.fit(normalized[train_idx], tercile[train_idx])
        accs.append(model.score(normalized[test_idx], tercile[test_idx]))
    return float(np.mean(accs)), float(np.std(accs))


def score_candidate(name, kind, values_by_batter, hitters_idx, normalized, hitters,
                     definition, notes):
    aligned = hitters_idx["batter"].map(values_by_batter)
    mask = aligned.notna().to_numpy()
    n = int(mask.sum())
    if n < MIN_N:
        return None, n
    vals = aligned[mask].to_numpy()
    tercile = es.tercile_labels(pd.Series(vals)).to_numpy()
    sub_norm = normalized[mask]
    lda_mean, lda_std = lda_cv(sub_norm, tercile)

    sub_hitters = hitters[mask]
    stand01 = (sub_hitters["stand"] == "L").astype(float).to_numpy()
    corrs = {}
    for label, other in [("r_ev_p90", sub_hitters["ev_p90"]),
                          ("r_log_prior_pa", sub_hitters["log_prior_pa"]),
                          ("r_stand", pd.Series(stand01, index=sub_hitters.index)),
                          ("r_swing_rate", sub_hitters["swing_rate"])]:
        r, _ = pearsonr(vals, other.to_numpy())
        corrs[label] = float(r)

    entry = {
        "kind": kind, "n": n,
        "lda_cv_mean": lda_mean, "lda_cv_std": lda_std,
        **corrs,
        "definition": definition, "notes": notes,
    }
    return entry, n, vals, mask


DEFINITIONS = {
    "swing_rate": ("Per-hitter rate of swinging at any pitch.",
                    "From hitter_stats.csv (already computed marginal)."),
    "contact_rate": ("Per-hitter rate of making contact given a swing.",
                      "From hitter_stats.csv."),
    "chase_rate": ("Per-hitter swing rate on pitches outside the strike zone (zone codes 11-14).",
                    "From hitter_stats.csv."),
    "zone_swing_rate": ("Per-hitter swing rate on pitches inside the strike zone (zone codes 1-9).",
                         "From hitter_stats.csv."),
    "ev_p90": ("Per-hitter 90th-percentile exit velocity.", "From hitter_stats.csv."),
    "la_mean": ("Per-hitter mean launch angle.", "From hitter_stats.csv."),
    "pull_rate": ("Per-hitter rate of pulling batted balls.", "From hitter_stats.csv."),
    "woba_level": ("Per-hitter empirical-Bayes wOBA level.", "From hitter_stats.csv."),
    "whiff_brk_minus_fb": ("Whiff rate on breaking balls minus on fastballs, over swings.",
                            f"Min {MIN_SWINGS_PER_SLICE} swings per pitch group; reused from embedding_alt_views.whiff_brk_minus_fb_by_batter."),
    "selectivity": ("zone_swing_rate minus chase_rate: swings on strikes minus swings on non-strikes.",
                     "Derived directly from the two existing hitter_stats.csv marginals, no new pitch-level pull."),
    "two_strike_gear": ("Swing rate with 2 strikes minus swing rate with fewer than 2 strikes.",
                         f"Min {MIN_PITCHES_PER_SLICE} pitches per slice."),
    "count_leverage_swing": ("Swing rate when behind in the count (strikes>balls) minus when ahead (balls>strikes).",
                              f"Min {MIN_PITCHES_PER_SLICE} pitches per slice; even counts excluded from both slices."),
    "ev_brk_minus_fb": ("Mean exit velocity on breaking balls minus on fastballs.",
                         f"Batted balls only (ev notna); min {MIN_BATTED_BALLS_PER_SLICE} per group."),
    "la_brk_minus_fb": ("Mean launch angle on breaking balls minus on fastballs.",
                         f"Batted balls only (la notna); min {MIN_BATTED_BALLS_PER_SLICE} per group."),
    "platoon_whiff": ("Whiff rate vs opposite-handed pitchers minus vs same-handed pitchers.",
                       f"Swings only; min {MIN_SWINGS_PER_SLICE} per group. No switch-hitter ambiguity: pitch-level `stand` is the side actually batted from."),
    "platoon_ev": ("Mean exit velocity vs opposite-handed pitchers minus vs same-handed pitchers.",
                    f"Batted balls only; min {MIN_BATTED_BALLS_PER_SLICE} per group."),
    "zone_height_swing": ("Swing rate on upper-third pitches minus on lower-third pitches.",
                           f"plate_z terciles computed over ALL pitches (global, not per-batter); min {MIN_PITCHES_PER_SLICE} pitches per group."),
    "velo_whiff_slope": ("Per-hitter OLS slope of whiff (0/1) on release_speed, over swings only.",
                          f"Min {MIN_SWINGS_FOR_SLOPE} swings with non-null release_speed and nonzero variance."),
    "two_strike_chase": ("Chase rate (swing on out-of-zone pitch) with 2 strikes minus chase rate with fewer than 2 strikes.",
                          f"Out-of-zone pitches only (zone>=11); min {MIN_SWINGS_PER_SLICE} per slice. Added because it isolates discipline "
                          "under two-strike pressure specifically on non-strikes, distinct from #3's mixed in+out-of-zone swing rate -- "
                          "a count x zone interaction with no dedicated model head."),
}


def main():
    emb, hitters, _ = es.load_hitters(es.DEFAULT_CHECKPOINT_DIR, es.DEFAULT_ARM,
                                       es.HITTER_STATS_PATH, es.NAMES_PATH)
    normalized, _ = es.unit_normalize(emb)

    contrasts = build_contrasts(hitters)

    candidates = {}
    final_series = {}

    for col in MARGINAL_COLUMNS:
        values_by_batter = hitters.set_index("batter")[col]
        definition, notes = DEFINITIONS[col]
        result = score_candidate(col, "marginal", values_by_batter, hitters,
                                  normalized, hitters, definition, notes)
        entry = result[0]
        if entry is None:
            print(f"SKIP {col}: n={result[1]} < {MIN_N}")
            continue
        candidates[col] = entry
        vals, mask = result[2], result[3]
        s = pd.Series(np.nan, index=hitters.index)
        s[mask] = vals
        final_series[col] = s

    for name, series in contrasts.items():
        definition, notes = DEFINITIONS[name]
        result = score_candidate(name, "contrast", series, hitters,
                                  normalized, hitters, definition, notes)
        entry = result[0]
        if entry is None:
            print(f"SKIP {name}: n={result[1]} < {MIN_N}")
            continue
        candidates[name] = entry
        vals, mask = result[2], result[3]
        s = pd.Series(np.nan, index=hitters.index)
        s[mask] = vals
        final_series[name] = s

    corr_df = pd.DataFrame(final_series)
    correlation_matrix = corr_df.corr(method="pearson", min_periods=MIN_N // 2)

    output = {
        "candidates": candidates,
        "candidate_correlation_matrix": {
            "columns": correlation_matrix.columns.tolist(),
            "matrix": correlation_matrix.round(4).where(pd.notna(correlation_matrix), None).values.tolist(),
        },
        "method": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "min_n": MIN_N,
            "season_filter": f"season <= {SEASON_MAX}",
            "pitch_groups": {k: sorted(v) for k, v in PITCH_GROUPS.items()},
            "min_swings_per_slice": MIN_SWINGS_PER_SLICE,
            "min_pitches_per_slice": MIN_PITCHES_PER_SLICE,
            "min_batted_balls_per_slice": MIN_BATTED_BALLS_PER_SLICE,
            "min_swings_for_slope": MIN_SWINGS_FOR_SLOPE,
            "embedding_checkpoint_dir": es.DEFAULT_CHECKPOINT_DIR,
            "embedding_arm": es.DEFAULT_ARM,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "axis_screen.json", "w") as f:
        json.dump(output, f, indent=2)

    ranked = sorted(
        [(k, v["lda_cv_mean"]) for k, v in candidates.items() if v["kind"] == "contrast"],
        key=lambda t: -t[1])
    print("\nContrast candidates ranked by lda_cv_mean:")
    for name, mean in ranked:
        print(f"  {name}: {mean:.3f} (n={candidates[name]['n']})")


if __name__ == "__main__":
    main()
