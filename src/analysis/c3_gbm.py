"""
Phase C.3 — XGBoost on hitter x context aggregates (architecture §3 Phase C.3).

The last Phase C baseline, and the one frozen rule #1 names last: "XGBoost with
context-interaction features". It is the strongest incumbent claim 1 must beat,
so it is deliberately built strong rather than convenient.

WHAT "context-interaction features" HAS TO MEAN HERE. §3 B.2 already hit the
tension and recorded it: the 48-dim context vector (src/features/context_features)
carries no hitter identity by design — that is the hitter tower's job — so context
alone cannot emit a hitter-level number, which is what claim 1 grades. C.3 is
therefore built at the unit the metric scores: one row per
(batter, target season, pitcher hand), whose features are the hitter's OWN
prior-window rates SLICED BY CONTEXT (pitcher hand, pitch group, two-strike
count state). Pitcher hand is a feature rather than two separate models, so the
trees learn hand x feature interactions directly — that interaction is the whole
point of the baseline. This reading is an interpretation of the plan, not a
frozen decision (decision log 2026-07-28).

TWO FEATURE SETS, both fitted and both reported:

  outcome — trailing side-specific wOBA and PA (own side and opposite side),
            batter hand, seasons of service. The same information C.1 and C.2
            saw; isolates "does a GBM beat empirical Bayes on identical inputs?"
  full    — outcome + process aggregates (swing/whiff rates, EV/LA/spray) and
            their context slices (by pitch group, by two-strike count).

  outcome -> full  measures whether PROCESS carries hitter-level projection
                   signal, which is the Phase D premise measured cheaply and in
                   advance. B.1 put process stabilization at 2-7x faster than
                   side-specific wOBA, so a baseline denied process features
                   would be a strawman, and frozen rule #1 exists to prevent
                   exactly that.

LEAKAGE DISCIPLINE (the gate that matters most here — rolling-window features
are the classic silent offender). Every feature for a row with target season s is
computed from seasons strictly in [s - 3, s), the same 3-season window C.1 and
C.2 use. Enforced in `window_features` by assertion, not by convention. The
evaluated season 2024 appears in no training row, as a target or in any feature
window; `build_training_frame` asserts it.

TUNING PROTOCOL. Training rows are target seasons 2016-2022; the number of
boosting rounds is chosen by early stopping on target season 2023, held out from
fitting. The evaluated season is never read for model selection. The final model
is then refit on 2016-2023 at the chosen round count, because a projection task
should not throw away its most recent season — the round count is the only
quantity reused, and that reuse is stated here rather than hidden (the same
accepted-and-declared compromise as B.2's shared VAL).

Hyperparameters are pre-registered small-data settings (depth 3, heavy
regularization) fixed in PARAMS before the first run: ~12k training rows is a
small-data GBM, and a tuned-on-the-eval-frame hyperparameter search is precisely
the leak the protocol above exists to prevent.

Training loss is PA-weighted squared error, matching the frozen §5.2 metric.
Targets are the OBSERVED side-specific wOBA, never C.2's posterior: training on
the incumbent's output makes C.2 the teacher and the head-to-head circular.

Missingness is never imputed. A hitter with no prior PA vs that hand gets NaN
features and XGBoost's learned default direction handles them, which is the
honest treatment of "we have no information about this hitter" and keeps C.3
covering exactly the same groups as C.1 and C.2 (a prerequisite for the paired
bootstrap, which asserts identical group sets).
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.eval_targets import drop_pitcher_batters

TRAILING_SEASONS = 3

# Round count is the only thing chosen by data; everything else is fixed here
# before the first run. Small-data settings: shallow trees, strong shrinkage.
PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5.0,   # in normalized-weight units (mean weight = 1)
    reg_lambda=5.0,
    n_estimators=2000,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=1,               # fixed thread count: reproducibility gate depends on it
)
EARLY_STOPPING_ROUNDS = 50
INNER_VAL_SEASON = 2023     # held out from fitting to choose the round count
MIN_TARGET_SEASON = 2016    # first season with any prior season to build features from

# Statcast pitch_type -> the three context groups the whiff/EV slices use.
# IN (intentional ball) is excluded from every process aggregate: it is not a
# swing decision, and including it deflates swing rate in proportion to a
# hitter's intentional-walk rate, which is a team's choice, not his skill.
PITCH_GROUPS = {
    "fastball": {"FF", "SI", "FC", "FA"},
    "breaking": {"SL", "CU", "KC", "ST", "SV", "CS", "KN", "SC", "EP"},
    "offspeed": {"CH", "FS", "FO"},
}
HARD_HIT_EV = 95.0  # the standard Statcast hard-hit threshold

KEY = ["batter", "season", "p_throws"]

OUTCOME_FEATURES = [
    "vs_lhp", "bats_left", "bats_switch",
    "own_pa", "own_woba", "opp_pa", "opp_woba", "total_pa", "seasons_seen",
]
PROCESS_FEATURES = [
    "own_pitches", "swing_rate", "whiff_rate",
    "bbip", "ev_mean", "la_mean", "spray_mean", "hard_hit_rate",
    "whiff_rate_fastball", "whiff_rate_breaking", "whiff_rate_offspeed",
    "ev_mean_fastball", "ev_mean_nonfastball",
    "swing_rate_two_strike", "whiff_rate_two_strike",
    "pooled_pitches", "pooled_whiff_rate", "pooled_ev_mean",
]
FEATURE_SETS = {
    "outcome": OUTCOME_FEATURES,
    "full": OUTCOME_FEATURES + PROCESS_FEATURES,
}


# --------------------------------------------------------------------------
# season-level aggregation: sums only, so a window is a plain sum of seasons
# --------------------------------------------------------------------------

def season_outcome(pa_df):
    """
    Per (batter, season, p_throws) wOBA numerator and denominator, plus the side
    the batter stood on. Sums rather than rates: a trailing window is then a sum
    over seasons and the rate is computed once, exactly, at the end.

    `stand` is taken per (batter, season, PITCHER HAND), not per season: a switch
    hitter stands L against RHP and R against LHP, so a per-season mode would
    label him by whichever hand he happened to face more (almost always RHP) and
    silently turn every switch hitter into a right-handed batter.
    """
    scored = pa_df[pa_df["in_denominator"]]
    return scored.groupby(KEY, as_index=False).agg(
        woba_points=("woba_points", "sum"),
        pa=("in_denominator", "sum"),
        stand=("stand", lambda s: s.mode().iat[0]),
    )


def season_process(pitch_df):
    """
    Per (batter, season, p_throws) process COUNTS and SUMS from the labeled pitch
    table: swings, whiffs, balls in play, EV/LA/spray totals, and the same broken
    out by pitch group and by two-strike count. Rates are derived after the
    window sum, so a hitter's window rate is the pooled rate, not a mean of
    seasonal rates (which would weight a 20-PA season like a 600-PA one).
    """
    df = pitch_df[pitch_df["pitch_type"] != "IN"].copy()
    group = pd.Series("other", index=df.index)
    for name, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = name
    df["pitch_group"] = group
    # `contact` is deliberately NULL on non-swings (labels.py nesting invariant), so a
    # whiff is a swing whose contact label is explicitly 0 — never a negated null.
    df["swing"] = df["swing"].astype(bool)
    assert df.loc[df["swing"], "contact"].notna().all(), "a swing carries no contact label"
    df["whiff"] = df["swing"] & (df["contact"].fillna(1).astype(int) == 0)
    df["two_strike"] = df["strikes"] == 2
    df["in_play"] = df["ev"].notna()
    df["hard_hit"] = df["ev"] >= HARD_HIT_EV

    agg = {
        "pitches": ("swing", "size"),
        "swings": ("swing", "sum"),
        "whiffs": ("whiff", "sum"),
        "bbip": ("in_play", "sum"),
        "ev_sum": ("ev", "sum"),
        "la_sum": ("la", "sum"),
        "spray_sum": ("spray", "sum"),
        "spray_n": ("spray", "count"),
        "hard_hits": ("hard_hit", "sum"),
        "two_strike_pitches": ("two_strike", "sum"),
    }
    out = df.groupby(KEY, as_index=False).agg(**agg)

    two = df[df["two_strike"]].groupby(KEY, as_index=False).agg(
        two_strike_swings=("swing", "sum"), two_strike_whiffs=("whiff", "sum"))
    out = out.merge(two, on=KEY, how="left")

    for name in PITCH_GROUPS:
        part = df[df["pitch_group"] == name]
        by_group = part.groupby(KEY, as_index=False).agg(**{
            f"swings_{name}": ("swing", "sum"),
            f"whiffs_{name}": ("whiff", "sum"),
            f"bbip_{name}": ("in_play", "sum"),
            f"ev_sum_{name}": ("ev", "sum"),
        })
        out = out.merge(by_group, on=KEY, how="left")

    count_columns = [c for c in out.columns if c not in KEY]
    out[count_columns] = out[count_columns].fillna(0.0).astype(float)
    return out


# --------------------------------------------------------------------------
# window features
# --------------------------------------------------------------------------

def _window_sum(season_table, target_season, n_seasons, keys):
    """Sum a season-level count table over [target_season - n_seasons, target_season)."""
    first = target_season - n_seasons
    window = season_table[(season_table["season"] >= first)
                          & (season_table["season"] < target_season)]
    assert window.empty or window["season"].max() < target_season, \
        f"feature window leaked target season {target_season}"
    value_columns = [c for c in window.columns
                     if c not in ("season", "stand") and c not in keys]
    return window.groupby(keys, as_index=False)[value_columns].sum()


def _rate(numerator, denominator):
    """Rate with an explicit NaN where there is no denominator — never a silent 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)


def window_features(groups, outcome_seasons, process_seasons, target_season,
                    n_seasons=TRAILING_SEASONS):
    """
    Build one feature row per (batter, p_throws) in `groups`, from seasons
    strictly before target_season.
    groups: frame with batter/p_throws needing a projection.
    Returns `groups` plus every column in FEATURE_SETS["full"].
    """
    outcome = _window_sum(outcome_seasons, target_season, n_seasons, ["batter", "p_throws"])
    process = _window_sum(process_seasons, target_season, n_seasons, ["batter", "p_throws"])

    frame = groups[["batter", "p_throws"]].drop_duplicates().copy()
    frame = frame.merge(outcome, on=["batter", "p_throws"], how="left")

    # opposite-side outcome: the same table joined on the flipped hand
    opposite = outcome.copy()
    opposite["p_throws"] = opposite["p_throws"].map({"L": "R", "R": "L"})
    frame = frame.merge(opposite.rename(columns={"woba_points": "opp_points", "pa": "opp_pa"}),
                        on=["batter", "p_throws"], how="left")

    # batter hand and service, from the window (a switch hitter shows both stands)
    window_stand = outcome_seasons[(outcome_seasons["season"] >= target_season - n_seasons)
                                   & (outcome_seasons["season"] < target_season)]
    hand = window_stand.groupby("batter")["stand"].agg(lambda s: set(s)).rename("stands")
    seasons_seen = window_stand.groupby("batter")["season"].nunique().rename("seasons_seen")
    frame = frame.merge(hand, on="batter", how="left").merge(seasons_seen, on="batter", how="left")
    stands = frame["stands"].apply(lambda s: s if isinstance(s, set) else set())
    frame["bats_switch"] = stands.apply(lambda s: float(len(s) > 1))
    frame["bats_left"] = stands.apply(lambda s: float(s == {"L"}))
    frame = frame.drop(columns=["stands"])

    frame["own_pa"] = frame["pa"].fillna(0.0)
    frame["opp_pa"] = frame["opp_pa"].fillna(0.0)
    frame["total_pa"] = frame["own_pa"] + frame["opp_pa"]
    frame["own_woba"] = _rate(frame["woba_points"], frame["own_pa"])
    frame["opp_woba"] = _rate(frame["opp_points"], frame["opp_pa"])
    frame["vs_lhp"] = (frame["p_throws"] == "L").astype(float)
    frame["seasons_seen"] = frame["seasons_seen"].fillna(0.0)

    frame = frame.merge(process, on=["batter", "p_throws"], how="left")
    frame["own_pitches"] = frame["pitches"].fillna(0.0)
    frame["bbip"] = frame["bbip"].fillna(0.0)
    frame["swing_rate"] = _rate(frame["swings"], frame["own_pitches"])
    frame["whiff_rate"] = _rate(frame["whiffs"], frame["swings"])
    frame["ev_mean"] = _rate(frame["ev_sum"], frame["bbip"])
    frame["la_mean"] = _rate(frame["la_sum"], frame["bbip"])
    frame["spray_mean"] = _rate(frame["spray_sum"], frame["spray_n"])
    frame["hard_hit_rate"] = _rate(frame["hard_hits"], frame["bbip"])
    frame["swing_rate_two_strike"] = _rate(frame["two_strike_swings"], frame["two_strike_pitches"])
    frame["whiff_rate_two_strike"] = _rate(frame["two_strike_whiffs"], frame["two_strike_swings"])
    for name in PITCH_GROUPS:
        frame[f"whiff_rate_{name}"] = _rate(frame[f"whiffs_{name}"], frame[f"swings_{name}"])
    frame["ev_mean_fastball"] = _rate(frame["ev_sum_fastball"], frame["bbip_fastball"])
    frame["ev_mean_nonfastball"] = _rate(
        frame["ev_sum_breaking"] + frame["ev_sum_offspeed"],
        frame["bbip_breaking"] + frame["bbip_offspeed"])

    # pooled (both hands) backbone: the more stable estimate of the same quantities
    pooled = _window_sum(process_seasons, target_season, n_seasons, ["batter"])
    pooled = pooled.rename(columns={"pitches": "pooled_pitches"})
    pooled["pooled_whiff_rate"] = _rate(pooled["whiffs"], pooled["swings"])
    pooled["pooled_ev_mean"] = _rate(pooled["ev_sum"], pooled["bbip"])
    frame = frame.merge(pooled[["batter", "pooled_pitches", "pooled_whiff_rate", "pooled_ev_mean"]],
                        on="batter", how="left")
    frame["pooled_pitches"] = frame["pooled_pitches"].fillna(0.0)

    out = frame[["batter", "p_throws"] + FEATURE_SETS["full"]]
    assert not out.duplicated(["batter", "p_throws"]).any(), "duplicate feature rows"
    assert len(out) == len(groups[["batter", "p_throws"]].drop_duplicates()), \
        "feature build changed the group count"
    return out


# --------------------------------------------------------------------------
# training / prediction frames
# --------------------------------------------------------------------------

def build_row_set(pa_df, outcome_seasons, process_seasons, target_season,
                  n_seasons=TRAILING_SEASONS, with_target=True):
    """
    Feature rows for every (batter, hand) active in target_season, with the
    observed side-specific wOBA and PA attached when with_target.
    """
    active = pa_df[pa_df["season"] == target_season]
    assert len(active) > 0, f"no PA in season {target_season}"
    groups = active[["batter", "p_throws"]].drop_duplicates()

    rows = window_features(groups, outcome_seasons, process_seasons, target_season, n_seasons)
    rows.insert(1, "season", target_season)

    if with_target:
        observed = outcome_seasons[outcome_seasons["season"] == target_season]
        rows = rows.merge(observed[KEY + ["woba_points", "pa"]], on=KEY, how="inner")
        rows["target_woba"] = rows["woba_points"] / rows["pa"]
        rows = rows.drop(columns=["woba_points"])
    return rows


def build_training_frame(pa_df, outcome_seasons, process_seasons, eval_season,
                         n_seasons=TRAILING_SEASONS, min_target_season=MIN_TARGET_SEASON):
    """
    Stacked feature rows for every target season strictly before eval_season.
    The split-boundary guard for the whole module: nothing at or after
    eval_season enters training, as a target or through a feature window.
    """
    seasons = sorted(s for s in outcome_seasons["season"].unique()
                     if min_target_season <= s < eval_season)
    assert seasons, f"no training target seasons before {eval_season}"

    frames = [build_row_set(pa_df, outcome_seasons, process_seasons, season, n_seasons)
              for season in seasons]
    stacked = pd.concat(frames, ignore_index=True)

    assert stacked["season"].max() < eval_season, "training frame contains the evaluated season"
    assert (stacked["pa"] > 0).all(), "training rows must have held-out PA to weight by"
    return stacked


def _matrix(rows, feature_set):
    columns = FEATURE_SETS[feature_set]
    matrix = rows[columns].astype(float)
    assert matrix.shape == (len(rows), len(columns)), \
        f"feature matrix shape {matrix.shape} != {(len(rows), len(columns))}"
    return matrix


def _weights(pa):
    """PA weights normalized to mean 1, so min_child_weight is in row units."""
    weight = np.asarray(pa, dtype=float)
    assert weight.sum() > 0, "total PA weight is zero"
    return weight / weight.mean()


def fit(training, feature_set="full", inner_val_season=INNER_VAL_SEASON, seed=0,
        params=None, n_rounds=None):
    """
    Fit the C.3 GBM: choose the round count by early stopping on
    inner_val_season, then refit on all training rows at that count.
    n_rounds forces the count and skips the search (used by the wiring gate).
    Returns (model, n_rounds).
    """
    assert feature_set in FEATURE_SETS, f"unknown feature set {feature_set!r}"
    assert inner_val_season in set(training["season"]), \
        f"inner-val season {inner_val_season} is not in the training frame"
    settings = dict(PARAMS if params is None else params)
    settings["random_state"] = seed

    if n_rounds is None:
        inner_train = training[training["season"] < inner_val_season]
        inner_val = training[training["season"] == inner_val_season]
        assert len(inner_train) > 0 and len(inner_val) > 0, "empty inner split"

        searcher = xgb.XGBRegressor(**settings, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                    eval_metric="rmse")
        searcher.fit(
            _matrix(inner_train, feature_set), inner_train["target_woba"],
            sample_weight=_weights(inner_train["pa"]),
            eval_set=[(_matrix(inner_val, feature_set), inner_val["target_woba"])],
            sample_weight_eval_set=[_weights(inner_val["pa"])],
            verbose=False,
        )
        n_rounds = int(searcher.best_iteration) + 1

    # refit on everything at the chosen round count: a projection model should not
    # discard its most recent season. Round count is the only quantity reused.
    settings["n_estimators"] = n_rounds
    model = xgb.XGBRegressor(**settings)
    model.fit(_matrix(training, feature_set), training["target_woba"],
              sample_weight=_weights(training["pa"]), verbose=False)
    return model, n_rounds


def predict(pa_df, process_seasons, eval_season, feature_set="full", seed=0,
            n_seasons=TRAILING_SEASONS, return_model=False):
    """
    Project side-specific wOBA for every hitter-hand active in eval_season.
    pa_df: PA-level eval targets; process_seasons: output of season_process on the
    labeled pitch table. Returns batter/season/p_throws/pred_woba, ready for
    claim1_eval — the same object C.1 and C.2 emit, over the same groups.
    """
    # pitchers taking their own at-bats are not the population claim 1 projects,
    # and their ~.15 wOBA rows would dominate a squared-error fit
    pa_df = drop_pitcher_batters(pa_df)

    outcome_seasons = season_outcome(pa_df)
    training = build_training_frame(pa_df, outcome_seasons, process_seasons,
                                    eval_season, n_seasons)
    model, n_rounds = fit(training, feature_set=feature_set, seed=seed)

    rows = build_row_set(pa_df, outcome_seasons, process_seasons, eval_season,
                         n_seasons, with_target=False)
    rows["pred_woba"] = model.predict(_matrix(rows, feature_set))

    assert rows["pred_woba"].notna().all(), "predictions contain nulls"
    assert np.isfinite(rows["pred_woba"]).all(), "predictions contain non-finite values"
    out = rows[KEY + ["pred_woba"]]
    return (out, model, n_rounds) if return_model else out


def shuffled_training(training, seed):
    """
    Training rows with the hitter -> target link destroyed, for the label-shuffle
    null: every feature row keeps its features but is handed another row's outcome.

    The (target, PA) PAIR is permuted together, and that is the whole subtlety.
    Permuting the target alone leaves each weight with a random target, which
    changes the PA-weighted mean of the labels — and that mean is the model's
    intercept. In this project's data it moves the intercept from .3186 to .2836,
    which lands much closer to the true level of LOW-EXPOSURE hitters (~.294) than
    league average does, so the "null" model wins the low stratum for a reason
    that has nothing to do with the link being broken. A null that hands the null
    model a free advantage is not a null. The assertion below is the guard.
    """
    order = np.random.default_rng(seed).permutation(len(training))
    shuffled = training.copy()
    shuffled[["target_woba", "pa"]] = training[["target_woba", "pa"]].to_numpy()[order]

    before = np.average(training["target_woba"], weights=training["pa"])
    after = np.average(shuffled["target_woba"], weights=shuffled["pa"])
    assert np.isclose(before, after), \
        f"shuffle moved the intercept ({before:.4f} -> {after:.4f}); permute the pair, not the target"
    return shuffled


def shuffle_null(pa_df, process_seasons, eval_season, feature_set="full", seeds=(0, 1, 2, 3, 4),
                 n_seasons=TRAILING_SEASONS):
    """
    Prediction frames from models trained on shuffled labels — the calibrated null
    for "is C.3's margin larger than what this fitting procedure produces from no
    hitter information at all?". Scored the same way as the real model, so the
    comparison is like for like. Returns {seed: predictions}.
    """
    pa_df = drop_pitcher_batters(pa_df)
    outcome_seasons = season_outcome(pa_df)
    training = build_training_frame(pa_df, outcome_seasons, process_seasons,
                                    eval_season, n_seasons)
    rows = build_row_set(pa_df, outcome_seasons, process_seasons, eval_season,
                         n_seasons, with_target=False)

    out = {}
    for seed in seeds:
        model, _ = fit(shuffled_training(training, seed), feature_set=feature_set, seed=seed)
        frame = rows[KEY].copy()
        frame["pred_woba"] = model.predict(_matrix(rows, feature_set))
        out[seed] = frame
    return out


def feature_importance(model, feature_set="full"):
    """Gain importance, reported for interpretation only — it is in-sample and
    biased toward high-cardinality splits (same caveat B.2 records). The
    outcome-vs-full comparison, not this table, is the evidence."""
    gains = model.get_booster().get_score(importance_type="gain")
    columns = FEATURE_SETS[feature_set]
    return (pd.DataFrame({"feature": columns,
                          "gain": [gains.get(c, 0.0) for c in columns]})
            .sort_values("gain", ascending=False, ignore_index=True))
