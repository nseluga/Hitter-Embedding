"""
Phase D.5 — the four auxiliary tables the query machinery needs (docs/phase-d5-spec.md §2).

v1 predicts p(swing), p(contact | swing), and contact quality on balls in play. None of
those is enough to finish a plate appearance, so D.5 supplies what the network does not:

  1. TAKE OUTCOMES. p(ball, called strike, hit by pitch | take, plate_x, plate_z), fit as
     four surfaces, one per (stand, p_throws). Three classes rather than ball/strike
     because HBP is a take, a terminal state, and inside the wOBA denominator -- a
     two-class model drops it and under-predicts every hitter (2026-08-08).

  2. THE CONTACT SPLIT, baseline form. p(foul, foul_tip, in_play | contact) from a league
     table indexed by (balls, strikes, in_zone). This is the stand-in until the three-class
     head is trained (2026-08-08). It is conditioned on count AND zone deliberately: those
     are the two effects with any documented magnitude, so a baseline carrying both makes
     the retrained comparison isolate the HITTER-specific part rather than rediscovering
     league structure the model would get for free.

  3. PITCHER REPERTOIRES. Real pitch rows grouped by (pitcher, stand, balls, strikes)
     (2026-08-08). Resampling whole rows rather than generating them is what keeps pitch
     physics, missingness flags, and the stand/count one-hots mutually consistent -- and
     because the cell already matches the query, no field is ever overwritten.

  4. THE BATTED-BALL OUTCOME TABLE. V[ev, la, spray] over {OUT, 1B, 2B, 3B, HR}. Stored as
     CATEGORY PROBABILITIES, not wOBA points, so a table fit on 2015-2023 gets scored under
     the eval season's weights instead of baking in a stale vintage. SF folds into OUT: both
     score zero and both sit inside the denominator.

ROW ALIGNMENT is the load-bearing assumption. `pitcher`, `description`, and `zone` are
deliberately outside the model's context (context_features.py:51-52), so they have to come
back from the labeled parquet. `model_dataset.build` applies one order-preserving filter to
that parquet, so re-reading it and applying the same filter reproduces the tensor row order
exactly. That is asserted, not assumed -- a silent misalignment here would attach every
pitch's outcome to a different pitch.

SMOOTHING. One estimator serves all four tables: shrink a cell toward its own coarser pool
with weight n/(n+alpha), where alpha is the variance-components ratio noise/signal that
`stabilization.py` already computes for the equal-weight point. A search found no published
empirical-Bayes treatment of pitch-usage or batted-ball-outcome tables, so this follows the
project's own C.2 precedent rather than an outside source, and that is stated rather than
implied (docs/phase-d5-spec.md §5.2).
"""

import numpy as np
import pandas as pd

from src.analysis.stabilization import stabilization_point_vc, variance_components
from src.data.eval_targets import primarily_pitchers
from src.data.model_dataset import MASKED, drop_pitcher_at_bats

# outcome classes, in the fixed order every array in this module uses
CONTACT_CLASSES = ("foul", "foul_tip", "in_play")
TAKE_CLASSES = ("ball", "called_strike", "hit_by_pitch")
BIP_CLASSES = ("OUT", "1B", "2B", "3B", "HR")

# `intent_ball` is excluded from every table and from the resampling pool: IBB sits outside
# the wOBA denominator (eval_targets.py:61), so a simulator that generated intentional balls
# would produce plate appearances the metric does not count (docs/phase-d5-spec.md §4.3)
EXCLUDED_DESCRIPTIONS = ("intent_ball",)
BALL_DESCRIPTIONS = ("ball", "blocked_ball")
CONTACT_DESCRIPTION_TO_CLASS = {"foul": 0, "foul_tip": 1, "hit_into_play": 2}
# the one description that is a ball in play, named so the D5-R15 unmeasured-share branch and
# the contact split cannot disagree about what "in play" means
IN_PLAY_DESCRIPTION = "hit_into_play"
TAKE_DESCRIPTION_TO_CLASS = {"ball": 0, "blocked_ball": 0, "called_strike": 1,
                             "hit_by_pitch": 2}

# SF scores zero and stays in the denominator, so it is an OUT for the outcome table
BIP_CATEGORY_TO_CLASS = {"OUT": 0, "SF": 0, "1B": 1, "2B": 2, "3B": 3, "HR": 4}

# take surface grid, in feet. ~1.2 inch resolution: the strike-zone edge is the whole
# point of this surface, so a coarse grid would blur exactly the region that matters.
# Values outside the box are clipped into the edge cells rather than dropped.
TAKE_GRID_X = (-2.5, 2.5, 0.1)
TAKE_GRID_Z = (0.0, 5.0, 0.1)

N_BALLS, N_STRIKES = 4, 3
JOIN_KEYS = ["game_pk", "at_bat_number"]
ALIGN_COLUMNS = ["batter", "season", "pitcher", "description", "zone", "balls", "strikes",
                 "stand", "p_throws", "plate_x", "plate_z", "game_pk", "at_bat_number"]


def align_pitch_frame(pitch_events_path, eval_targets_path, season_tensor):
    """
    The labeled parquet, filtered and ordered to match the built Phase D tensors row for row.
    season_tensor: the tensors' season array, used as the alignment assertion.
    Returns a DataFrame carrying the columns the model's context deliberately excludes.
    """
    frame = pd.read_parquet(pitch_events_path, columns=ALIGN_COLUMNS)
    excluded = primarily_pitchers(pd.read_parquet(eval_targets_path))
    frame = drop_pitcher_at_bats(frame, excluded).reset_index(drop=True)

    season = np.asarray(season_tensor)
    assert len(frame) == len(season), \
        f"parquet has {len(frame)} rows against the tensors' {len(season)}; rebuild"
    assert bool((frame["season"].to_numpy() == season).all()), \
        "parquet and tensor row order disagree; every join downstream would be off by rows"
    return frame


def shrink(counts, alpha):
    """
    Shrink each cell's class distribution toward the pooled distribution over all cells.
    counts: (n_cells, n_classes) integer counts. Returns probabilities of the same shape.
    A cell with no observations returns the pool exactly, which is the intended backoff.
    """
    counts = np.asarray(counts, dtype="float64")
    totals = counts.sum(axis=1, keepdims=True)
    pool = counts.sum(axis=0)
    pool = pool / pool.sum() if pool.sum() > 0 else np.full(counts.shape[1],
                                                            1.0 / counts.shape[1])
    return (counts + alpha * pool) / (totals + alpha)


def fit_alpha(counts):
    """
    The shrinkage weight, as the variance-components ratio noise/signal per class.
    counts: (n_cells, n_classes). Returns the median over classes, so one loud class
    cannot set the smoothing for the whole table. inf (a class with no between-cell
    signal) is dropped before the median rather than propagating.
    """
    counts = np.asarray(counts, dtype="float64")
    totals = counts.sum(axis=1)
    usable = totals > 0
    if usable.sum() < 2:
        return 1.0

    alphas = []
    for klass in range(counts.shape[1]):
        rate = counts[usable, klass] / totals[usable]
        # a 0/1 indicator's population variance is p(1-p), so the sufficient statistics
        # come straight from the counts without revisiting the raw rows
        stats = pd.DataFrame({"n": totals[usable], "mean": rate,
                              "ss": totals[usable] * rate * (1.0 - rate)})
        signal, noise = variance_components(stats)
        point = stabilization_point_vc(signal, noise, threshold=0.5)
        if np.isfinite(point):
            alphas.append(point)
    return float(np.median(alphas)) if alphas else 1.0


def _grid_index(values, spec):
    """Bin index on a fixed grid, clipping out-of-range values into the edge cells."""
    low, high, step = spec
    n = int(round((high - low) / step))
    index = np.floor((np.asarray(values, dtype="float64") - low) / step)
    return np.clip(index, 0, n - 1).astype("int64"), n


def fit_take_surfaces(frame, train_mask):
    """
    p(ball, called strike, hit by pitch | take, location), one surface per handedness cell.
    Returns {(stand, p_throws): (n_x, n_z, 3) probability array} plus the fitted alpha.
    Four surfaces rather than one because taken-pitch location differs markedly across
    handedness combinations (2026-08-08; Deshpande & Wyner 2017 Fig. 3).
    """
    klass = frame["description"].map(TAKE_DESCRIPTION_TO_CLASS)
    usable = train_mask & klass.notna().to_numpy()
    x_index, n_x = _grid_index(frame["plate_x"], TAKE_GRID_X)
    z_index, n_z = _grid_index(frame["plate_z"], TAKE_GRID_Z)

    surfaces, alphas = {}, []
    for stand in ("L", "R"):
        for throws in ("L", "R"):
            cell = usable & (frame["stand"] == stand).to_numpy() \
                & (frame["p_throws"] == throws).to_numpy()
            flat = x_index[cell] * n_z + z_index[cell]
            counts = np.zeros((n_x * n_z, len(TAKE_CLASSES)), dtype="int64")
            np.add.at(counts, (flat, klass[cell].to_numpy(dtype="int64")), 1)
            alpha = fit_alpha(counts)
            alphas.append(alpha)
            surfaces[(stand, throws)] = shrink(counts, alpha).reshape(n_x, n_z,
                                                                     len(TAKE_CLASSES))
    return surfaces, float(np.median(alphas))


def fit_take_count_offsets(surfaces, frame, train_mask, n_iterations=32):
    """
    Per-count multiplicative adjustments to the take surfaces (D5-R1). Returns (4, 3, 3).

    The location surfaces are count-BLIND. They are right on average -- the marginal error is
    +0.00019 -- and wrong in every count, because umpires call the same location differently at
    3-0 than at 0-2, and the count chain visits all twelve counts. This is the LIGHT fix: twelve
    per-count factors on top of the four existing surfaces, rather than 48 separate surfaces.

    Fitted by raking (iterative proportional fitting) against the observed per-count class
    marginals, with the LOCATION shape held fixed. Nothing about where pitches are taken is
    re-estimated and the 2,500 spatial cell counts are untouched, so D5-R14's shrink-toward-the-
    global-marginal anti-prior stays latent rather than being made to bind by thin cells.

    Escalation path, if this fails to close the walk/strikeout gap: 48 separate surfaces, one per
    (stand, p_throws, balls, strikes). That drops average cell counts to ~32, at which point
    shrinking toward a global ~32% called-strike marginal actively hurts and the neighbour-
    smoothing fix becomes mandatory rather than optional.
    """
    klass = frame["description"].map(TAKE_DESCRIPTION_TO_CLASS)
    usable = train_mask & klass.notna().to_numpy()
    rows = np.flatnonzero(usable)
    base = take_probabilities(surfaces, frame, rows)              # (n, 3), count-blind
    observed_class = klass.to_numpy()[rows].astype("int64")
    balls = frame["balls"].to_numpy()[rows]
    strikes = frame["strikes"].to_numpy()[rows]

    offsets = np.ones((N_BALLS, N_STRIKES, len(TAKE_CLASSES)), dtype="float64")
    for b in range(N_BALLS):
        for s in range(N_STRIKES):
            cell = (balls == b) & (strikes == s)
            if not cell.any():
                continue
            predicted = base[cell]
            target = np.bincount(observed_class[cell], minlength=len(TAKE_CLASSES)) / cell.sum()
            factor = np.ones(len(TAKE_CLASSES))
            for _ in range(n_iterations):
                adjusted = predicted * factor
                adjusted /= adjusted.sum(axis=1, keepdims=True)
                current = adjusted.mean(axis=0)
                # a class with no observations in this count stays where the surface put it
                step = np.where((current > 0) & (target > 0), target / np.maximum(current, 1e-12), 1.0)
                factor = factor * step
            offsets[b, s] = factor
    return offsets


def take_probabilities(surfaces, frame, rows, count_offsets=None):
    """
    Look up (n, 3) take-class probabilities for the given row positions.
    count_offsets: optional (4, 3, 3) multiplicative adjustment from
    `fit_take_count_offsets`; applied then renormalised, so the result is still a distribution.
    """
    subset = frame.iloc[rows]
    x_index, _ = _grid_index(subset["plate_x"], TAKE_GRID_X)
    z_index, _ = _grid_index(subset["plate_z"], TAKE_GRID_Z)
    out = np.empty((len(subset), len(TAKE_CLASSES)), dtype="float64")
    stands = subset["stand"].to_numpy()
    throws = subset["p_throws"].to_numpy()
    for key, surface in surfaces.items():
        cell = (stands == key[0]) & (throws == key[1])
        if cell.any():
            out[cell] = surface[x_index[cell], z_index[cell]]
    if count_offsets is not None:
        out = out * np.asarray(count_offsets)[subset["balls"].to_numpy(),
                                              subset["strikes"].to_numpy()]
        out /= out.sum(axis=1, keepdims=True)
    return out


def fit_contact_split(frame, train_mask):
    """
    League p(foul, foul_tip, in_play | contact) indexed by (balls, strikes, in_zone).
    The baseline stand-in for the three-class head. Returns a (4, 3, 2, 3) array.
    """
    klass = frame["description"].map(CONTACT_DESCRIPTION_TO_CLASS)
    cell = train_mask & klass.notna().to_numpy()
    in_zone = (frame["zone"].to_numpy() <= 9).astype("int64")  # Statcast 1-9 is in-zone
    flat = ((frame["balls"].to_numpy() * N_STRIKES + frame["strikes"].to_numpy()) * 2
            + in_zone)[cell]

    counts = np.zeros((N_BALLS * N_STRIKES * 2, len(CONTACT_CLASSES)), dtype="int64")
    np.add.at(counts, (flat, klass[cell].to_numpy(dtype="int64")), 1)
    alpha = fit_alpha(counts)
    return shrink(counts, alpha).reshape(N_BALLS, N_STRIKES, 2, len(CONTACT_CLASSES))


def fit_outcome_table(frame, bins, train_mask, pa_df, n_bins):
    """
    V[ev, la, spray] over {OUT, 1B, 2B, 3B, HR}, as category probabilities.
    bins: dict of the three quality bin arrays, MASKED off balls in play.
    Shrunk hierarchically -- a (ev, la, spray) cell backs off to its (ev, la) marginal,
    which backs off to the global pool -- because the bins are equal-mass marginally and
    not jointly, so the joint cells are far more uneven than their 73.9 average suggests.
    """
    scored = (bins["ev"] != MASKED) & (bins["la"] != MASKED) & (bins["spray"] != MASKED)
    rows = np.flatnonzero(train_mask & scored)

    outcome = pa_df.drop_duplicates(JOIN_KEYS).set_index(JOIN_KEYS)["woba_category"]
    assert outcome.index.is_unique, "a plate appearance key appears twice in the PA table"
    keys = pd.MultiIndex.from_frame(frame.iloc[rows][JOIN_KEYS])
    klass = outcome.reindex(keys).map(BIP_CATEGORY_TO_CLASS)
    # unjoined rows and PAs ending in SH or INT carry no batted-ball value and drop out
    joined = klass.notna().to_numpy()
    rows, klass = rows[joined], klass.to_numpy()[joined].astype("int64")

    shape = (n_bins, n_bins, n_bins)
    flat = np.ravel_multi_index((bins["ev"][rows], bins["la"][rows], bins["spray"][rows]),
                                shape)
    counts = np.zeros((n_bins ** 3, len(BIP_CLASSES)), dtype="int64")
    np.add.at(counts, (flat, klass), 1)

    # the (ev, la) marginal is the intermediate pool the joint cell backs off to
    marginal = counts.reshape(n_bins, n_bins, n_bins, -1).sum(axis=2)
    marginal = shrink(marginal.reshape(n_bins * n_bins, -1), fit_alpha(marginal.reshape(
        n_bins * n_bins, -1)))
    marginal = np.repeat(marginal, n_bins, axis=0)

    alpha = fit_alpha(counts)
    totals = counts.sum(axis=1, keepdims=True)
    table = (counts + alpha * marginal) / (totals + alpha)

    unmeasured = _unmeasured_outcomes(frame, train_mask, scored, outcome, len(rows))
    return (table.reshape(n_bins, n_bins, n_bins, len(BIP_CLASSES)), int(len(rows)),
            unmeasured)


def _unmeasured_outcomes(frame, train_mask, scored, outcome, n_measured):
    """
    The in-play share `V` is NOT fit on, and how that share actually hit (D5-R15).

    `V` is fit on balls in play carrying all three quality bins. The excluded share is not
    missing at random -- it hit materially worse than the retained share -- so a composition that
    values EVERY ball in play through `V` values the whole population at the retained subset's
    rate and runs high by that difference at every plate appearance.

    Returned as a CATEGORY distribution rather than a wOBA number so the unmeasured share is
    converted by the same season weights as `V` (see `woba_points_table`). A wOBA constant here
    would silently mix train-season weights into an eval-season number.

    Treated as a fixed LEAGUE rate, not a hitter effect: whether Statcast tracked your ball is
    not a skill worth 1,763 parameters. The caveat, stated rather than solved: weak contact is a
    skill and weak contact is likelier to go untracked, so a league constant slightly
    under-penalizes weak-contact hitters.

    Rejected alternative: reweighting the retained sample to match the full population. That
    distorts every cell's conditional in order to fix an aggregate level -- imputation by another
    name, which the Phase B missingness rule forbids.
    """
    in_play = frame["description"].to_numpy() == IN_PLAY_DESCRIPTION
    candidates = np.flatnonzero(train_mask & in_play & ~scored)
    keys = pd.MultiIndex.from_frame(frame.iloc[candidates][JOIN_KEYS])
    klass = outcome.reindex(keys).map(BIP_CATEGORY_TO_CLASS)
    klass = klass.to_numpy()[klass.notna().to_numpy()].astype("int64")

    total = n_measured + len(klass)
    assert total > 0, "no train balls in play joined to a plate-appearance outcome"
    categories = np.bincount(klass, minlength=len(BIP_CLASSES)).astype("float64")
    return {
        "measured_share": n_measured / total,
        "n_measured": int(n_measured),
        "n_unmeasured": int(len(klass)),
        # a degenerate empty group would divide by zero; a share of 0 makes the branch inert
        "categories": (categories / len(klass)).tolist() if len(klass) else
                      [0.0] * len(BIP_CLASSES),
    }


def woba_points_table(table, weights):
    """Collapse V's category probabilities to wOBA points under one season's weights."""
    from src.data.eval_targets import CATEGORY_WEIGHT_KEY

    values = np.array([weights.get(CATEGORY_WEIGHT_KEY[name], 0.0)
                       if name in CATEGORY_WEIGHT_KEY else 0.0 for name in BIP_CLASSES])
    return table @ values


class Repertoire:
    """
    Real pitch rows indexed by (pitcher, stand, balls, strikes), with backoff.
    Stored as one sorted row-index array plus cell boundaries rather than a dict of
    arrays: the train table is 5.9M rows and the cell count is in the tens of thousands.
    """

    def __init__(self, order, starts, pitcher_ids, n_pitchers):
        self.order, self.starts = order, starts
        self.pitcher_ids, self.n_pitchers = pitcher_ids, n_pitchers

    def _cell(self, pitcher_slot, stand_slot, balls, strikes):
        return ((pitcher_slot * 2 + stand_slot) * N_BALLS + balls) * N_STRIKES + strikes

    def rows(self, pitcher_slot, stand_slot, balls, strikes):
        """Row indices for one cell, empty if the pitcher never faced that situation."""
        cell = self._cell(pitcher_slot, stand_slot, balls, strikes)
        return self.order[self.starts[cell]:self.starts[cell + 1]]

    def sample(self, pitcher_slot, stand_slot, balls, strikes, n, generator):
        """
        Up to n rows for a cell, backing off when the cell is thin.
        Backoff order per docs/phase-d5-spec.md §2.3: exact cell, then the pitcher's
        rows at that strike count regardless of balls, then all his rows to that stand.
        Returns (rows, level) so the caller can report how often backoff fired.
        """
        exact = self.rows(pitcher_slot, stand_slot, balls, strikes)
        if len(exact) >= n:
            return generator.choice(exact, size=n, replace=False), 0

        wider = np.concatenate([self.rows(pitcher_slot, stand_slot, b, strikes)
                                for b in range(N_BALLS)])
        if len(wider) >= n:
            return generator.choice(wider, size=n, replace=False), 1

        widest = np.concatenate([self.rows(pitcher_slot, stand_slot, b, s)
                                 for b in range(N_BALLS) for s in range(N_STRIKES)])
        if len(widest) == 0:
            return widest, 3
        # exactly n rows even from a thin pool, so callers get a rectangular grid and can
        # batch across pitchers; drawing with replacement re-weights nothing, since the
        # cell mean is over whatever rows the pitcher actually has either way
        return generator.choice(widest, size=n, replace=True), 2


def build_repertoire(frame, train_mask):
    """
    Index the train-season pitch rows by (pitcher, stand, balls, strikes).
    Excludes intentional balls, which the simulator never generates.
    Returns a Repertoire.
    """
    keep = train_mask & ~frame["description"].isin(EXCLUDED_DESCRIPTIONS).to_numpy()
    rows = np.flatnonzero(keep)

    pitchers = frame["pitcher"].to_numpy()[rows]
    pitcher_ids = np.unique(pitchers)
    slot = np.searchsorted(pitcher_ids, pitchers)
    stand_slot = (frame["stand"].to_numpy()[rows] == "R").astype("int64")

    cell = (((slot * 2 + stand_slot) * N_BALLS + frame["balls"].to_numpy()[rows])
            * N_STRIKES + frame["strikes"].to_numpy()[rows])
    n_cells = len(pitcher_ids) * 2 * N_BALLS * N_STRIKES

    order_within = np.argsort(cell, kind="stable")
    starts = np.searchsorted(cell[order_within], np.arange(n_cells + 1))
    return Repertoire(rows[order_within], starts, pitcher_ids, len(pitcher_ids))


def pitcher_weights(pa_df, seasons, repertoire):
    """
    Batters faced per pitcher over `seasons`, normalized within each throwing hand.
    Returns {hand: (slots, weights)} over pitchers present in the repertoire, so the
    simulated opponent distribution is the one hitters actually face rather than one
    weighted by roster count (2026-08-08).
    """
    window = pa_df[pa_df["season"].isin(seasons)]
    faced = window.groupby(["pitcher", "p_throws"]).size().reset_index(name="bf")
    faced = faced[faced["pitcher"].isin(repertoire.pitcher_ids)]

    out = {}
    for hand in ("L", "R"):
        side = faced[faced["p_throws"] == hand]
        slots = np.searchsorted(repertoire.pitcher_ids, side["pitcher"].to_numpy())
        weight = side["bf"].to_numpy(dtype="float64")
        assert weight.sum() > 0, f"no {hand}-handed pitchers in the repertoire"
        out[hand] = (slots, weight / weight.sum())
    return out
