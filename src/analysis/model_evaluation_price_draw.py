"""
Phase E.8 — price the resampler's draw gap through the exact count chain.

E.7 measured a gap between the take probabilities the resampler's draw carries and the ones
the same pitchers' own cells carry. A gap in P(ball | take) at 1-0 is not worth the same as
one at 3-1, so a table of gaps is not yet an answer: the chain has to convert them into
walks. That is what this does.

The vehicle is a LEAGUE-AVERAGE synthetic chain, built from real train-window pitch
descriptions rather than from the model. Every transition probability is an observed
frequency, so the baseline it produces is the league's own walk rate and not a model output.
Then each count's take mass is shifted by E.7's measured gap and `absorbing_rates` is
re-solved. The difference is what the draw is worth in walks, exactly -- the chain is solved
by backward induction, not simulated, so nothing here carries sampling noise.

This prices ONE channel. It does not claim the channel is the whole residual, and the
report states the share it covers rather than implying closure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.model import query, query_tables as qt

SWING_MISS = ("swinging_strike", "swinging_strike_blocked", "missed_bunt")
FOUL = ("foul", "bunt_foul")
STRIKE_LIKE = ("called_strike", "foul_tip", "foul_bunt", "bunt_foul_tip")


def league_chain(frame, train_mask, shares):
    """
    Per-count transition frequencies from real pitches, handedness-weighted like the
    composition. Returns solve_chain-shaped (4, 3) arrays plus the per-count take mass.

    foul_tip is a strike, not a foul: it is caught by definition, so on two strikes it ends
    the plate appearance. Folding it into `foul` would hand the chain a self-loop that real
    baseball does not have.
    """
    description = frame["description"].to_numpy()
    balls, strikes = frame["balls"].to_numpy(), frame["strikes"].to_numpy()
    stand, hand = frame["stand"].to_numpy(), frame["p_throws"].to_numpy()
    keep = train_mask & ~np.isin(description, qt.EXCLUDED_DESCRIPTIONS)

    channels = {
        "ball": np.isin(description, qt.BALL_DESCRIPTIONS),
        "hbp": description == "hit_by_pitch",
        "foul": np.isin(description, FOUL),
        "bip": description == qt.IN_PLAY_DESCRIPTION,
    }
    channels["strike"] = np.isin(description, STRIKE_LIKE) | np.isin(description, SWING_MISS)
    take = channels["ball"] | channels["hbp"] | (description == "called_strike")

    out = {name: np.zeros((qt.N_BALLS, qt.N_STRIKES)) for name in
           ("ball", "strike", "hbp", "foul", "bip", "take_mass")}
    for b, s in query.COUNT_STATES:
        cell = keep & (balls == b) & (strikes == s)
        total = 0.0
        for name in out:
            out[name][b, s] = 0.0
        for (cell_stand, cell_hand), share in shares.items():
            sub = cell & (stand == cell_stand) & (hand == cell_hand)
            n = sub.sum()
            if n == 0:
                continue
            total += share
            for name, mask in channels.items():
                out[name][b, s] += share * (mask & sub).sum() / n
            out["take_mass"][b, s] += share * (take & sub).sum() / n
        for name in out:
            out[name][b, s] /= total

    covered = sum(out[name] for name in ("ball", "strike", "hbp", "foul", "bip"))
    assert np.allclose(covered[np.array(query.COUNT_STATES).T[0],
                               np.array(query.COUNT_STATES).T[1]], 1.0, atol=1e-12), \
        f"pitch descriptions do not partition the outcome space: {covered}"
    return out


def to_aggregates(chain):
    """solve_chain's dict, with zero wOBA points on balls in play (only rates are read)."""
    return {"ball": chain["ball"], "strike": chain["strike"], "hbp": chain["hbp"],
            "foul": chain["foul"], "bip": chain["bip"],
            "bip_points": np.zeros_like(chain["bip"])}


def apply_draw_gap(chain, by_count, sign=1.0):
    """
    Shift each count's take mass by E.7's measured gap, renormalising within the take.

    The gap is in P(class | take), so it is multiplied by that count's take mass to land in
    the chain's units. Called strikes absorb the ball shift, which is what the surface
    itself does -- the three take classes sum to 1 by construction.
    """
    shifted = {name: array.copy() for name, array in chain.items()}
    for row in by_count.itertuples():
        b, s = int(row.balls), int(row.strikes)
        mass = chain["take_mass"][b, s]
        shifted["ball"][b, s] += sign * mass * row.draw_gap_ball
        shifted["hbp"][b, s] += sign * mass * row.draw_gap_hbp
        shifted["strike"][b, s] -= sign * mass * (row.draw_gap_ball + row.draw_gap_hbp)
    return shifted


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.8 — what the resampler's draw gap is worth in walks.")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default="results/model_evaluation")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    season = np.load(Path(args.data_dir) / "season.npy", mmap_mode="r")
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, np.asarray(season))
    pa_df = pd.read_parquet(args.eval_targets)
    shares = query.handedness_shares(pa_df, manifest["train_seasons"])

    audit = pd.read_csv(out_dir / "resampler_audit_resampler_audit.csv")
    audit["share"] = [shares[(row.stand, row.p_throws)] for row in audit.itertuples()]
    by_count = audit.groupby(["balls", "strikes"]).apply(
        lambda part: pd.Series({
            name: float(np.average(part[name], weights=part["share"]))
            for name in ("draw_gap_ball", "draw_gap_hbp", "draw_gap_called_strike")
        }), include_groups=False).reset_index()

    train_mask = np.isin(frame["season"].to_numpy(), manifest["train_seasons"])
    chain = league_chain(frame, train_mask, shares)
    base = query.absorbing_rates(to_aggregates(chain))
    shifted = query.absorbing_rates(to_aggregates(apply_draw_gap(chain, by_count)))

    # per-count attribution: one count's gap at a time, so the contributions are additive
    # only to the extent the chain is locally linear -- reported, not assumed
    per_count = []
    for row in by_count.itertuples():
        one = by_count[(by_count["balls"] == row.balls) & (by_count["strikes"] == row.strikes)]
        alone = query.absorbing_rates(to_aggregates(apply_draw_gap(chain, one)))
        per_count.append({"balls": int(row.balls), "strikes": int(row.strikes),
                          "draw_gap_ball": float(row.draw_gap_ball),
                          "delta_bb": float(alone["bb"][0, 0] - base["bb"][0, 0]),
                          "delta_k": float(alone["k"][0, 0] - base["k"][0, 0])})
    per_count = pd.DataFrame(per_count).sort_values("delta_bb", ascending=False)
    per_count.to_csv(out_dir / "draw_price_draw_price_by_count.csv", index=False)

    matched = 0.08312   # E.1's population-matched observed walk rate
    modelled = 0.08538  # the shipped rebuild_baseline composition
    total_delta = float(shifted["bb"][0, 0] - base["bb"][0, 0])
    summary = {
        "league_chain_bb": float(base["bb"][0, 0]), "league_chain_k": float(base["k"][0, 0]),
        "shifted_bb": float(shifted["bb"][0, 0]), "shifted_k": float(shifted["k"][0, 0]),
        "delta_bb_from_draw": total_delta,
        "delta_k_from_draw": float(shifted["k"][0, 0] - base["k"][0, 0]),
        "sum_of_per_count_delta_bb": float(per_count["delta_bb"].sum()),
        "residual_walk_excess": modelled - matched,
        "share_of_residual_explained": total_delta / (modelled - matched),
    }
    (out_dir / "draw_price_draw_price_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- what one count's draw gap is worth in walks --")
    print(per_count.to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
