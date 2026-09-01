"""
Phase E.7 — is the resampled pitch mix the real pitch mix? (docs/phase-e-spec.md §8 fork)

Entered by E.6's PRE-REGISTERED fork. E.6 scored the swing head on real held-out pitches
with the resampler removed and found it calibrated to +0.27% overall, over-predicting
swings rather than under-predicting them in every three-ball count. A head that swings too
OFTEN cannot manufacture an excess of walks, so the pre-registered reading hands ownership
to the resampler and this module asks which part of it.

The chain's ball mass at count (b, s) is  (1 - p_swing) * P(ball | take, location).  The
hitter-dependent factor is held fixed, so the only thing compared here is the second term:
the take surface evaluated on the pitches `_sample_grid` draws, against the take surface
evaluated on the pitches those same pitchers actually threw at that count. Same surface,
same pitcher pool, same batters-faced weights. Any gap is the DRAW, not the model and not
the surface.

Three references per count, because they answer different questions:
  sampled   what the composition actually consumed
  exact     the pitcher pool's own rate at that cell, weighted the same way
  league    the raw train-window rate at that count, unweighted by pool membership

`sampled` minus `exact` isolates backoff and the thin-cell draw. `exact` minus `league`
isolates the pool weighting. Read-only; nothing trains and no model is loaded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import claim1_eval
from src.model import query, query_tables as qt

DEFAULT_OUT_DIR = "results/phase_e"
TAKE_CLASSES = ("ball", "called_strike", "hbp")


def _take_means(tables, frame, rows, offsets):
    """Mean P(ball / called strike / hbp | take) over `rows`, plus the in-zone share."""
    if len(rows) == 0:
        return {name: float("nan") for name in TAKE_CLASSES} | {"in_zone": float("nan")}
    take = qt.take_probabilities(tables["take"], frame, rows, offsets)
    out = {name: float(take[:, index].mean()) for index, name in enumerate(TAKE_CLASSES)}
    out["in_zone"] = float((frame["zone"].to_numpy()[rows] <= 9).mean())
    return out


def resampler_audit(tables, frame, seed=0, use_offsets=False):
    """
    Per (count, batter stand, pitcher hand): the sampled, exact-cell and league take rates.

    The sampled draw is reproduced with `_sample_grid` under the same generator seeding
    `predict` uses, so this audits the draw the composition consumed rather than a fresh
    one that happens to look similar.
    """
    offsets = tables.get("take_count_offsets") if use_offsets else None
    repertoire, pool = tables["repertoire"], tables["pitcher_weights"]
    train_mask = np.isin(frame["season"].to_numpy(), tables["_train_seasons"])
    excluded = frame["description"].isin(qt.EXCLUDED_DESCRIPTIONS).to_numpy()
    zone = frame["zone"].to_numpy()
    stand_column = frame["stand"].to_numpy()
    balls_column, strikes_column = frame["balls"].to_numpy(), frame["strikes"].to_numpy()

    rows = []
    generator = np.random.default_rng(seed)
    for hand, (slots, probability) in sorted(pool.items()):
        for stand in ("L", "R"):
            stand_slot = 1 if stand == "R" else 0
            grid, usable, levels = qt._sample_grid(repertoire, slots, stand_slot,
                                                   query.DEFAULT_N_PITCHES, generator) \
                if hasattr(qt, "_sample_grid") else query._sample_grid(
                    repertoire, slots, stand_slot, query.DEFAULT_N_PITCHES, generator)
            weight = probability[usable] / probability[usable].sum()
            for position, (balls, strikes) in enumerate(query.COUNT_STATES):
                drawn = grid[usable][:, position, :]                       # (P, 6)
                per_pitcher = qt.take_probabilities(
                    tables["take"], frame, drawn.reshape(-1), offsets
                ).reshape(len(drawn), query.DEFAULT_N_PITCHES, 3).mean(axis=1)
                sampled = {name: float(weight @ per_pitcher[:, index])
                           for index, name in enumerate(TAKE_CLASSES)}
                sampled["in_zone"] = float(
                    weight @ (zone[drawn] <= 9).mean(axis=1))

                # the same pitchers' OWN rate at this exact cell, same weights, no draw
                exact_rows, exact_weight = [], []
                for index, slot in enumerate(np.flatnonzero(usable)):
                    cell = repertoire.rows(slots[slot], stand_slot, balls, strikes)
                    if len(cell):
                        exact_rows.append(cell)
                        exact_weight.append(np.full(len(cell), weight[index] / len(cell)))
                if exact_rows:
                    flat = np.concatenate(exact_rows)
                    flat_weight = np.concatenate(exact_weight)
                    flat_weight = flat_weight / flat_weight.sum()
                    take = qt.take_probabilities(tables["take"], frame, flat, offsets)
                    exact = {name: float(flat_weight @ take[:, index])
                             for index, name in enumerate(TAKE_CLASSES)}
                    exact["in_zone"] = float(flat_weight @ (zone[flat] <= 9))
                    n_exact = int(len(flat))
                else:
                    exact = {name: float("nan") for name in TAKE_CLASSES} | {"in_zone": float("nan")}
                    n_exact = 0

                mask = (train_mask & ~excluded & (stand_column == stand)
                        & (balls_column == balls) & (strikes_column == strikes))
                league = _take_means(tables, frame, np.flatnonzero(mask), offsets)

                rows.append({
                    "p_throws": hand, "stand": stand, "balls": balls, "strikes": strikes,
                    "n_pitchers": int(usable.sum()), "n_exact_pitches": n_exact,
                    "n_league_pitches": int(mask.sum()),
                    **{f"sampled_{k}": v for k, v in sampled.items()},
                    **{f"exact_{k}": v for k, v in exact.items()},
                    **{f"league_{k}": v for k, v in league.items()},
                    "backoff_exact": int(levels[0]), "backoff_strikes": int(levels[1]),
                    "backoff_stand": int(levels[2]), "backoff_empty": int(levels[3]),
                })
    table = pd.DataFrame(rows)
    for name in list(TAKE_CLASSES) + ["in_zone"]:
        table[f"draw_gap_{name}"] = table[f"sampled_{name}"] - table[f"exact_{name}"]
        table[f"pool_gap_{name}"] = table[f"exact_{name}"] - table[f"league_{name}"]
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.7 — audit the resampled pitch mix against the real one.")
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--take-count-offsets", action="store_true")
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    season = np.load(Path(args.data_dir) / "season.npy", mmap_mode="r")
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, np.asarray(season))
    pa_df = pd.read_parquet(args.eval_targets)

    print("fitting the take surface, repertoire and pitcher weights")
    train_mask = np.isin(frame["season"].to_numpy(), manifest["train_seasons"])
    surfaces, alpha = qt.fit_take_surfaces(frame, train_mask)
    tables = {
        "take": surfaces,
        "take_count_offsets": qt.fit_take_count_offsets(surfaces, frame, train_mask),
        "repertoire": qt.build_repertoire(frame, train_mask),
        "_train_seasons": manifest["train_seasons"],
    }
    tables["pitcher_weights"] = qt.pitcher_weights(pa_df, manifest["train_seasons"],
                                                   tables["repertoire"])

    print("auditing the draw")
    table = resampler_audit(tables, frame, seed=args.seed,
                            use_offsets=args.take_count_offsets)
    suffix = "_offsets" if args.take_count_offsets else ""
    table.to_csv(out_dir / f"e7_resampler_audit{suffix}.csv", index=False)

    # collapse to one row per count over the four (stand, hand) cells, weighted by the
    # handedness shares the composition itself uses -- an unweighted mean over the four
    # would give LHB-vs-LHP, 7.6% of real plate appearances, the same say as RHB-vs-RHP
    shares = query.handedness_shares(pa_df, manifest["train_seasons"])
    table["share"] = [shares[(row.stand, row.p_throws)] for row in table.itertuples()]
    by_count = table.groupby(["balls", "strikes"]).apply(
        lambda part: pd.Series({
            "share": part["share"].sum(),
            **{f"{prefix}_{name}": float(np.average(part[f"{prefix}_{name}"],
                                                    weights=part["share"]))
               for prefix in ("sampled", "exact", "league")
               for name in ("ball", "in_zone")},
        }), include_groups=False).reset_index()
    by_count["draw_gap_ball"] = by_count["sampled_ball"] - by_count["exact_ball"]
    by_count["pool_gap_ball"] = by_count["exact_ball"] - by_count["league_ball"]
    by_count.to_csv(out_dir / f"e7_resampler_by_count{suffix}.csv", index=False)

    summary = {
        "eval_season": args.eval_season, "take_count_offsets": args.take_count_offsets,
        "take_alpha": float(alpha),
        "overall_sampled_ball": float(np.average(table["sampled_ball"],
                                                 weights=table["share"])),
        "overall_exact_ball": float(np.average(table["exact_ball"],
                                               weights=table["share"])),
        "overall_league_ball": float(np.average(table["league_ball"],
                                                weights=table["share"])),
        "three_ball_draw_gap": float(
            by_count[by_count["balls"] == 3]["draw_gap_ball"].mean()),
        "backoff_levels": [int(table[f"backoff_{name}"].iloc[::12].sum())
                           for name in ("exact", "strikes", "stand", "empty")],
    }
    (out_dir / f"e7_resampler_summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("\n-- P(ball | take) by count: sampled draw vs the pool's own cell vs league --")
    print(by_count.to_string(index=False))
    print(f"\nwrote {out_dir / f'e7_resampler_by_count{suffix}.csv'}")


if __name__ == "__main__":
    main()
