"""
Phase E.9 — the resampler's SECOND channel: does the drawn pitch mix also change take mass?

E.8 priced one channel of the draw gap: the take surface returns a different P(ball | take)
on the pitches the resampler draws than on the pitches those pitchers actually threw. It
covered 41% of the residual walk excess. A second channel is untouched by that calculation,
because E.8 held the take MASS at its league value:

    ball mass = (1 - p_swing) * P(ball | take, location)
                 \__________/   \____________________/
                  channel 2          channel 1 (E.8)

The swing head is calibrated on real pitches (E.6, +0.27%), but in the composition it is
asked about RESAMPLED pitches. If those sit further out of the zone, a correctly calibrated
head returns a lower p_swing, take mass rises, and walks rise with it -- with no defect
anywhere in the model. This measures that, paired: the same hitters, the same pitchers, the
same counts, scored on the drawn rows and then on rows drawn from the pitcher's own exact
cell with no backoff.

Then the measured take-mass gap goes through E.8's league chain, so both channels are priced
in the same units and against the same denominator.

SAMPLED, not exhaustive: a subset of pitchers and hitters (see --n-pitchers / --n-hitters).
The estimand is a PAIRED difference over identical hitters and pitchers, where the hitter
effect cancels, so a small panel is enough -- but the shortcut is stated here and in the
summary rather than left for a reader to infer. Read-only; nothing trains.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import claim1_eval
from src.analysis.model_evaluation_price_draw import league_chain, to_aggregates
from src.model import query, query_tables as qt

DEFAULT_OUT_DIR = "results/model_evaluation"
N_EXACT = 24   # reference draws per cell; 4x the composition's 6, to quiet the reference side


def paired_rows(repertoire, slots, stand_slot, n_exact, generator):
    """
    Per (pitcher, count): the composition's 6 drawn rows, and `n_exact` rows from the
    pitcher's OWN cell with backoff disabled. Pitchers missing any exact cell are dropped,
    so every comparison is within-pitcher and no cell is compared against a substitute.
    """
    drawn, exact, keep = [], [], []
    for slot in slots:
        cells_drawn, cells_exact, ok = [], [], True
        for balls, strikes in query.COUNT_STATES:
            picked, level = repertoire.sample(slot, stand_slot, balls, strikes, 6, generator)
            own = repertoire.rows(slot, stand_slot, balls, strikes)
            if len(picked) == 0 or len(own) == 0:
                ok = False
                break
            cells_drawn.append(picked)
            cells_exact.append(generator.choice(own, size=n_exact,
                                                replace=len(own) < n_exact))
        keep.append(ok)
        if ok:
            drawn.append(np.stack(cells_drawn))
            exact.append(np.stack(cells_exact))
    return np.array(drawn), np.array(exact), np.array(keep)


def score_take_mass(models, context_memmap, hitter_rows, pitch_rows, batch=32768):
    """
    Mean take mass (1 - p_swing) over `hitter_rows`, per pitch row. Probabilities are
    averaged across the ensemble before the mean, matching `query.expected_woba`.
    """
    unique, inverse = np.unique(pitch_rows.reshape(-1), return_inverse=True)
    context = torch.from_numpy(np.ascontiguousarray(context_memmap[unique]))
    total = np.zeros(len(unique), dtype="float64")
    with torch.no_grad():
        for hitter_row in hitter_rows:
            hitter = torch.full((len(unique),), int(hitter_row), dtype=torch.int64)
            out = np.zeros(len(unique), dtype="float64")
            for start in range(0, len(unique), batch):
                stop = min(start + batch, len(unique))
                seed_total = None
                for model in models:
                    p = torch.sigmoid(
                        model.head_swing(query._trunk(model, hitter[start:stop],
                                                      context[start:stop])).squeeze(-1))
                    assert p.shape == (stop - start,), \
                        f"swing head returned {tuple(p.shape)}, expected {(stop - start,)}"
                    seed_total = p if seed_total is None else seed_total + p
                out[start:stop] = (seed_total / len(models)).double().numpy()
            total += 1.0 - out
    return (total / len(hitter_rows))[inverse].reshape(pitch_rows.shape)


def main():
    parser = argparse.ArgumentParser(
        description="Phase E.9 — take-mass shift from the resampled pitch mix.")
    parser.add_argument("--arm", default="rebuild_baseline")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-season", type=int, default=2024)
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--data-dir", default="data/processed/phase_d")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--pitch-events", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--eval-targets", default="data/processed/eval_targets_pa.parquet")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-pitchers", type=int, default=200)
    parser.add_argument("--n-hitters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    claim1_eval.assert_not_test_season(args.eval_season, final_run=args.final_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(args.data_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    season = np.load(directory / "season.npy", mmap_mode="r")
    context_memmap = np.load(directory / "context.npy", mmap_mode="r")
    frame = qt.align_pitch_frame(args.pitch_events, args.eval_targets, np.asarray(season))
    pa_df = pd.read_parquet(args.eval_targets)
    train_mask = np.isin(frame["season"].to_numpy(), manifest["train_seasons"])
    shares = query.handedness_shares(pa_df, manifest["train_seasons"])

    print("fitting the repertoire and pitcher weights")
    repertoire = qt.build_repertoire(frame, train_mask)
    pool = qt.pitcher_weights(pa_df, manifest["train_seasons"], repertoire)

    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = query.load_ensemble(paths, manifest, context_memmap.shape[1])

    # hitters spanning exposure, not the top of the board: a take-mass gap read off four
    # high-volume regulars would not represent the low-exposure stratum the claim lives in
    trained = sorted(int(row) for row in manifest["vocabulary"].values() if int(row) != 0)
    hitter_rows = [trained[int(q * (len(trained) - 1))]
                   for q in np.linspace(0.1, 0.9, args.n_hitters)]

    generator = np.random.default_rng(args.seed)
    rows = []
    for hand, (slots, probability) in sorted(pool.items()):
        order = np.argsort(-probability)[:args.n_pitchers]
        for stand in ("L", "R"):
            stand_slot = 1 if stand == "R" else 0
            drawn, exact, keep = paired_rows(repertoire, np.asarray(slots)[order],
                                             stand_slot, N_EXACT, generator)
            weight = probability[order][keep]
            weight = weight / weight.sum()
            take_drawn = score_take_mass(models, context_memmap, hitter_rows, drawn)
            take_exact = score_take_mass(models, context_memmap, hitter_rows, exact)
            for position, (balls, strikes) in enumerate(query.COUNT_STATES):
                rows.append({
                    "p_throws": hand, "stand": stand, "balls": balls, "strikes": strikes,
                    "n_pitchers": int(keep.sum()),
                    "take_mass_drawn": float(weight @ take_drawn[:, position].mean(axis=1)),
                    "take_mass_exact": float(weight @ take_exact[:, position].mean(axis=1)),
                })
            print(f"  {hand}HP vs {stand}HB: {keep.sum()} pitchers paired")

    table = pd.DataFrame(rows)
    table["take_mass_gap"] = table["take_mass_drawn"] - table["take_mass_exact"]
    table["share"] = [shares[(row.stand, row.p_throws)] for row in table.itertuples()]
    table.to_csv(out_dir / "take_mass_take_mass.csv", index=False)

    by_count = table.groupby(["balls", "strikes"]).apply(
        lambda part: pd.Series({
            name: float(np.average(part[name], weights=part["share"]))
            for name in ("take_mass_drawn", "take_mass_exact", "take_mass_gap")
        }), include_groups=False).reset_index()

    # price it through E.8's league chain: extra take mass buys ball / called-strike / hbp
    # in the surface's own proportions at that count, so the split is not a free parameter
    chain = league_chain(frame, train_mask, shares)
    base = query.absorbing_rates(to_aggregates(chain))
    shifted = {name: array.copy() for name, array in chain.items()}
    for row in by_count.itertuples():
        b, s = int(row.balls), int(row.strikes)
        mass = chain["take_mass"][b, s]
        if mass <= 0:
            continue
        for name in ("ball", "hbp"):
            shifted[name][b, s] += row.take_mass_gap * chain[name][b, s] / mass
        called = chain["take_mass"][b, s] - chain["ball"][b, s] - chain["hbp"][b, s]
        shifted["strike"][b, s] += row.take_mass_gap * called / mass
        # the mass has to come from somewhere: swings lose exactly what takes gain,
        # split across the swing outcomes in their own observed proportions
        swing = 1.0 - mass
        for name in ("foul", "bip"):
            shifted[name][b, s] -= row.take_mass_gap * chain[name][b, s] / swing
        shifted["strike"][b, s] -= row.take_mass_gap * (
            chain["strike"][b, s] - called) / swing
    after = query.absorbing_rates(to_aggregates(shifted))

    channel_2 = float(after["bb"][0, 0] - base["bb"][0, 0])
    channel_1 = json.loads((out_dir / "draw_price_draw_price_summary.json").read_text())
    residual = channel_1["residual_walk_excess"]
    summary = {
        "arm": args.arm, "n_pitchers_requested": args.n_pitchers,
        "n_hitters": args.n_hitters, "n_exact_reference_draws": N_EXACT,
        "coverage_note": "sampled panel of pitchers and hitters; paired within both",
        "overall_take_mass_drawn": float(np.average(table["take_mass_drawn"],
                                                    weights=table["share"])),
        "overall_take_mass_exact": float(np.average(table["take_mass_exact"],
                                                    weights=table["share"])),
        "delta_bb_channel_1_surface": channel_1["delta_bb_from_draw"],
        "delta_bb_channel_2_take_mass": channel_2,
        "delta_bb_both_channels": channel_1["delta_bb_from_draw"] + channel_2,
        "residual_walk_excess": residual,
        "share_of_residual_explained": (channel_1["delta_bb_from_draw"] + channel_2) / residual,
        "delta_k_channel_2": float(after["k"][0, 0] - base["k"][0, 0]),
    }
    by_count.to_csv(out_dir / "take_mass_take_mass_by_count.csv", index=False)
    (out_dir / "take_mass_take_mass_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- take mass: drawn vs the pitcher's own cell --")
    print(by_count.to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
