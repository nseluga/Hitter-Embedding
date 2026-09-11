"""
Pitcher-type queries: which hitters do better against which kinds of pitcher.

Exploratory and descriptive, on the 2024 validation season. Handedness is the one
validated query; these axes are exposed queries until a sealed-season pass scores them.

Axes (one each for velocity, location, mix), per pitcher, from TRAIN-season pitches only:
  fb_velo    mean release_speed of four-seamers and sinkers
  fb_height  mean plate_z of four-seamers and sinkers
  brk_share  share of pitches that are breaking balls

Stage `ceiling` (data only): per hitter, observed wOBA vs the top BF-weighted tercile of
an axis minus vs the bottom tercile, within one pitcher hand. Split-half reliability of
that contrast (game_pk parity, Spearman-Brown) says whether any hitter-level claim is
measurable at all.

Stage `query` (hours, 5-seed ensemble): one composition pass that keeps every hitter's
predicted wOBA against every individual pitcher (W(0,0) matrix), so each axis curve and
named-pitcher example afterwards is a reweighting, not another pass.

Run:  PYTHONPATH=. .venv/bin/python -m src.analysis.pitcher_type_query --stage ceiling
      PYTHONPATH=. .venv/bin/python -m src.analysis.pitcher_type_query --stage query
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_evaluation_platoon_ceiling import spearman_brown
from src.data import eval_targets

PITCHES_PATH = "data/processed/pitch_events_labeled.parquet"
EVAL_TARGETS_PATH = "data/processed/eval_targets_pa.parquet"
SPLIT_CONFIG_PATH = "src/config/split_config.json"
DEFAULT_OUT_DIR = "results/pitcher_type_query"
EVAL_SEASON = 2024

FASTBALLS = ("FF", "SI")
BREAKING = ("SL", "ST", "SV", "CU", "KC", "CS")
AXES = ("fb_velo", "fb_height", "brk_share")
MIN_TRAIN_PITCHES = 200      # pitcher needs this many train pitches for a stable attribute
MIN_TRAIN_FASTBALLS = 50
MIN_HALF_PA = 10
N_BOOT = 1000


def train_seasons(path=SPLIT_CONFIG_PATH):
    with open(path) as handle:
        return json.load(handle)["split"]["train"]


def pitcher_attributes(pitches, seasons):
    """
    One row per (pitcher, p_throws) with the three axes, from `seasons` only.
    Pitchers below the pitch floors get no row: an attribute on a handful of pitches
    would place them in a tercile by noise.
    """
    window = pitches[pitches["season"].isin(seasons)]
    assert len(window), "no train-season pitches"
    fastball = window["pitch_type"].isin(FASTBALLS)
    grouped = window.assign(
        is_fb=fastball, is_brk=window["pitch_type"].isin(BREAKING),
        fb_speed=window["release_speed"].where(fastball),
        fb_z=window["plate_z"].where(fastball),
    ).groupby(["pitcher", "p_throws"])
    out = grouped.agg(n_pitches=("is_fb", "size"), n_fb=("is_fb", "sum"),
                      fb_velo=("fb_speed", "mean"), fb_height=("fb_z", "mean"),
                      brk_share=("is_brk", "mean")).reset_index()
    out = out[(out["n_pitches"] >= MIN_TRAIN_PITCHES) & (out["n_fb"] >= MIN_TRAIN_FASTBALLS)]
    assert out[list(AXES)].notna().all().all(), "an attribute is missing after the floors"
    return out.reset_index(drop=True)


def weighted_terciles(values, weight):
    """
    Tercile label (0 low, 1 mid, 2 high) where each tercile holds a third of the WEIGHT,
    so 'top tercile' means a third of the plate appearances hitters actually saw.
    """
    values = np.asarray(values, dtype="float64")
    weight = np.asarray(weight, dtype="float64")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weight[order]) / weight.sum()
    labels = np.empty(len(values), dtype="int64")
    labels[order] = np.minimum((cumulative * 3 - 1e-12).astype("int64"), 2)
    return labels


def label_pitchers(attributes, pa_train):
    """
    Attach batters-faced over the train window and each axis's tercile, within hand.
    """
    bf = pa_train.groupby(["pitcher", "p_throws"]).size().rename("bf").reset_index()
    out = attributes.merge(bf, on=["pitcher", "p_throws"], how="inner")
    for axis in AXES:
        out[f"{axis}_tercile"] = -1
        for hand in ("L", "R"):
            side = out["p_throws"] == hand
            out.loc[side, f"{axis}_tercile"] = weighted_terciles(
                out.loc[side, axis], out.loc[side, "bf"])
    return out


def contrast_pa(pa_eval, labels, axis, hand):
    """Eval-season denominator PAs vs `hand` pitchers in the top or bottom tercile of `axis`."""
    part = pa_eval[(pa_eval["p_throws"] == hand) & pa_eval["in_denominator"]]
    tag = labels.loc[labels["p_throws"] == hand, ["pitcher", f"{axis}_tercile"]]
    part = part.merge(tag, on="pitcher", how="left")
    covered = float(part[f"{axis}_tercile"].notna().mean())
    part = part[part[f"{axis}_tercile"].isin([0, 2])]
    part = part.assign(arm=np.where(part[f"{axis}_tercile"] == 2, "hi", "lo"))
    return part, covered


def split_half(part, min_half_pa=MIN_HALF_PA, seed=0):
    """
    Split-half reliability of each hitter's hi-minus-lo wOBA contrast, by stand and pooled
    (pooled centres each stand first, so the stand main effect cannot inflate it).
    Same method as `model_evaluation_platoon_ceiling.split_half_reliability`, with the
    tercile arm in place of pitcher hand.
    """
    part = part.assign(half=np.where(part["game_pk"].to_numpy() % 2 == 0, "A", "B"))
    stand = part.groupby("batter")["stand"].agg(lambda s: s.mode().iat[0])
    grouped = (part.groupby(["batter", "half", "arm"])["woba_points"]
               .agg(points="sum", denom="size").reset_index())
    wide = grouped.pivot_table(index="batter", columns=["half", "arm"],
                               values=["points", "denom"]).dropna()
    enough = np.ones(len(wide), dtype=bool)
    for half in ("A", "B"):
        for arm in ("hi", "lo"):
            enough &= wide[("denom", half, arm)].to_numpy() >= min_half_pa
    wide = wide[enough]
    delta = {half: (wide[("points", half, "hi")] / wide[("denom", half, "hi")]
                    - wide[("points", half, "lo")] / wide[("denom", half, "lo")]).to_numpy()
             for half in ("A", "B")}
    side = stand.reindex(wide.index).to_numpy()

    rows = []
    for label in ("L", "R", "pooled"):
        mask = np.ones(len(wide), dtype=bool) if label == "pooled" else side == label
        a, b = delta["A"][mask].copy(), delta["B"][mask].copy()
        if mask.sum() < 20:
            rows.append({"stand": label, "n_hitters": int(mask.sum())})
            continue
        if label == "pooled":
            for value in np.unique(side):
                cell = side[mask] == value
                a[cell] -= a[cell].mean()
                b[cell] -= b[cell].mean()
        half_r = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        rng = np.random.default_rng(seed)
        boot = np.array([spearman_brown(float(pd.Series(a[i]).corr(pd.Series(b[i]),
                                                                    method="spearman")))
                         for i in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))])
        boot = boot[np.isfinite(boot)]
        rows.append({"stand": label, "n_hitters": int(mask.sum()),
                     "half_split_spearman": half_r,
                     "reliability": float(spearman_brown(half_r)),
                     "reliability_ci_low": float(np.percentile(boot, 2.5)),
                     "reliability_ci_high": float(np.percentile(boot, 97.5))})
    return rows


def ceiling_stage(pitches, pa_df, seasons, eval_season=EVAL_SEASON):
    pa_df = eval_targets.drop_pitcher_batters(pa_df)
    attributes = pitcher_attributes(pitches, seasons)
    labels = label_pitchers(attributes, pa_df[pa_df["season"].isin(seasons)])
    pa_eval = pa_df[pa_df["season"] == eval_season]

    rows = []
    for axis in AXES:
        for hand in ("R", "L"):
            part, covered = contrast_pa(pa_eval, labels, axis, hand)
            for row in split_half(part):
                # "same" = hitter stands on the pitcher's side
                row.update({"axis": axis, "p_throws": hand, "eval_pa_covered": covered,
                            "matchup": {"pooled": "both"}.get(
                                row["stand"], "same" if row["stand"] == hand else "opposite")})
                rows.append(row)
    summary = labels.groupby("p_throws")[list(AXES)].describe().T
    return labels, pd.DataFrame(rows), summary


def query_stage(args):
    """
    The canonical 2024 composition with the per-pitcher matrix kept. Writes
    `matrix_w00.npz` (per group: batter, pitcher, bf_weight, w00) and the ordinary
    predictions, and checks the matrix reproduces them before anything is written.
    """
    from src.model import loader, query, query_tables as qt

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors, manifest = loader.load_tensors(args.data_dir)
    assert EVAL_SEASON not in manifest["train_seasons"], "eval season inside the train window"
    frame = qt.align_pitch_frame(args.pitches, args.eval_targets, tensors["season"])
    pa_df = pd.read_parquet(args.eval_targets)
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    paths = [Path(args.checkpoint_dir) / f"{args.arm}_s{seed}.pt" for seed in args.seeds]
    models = query.load_ensemble(paths, manifest, tensors["context"].shape[1])
    prior = query.default_cold_start_prior(models, args.hitter_stats)

    predictions, diagnostics = query.predict(
        models, tensors, manifest, frame, tables, pa_df, EVAL_SEASON,
        batters=args.batters, cold_start_prior=prior, keep_matrix=True)
    groups = diagnostics.pop("per_pitcher")

    arrays = {}
    for group in groups:
        key = f"{group['stand']}HB_vs_{group['p_throws']}HP"
        pred = group["w00"] @ group["bf_weight"] / group["bf_weight"].sum()
        expected = (predictions[predictions["p_throws"] == group["p_throws"]]
                    .set_index("batter").loc[group["batter"], "pred_woba"].to_numpy())
        assert np.allclose(pred, expected, atol=1e-10), f"{key}: matrix != pred_woba"
        for name in ("batter", "pitcher", "bf_weight", "w00"):
            arrays[f"{key}__{name}"] = np.asarray(group[name])
    np.savez_compressed(out / "matrix_w00.npz", **arrays)
    predictions.to_csv(out / "query_predictions.csv", index=False)
    (out / "query_diagnostics.json").write_text(json.dumps(
        {**diagnostics, "arm": args.arm, "seeds": args.seeds, "eval_season": EVAL_SEASON,
         "data_dir": args.data_dir, "hitter_stats": args.hitter_stats}, indent=2))
    print(f"wrote {out / 'matrix_w00.npz'}: " + ", ".join(
        f"{g['stand']}HB_vs_{g['p_throws']}HP {g['w00'].shape}" for g in groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=["ceiling", "query"], default="ceiling")
    parser.add_argument("--pitches", default=PITCHES_PATH)
    parser.add_argument("--eval-targets", default=EVAL_TARGETS_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--arm", default="embedding_sgd_sgd_lr1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-dir", default="data/processed/phase_d5")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--hitter-stats", default="results/model_visualization/hitter_stats.csv")
    parser.add_argument("--batters", type=int, nargs="+", default=None,
                        help="smoke-test subset; write to a /tmp --out-dir")
    args = parser.parse_args()

    if args.stage == "query":
        query_stage(args)
        return

    seasons = train_seasons()
    assert EVAL_SEASON not in seasons, "eval season is inside the train window"
    pitches = pd.read_parquet(args.pitches, columns=[
        "pitcher", "p_throws", "season", "pitch_type", "release_speed", "plate_z"])
    pa_df = pd.read_parquet(args.eval_targets)
    labels, ceiling, summary = ceiling_stage(pitches, pa_df, seasons)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / "pitcher_attributes.csv", index=False)
    ceiling.to_csv(out / "ceiling.csv", index=False)
    summary.to_csv(out / "pitcher_attribute_summary.csv")
    print(ceiling.to_string(index=False))


if __name__ == "__main__":
    main()
