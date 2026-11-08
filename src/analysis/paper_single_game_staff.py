"""
Draft-3 single-game example: rank the Phillies' available hitters against the Braves' projected
staff for 2025-08-30 (Chris Sale starting), next to their generic projection against left-handers.
Descriptive only, not graded. Final model (clean_dim64_final, trained 2015-2024), five-seed ensemble in one
call, exactly as the cached final point predictions were made.

Staff weights (stated as conventions in the paper):
  starter   Sale's average batters faced per start over the training seasons
  bullpen   pitchers who relieved for ATL in 2025 before the game AND in the 30 days before it
            (active-roster proxy), shares by 2025 relief BF to date, scaled so the bullpen
            total equals ATL's average relief BF per game in 2025 to date
  rest      drop relievers who pitched on both of the previous two days, or threw 25+ pitches
            the day before
Pitchers the model cannot query (no training-season pitches) drop out; their weight is reported.

Run: .venv/bin/python -m src.analysis.paper_single_game_staff
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import loader, query, query_tables as qt

GAME_DATE, TEAM, OPP, STARTER = pd.Timestamp("2025-08-30"), "PHI", "ATL", 519242  # Sale
ARM, SEEDS, DATA_DIR = "clean_dim64_final", [0, 1, 2, 3, 4], "data/processed/phase_d5_clean_final"
STATS = "results/model_visualization_final_clean/hitter_stats.csv"
RAW = "data/raw/statcast/snapshot_2026-07-14"
OUT = Path("results/paper_draft3")
ACTIVE_DAYS, HEAVY_PITCHES = 30, 25


def raw(season, cols):
    d = pd.read_parquet(f"{RAW}/season={season}.parquet", columns=cols)
    d = d[d.game_type == "R"].copy()
    d["fld"] = d.home_team.where(d.inning_topbot == "Top", d.away_team)
    d["bat"] = d.away_team.where(d.inning_topbot == "Top", d.home_team)
    d["game_date"] = pd.to_datetime(d.game_date)
    return d


def starter_bf_per_start(pitcher):
    bf = starts = 0
    for season in range(2015, 2025):
        d = raw(season, ["game_pk", "game_type", "inning_topbot", "pitcher", "at_bat_number",
                         "home_team", "away_team", "game_date"])
        first = d.sort_values("at_bat_number").groupby(["game_pk", "fld"]).pitcher.first()
        mine = d[d.pitcher == pitcher].groupby(["game_pk", "fld"]).at_bat_number.nunique()
        started = [first.get(k) == pitcher for k in mine.index]
        bf, starts = bf + int(mine[started].sum()), starts + int(sum(started))
    return bf / starts, starts


def staff_weights():
    d = raw(2025, ["game_pk", "game_type", "inning_topbot", "pitcher", "batter", "at_bat_number",
                   "pitch_number", "home_team", "away_team", "game_date", "player_name", "p_throws"])
    app = (d.groupby(["game_pk", "game_date", "fld", "pitcher"])
           .agg(pitches=("pitch_number", "size"), bf=("at_bat_number", "nunique"),
                first_ab=("at_bat_number", "min"), name=("player_name", "first")).reset_index())
    app["starter"] = app.groupby(["game_pk", "fld"]).first_ab.transform("min") == app.first_ab
    before = app[(app.fld == OPP) & (app.game_date < GAME_DATE)]
    relief = before[~before.starter]
    recent = set(relief[relief.game_date >= GAME_DATE - pd.Timedelta(days=ACTIVE_DAYS)].pitcher)
    pen = relief[relief.pitcher.isin(recent) & (relief.pitcher != STARTER)]
    d1 = before[before.game_date == GAME_DATE - pd.Timedelta(days=1)]
    d2 = before[before.game_date == GAME_DATE - pd.Timedelta(days=2)]
    rested_out = (set(d1.pitcher) & set(d2.pitcher)) | set(d1[d1.pitches >= HEAVY_PITCHES].pitcher)
    shares = pen.groupby("pitcher").bf.sum()
    names = before.groupby("pitcher").name.first()
    hands = d[d.fld == OPP].groupby("pitcher").p_throws.agg(lambda h: h.mode().iat[0])
    removed = sorted(names[p] for p in rested_out if p in shares.index)
    shares = shares.drop([p for p in rested_out if p in shares.index])
    relief_per_game = relief.bf.sum() / before.game_pk.nunique()
    sp_bf, sp_starts = starter_bf_per_start(STARTER)
    weights = pd.concat([pd.Series({STARTER: sp_bf}),
                         shares / shares.sum() * relief_per_game])
    # who started for PHI in the surrounding week = the hitters available that day
    window = d[(d.bat == TEAM) & d.game_date.between(GAME_DATE - pd.Timedelta(days=4),
                                                     GAME_DATE + pd.Timedelta(days=4))]
    lineups = (window.drop_duplicates(["game_pk", "at_bat_number"]).sort_values("at_bat_number")
               .groupby("game_pk").batter.apply(lambda s: list(dict.fromkeys(s))[:9]))
    hitters = sorted({b for lineup in lineups for b in lineup})
    game = d[(d.game_date == GAME_DATE) & (d.bat == TEAM)]
    started = list(dict.fromkeys(game.sort_values("at_bat_number").batter))[:9]
    meta = {"starter_bf_per_start": sp_bf, "starter_training_starts": sp_starts,
            "relief_bf_per_game": float(relief_per_game), "rest_rule_removed": removed,
            "n_relievers": int(len(shares)), "game_pk": int(game.game_pk.iloc[0])}
    return weights, names, hands, hitters, started, meta


def per_pitcher_matrix(hitters):
    """W(0,0) for each hitter vs every queryable pitcher, plus generic predictions, from the
    five-seed ensemble in one call (head distributions averaged before the chain), which is
    how the cached final point predictions were made."""
    tensors, manifest = loader.load_tensors(DATA_DIR)
    frame = qt.align_pitch_frame("data/processed/pitch_events_labeled.parquet",
                                 "data/processed/eval_targets_pa.parquet", tensors["season"],
                                 career_pitchers=manifest.get("career_pitchers_excluded", False))
    pa_df = pd.read_parquet("data/processed/eval_targets_pa.parquet")
    tables = query.build_tables(frame, tensors, manifest, pa_df)
    models = query.load_ensemble([Path("results/checkpoints") / f"{ARM}_s{seed}.pt"
                                  for seed in SEEDS], manifest, tensors["context"].shape[1])
    prior = query.default_cold_start_prior(models, STATS)
    p, diag = query.predict(models, tensors, manifest, frame, tables, pa_df, 2025,
                            batters=hitters, cold_start_prior=prior, keep_matrix=True)
    # a pitcher can sit in both hand groups (a hand recorded both ways), so key on the hand too
    long = [pd.DataFrame(g["w00"], index=g["batter"], columns=g["pitcher"]).stack()
            .rename("w00").rename_axis(["batter", "pitcher"]).reset_index().assign(p_throws=g["p_throws"])
            for g in diag["per_pitcher"]]
    return (pd.concat(long).set_index(["batter", "pitcher", "p_throws"]).w00,
            p.set_index(["batter", "p_throws"]).pred_woba)


def main():
    weights, pitcher_names, hands, hitters, started, meta = staff_weights()
    w00, generic = per_pitcher_matrix(hitters)

    cached = (pd.read_csv("results/model_v1/model_v1_predictions_clean_dim64_final.csv")
              .set_index(["batter", "p_throws"]).pred_woba)
    gap = (generic - cached.reindex(generic.index)).abs().max()
    assert gap < 1e-6, f"generic projections do not reproduce the cached final run (max gap {gap})"

    keys = set(zip(w00.index.get_level_values("pitcher"), w00.index.get_level_values("p_throws")))
    queryable = [p for p in weights.index if (p, hands[p]) in keys]
    dropped = weights.drop(queryable)
    w = weights[queryable]
    rows = []
    names = pd.read_csv("data/processed/hitter_names_clean.csv").set_index("batter")["name"]
    for b in hitters:
        vals = pd.Series([w00.get((b, p, hands[p]), np.nan) for p in queryable], index=queryable)
        assert vals.notna().all(), f"hitter {b} missing a staff pitcher"
        rows.append({"batter": b, "name": names.get(b, str(b)), "started": b in started,
                     "vs_staff": float(vals @ w / w.sum()),
                     "vs_lhp_generic": float(generic.loc[(b, "L")])})
    table = pd.DataFrame(rows).sort_values("vs_staff", ascending=False)
    table["rank_staff"] = range(1, len(table) + 1)
    table["rank_generic"] = table.vs_lhp_generic.rank(ascending=False).astype(int)

    staff = pd.DataFrame({"pitcher": weights.index, "name": pitcher_names.reindex(weights.index)
                          .fillna("Sale, Chris").values, "weight": weights.values,
                          "queryable": weights.index.isin(queryable)})
    meta.update({"weight_dropped_share": float(dropped.sum() / weights.sum()),
                 "dropped_pitchers": staff[~staff.queryable].name.tolist(),
                 "max_generic_gap_vs_cached": float(gap)})
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "single_game_staff.csv", index=False)
    staff.to_csv(OUT / "single_game_staff_weights.csv", index=False)
    (OUT / "single_game_staff.json").write_text(json.dumps(meta, indent=2, default=str))
    print(table.to_string(index=False, float_format="%.3f"))
    print(staff.to_string(index=False, float_format="%.1f"))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
