"""Export a pitcher-type query matrix for the public matchup explorer (explorer/).

Descriptive only: model predictions, not validated (one-season reliability of matchup
contrasts ~0, see <res>/ceiling.csv). Arm, matchup season, train window and hitter stats are
read from <res>/query_diagnostics.json (written by pitcher_type_query --stage query), so the
page label always names the build that produced the matrix.
The browser does all math (double-centering, type averages) from the raw W(0,0) matrices.

Run: PYTHONPATH=. .venv/bin/python -m src.analysis.matchup_explorer_data [--res results/pitcher_type_query_2025]
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.pitcher_type_query import BREAKING, FASTBALLS, PITCHES_PATH

RES = Path("results/pitcher_type_query_2025")  # final model (clean_dim64_final) on 2025 matchups
OUT = Path("explorer/data/matchups.json")
GROUPS = [f"{s}HB_vs_{h}HP" for h in "RL" for s in "LR"]
LO, HI = 0.0, 1.0  # fixed wOBA range for uint16 quantization
BASE = ("fb_velo", "fb_height", "brk_share")
OFFSPEED = ("CH", "FS", "FO", "SC")
MIN_SUB = 20  # min pitches of a subgroup (breaking balls, non-fastballs) for a trait built on it
# Extra explorer axes. Horizontal traits are sign-flipped for LHP so that + means glove side
# (or arm side, where named) for both hands. Inches for movement and plate_x.
EXTRA = ("fb_ivb", "fb_run", "fb_spin", "si_share", "rel_height", "extension", "rel_side",
         "off_share", "velo_gap", "mix_depth", "brk_sweep", "zone_rate", "glove_side",
         "fb_up", "brk_down", "brk_height", "off_down")  # last four: location by pitch class
PITCH_COLS = ["pitcher", "p_throws", "season", "pitch_type", "release_speed", "pfx_x", "pfx_z", "release_pos_x",
              "release_pos_z", "release_extension", "release_spin_rate", "plate_x", "plate_z", "zone"]
RAW_DIR = Path("data/raw/statcast/snapshot_2026-07-14")  # sz_top/sz_bot (batter's zone) live only in raw
KEYS = ["game_pk", "at_bat_number", "pitch_number"]


def load_pitches(seasons):
    """Processed train-season pitches plus each pitch's batter zone (sz_top, sz_bot) from the raw snapshot."""
    p = pd.read_parquet(PITCHES_PATH, columns=PITCH_COLS + KEYS)
    p = p[p["season"].isin(seasons)]
    raw = pd.concat(pd.read_parquet(RAW_DIR / f"season={s}.parquet", columns=KEYS + ["sz_top", "sz_bot"]) for s in seasons)
    raw = raw.astype({k: "int64" for k in KEYS}).drop_duplicates(KEYS)
    out = p.astype({k: "int64" for k in KEYS}).merge(raw, on=KEYS, how="left")
    assert len(out) == len(p), "raw join duplicated pitches"
    assert out["sz_top"].notna().mean() > 0.95, "most pitches lack a batter zone after the join"
    return out


def extra_attributes(pitches, seasons):
    """One row per (pitcher, p_throws) with the EXTRA traits, from `seasons` only (never 2025).
    `pitches` needs PITCH_COLS plus sz_top/sz_bot (see load_pitches)."""
    d = pitches[pitches["season"].isin(seasons)].copy()
    num = [c for c in PITCH_COLS if c not in ("pitcher", "p_throws", "season", "pitch_type")] + ["sz_top", "sz_bot"]
    d[num] = d[num].astype("float64")
    s = np.where(d["p_throws"] == "R", 1.0, -1.0)  # glove side = +x for RHP (catcher's view)
    fb, brk, off = (d["pitch_type"].isin(t) for t in (FASTBALLS, BREAKING, OFFSPEED))
    zn = ((d["plate_z"] - d["sz_bot"]) / (d["sz_top"] - d["sz_bot"])).where(d["sz_top"] > d["sz_bot"])  # 0 = zone bottom, 1 = top
    d = d.assign(
        ivb=(d["pfx_z"] * 12).where(fb), run=(-s * d["pfx_x"] * 12).where(fb), spin=d["release_spin_rate"].where(fb),
        is_si=(d["pitch_type"] == "SI").where(fb).astype("float64"), is_off=off,
        side=-s * d["release_pos_x"], sweep=(s * d["pfx_x"] * 12).where(brk), brk=brk,
        fbv=d["release_speed"].where(fb), secv=d["release_speed"].where(~fb),
        in_zone=d["zone"].between(1, 9).where(d["zone"].notna()).astype("float64"), glove=s * d["plate_x"] * 12,
        up=(zn > 2 / 3).astype("float64").where(fb & zn.notna()), brk_z=d["plate_z"].where(brk),
        brk_lo=(zn < 0).astype("float64").where(brk & zn.notna()), off_lo=(zn < 0).astype("float64").where(off & zn.notna()))
    g = d.groupby(["pitcher", "p_throws"])
    out = g.agg(fb_ivb=("ivb", "mean"), fb_run=("run", "mean"), fb_spin=("spin", "mean"), si_share=("is_si", "mean"),
                rel_height=("release_pos_z", "mean"), extension=("release_extension", "mean"), rel_side=("side", "mean"),
                off_share=("is_off", "mean"), n_brk=("brk", "sum"), n_off=("is_off", "sum"), brk_sweep=("sweep", "mean"),
                fbv=("fbv", "mean"), secv=("secv", "mean"), n_sec=("secv", "count"),
                zone_rate=("in_zone", "mean"), glove_side=("glove", "mean"),
                fb_up=("up", "mean"), brk_down=("brk_lo", "mean"), brk_height=("brk_z", "mean"), off_down=("off_lo", "mean"))
    out["velo_gap"] = (out["fbv"] - out["secv"]).where(out["n_sec"] >= MIN_SUB)
    for c in ("brk_sweep", "brk_down", "brk_height"):
        out[c] = out[c].where(out["n_brk"] >= MIN_SUB)
    out["off_down"] = out["off_down"].where(out["n_off"] >= MIN_SUB)
    share = d.groupby(["pitcher", "p_throws"])["pitch_type"].value_counts(normalize=True)
    out["mix_depth"] = 1 / (share ** 2).groupby(level=[0, 1]).sum()  # effective number of pitches
    return out[list(EXTRA)].reset_index()


def quantize(w):
    """Map wOBA values (clipped to [LO, HI]) to little-endian uint16 0..65535 so the matrix ships compactly to the browser."""
    return np.round((np.clip(w, LO, HI) - LO) / (HI - LO) * 65535).astype("<u2")


def dequantize(q):
    """Inverse of quantize: uint16 codes back to wOBA floats (to within quantization step)."""
    return q.astype(float) / 65535 * (HI - LO) + LO


def double_center(w, bf):
    """Matchup effect: w minus hitter's BF-weighted avg minus pitcher's avg over hitters, plus grand mean."""
    # w [H, P]: H hitters x P pitchers; bf [P]: pitcher BF weights (so p sums to 1).
    # r [H]: hitter's BF-weighted average over pitchers; c [P]: pitcher's mean over hitters.
    p = bf / bf.sum()
    r, c = w @ p, w.mean(0)
    return w - r[:, None] - c[None, :] + c @ p


def _b64(a):
    """Base64 text of an array's raw bytes (row-major), for embedding in JSON."""
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _r(x, d=3):
    """Round to d places, with NaN becoming None (JSON null)."""
    return None if pd.isna(x) else round(float(x), d)


def lookup_names(ids):
    """{MLBAM id: "First Last"} via pybaseball; returns {} if the lookup fails (names are cosmetic)."""
    try:
        from pybaseball import playerid_reverse_lookup
        lk = playerid_reverse_lookup(list(ids), key_type="mlbam")
        return lk.set_index("key_mlbam").apply(lambda r: f"{r.name_first} {r.name_last}".title(), axis=1).to_dict()
    except Exception as e:  # names are cosmetic; fall back to ids
        print(f"name lookup failed: {e}")
        return {}


def build(res=RES):
    """Assemble the explorer payload: per-handedness-group quantized W matrices, hitter and pitcher records, and metadata. Returns a JSON-ready dict."""
    # Inputs: precomputed matrices and pitcher attributes; the extra traits are recomputed from train-season pitches here.
    diag = json.loads((res / "query_diagnostics.json").read_text())
    z = np.load(res / "matrix_w00.npz")
    attrs = pd.read_csv(res / "pitcher_attributes.csv")
    seasons = diag["train_seasons"]  # traits from the model's train window only, never the matchup season
    extra = extra_attributes(load_pitches(seasons), seasons)
    extra.to_csv(res / "pitcher_attributes_extra.csv", index=False)  # also read by matchup_figures
    attrs = attrs.merge(extra, on=["pitcher", "p_throws"], how="left")
    hnames = pd.read_csv("data/processed/hitter_names.csv").set_index("batter")["name"]
    hs = pd.read_csv(diag["hitter_stats"]).set_index("batter")

    # One entry per batter-hand vs pitcher-hand group; each stores batter ids, pitcher ids, BF weights, and the quantized [hitters, pitchers] matrix.
    groups, bats, pits = {}, set(), {"R": set(), "L": set()}
    for g in GROUPS:
        bat, pit, bf, w = (z[f"{g}__{k}"] for k in ("batter", "pitcher", "bf_weight", "w00"))
        assert w.min() >= LO and w.max() <= HI, g
        groups[g] = dict(batters=bat.tolist(), pitchers=pit.tolist(), bf=bf.tolist(),
                         shape=list(w.shape), w=_b64(quantize(w)))
        bats |= set(bat.tolist())
        # g[-3] is the pitcher hand letter in names like "LHB_vs_RHP".
        pits[g[-3]] |= set(pit.tolist())

    # Look up names only for ids lacking a name in hitter_names.csv, plus all pitchers.
    missing = [b for b in bats if b not in hnames.index]
    all_pit = pits["R"] | pits["L"]
    lk = lookup_names(missing + sorted(all_pit))

    # Hitter records: name, stand, three scouting stats, and prior PA vs each hand (zero/None when no stats row).
    hitters = {}
    for b in sorted(bats):
        row = hs.loc[b] if b in hs.index else None
        hitters[str(b)] = dict(
            name=hnames.get(b, lk.get(b, str(b))),
            stand=None if row is None else row["stand"],
            **{k: (None if row is None else _r(row[k])) for k in ("ev_p90", "contact_rate", "chase_rate")},
            pa_L=0 if row is None else int(row["prior_pa_L"]), pa_R=0 if row is None else int(row["prior_pa_R"]))

    pitchers = {}
    for hand in "RL":  # one id can appear under both hands; key by hand
        a = attrs[attrs.p_throws == hand].set_index("pitcher")
        for p in sorted(pits[hand]):
            row = a.loc[p] if p in a.index else None
            pitchers[f"{hand}{p}"] = dict(
                name=lk.get(p, str(p)),
                bf=0 if row is None else int(row["bf"]),  # train-season BF: type percentiles + type averaging
                # full-ish precision: rounding creates ties that shift percentile boundaries
                **{k: (None if row is None else _r(row[k], 5)) for k in BASE + EXTRA})

    meta = dict(build=f"{diag['arm']} ({len(diag['seeds'])}-seed ensemble, trained {seasons[0]}\u2013{seasons[-1]}), "
                      f"{diag['eval_season']} matchups",
                eval_season=diag["eval_season"], train_seasons=[seasons[0], seasons[-1]],
                generated=pd.Timestamp.now().strftime("%Y-%m-%d"),
                quant=dict(lo=LO, hi=HI, dtype="uint16 little-endian, row-major hitters x pitchers"),
                note="Model predictions, descriptive. One-season data cannot validate matchup effects.")
    return dict(meta=meta, groups=groups, hitters=hitters, pitchers=pitchers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=RES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = build(ap.parse_args().res)
    OUT.write_text(json.dumps(d, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {len(d['hitters'])} hitters, {len(d['pitchers'])} pitcher-hands")
