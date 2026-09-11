"""Export the 2024 pitcher-type query matrix for the public matchup explorer (explorer/).

Descriptive only: model predictions from the 2024 exploration build, not validated
(one-season reliability of matchup contrasts ~0, see results/pitcher_type_query/ceiling.csv).
The browser does all math (double-centering, type averages) from the raw W(0,0) matrices.

Run: PYTHONPATH=. .venv/bin/python -m src.analysis.matchup_explorer_data
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results/pitcher_type_query")
OUT = Path("explorer/data/matchups.json")
GROUPS = [f"{s}HB_vs_{h}HP" for h in "RL" for s in "LR"]
LO, HI = 0.0, 1.0  # fixed wOBA range for uint16 quantization


def quantize(w):
    return np.round((np.clip(w, LO, HI) - LO) / (HI - LO) * 65535).astype("<u2")


def dequantize(q):
    return q.astype(float) / 65535 * (HI - LO) + LO


def double_center(w, bf):
    """Matchup effect: w minus hitter's BF-weighted avg minus pitcher's avg over hitters, plus grand mean."""
    p = bf / bf.sum()
    r, c = w @ p, w.mean(0)
    return w - r[:, None] - c[None, :] + c @ p


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _r(x, d=3):
    return None if pd.isna(x) else round(float(x), d)


def lookup_names(ids):
    try:
        from pybaseball import playerid_reverse_lookup
        lk = playerid_reverse_lookup(list(ids), key_type="mlbam")
        return lk.set_index("key_mlbam").apply(lambda r: f"{r.name_first} {r.name_last}".title(), axis=1).to_dict()
    except Exception as e:  # names are cosmetic; fall back to ids
        print(f"name lookup failed: {e}")
        return {}


def build():
    z = np.load(RES / "matrix_w00.npz")
    attrs = pd.read_csv(RES / "pitcher_attributes.csv")
    hnames = pd.read_csv("data/processed/hitter_names.csv").set_index("batter")["name"]
    hs = pd.read_csv("results/model_visualization/hitter_stats.csv").set_index("batter")

    groups, bats, pits = {}, set(), {"R": set(), "L": set()}
    for g in GROUPS:
        bat, pit, bf, w = (z[f"{g}__{k}"] for k in ("batter", "pitcher", "bf_weight", "w00"))
        assert w.min() >= LO and w.max() <= HI, g
        groups[g] = dict(batters=bat.tolist(), pitchers=pit.tolist(), bf=bf.tolist(),
                         shape=list(w.shape), w=_b64(quantize(w)))
        bats |= set(bat.tolist())
        pits[g[-3]] |= set(pit.tolist())

    missing = [b for b in bats if b not in hnames.index]
    all_pit = pits["R"] | pits["L"]
    lk = lookup_names(missing + sorted(all_pit))

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
                **{k: (None if row is None else _r(row[k], 5)) for k in ("fb_velo", "fb_height", "brk_share")})

    meta = dict(build="embedding_sgd_sgd_lr1 (2024 exploration build)",
                generated=pd.Timestamp.now().strftime("%Y-%m-%d"),
                quant=dict(lo=LO, hi=HI, dtype="uint16 little-endian, row-major hitters x pitchers"),
                note="Model predictions, descriptive. One-season data cannot validate matchup effects.")
    return dict(meta=meta, groups=groups, hitters=hitters, pitchers=pitchers)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = build()
    OUT.write_text(json.dumps(d, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {len(d['hitters'])} hitters, {len(d['pitchers'])} pitcher-hands")
