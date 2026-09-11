"""Static pitcher-type matchup figures for notebook 09 (the interactive version lives in explorer/).

Matchup effect = double-centered W(0,0): projected wOBA vs the pitcher minus the hitter's own
average vs that hand minus the pitcher's average over hitters. RHP only. 2024 exploration build,
descriptive, not validated (one-season ceiling ~0, results/pitcher_type_query/ceiling.csv).

Run: PYTHONPATH=. .venv/bin/python -m src.analysis.matchup_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.analysis.matchup_explorer_data import double_center, lookup_names

RES = Path("results/pitcher_type_query")
OUT = RES / "figures"
INK, MUTED = "#3a3f47", "#8a9099"
CAVEAT = "2024 exploration build. Model matchup effect, descriptive; one season cannot validate it."
# Location is by pitch class against each batter's zone (fb_up, brk_down from matchup_explorer_data),
# not average fastball height. Each type = BF-weighted within-RHP percentile ranges on the axes it names.
TYPES = {
    "Hard throwers (top 20% velo)": {"fb_velo": (.8, 1)},
    "Soft tossers (bottom 20% velo)": {"fb_velo": (0, .2)},
    "FB up (top 20% FB up share)": {"fb_up": (.8, 1)},
    "FB low (bottom 20% FB up share)": {"fb_up": (0, .2)},
    "Brk buried (top 20% brk below zone)": {"brk_down": (.8, 1)},
    "Brk in zone (bottom 20% brk below zone)": {"brk_down": (0, .2)},
    "Breaking-ball heavy (top 20%)": {"brk_share": (.8, 1)},
    "Fastball heavy (bottom 20% brk)": {"brk_share": (0, .2)},
    "North-south (FB up & brk buried top 40%)": {"fb_up": (.6, 1), "brk_down": (.6, 1)},
    "Power elevators (velo & FB up top 40%)": {"fb_velo": (.6, 1), "fb_up": (.6, 1)},
    "Soft breakers (velo bottom 40%, brk top 40%)": {"fb_velo": (0, .4), "brk_share": (.6, 1)},
}
AXES = ("fb_velo", "fb_up", "brk_down", "brk_share")


def pct(s, bf):
    """BF-weighted percentile of each pitcher within hand."""
    o = s.dropna().sort_values()  # pitchers missing the trait get NaN and fall in no type
    return (bf.reindex(o.index).cumsum() / bf.reindex(o.index).sum()).reindex(s.index)


def load():
    z = np.load(RES / "matrix_w00.npz")
    attrs = pd.read_csv(RES / "pitcher_attributes.csv")
    attrs = attrs.merge(pd.read_csv(RES / "pitcher_attributes_extra.csv"), on=["pitcher", "p_throws"], how="left")
    attrs = attrs[attrs.p_throws == "R"].set_index("pitcher")  # one id can appear under both hands
    names = pd.read_csv("data/processed/hitter_names.csv").set_index("batter")["name"]
    hs = pd.read_csv("results/model_visualization/hitter_stats.csv").set_index("batter")
    I_parts, lvl = [], []
    for stand in "LR":
        k = f"{stand}HB_vs_RHP"
        bat, pit, bf, w = (z[f"{k}__{s}"] for s in ("batter", "pitcher", "bf_weight", "w00"))
        I_parts.append(pd.DataFrame(double_center(w, bf) * 1000, index=bat, columns=pit))
        lvl.append(pd.Series(w @ (bf / bf.sum()) * 1000, index=bat))
    I, lvl = pd.concat(I_parts), pd.concat(lvl)  # NaN where a pitcher is absent from a stand group
    a = attrs.reindex(I.columns).dropna(subset=["fb_velo", "fb_up", "brk_share"])  # brk_down may be NaN
    bf = a["bf"].astype(float)
    P = {ax: pct(a[ax], bf) for ax in AXES}
    S = {}
    for t, r in TYPES.items():
        m = np.logical_and.reduce([(P[ax] >= lo) & (P[ax] <= hi) for ax, (lo, hi) in r.items()])
        ids = a.index[m]
        X, wt = I[ids], bf[ids]
        S[t] = (X.fillna(0) @ wt) / (X.notna() @ wt)
    S = pd.DataFrame(S).astype(float)
    lk = lookup_names([b for b in S.index if b not in names.index])
    nm = pd.Series({b: names.get(b, lk.get(b, str(b))) for b in S.index})
    stand = hs["stand"].reindex(S.index).fillna("?")
    pa_r = hs["prior_pa_R"].reindex(S.index).fillna(0)
    return I, a, bf, S, lvl, nm, stand, pa_r


def style(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(colors=INK, labelsize=8)


def place_labels(a, pts):
    """Greedy de-overlap: push a label down until its box clears every label already placed."""
    # ponytail: greedy vertical nudge, swap for adjustText if maps get denser
    a.figure.canvas.draw()
    k, placed = a.figure.dpi / 72, []
    for x, y, t in sorted(pts, key=lambda r: -r[1]):
        px, py = a.transData.transform((x, y))
        w, dy = len(t) * 4.2 * k, 0.0
        while any(px < qx + qw and qx < px + w and abs(py - dy - qy) < 9 * k for qx, qy, qw in placed):
            dy += 9 * k
        placed.append((px, py - dy, w))
        a.annotate(t, (x, y), xytext=(3, 2 - dy / k), textcoords="offset points", fontsize=7, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=.4) if dy else None)


MAPS = {  # one hitter map per pitcher axis: x = matchup contrast between the two ends
    "velo": ("Hard throwers (top 20% velo)", "Soft tossers (bottom 20% velo)", "better vs hard throwers", "better vs soft tossers"),
    "mix": ("Breaking-ball heavy (top 20%)", "Fastball heavy (bottom 20% brk)", "better vs breaking-ball heavy", "better vs fastball heavy"),
    "location": ("FB up (top 20% FB up share)", "FB low (bottom 20% FB up share)", "better vs fastballs up", "better vs fastballs low"),
    "brk_location": ("Brk buried (top 20% brk below zone)", "Brk in zone (bottom 20% brk below zone)",
                     "better vs buried breaking balls", "better vs breaking balls in the zone"),
}


SHORT = {"velo": ("hard throwers", "soft tossers"), "mix": ("breaking-ball heavy", "fastball heavy"),
         "location": ("fastballs up", "fastballs low"), "brk_location": ("buried breaking balls", "breaking balls in zone")}
UNITS = {"fb_velo": ("fastball velocity (mph)", "{:.0f}"), "fb_up": ("share of fastballs up in zone", "{:.0%}"),
         "brk_down": ("share of breaking balls below zone", "{:.0%}"),
         "brk_share": ("breaking-ball share", "{:.0%}")}
EXAMPLES = (545361, 592450, 660271, 665742, 518692, 605141)  # well-known regulars, not picked for extreme maps
LIM = 20  # fixed color range (wOBA pts) so hitters compare on one scale


def contrast(S, key):
    hi, lo, *_ = MAPS[key]
    return S[hi] - S[lo]


def woba_colors(a, lvl, **kw):
    """Scatter colored by the hitter's own projected wOBA vs RHP (red = better)."""
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=float(np.median(lvl)), vmin=float(lvl.quantile(.02)), vmax=float(lvl.quantile(.98)))
    sc = a.scatter(c=lvl, cmap="RdBu_r", norm=norm, alpha=.75, lw=0, s=16, **kw)
    cb = a.figure.colorbar(sc, ax=a, shrink=.5, pad=.01)
    cb.set_label("His projected wOBA vs RHP (x1000)", fontsize=8, color=INK)


def label_far(a, x, y, nm, n=28):
    far = (np.abs(x.rank(pct=True) - .5) + np.abs(y.rank(pct=True) - .5)).sort_values(ascending=False)
    place_labels(a, [(x[b], y[b], nm[b]) for b in far.index[:n]])


def hitter_map(key, S, lvl, nm, pa_r):
    """x = matchup contrast on one pitcher axis, y = hitter's own level vs RHP, color = same level."""
    hi, lo, rlab, llab = MAPS[key]
    keep = pa_r >= 300
    x, y = contrast(S, key)[keep], lvl.reindex(S.index)[keep]
    fig, a = plt.subplots(figsize=(10, 7))
    style(a)
    a.axvline(0, color=MUTED, lw=.8)
    woba_colors(a, y, x=x, y=y)
    label_far(a, x, y, nm)
    kw = dict(fontsize=9, color=MUTED, style="italic", transform=a.transAxes, va="top")
    a.text(.98, .99, f"{rlab} →", ha="right", **kw)
    a.text(.02, .99, f"← {llab}", ha="left", **kw)
    a.set_xlabel(f"Matchup effect: {hi.split(' (')[0].lower()} minus {lo.split(' (')[0].lower()} (wOBA pts)", fontsize=9, color=INK)
    a.set_ylabel("Hitter's own projected wOBA vs RHP (x1000)", fontsize=9, color=INK)
    a.set_title(f"{key.replace('_', ' ').title()}: {len(x)} hitters with 300+ prior PA vs RHP.", fontsize=9, color=INK)
    fig.text(.5, .005, CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, 1))
    return fig


PAIRS = (("location", "brk_location"), ("velo", "mix"), ("velo", "location"))  # first = north-south


def pair_map(kx, ky, S, lvl, nm, pa_r):
    """x, y = matchup contrasts on two pitcher axes; each quadrant = the pitcher combo that suits him."""
    keep = pa_r >= 300
    x, y, v = contrast(S, kx)[keep], contrast(S, ky)[keep], lvl.reindex(S.index)[keep]
    fig, a = plt.subplots(figsize=(10, 8))
    style(a)
    a.axvline(0, color=MUTED, lw=.8)
    a.axhline(0, color=MUTED, lw=.8)
    woba_colors(a, v, x=x, y=y)
    label_far(a, x, y, nm, 30)
    (xh, xl), (yh, yl) = SHORT[kx], SHORT[ky]
    kw = dict(fontsize=9, color=MUTED, style="italic", transform=a.transAxes)
    for tx, ty, ha, va, s1, s2 in ((.98, .99, "right", "top", xh, yh), (.02, .99, "left", "top", xl, yh),
                                   (.98, .01, "right", "bottom", xh, yl), (.02, .01, "left", "bottom", xl, yl)):
        a.text(tx, ty, f"better vs {s1}\n& {s2}", ha=ha, va=va, **kw)
    a.set_xlabel(f"Matchup effect: {xh} minus {xl} (wOBA pts)", fontsize=9, color=INK)
    a.set_ylabel(f"Matchup effect: {yh} minus {yl} (wOBA pts)", fontsize=9, color=INK)
    r = np.corrcoef(x, y)[0, 1]
    a.set_title(f"{kx.replace('_', ' ').title()} x {ky.replace('_', ' ')}: {len(x)} hitters with 300+ prior PA vs RHP (top vs bottom 20% on each axis; r = {r:.2f}).",
                fontsize=9, color=INK)
    fig.text(.5, .005, CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, 1))
    return fig


def smooth2d(e, a, bf, xa, ya, n=30):
    """BF-weighted Gaussian kernel average of one hitter's effects over a 2D pitcher-attribute grid.
    Returns grid axes, smoothed effect, and support (share of peak kernel BF; low = few pitchers nearby)."""
    ok = (e.notna() & a[xa].notna() & a[ya].notna()).to_numpy()
    x, y, w, v = a[xa].to_numpy()[ok], a[ya].to_numpy()[ok], bf.to_numpy()[ok], e.to_numpy()[ok]
    gx, gy = (np.quantile(q, np.linspace(.05, .95, n)) for q in (x, y))
    Kx = np.exp(-.5 * ((gx[:, None] - x[None]) / (.35 * x.std())) ** 2)
    Ky = np.exp(-.5 * ((gy[:, None] - y[None]) / (.35 * y.std())) ** 2)
    num, den = (Ky * w) @ (Kx * v).T, (Ky * w) @ Kx.T  # [ny, nx]
    return gx, gy, num / den, den / den.max()


def profiles(xa, ya, I, a, bf, nm, stand, bids=EXAMPLES):
    """One 2D matchup profile per example hitter: color = effect vs pitchers near that (x, y)."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax_, b in zip(axs.flat, bids):
        gx, gy, G, sup = smooth2d(I.loc[b, a.index], a, bf, xa, ya)
        G = np.where(sup < .08, np.nan, G)  # ponytail: fixed 8% support cutoff, tune if edges look noisy
        ax_.set_facecolor("#e6e4df")
        im = ax_.pcolormesh(gx, gy, G, cmap="RdBu_r", vmin=-LIM, vmax=LIM, shading="nearest")
        ok = I.loc[b, a.index].notna() & a[xa].notna() & a[ya].notna()
        ax_.scatter(a.loc[ok, xa], a.loc[ok, ya], s=2, c=INK, alpha=.15, lw=0)
        ax_.set_xlim(gx[0], gx[-1]); ax_.set_ylim(gy[0], gy[-1])
        ax_.set_title(f"{nm[b]} ({stand[b]}HB)", fontsize=9, color=INK)
        style(ax_)
    for ax_ in axs[-1]:
        lab, fmt = UNITS[xa]
        ax_.set_xlabel(f"RHP {lab} →", fontsize=9, color=INK)
        ax_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _, f=fmt: f.format(t)))
    for ax_ in axs[:, 0]:
        lab, fmt = UNITS[ya]
        ax_.set_ylabel(f"RHP {lab} →", fontsize=9, color=INK)
        ax_.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _, f=fmt: f.format(t)))
    cb = fig.colorbar(im, ax=axs, shrink=.5, pad=.02)
    cb.set_label(f"Matchup effect (wOBA pts, fixed ±{LIM})\nred = better than his own RHP average", fontsize=8, color=INK)
    fig.suptitle(f"Hitter matchup profiles: {UNITS[xa][0]} x {UNITS[ya][0]} (RHP). Dots = pitchers; gray = too few pitchers nearby.",
                 fontsize=10, color=INK)
    fig.text(.5, .005, CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I, a, bf, S, lvl, nm, stand, pa_r = load()
    S.assign(name=nm, stand=stand, prior_pa_R=pa_r).to_csv(RES / "matchup_by_type.csv")
    figs = {**{f"profile_{x}_{y}": profiles(x, y, I, a, bf, nm, stand)
               for x, y in (("fb_up", "brk_down"), ("fb_velo", "fb_up"), ("fb_velo", "brk_share"))},
            **{f"map_{k}": hitter_map(k, S, lvl, nm, pa_r) for k in MAPS},
            **{f"pair_{x}_{y}": pair_map(x, y, S, lvl, nm, pa_r) for x, y in PAIRS}}
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(figs)} figures to {OUT}")


if __name__ == "__main__":
    main()
