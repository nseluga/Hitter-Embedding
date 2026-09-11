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
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from src.analysis.matchup_explorer_data import double_center, lookup_names

RES = Path("results/pitcher_type_query")
OUT = RES / "figures"
INK, MUTED = "#3a3f47", "#8a9099"
CAVEAT = "2024 exploration build. Model matchup effect, descriptive; one season cannot validate it."
TYPES = {  # BF-weighted within-RHP percentile ranges (velo, height, brk), same presets as explorer/app.js
    "Hard throwers (top 20% velo)": (.8, 1, 0, 1, 0, 1),
    "Soft tossers (bottom 20% velo)": (0, .2, 0, 1, 0, 1),
    "Elevators (top 20% FB height)": (0, 1, .8, 1, 0, 1),
    "Low-zone (bottom 20% FB height)": (0, 1, 0, .2, 0, 1),
    "Breaking-ball heavy (top 20%)": (0, 1, 0, 1, .8, 1),
    "Fastball heavy (bottom 20% brk)": (0, 1, 0, 1, 0, .2),
    "Power elevators (velo & height top 40%)": (.6, 1, .6, 1, 0, 1),
    "Soft breakers (velo bottom 40%, brk top 40%)": (0, .4, 0, 1, .6, 1),
}
AXES = ("fb_velo", "fb_height", "brk_share")


def pct(s, bf):
    """BF-weighted percentile of each pitcher within hand."""
    o = s.sort_values()
    return (bf.reindex(o.index).cumsum() / bf.sum()).reindex(s.index)


def load():
    z = np.load(RES / "matrix_w00.npz")
    attrs = pd.read_csv(RES / "pitcher_attributes.csv")
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
    a = attrs.reindex(I.columns).dropna(subset=list(AXES))
    bf = a["bf"].astype(float)
    P = {ax: pct(a[ax], bf) for ax in AXES}
    S = {}
    for t, r in TYPES.items():
        m = np.logical_and.reduce([(P[ax] >= r[2 * i]) & (P[ax] <= r[2 * i + 1]) for i, ax in enumerate(AXES)])
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


def heatmap(S, nm, stand, pa_r):
    """45 regulars x 8 pitcher types, rows clustered so similar profiles sit together."""
    M = S.loc[pa_r.sort_values(ascending=False).index[:45]]
    M = M.iloc[leaves_list(linkage(M.to_numpy(), "average", metric="correlation"))]
    fig, a = plt.subplots(figsize=(9.5, 12))
    lim = np.nanpercentile(np.abs(M.to_numpy()), 97)
    im = a.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    a.set_xticks(range(M.shape[1]), [c.split(" (")[0].replace(" ", "\n", 1) for c in M.columns], fontsize=7.5, color=INK)
    a.xaxis.tick_top()
    a.set_yticks(range(len(M)), [f"{nm[b]} ({stand[b]})" for b in M.index], fontsize=7.5, color=INK)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M.iat[i, j]:+.0f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(M.iat[i, j]) > lim * .6 else INK)
    cb = fig.colorbar(im, ax=a, shrink=.4, pad=.02)
    cb.set_label("Matchup effect vs type (wOBA pts)\nred = better than his own RHP average", fontsize=8, color=INK)
    fig.text(.5, .005, "45 hitters with the most prior PA vs RHP. " + CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .015, 1, 1))
    return fig


MAPS = {  # one hitter map per pitcher axis: x = matchup contrast between the two ends
    "velo": ("Hard throwers (top 20% velo)", "Soft tossers (bottom 20% velo)", "better vs hard throwers", "better vs soft tossers"),
    "mix": ("Breaking-ball heavy (top 20%)", "Fastball heavy (bottom 20% brk)", "better vs breaking-ball heavy", "better vs fastball heavy"),
    "location": ("Elevators (top 20% FB height)", "Low-zone (bottom 20% FB height)", "better vs elevators", "better vs low-zone"),
}


def hitter_map(key, S, lvl, nm, stand, pa_r):
    """x = matchup contrast on one pitcher axis, y = hitter's own level vs RHP."""
    hi, lo, rlab, llab = MAPS[key]
    keep = pa_r >= 300
    x, y = (S[hi] - S[lo])[keep], lvl.reindex(S.index)[keep]
    fig, a = plt.subplots(figsize=(10, 7))
    style(a)
    a.axvline(0, color=MUTED, lw=.8)
    a.scatter(x, y, s=14, c=np.where(stand[x.index] == "L", "#2f6690", "#c8553d"), alpha=.55, lw=0)
    far = (np.abs(x.rank(pct=True) - .5) * 2 + (y.rank(pct=True) > .9)).sort_values(ascending=False)
    place_labels(a, [(x[b], y[b], nm[b]) for b in far.index[:28]])
    kw = dict(fontsize=9, color=MUTED, style="italic", transform=a.transAxes, va="top")
    a.text(.98, .99, f"{rlab} →", ha="right", **kw)
    a.text(.02, .99, f"← {llab}", ha="left", **kw)
    a.set_xlabel(f"Matchup effect: {hi.split(' (')[0].lower()} minus {lo.split(' (')[0].lower()} (wOBA pts)", fontsize=9, color=INK)
    a.set_ylabel("Hitter's own projected wOBA vs RHP (x1000)", fontsize=9, color=INK)
    a.set_title(f"{key.title()}: {len(x)} hitters with 300+ prior PA vs RHP. Blue = LHB, red = RHB.", fontsize=9, color=INK)
    fig.text(.5, .005, CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, 1))
    return fig


def gradients(I, a, bf, nm, pa_r):
    """12 most positive and 12 most negative regulars per axis, effect smoothed along the pitcher axis."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 8.5))
    regs = pa_r[pa_r >= 300].index
    for ax_, (col, lab, fmt) in zip(axs, [("fb_velo", "fastball velocity (mph)", "{:.0f}"),
                                          ("fb_height", "fastball height (ft)", "{:.1f}"),
                                          ("brk_share", "breaking-ball share", "{:.0%}")]):
        v = a[col].to_numpy()
        grid = np.quantile(v, np.linspace(.05, .95, 40))
        K = bf.to_numpy()[None, :] * np.exp(-.5 * ((grid[:, None] - v[None, :]) / (.35 * v.std())) ** 2)
        X = I.loc[regs, a.index]
        G = (X.fillna(0).to_numpy(float) @ K.T) / (X.notna().to_numpy() @ K.T)
        o = np.argsort(G[:, -8:].mean(1) - G[:, :8].mean(1))
        pick = np.r_[o[:12], o[-12:]]
        lim = np.nanpercentile(np.abs(G[pick]), 97)
        ax_.imshow(G[pick], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax_.set_yticks(range(24), [nm[regs[i]] for i in pick], fontsize=7.5, color=INK)
        ax_.axhline(11.5, color=INK, lw=1.2)
        ticks = np.linspace(0, 39, 5).astype(int)
        ax_.set_xticks(ticks, [fmt.format(grid[t]) for t in ticks], fontsize=8, color=INK)
        ax_.set_xlabel(f"RHP {lab} →", fontsize=9, color=INK)
        ax_.set_title("top 12: worse as it rises · bottom 12: better", fontsize=8, color=MUTED)
    fig.suptitle("Matchup effect along each pitcher axis (red = better than own RHP average)", fontsize=10, color=INK)
    fig.text(.5, .005, "Hitters with 300+ prior PA vs RHP. " + CAVEAT, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, .97))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I, a, bf, S, lvl, nm, stand, pa_r = load()
    S.assign(name=nm, stand=stand, prior_pa_R=pa_r).to_csv(RES / "matchup_by_type.csv")
    figs = {"type_heatmap": heatmap(S, nm, stand, pa_r), "gradients": gradients(I, a, bf, nm, pa_r),
            **{f"map_{k}": hitter_map(k, S, lvl, nm, stand, pa_r) for k in MAPS}}
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(figs)} figures to {OUT}")


if __name__ == "__main__":
    main()
