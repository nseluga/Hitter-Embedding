"""Draft-3 figures, all on the CLEAN build (selection arm clean_clean_dim64, refit clean_dim64_final).

Every number drawn is read from a results/ file; nothing here is a hard-coded result.
Run from the repo root:  PYTHONPATH=. .venv/bin/python paper/draft3/make_figs3.py [name ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/draft3/figs"
RES = ROOT / "results"
VC = RES / "v_chain_clean"

MODEL, EB, GBM = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FAINT, GRID = "#2b2b2b", "#7a7a7a", "#bdbdbd", "#ebebeb"
# diverging: two hues, gray midpoint (deliberately not the method colours)
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#3a5f8f", "#9db4cf", "#efefef", "#e0a98f", "#a8452b"])
SEQ_BLUES = ["#a9c9ee", "#5b9be0", "#1f5aa6"]  # one hue, light -> dark

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


def tidy(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ 1. reliability curve
def fig_reliability():
    panel = pd.read_csv(RES / "feature_screening/stabilization_panel.csv")
    row = panel[panel["metric"] == "wOBA vs LHP"].iloc[0]
    n_star, lo, hi = row["vc_n50"], row["vc_n50_lo"], row["vc_n50_hi"]
    sys.path.insert(0, str(ROOT))
    from src.analysis.claim1_eval import STRATUM_BOUNDARIES
    cut_lo, cut_hi = STRATUM_BOUNDARIES
    max_pa = 1200
    n = np.linspace(0, max_pa, 1201)
    rel = n / (n + n_star)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    tidy(ax)
    for x0, x1, shade, label, xt in ((0, cut_lo, "#f4f4f4", "Low", cut_lo / 2),
                                     (cut_lo, cut_hi, "#e9e9e9", "Medium", (n_star + cut_hi) / 2 + 20),
                                     (cut_hi, max_pa, "#f4f4f4", "High", (cut_hi + max_pa) / 2)):
        ax.axvspan(x0, x1, color=shade, lw=0, zorder=0)
        ax.text(xt, 0.985, label, ha="center", va="top", color=MUTED, fontsize=7.5)
    for cut in (cut_lo, cut_hi):
        ax.axvline(cut, color=FAINT, lw=0.6, zorder=1)
    ax.axvspan(lo, hi, color=MODEL, alpha=0.12, lw=0, zorder=1)
    ax.plot(n, rel, color=INK, lw=1.3, zorder=3, label="Split-half reliability, n / (n + n*)")
    ax.plot(n, np.sqrt(rel), color=MUTED, lw=1.1, ls="--", zorder=3,
            label="Rank-correlation ceiling, √reliability")
    ax.axvline(n_star, color=MODEL, lw=0.9, zorder=2)
    ax.plot([n_star], [0.5], "o", ms=3.5, color=MODEL, zorder=4)
    ax.annotate(f"n* = {n_star:.0f} PA\n95% CI [{lo:.0f}, {hi:.0f}]", (n_star, 0.5), xytext=(8, -26),
                textcoords="offset points", fontsize=7.5, color=MODEL)
    ax.set_xlim(0, max_pa)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, cut_lo, round(n_star), cut_hi, 800, 1200])
    ax.set_xlabel("Prior plate appearances vs LHP (n)")
    ax.set_ylabel("Reliability")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.04))
    save(fig, "fig_reliability.png")


# ------------------------------------------------------------------ 2. calibration 2025
def fig_calibration_2025():
    cal = pd.read_csv(RES / "model_evaluation_final_clean/calibration.csv").set_index("stratum")
    bins = pd.read_csv(RES / "model_evaluation_final_clean/calibration_reliability.csv")
    span = (bins["mean_pred"].min(), bins["mean_pred"].max())
    lim = (0.22, 0.42)
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.95), sharex=True, sharey=True)
    for ax, stratum in zip(axes, ["low", "medium", "high", "all"]):
        r = cal.loc[stratum]
        tidy(ax, grid="both")
        ax.plot(lim, lim, color=FAINT, lw=0.9, ls="--", label="Perfect calibration")
        x = np.linspace(*lim, 50)
        ax.plot(x, r["intercept"] + r["slope"] * x, color=MODEL, lw=0.8, alpha=0.35)
        xs = np.linspace(*span, 50)
        ax.plot(xs, r["intercept"] + r["slope"] * xs, color=MODEL, lw=1.6, label="Fitted line")
        if stratum == "all":
            ax.plot(bins["mean_pred"], bins["mean_obs"], "o", ms=2.8, color=INK, label="Binned hitters")
        ax.text(0.04, 0.96, f"{stratum.capitalize()} (n = {int(r['n_hitters'])})", transform=ax.transAxes,
                va="top", fontsize=7.5)
        ax.text(0.96, 0.04, f"slope {r['slope']:.2f}\n± {r['slope_se']:.2f} SE", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=7.5, color=MODEL)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([0.25, 0.30, 0.35, 0.40])
        ax.set_xticklabels([".25", ".30", ".35", ".40"])
        ax.set_yticks([0.25, 0.30, 0.35, 0.40])
        ax.set_yticklabels([".25", ".30", ".35", ".40"])
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed 2025 wOBA")
    fig.supxlabel("Projected wOBA (side-specific)", fontsize=8, y=0.0)
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [l for ax in axes for l in ax.get_legend_handles_labels()[1]]
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.83))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.82, wspace=0.08)
    save(fig, "fig_calibration_2025.png")


# ------------------------------------------------------------------ 3. head partial heatmap
HEAD_LABELS = {"swing": "Swing head", "contact": "Contact head", "split_inplay": "In-play head",
               "ev": "Exit velocity head", "la": "Launch angle head", "spray": "Spray head"}
STAT_LABELS = {"swing_rate": "Swing rate", "whiff_rate": "Whiff rate", "contact_rate": "Contact rate",
               "chase_rate": "Chase rate", "zone_swing_rate": "In-zone swing rate",
               "ev_mean": "Mean exit velo", "ev_p90": "90th-pct exit velo", "la_mean": "Mean launch angle",
               "bat_speed_mean": "Bat speed", "pull_rate": "Pull rate", "woba_level": "wOBA",
               "obs_platoon_diff": "Observed platoon split"}


def fig_head_partial():
    d = pd.read_csv(VC / "head_partial_loadings.csv")
    r = d.pivot(index="head", columns="stat", values="r_partial").reindex(
        index=list(HEAD_LABELS), columns=list(STAT_LABELS))
    sig = d.assign(s=(d["ci_low"] > 0) | (d["ci_high"] < 0)).pivot(
        index="head", columns="stat", values="s").reindex(index=list(HEAD_LABELS), columns=list(STAT_LABELS))
    vmax = np.ceil(np.abs(r.to_numpy()).max() * 10) / 10
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    im = ax.imshow(r.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.iat[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-0.", "−.").replace("0.", "."), ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.45 else (INK if sig.iat[i, j] else MUTED))
    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels([STAT_LABELS[c] for c in r.columns], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(r.shape[0]))
    ax.set_yticklabels([HEAD_LABELS[h] for h in r.index])
    ax.tick_params(length=0, labelcolor=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("Partial correlation,\nlog prior PA held fixed")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, "fig_head_partial.png")


# ------------------------------------------------------------------ 4. model vs EB disagreement
LABEL_HITTERS = {  # name -> (label offset in points, ha); styling only
    "Robbie Grossman": ((-4, 15), "left"), "Luis Robert": ((10, -2), "left"),
    "Aaron Judge": ((-12, 14), "right"), "José Altuve": ((-14, 2), "right"),
    "Marcell Ozuna": ((-14, -12), "right"),
}


def fig_disagreement():
    d = pd.read_csv(VC / "disagreement.csv")
    summ = json.loads((VC / "disagreement_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    tidy(ax, grid="both")
    top = max(d["rank_eb"].max(), d["rank_model"].max())
    ax.plot([1, top], [1, top], color=FAINT, lw=0.9, ls="--", zorder=1)
    for stratum, color in zip(("low", "medium", "high"), SEQ_BLUES):
        s = d[d["stratum"] == stratum]
        st = summ[stratum]
        ax.scatter(s["rank_eb"], s["rank_model"], s=9, color=color, lw=0, alpha=0.85, zorder=2,
                   label=f"{stratum.capitalize()}: ρ = {st['spearman']:.2f} (n = {st['n_hitters']})")
    import matplotlib.patheffects as pe
    for name, ((dx, dy), ha) in LABEL_HITTERS.items():
        r = d[d["name"] == name].iloc[0]
        ax.scatter([r["rank_eb"]], [r["rank_model"]], s=18, facecolor="none", edgecolor=INK, lw=0.8, zorder=3)
        ax.annotate(name, (r["rank_eb"], r["rank_model"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center", ha=ha, zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=1, shrinkB=3))
    ax.set_xlim(-5, top + 10)
    ax.set_ylim(-22, top + 26)
    ax.set_xlabel("Rank by empirical Bayes projection (1 = most favors LHP, within batting side)")
    ax.set_ylabel("Rank by model projection")
    p = summ["pooled"]
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              title=f"Spearman ρ by prior-PA stratum, all hitters {p['spearman']:.2f} (n = {p['n_hitters']})",
              title_fontsize=7.5, handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig_disagreement.png")


# ------------------------------------------------------------------ 5. location surfaces
SURFACE_ORDER = ["Joc Pederson", "Kyle Schwarber", "Juan Soto", "Mike Trout", "Alec Bohm"]


def fig_surfaces():
    sys.path.insert(0, str(ROOT))
    from src.analysis.model_visualization_anchors import LOCATION_X_EDGES, LOCATION_Z_EDGES, N_LOCATION_BINS
    loc = pd.read_csv(VC / "surfaces_location.csv")
    loc = loc[loc["quantity"] == "q"]
    stand = pd.read_csv(VC / "hitter_stats.csv").set_index("batter")["stand"]
    vmax = np.ceil(np.abs(loc["delta"]).max() * 1000 / 10) * 10
    extent = (LOCATION_X_EDGES[0], LOCATION_X_EDGES[-1], LOCATION_Z_EDGES[0], LOCATION_Z_EDGES[-1])
    hands = ["L", "R"]
    # not sharex: the In/Out tick labels differ by batting side
    fig, axes = plt.subplots(2, 5, figsize=(6.5, 2.85), sharey=True)
    for i, hand in enumerate(hands):
        for j, name in enumerate(SURFACE_ORDER):
            ax = axes[i, j]
            cell = loc[(loc["anchor_name"] == name) & (loc["p_throws"] == hand)]
            grid = np.full((N_LOCATION_BINS, N_LOCATION_BINS), np.nan)
            for _, r in cell.iterrows():
                grid[int(r["z_bin"]), int(r["x_bin"])] = r["delta"] * 1000
            im = ax.imshow(grid, cmap=DIVERGING, vmin=-vmax, vmax=vmax, origin="lower", extent=extent)
            # rule-book zone outline (17-in plate, typical 1.5-3.5 ft heights) for orientation
            ax.add_patch(plt.Rectangle((-17 / 24, 1.5), 17 / 12, 2.0, fill=False, ec=INK, lw=0.6, ls=":"))
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            b = cell["anchor"].iat[0]
            side = stand[b]
            ax.set_xticks([LOCATION_X_EDGES[0] + 0.3, LOCATION_X_EDGES[-1] - 0.3])
            # catcher's view: plate_x < 0 is the third-base side, i.e. inside to a right-handed batter
            ax.set_xticklabels(["In", "Out"] if side == "R" else ["Out", "In"], fontsize=7)
            ax.set_yticks([1.5, 2.5, 3.5])
            if i == 0:
                ax.set_title(f"{name.split()[-1]} ({side}HB)", fontsize=8, pad=3)
        axes[i, 0].set_ylabel(f"vs {hand}HP\nHeight (ft)")
    fig.supxlabel("Horizontal location, catcher's view (in = toward the batter)", fontsize=8, y=0.03)
    fig.subplots_adjust(wspace=0.06, hspace=0.14, right=0.86, top=0.92, bottom=0.12)
    cax = fig.add_axes([0.885, 0.15, 0.015, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Expected wOBA on contact,\nhitter minus league (points)", fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, "fig_surfaces.png")


# ------------------------------------------------------------------ 6. hard-thrower matchup (clean query)
PTQ = RES / "pitcher_type_query_clean"
MIN_PRIOR_PA_R = 300  # same regular-hitter floor as src/analysis/matchup_figures.py maps


def hard_thrower_effect(res_dir=PTQ, hitter_stats=VC / "hitter_stats.csv"):
    """Per-hitter matchup effect vs hard throwers (top 20% BF-weighted fb_velo among RHP), wOBA points.
    Same definition as src/analysis/matchup_figures.py: double-centred W(0,0), BF-weighted over the type."""
    sys.path.insert(0, str(ROOT))
    from src.analysis.matchup_explorer_data import double_center
    from src.analysis.matchup_figures import pct
    z = np.load(res_dir / "matrix_w00.npz")
    attrs = pd.read_csv(res_dir / "pitcher_attributes.csv")
    attrs = attrs[attrs["p_throws"] == "R"].set_index("pitcher")
    parts = []
    for stand in "LR":
        k = f"{stand}HB_vs_RHP"
        bat, pit, bf, w = (z[f"{k}__{s}"] for s in ("batter", "pitcher", "bf_weight", "w00"))
        parts.append(pd.DataFrame(double_center(w, bf) * 1000, index=bat, columns=pit))
    effect = pd.concat(parts)
    a = attrs.reindex(effect.columns).dropna(subset=["fb_velo"])
    p = pct(a["fb_velo"], a["bf"].astype(float))
    out = {}
    for label, (lo, hi) in {"hard": (0.8, 1.0), "soft": (0.0, 0.2)}.items():
        ids = a.index[(p >= lo) & (p <= hi)]
        X, wt = effect[ids], a.loc[ids, "bf"].astype(float)
        out[label] = (X.fillna(0) @ wt) / (X.notna() @ wt)
    df = pd.DataFrame(out)
    hs = pd.read_csv(hitter_stats).set_index("batter")
    names = pd.read_csv(ROOT / "data/processed/hitter_names_clean.csv").set_index("batter")["name"]
    df["name"] = names.reindex(df.index)
    df["stand"] = hs["stand"].reindex(df.index)
    df["prior_pa_R"] = hs["prior_pa_R"].reindex(df.index).fillna(0)
    return df


def fig_velocity():
    if not (PTQ / "matrix_w00.npz").exists():
        print("skip fig_velocity: run the clean pitcher_type_query --stage query first (see figs3 report)")
        return None
    df = hard_thrower_effect()
    reg = df[df["prior_pa_R"] >= MIN_PRIOR_PA_R].sort_values("hard")
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    tidy(ax, grid="x")
    y = np.arange(len(reg))
    ax.axvline(0, color=FAINT, lw=0.8)
    ax.scatter(reg["hard"], y, s=5, color=MODEL, lw=0, alpha=0.75)
    ends = pd.concat([reg.head(5), reg.tail(5)])
    for b, r in ends.iterrows():
        yy = y[reg.index.get_loc(b)]
        left = r["hard"] < 0
        ax.annotate(f"{r['name']} ({r['hard']:+.1f})", (r["hard"], yy),
                    xytext=(5 if left else -5, 0), textcoords="offset points",
                    ha="left" if left else "right", va="center", fontsize=7, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel(f"{len(reg)} hitters, ordered by effect")
    ax.set_xlabel("Projected matchup effect vs hard-throwing RHP (wOBA points)")
    save(fig, "fig_velocity.png")
    return df, reg


# ------------------------------------------------------------------ 7. miss vs damage
def miss_damage_axes():
    """Recompute the clean LDA axes exactly as src/analysis/embedding_alt_views.run does."""
    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)  # the alt-views helpers read repo-relative default paths
    from src.analysis import embedding_alt_views as av
    from src.analysis.embedding_structure import (DEFAULT_ARM, DEFAULT_CHECKPOINT_DIR, HITTER_STATS_PATH,
                                                  NAMES_PATH, load_hitters, unit_normalize)
    from results.embedding_structure import axis_screen
    embedding, hitters, _ = load_hitters(DEFAULT_CHECKPOINT_DIR, DEFAULT_ARM, HITTER_STATS_PATH, NAMES_PATH)
    normalized, _ = unit_normalize(embedding)
    contrasts = axis_screen.build_contrasts(hitters)
    contrasts["whiff_rate_brk"], contrasts["whiff_rate_fb"] = av.whiff_rate_components_by_batter()
    contrasts["ev_p90_brk"], contrasts["ev_p90_fb"] = av.ev_p90_components_by_batter()
    out = {}
    for pid, xm, ym, ex, ey in av.AXIS_PAIRINGS:
        if pid in ("spin_vs_fastball", "spin_vs_fastball_damage"):
            summary, xp, yp, *_ = av.run_pairing(pid, xm, ym, ex, ey, normalized, hitters, contrasts)
            out[pid] = (summary, xp, yp)
    return out, DEFAULT_ARM


def fig_miss_damage():
    stored = json.loads((VC / "embedding_structure/embedding_alt_views.json").read_text())["pairings"]
    axes_data, arm = miss_damage_axes()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.05), sharex=True, sharey=True)
    titles = {"spin_vs_fastball": ("Whiff (miss)", "Breaking-ball whiff axis", "Fastball whiff axis"),
              "spin_vs_fastball_damage": ("Exit velocity (damage)", "Breaking-ball power axis",
                                          "Fastball power axis")}
    for ax, (pid, (summary, xp, yp)) in zip(axes, axes_data.items()):
        corr = summary["axis_correlation"]
        # the recomputed axes must be the ones the clean JSON describes
        assert abs(corr["lda_projected"] - stored[pid]["axis_correlation"]["lda_projected"]) < 1e-6, pid
        zx, zy = (xp - xp.mean()) / xp.std(), (yp - yp.mean()) / yp.std()
        tidy(ax, grid="both")
        ax.plot([-4, 4], [-4, 4], color=FAINT, lw=0.9, ls="--")
        ax.scatter(zx, zy, s=4, color=MODEL, lw=0, alpha=0.45, zorder=2)
        head, xl, yl = titles[pid]
        ax.set_title(head, fontsize=8, loc="left")
        ax.text(0.97, 0.03, f"embedding axes r = {corr['lda_projected']:.2f}\n"
                f"observed hitter stats r = {corr['raw_metric']:.2f}\nn = {summary['n_hitters_plotted']} hitters",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_xlabel(f"{xl} (SD units)")
        ax.set_ylabel(f"{yl} (SD units)")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-4.6, 3.3)  # room under the cloud for the stats text
        ax.set_aspect("equal")
    axes[1].yaxis.set_tick_params(labelleft=True)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_miss_damage.png")
    return axes_data, arm


# ------------------------------------------------------------------ 8. level query, held-out 2024
def fig_level_query():
    d = pd.read_csv(VC / "paper_figures/level_query_heldout.csv")
    j = json.loads((VC / "paper_figures/level_query_heldout.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    tidy(a, grid="both"); tidy(b, grid="x")
    for stratum, color in zip(("low", "medium", "high"), SEQ_BLUES):
        s = d[d["stratum"] == stratum]
        a.scatter(s["level_query"], s["woba_2024"], s=np.clip(s["pa_2024"] / 25, 2, 30), color=color,
                  lw=0, alpha=0.8, label=f"{stratum.capitalize()} (n = {j[stratum]['n']})")
    a.set_xlabel("Level query, projected wOBA against all pitchers")
    a.set_ylabel("Observed 2024 wOBA (held out)")
    a.legend(loc="upper left", handletextpad=0.2, markerscale=1.5)
    rows = [("Pooled", j["pooled"]), ("Low", j["low"]), ("Medium", j["medium"]), ("High", j["high"])]
    for i, (lab, r) in enumerate(rows):
        c = INK if lab == "Pooled" else SEQ_BLUES[i - 1]
        b.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=2, solid_capstyle="round")
        b.scatter([r["r_partial"]], [i], s=40, color=c, edgecolor="white", lw=1.2, zorder=3)
        b.annotate(f"{r['r_partial']:.2f}", (r["r_partial"], i), xytext=(0, 7), textcoords="offset points",
                   ha="center", fontsize=7.5)
    b.axvline(0, color=FAINT, lw=0.8)
    b.set_yticks(range(len(rows)), [l for l, _ in rows])
    b.set_ylim(len(rows) - 0.5, -0.6)
    b.set_xlim(-0.05, 0.75)
    b.set_xlabel("Partial r, given log prior PA")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_level_query.png")


FIGS = {"reliability": fig_reliability, "calibration": fig_calibration_2025, "head_partial": fig_head_partial,
        "disagreement": fig_disagreement, "surfaces": fig_surfaces, "velocity": fig_velocity,
        "miss_damage": fig_miss_damage, "level_query": fig_level_query}

if __name__ == "__main__":
    for key in sys.argv[1:] or FIGS:
        FIGS[key]()
