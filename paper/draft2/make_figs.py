"""Draft-2 paper figures (ceiling, case studies). Run from repo root with the venv python."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "paper/draft2/figs"
C = {"model": "#2a78d6", "eb": "#eb6834", "gbm": "#1baf7a"}  # dataviz default slots 1-3
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "xtick.color": "#444", "ytick.color": "#444"})


def ceiling():
    ex = pd.read_csv("results/measurement_ceiling_clean/differential_exhibit.csv")
    ex = ex[ex["exhibit_column"] == "pooled"].set_index("model")
    lev = json.load(open("results/measurement_ceiling_clean/level_ceiling_level_ceiling.json"))["intersection"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1, 2.2]})
    a.bar([0], [lev["achieved_rank_corr_weighted"]], width=0.5, color=C["model"])
    a.axhline(lev["ceiling_rank_corr"], color="#333", lw=1.2, ls="--")
    a.text(0.3, lev["ceiling_rank_corr"] + 0.02, f"ceiling {lev['ceiling_rank_corr']:.2f}", fontsize=8, ha="right")
    a.text(0, lev["achieved_rank_corr_weighted"] + 0.02, f"{lev['achieved_rank_corr_weighted']:.2f}", ha="center", fontsize=8)
    a.set_xticks([0], ["Model"]); a.set_xlim(-0.6, 0.6); a.set_ylim(0, 0.85)
    a.set_ylabel("Rank correlation with 2024 outcome"); a.set_title("Level of wOBA", fontsize=9)
    rows = [("model_v1_model", "Model", C["model"]), ("eb_bivariate", "Empirical Bayes", C["eb"]),
            ("gbm_full", "Boosted", C["gbm"])]
    for i, (k, lab, col) in enumerate(rows):
        r = ex.loc[k]
        b.bar(i, r["rank_corr_within_stand"], width=0.5, color=col)
        b.errorbar(i, r["rank_corr_within_stand"], yerr=[[r["rank_corr_within_stand"] - r["rank_corr_ci_low"]],
                   [r["rank_corr_ci_high"] - r["rank_corr_within_stand"]]], color="#333", capsize=3, lw=1)
    main_c = ex.loc["model_v1_model", "rank_ceiling_analytic_pearson"]
    b.axhline(main_c, color="#333", lw=1.2, ls="--"); b.axhline(0.1671, color="#999", lw=1, ls=":")
    b.text(2.45, main_c + 0.01, f"ceiling, main {main_c:.2f}", fontsize=8, ha="right")
    b.text(2.45, 0.1671 + 0.01, "ceiling, alternative 0.17", fontsize=8, ha="right", color="#666")
    b.set_xticks(range(3), [r[1] for r in rows]); b.set_ylim(0, 0.85)
    b.set_title("Platoon split (L minus R), side average removed", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ceiling.png", dpi=200); plt.close(fig)


def cases():
    d = pd.read_csv("results/case_studies_2025/case_studies_2025.csv")
    keep = ["Joc Pederson", "Kyle Schwarber", "Juan Soto", "Mike Trout", "Alec Bohm",
            "Aaron Judge", "José Altuve", "Marcell Ozuna",
            "Riley Adams", "Nick Allen", "Lenyn Sosa", "Luis Robert", "Chas Mccormick", "Jordan Walker"]
    d = d.set_index("name").loc[keep].reset_index()
    d["name"] = d["name"].replace({"Chas Mccormick": "Chas McCormick"})
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    y = list(range(len(d)))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.errorbar(r["delta_obs"], yi, xerr=1.96 * r["se_obs"], fmt="o", color="#333", ms=4, lw=1, capsize=2)
        ax.plot(r["delta_pred"], yi + 0.18, "D", color=C["model"], ms=5)
        ax.plot(r["delta_eb"], yi - 0.18, "s", color=C["eb"], ms=5)
    ax.axvline(0, color="#bbb", lw=0.8)
    # rows bottom-up: medium-band disagreements (0-5), named disagreements (6-8), anchors (9-13)
    for cut in (5.5, 8.5):
        ax.axhline(cut, color="#ddd", lw=0.8)
    ax.set_yticks(y, [f"{n} ({s})" for n, s in zip(d["name"], d["stand"])])
    ax.set_xlabel("2025 platoon split, wOBA vs LHP minus wOBA vs RHP")
    ax.plot([], [], "o", color="#333", label="Observed, ±1.96 SE")
    ax.plot([], [], "D", color=C["model"], label="Model")
    ax.plot([], [], "s", color=C["eb"], label="Empirical Bayes")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cases.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ceiling(); cases()
