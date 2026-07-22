"""
Phase B.1 report generator. Reuses the tested stabilization estimators to emit the
signal-per-PA analysis as durable artifacts under results/phase_b/: the full panel,
the common-PA-axis ranking (#4), the spray-clipping check (#5), the wOBA survivorship
decomposition, and three figures. Deterministic and seeded; rerun to regenerate.

Run: python -m src.analysis.b1_report
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import stabilization as stab

OUT = "results/phase_b"
MAX_TRAIN_SEASON = 2023
GRID = [10, 25, 50, 100, 200, 400, 800]
N_BOOT = 300
SEED = 0
PROCESS_COLOR = "#4c8dff"
OUTCOME_COLOR = "#e0574a"


def panel_row(label, df, value_col, unit):
    """Full stabilization panel for one metric: VC point + CI at both thresholds,
    and split-half random (optimistic) vs sequential (across-time) at 0.5."""
    stats = stab.hitter_stats(df, "batter", value_col)
    p50, lo50, hi50 = stab.stabilization_ci(stats, 0.5, n_boot=N_BOOT, seed=SEED)
    p70, lo70, hi70 = stab.stabilization_ci(stats, 0.7, n_boot=N_BOOT, seed=SEED)
    metric = stab.mean_metric(value_col)
    rand = stab.stabilization_point(stab.stabilization_curve(df, "batter", metric, GRID, seed=SEED, n_resamples=5), 0.5)
    seq = stab.stabilization_point(stab.stabilization_curve(df, "batter", metric, GRID, seed=SEED, split="sequential"), 0.5)
    return {"metric": label, "unit": unit, "n_hitters": len(stats),
            "vc_n50": p50, "vc_n50_lo": lo50, "vc_n50_hi": hi50,
            "vc_n70": p70, "vc_n70_lo": lo70, "vc_n70_hi": hi70,
            "splithalf_random_50": rand, "splithalf_sequential_50": seq}


def build_panel(labeled, pa):
    swings = labeled[labeled["swing"] == 1].copy()
    swings["whiff"] = (swings["contact"] == 0).astype(float)
    in_play = labeled[labeled["ev"].notna()]
    sprayed = labeled[labeled["spray"].notna()]
    rows = [
        panel_row("swing rate", labeled, "swing", "pitches"),
        panel_row("whiff rate", swings, "whiff", "swings"),
        panel_row("exit velocity", in_play, "ev", "BBIP"),
        panel_row("launch angle", in_play, "la", "BBIP"),
        panel_row("spray angle", sprayed, "spray", "BBIP"),
        panel_row("whiff vs LHP", swings[swings["p_throws"] == "L"], "whiff", "swings"),
        panel_row("whiff vs RHP", swings[swings["p_throws"] == "R"], "whiff", "swings"),
        panel_row("exit velocity vs LHP", in_play[in_play["p_throws"] == "L"], "ev", "BBIP"),
        panel_row("exit velocity vs RHP", in_play[in_play["p_throws"] == "R"], "ev", "BBIP"),
        panel_row("wOBA vs LHP", pa[pa["p_throws"] == "L"], "woba_points", "PA"),
        panel_row("wOBA vs RHP", pa[pa["p_throws"] == "R"], "woba_points", "PA"),
    ]
    return pd.DataFrame(rows)


def conversion_factors(labeled):
    """League-average observations-per-PA on the modeling table, to put each metric's
    native-unit stabilization point on a common PA axis. Approximate: uses pooled
    league rates, not per-hitter rates."""
    n_pa = labeled.drop_duplicates(["game_pk", "at_bat_number"]).shape[0]
    return {"pitches": len(labeled) / n_pa,
            "swings": labeled["swing"].sum() / n_pa,
            "BBIP": labeled["ev"].notna().sum() / n_pa,
            "PA": 1.0}


def ranking(panel, factors):
    """Common-PA-axis ranking (#4): convert each metric's VC n*(0.5) to PA-equivalent
    and sort fastest-first. Only the core one-per-head metrics, not the matched cuts."""
    core = ["swing rate", "whiff rate", "exit velocity", "launch angle", "spray angle", "wOBA vs LHP", "wOBA vs RHP"]
    rows = []
    for _, r in panel[panel["metric"].isin(core)].iterrows():
        per_pa = factors[r["unit"]]
        rows.append({"metric": r["metric"], "unit": r["unit"], "vc_n50_native": r["vc_n50"],
                     "obs_per_pa": per_pa, "vc_n50_pa_equiv": r["vc_n50"] / per_pa,
                     "kind": "outcome" if "wOBA" in r["metric"] else "process"})
    return pd.DataFrame(rows).sort_values("vc_n50_pa_equiv").reset_index(drop=True)


def spray_clipping(labeled):
    """Spray-clipping check (#5): does clipping the ~1% near-plate artifact (|spray|>90)
    speed up spray stabilization? Compare unclipped / clipped-at-90 / dropped(>90)."""
    sprayed = labeled[labeled["spray"].notna()].copy()
    treatments = {
        "unclipped": sprayed,
        "clipped_90": sprayed.assign(spray=sprayed["spray"].clip(-90, 90)),
        "dropped_gt90": sprayed[sprayed["spray"].abs() <= 90],
    }
    rows = []
    for name, df in treatments.items():
        stats = stab.hitter_stats(df, "batter", "spray")
        p50, lo, hi = stab.stabilization_ci(stats, 0.5, n_boot=N_BOOT, seed=SEED)
        rand = stab.stabilization_point(stab.stabilization_curve(df, "batter", stab.mean_metric("spray"), GRID, seed=SEED, n_resamples=5), 0.5)
        rows.append({"treatment": name, "n_bbip": len(df), "n_hitters": len(stats),
                     "vc_n50": p50, "vc_n50_lo": lo, "vc_n50_hi": hi, "splithalf_random_50": rand})
    return pd.DataFrame(rows)


def woba_survivorship(pa):
    """wOBA VC stabilization by population floor: shows n* doubling as low-exposure
    hitters are dropped, because restricting to durable regulars halves the signal
    variance. This is why VC (all hitters) and split-half (durable tip) diverge."""
    rows = []
    for hand in ["L", "R"]:
        side = pa[pa["p_throws"] == hand]
        counts = side.groupby("batter").size()
        for floor in [0, 400, 800]:
            keep = counts[counts >= floor].index
            stats = stab.hitter_stats(side[side["batter"].isin(keep)], "batter", "woba_points")
            signal, noise = stab.variance_components(stats)
            rows.append({"hand": hand, "pa_floor": floor, "n_hitters": len(keep),
                         "signal_var": signal, "noise_var": noise,
                         "vc_n50": stab.stabilization_point_vc(signal, noise, 0.5)})
    return pd.DataFrame(rows)


def fig_ranking(rank, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PROCESS_COLOR if k == "process" else OUTCOME_COLOR for k in rank["kind"]]
    ax.barh(rank["metric"], rank["vc_n50_pa_equiv"], color=colors)
    for y, v in enumerate(rank["vc_n50_pa_equiv"]):
        ax.text(v + 3, y, f"{v:.0f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("stabilization point, PA-equivalent (r=0.5, variance-components)")
    ax.set_title("Signal-per-PA: process (blue) stabilizes faster than the outcome (red)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_survivorship(surv, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    floors = [0, 400, 800]
    x = np.arange(len(floors))
    for i, hand in enumerate(["L", "R"]):
        vals = [surv[(surv.hand == hand) & (surv.pa_floor == f)]["vc_n50"].iloc[0] for f in floors]
        ax.bar(x + i * 0.35, vals, 0.35, label=f"vs {hand}HP",
               color=OUTCOME_COLOR if hand == "L" else "#e0a030")
        for xi, v in zip(x + i * 0.35, vals):
            ax.text(xi, v + 4, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels([f">= {f} side-PA\n(fewer, more durable)" if f else "all hitters" for f in floors])
    ax.set_ylabel("wOBA stabilization n* (PA, r=0.5)")
    ax.set_title("Survivorship: restricting to durable hitters doubles wOBA n*")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_spray(spray, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(spray["treatment"], spray["vc_n50"], color=PROCESS_COLOR)
    for x, v in enumerate(spray["vc_n50"]):
        ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("spray stabilization n* (BBIP, r=0.5)")
    ax.set_title("Spray-clipping check: near-plate artifact treatment")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    labeled = pd.read_parquet("data/processed/pitch_events_labeled.parquet",
                              columns=["batter", "season", "p_throws", "game_pk", "at_bat_number",
                                       "swing", "contact", "ev", "la", "spray"])
    labeled = labeled[labeled["season"] <= MAX_TRAIN_SEASON]
    pa = pd.read_parquet("data/processed/eval_targets_pa.parquet",
                         columns=["batter", "season", "p_throws", "woba_points", "in_denominator"])
    pa = pa[(pa["season"] <= MAX_TRAIN_SEASON) & (pa["in_denominator"] == 1)]

    panel = build_panel(labeled, pa)
    factors = conversion_factors(labeled)
    rank = ranking(panel, factors)
    spray = spray_clipping(labeled)
    surv = woba_survivorship(pa)

    panel.to_csv(f"{OUT}/stabilization_panel.csv", index=False)
    rank.to_csv(f"{OUT}/signal_per_pa_ranking.csv", index=False)
    spray.to_csv(f"{OUT}/spray_clipping.csv", index=False)
    surv.to_csv(f"{OUT}/woba_survivorship.csv", index=False)
    pd.DataFrame([factors]).to_csv(f"{OUT}/conversion_factors.csv", index=False)

    fig_ranking(rank, f"{OUT}/fig_signal_per_pa_ranking.png")
    fig_survivorship(surv, f"{OUT}/fig_woba_survivorship.png")
    fig_spray(spray, f"{OUT}/fig_spray_clipping.png")

    print("conversion factors (per PA):", {k: round(v, 3) for k, v in factors.items()})
    print("\nsignal-per-PA ranking:\n", rank.to_string(index=False))
    print("\nspray clipping:\n", spray.to_string(index=False))
    print("\nwOBA survivorship:\n", surv.to_string(index=False))
    print(f"\nwrote CSVs + 3 figures to {OUT}/")


if __name__ == "__main__":
    main()
