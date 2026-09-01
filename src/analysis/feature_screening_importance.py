"""
Phase B step 2 — GBM feature screen for the context tower (architecture §3 B.2, §4).

Cheap triage, not a verdict. For each of the five per-pitch process heads (§1.2)
we fit an XGBoost model on the *exact* 48-dim context vector the DL context tower
will consume (src/features/context_features.py) and ask which context features
carry out-of-sample signal for that head. Output is a proposed keep / flag / drop
feature list: a feature contributing nothing to every head needs a specific
argument to enter the DL model (§3). The definitive feature-decision stays the DL
common-window ablation (B.3, §4) — a GBM null does not prove the DL model cannot
use a feature (different model class, different interaction capacity).

Why the process heads and not the "claim-1 task" the plan names in §3: claim-1 is
hitter-level side-specific wOBA projection, and the context vector `c` carries no
hitter identity by design (that is the hitter tower's job). Context alone cannot
produce a hitter-level number, so the screen's real question is which context
features feed the per-pitch heads the shared DL trunk is built from.

Discipline (frozen split, §2.2):
  - vectorizer fit on TRAIN only (context_features.fit_on_train), applied unchanged;
  - each model trained on TRAIN seasons, early-stopped on VAL;
  - TEST (2025) is never read — not for tuning, not for importance;
  - importance is measured OUT-OF-SAMPLE on VAL (permutation + SHAP). In-sample
    importance is meaningless, and native XGBoost gain is in-sample and biased
    toward high-cardinality splits, so neither is used for the decision.

Known, accepted limitation: VAL is used both to early-stop and to measure
importance. A third split would waste data against the frozen 3-way split; for a
triage screen the shared use is acceptable and is stated, not hidden.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

from src.features import context_features as cf

# --- the five heads, each on its §1.2 nesting subset ---
# task: "clf" (binary) or "reg" (continuous). subset: which rows the head sees.
HEADS = {
    "swing":  {"task": "clf", "unit": "pitch"},   # all pitches
    "whiff":  {"task": "clf", "unit": "swing"},    # swings only; y = missed the ball
    "ev":     {"task": "reg", "unit": "bbip"},     # balls in play only
    "la":     {"task": "reg", "unit": "bbip"},
    "spray":  {"task": "reg", "unit": "bbip"},     # clipped label (|spray|<=90)
}

# importance is measured on a fixed VAL subsample so runtime is bounded and the
# result is deterministic; 200k held-out rows is ample for a stable ranking.
PERM_VAL_CAP = 200_000
PERM_REPEATS = 5
SHAP_SAMPLE = 2_000

# frozen-context inclusions (decisions 2 and the handedness context): their
# importance is informational, they enter regardless of the screen.
FROZEN_IN = {"count", "p_throws", "stand"}


def feature_groups(params):
    """
    Map the 48 context columns to source-feature groups — the actual decision unit.
    A categorical's importance must not fragment across its one-hot columns, and an
    optional continuous travels with its own missingness flag (you cannot keep the
    value without the flag). Permutation shuffles a whole group's columns jointly,
    so a one-hot block stays a valid one-hot after permutation.
    """
    names = cf.feature_names(params)
    idx = {n: i for i, n in enumerate(names)}
    groups = {}
    for c in cf.CORE_CONTINUOUS:
        groups[c] = [idx[c]]
    for c in cf.OPTIONAL_CONTINUOUS:
        groups[c] = [idx[c], idx[f"{c}_missing"]]
    groups["spin_axis"] = [idx["spin_axis_sin"], idx["spin_axis_cos"], idx["spin_axis_missing"]]
    groups["pitch_type"] = [i for n, i in idx.items() if n.startswith("pitch_type=")]
    groups["p_throws"] = [i for n, i in idx.items() if n.startswith("p_throws=")]
    groups["stand"] = [i for n, i in idx.items() if n.startswith("stand=")]
    groups["count"] = [i for n, i in idx.items() if n.startswith("balls=") or n.startswith("strikes=")]

    covered = sorted(i for cols in groups.values() for i in cols)
    assert covered == list(range(len(names))), "feature groups do not partition the 48 columns"
    return groups


def head_target(df, head):
    """
    The row mask and target vector for one head, honoring the §1.2 nesting.
    swing: all pitches. whiff: swings only, y=1 on a miss. ev/la/spray: in-play
    rows where that measurement is present (spray already clip-nulled in labels).
    """
    if head == "swing":
        mask = df["swing"].notna()
        y = df.loc[mask, "swing"].to_numpy(dtype="float32")
    elif head == "whiff":
        mask = df["swing"] == 1
        y = (df.loc[mask, "contact"] == 0).to_numpy(dtype="float32")
    else:  # ev / la / spray
        mask = df[head].notna()
        y = df.loc[mask, head].to_numpy(dtype="float32")
    return mask.to_numpy(), y


def fit_head(X_tr, y_tr, X_val, y_val, task, seed=0):
    """
    Train one XGBoost head, early-stopped on VAL (that is the 'tuned on val' step).
    Light fixed hyperparameters — this is a triage screen, not a tuned competitor;
    a full sweep would over-fit the screen to VAL and burn budget for no decision.
    """
    common = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  tree_method="hist", n_jobs=-1, random_state=seed,
                  early_stopping_rounds=30)
    if task == "clf":
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    else:
        model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _metric(model, X, y, task):
    """Primary held-out metric: log-loss for a classifier, RMSE for a regressor (lower better)."""
    if task == "clf":
        p = model.predict_proba(X)[:, 1]
        return float(log_loss(y, p, labels=[0, 1]))
    return float(root_mean_squared_error(y, model.predict(X)))


def base_rate_metric(y, task):
    """
    The metric of the trivial predictor (base rate for a classifier, mean for a
    regressor). The model must beat this on VAL — the loss-scale sanity gate.
    """
    if task == "clf":
        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return float(log_loss(y, np.full(len(y), p), labels=[0, 1]))
    return float(root_mean_squared_error(y, np.full(len(y), y.mean())))


def permutation_importance(model, X, y, groups, task, seed=0, n_repeats=PERM_REPEATS):
    """
    Grouped permutation importance on held-out X: for each group, shuffle that
    group's columns jointly across rows and measure how much the primary metric
    worsens. Returns {group: (mean_increase, std_increase)}. A joint row shuffle
    keeps one-hot blocks valid and measures the group's combined contribution.
    """
    rng = np.random.default_rng(seed)
    baseline = _metric(model, X, y, task)
    out = {}
    for name, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, cols] = X[np.ix_(perm, cols)]
            deltas.append(_metric(model, Xp, y, task) - baseline)
        out[name] = (float(np.mean(deltas)), float(np.std(deltas)))
    return baseline, out


def shap_importance(model, X_sample, groups):
    """
    Mean |SHAP| per group on a small VAL sample (TreeExplainer is exact and fast
    for trees). Complements permutation with per-feature attribution; grouped by
    summing column contributions so it is comparable to the permutation ranking.
    """
    explainer = shap_explainer(model)
    vals = explainer.shap_values(X_sample)
    mean_abs = np.abs(vals).mean(axis=0)
    return {name: float(mean_abs[cols].sum()) for name, cols in groups.items()}


def shap_explainer(model):
    """Isolated so the gate can stub it; TreeExplainer over the fitted booster."""
    import shap
    return shap.TreeExplainer(model)


def screen_head(df_ctx, X_all, params, splits, head, seed=0):
    """
    Run the full screen for one head and return its result dict: held-out metric,
    base-rate metric, and the permutation + SHAP importance tables. df_ctx carries
    the label columns aligned row-for-row with X_all (the vectorized context).
    """
    task = HEADS[head]["task"]
    groups = feature_groups(params)
    season = df_ctx["season"].to_numpy()
    train_seasons = set(splits["split"]["train"])
    val_seasons = set(splits["split"]["val"])

    mask, y = head_target(df_ctx, head)
    in_train = mask & np.isin(season, list(train_seasons))
    in_val = mask & np.isin(season, list(val_seasons))
    # split-boundary gate: this head touches only train/val seasons, never test
    used = set(np.unique(season[in_train | in_val]).tolist())
    assert used.issubset(train_seasons | val_seasons), f"{head}: selected a non-train/val season {used}"
    X_tr, y_tr = X_all[in_train], y[np.isin(season[mask], list(train_seasons))]
    X_val, y_val = X_all[in_val], y[np.isin(season[mask], list(val_seasons))]

    model = fit_head(X_tr, y_tr, X_val, y_val, task, seed=seed)

    # fixed, seeded VAL subsample for importance (bounded runtime, deterministic)
    rng = np.random.default_rng(seed)
    take = rng.permutation(len(X_val))[:min(PERM_VAL_CAP, len(X_val))]
    Xv, yv = X_val[take], y_val[take]

    baseline, perm = permutation_importance(model, Xv, yv, groups, task, seed=seed)
    shap_take = take[:min(SHAP_SAMPLE, len(take))]
    shap_vals = shap_importance(model, X_val[shap_take], groups)

    return {
        "head": head,
        "task": task,
        "n_train": int(in_train.sum()),
        "n_val": int(in_val.sum()),
        "n_val_scored": int(len(take)),
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "val_metric": baseline,
        "base_rate_metric": base_rate_metric(y_val, task),
        "importance": {
            g: {"perm": perm[g][0], "perm_std": perm[g][1],
                "perm_norm": perm[g][0] / baseline if baseline else float("nan"),
                "shap": shap_vals[g]}
            for g in groups
        },
    }


def importance_frame(result):
    """Flatten one head's importance dict to a tidy, ranked DataFrame."""
    rows = [{"group": g, **v} for g, v in result["importance"].items()]
    return pd.DataFrame(rows).sort_values("perm", ascending=False).reset_index(drop=True)


def decision_table(results, keep_frac=0.01):
    """
    Combine per-head permutation importance into a keep / flag / drop proposal.
    A feature is KEEP if it clears `keep_frac` of the head's baseline metric for at
    least one head, or is a frozen-in context (count / handedness / stand). Anything
    else is FLAG (near-zero everywhere) — a candidate for exclusion that needs a
    specific argument, per §3. Nothing is auto-dropped: the DL ablation (B.3) rules.
    """
    groups = list(results[0]["importance"].keys())
    rows = []
    for g in groups:
        norms = {r["head"]: r["importance"][g]["perm_norm"] for r in results}
        best = max(norms.values())
        frozen = g in FROZEN_IN
        verdict = "keep" if (frozen or best >= keep_frac) else "flag"
        rows.append({
            "group": g, "verdict": verdict, "frozen_in": frozen,
            "best_head": max(norms, key=norms.get), "best_perm_norm": best,
            **{f"norm_{h}": norms[h] for h in sorted(norms)},
        })
    return pd.DataFrame(rows).sort_values(["verdict", "best_perm_norm"], ascending=[True, False]).reset_index(drop=True)


def load_context(labeled_path, params):
    """
    Load the labeled table, vectorize its context with the train-fit params, and
    return (df with labels+season, X). X rows align with df rows one-for-one.
    """
    need = list(dict.fromkeys(
        cf.STANDARDIZED + cf.CIRCULAR + cf.CATEGORICAL
        + ["season", "swing", "contact", "ev", "la", "spray"]
    ))
    df = pd.read_parquet(labeled_path, columns=need).reset_index(drop=True)
    X, _ = cf.transform(df, params)
    return df, X


def _run_gates(df, X, params, splits):
    """
    ml-engineer real-data gates, printed before any head is fit and any result
    recorded. Config split-boundary disjointness + a decode-a-batch eyeball so a
    join or feature/label misalignment shows up against the source rows.
    """
    tr, va, te = (set(splits["split"][k]) for k in ("train", "val", "test"))
    assert tr.isdisjoint(va), "train/val season overlap"
    assert te.isdisjoint(tr | va), "test season leaks into train/val"
    assert X.shape[0] == len(df), "context matrix and label frame are misaligned"
    print(f"gate: seasons train={sorted(tr)} val={sorted(va)} test(untouched)={sorted(te)}")
    print(f"gate: context matrix {X.shape}, non-finite={int((~np.isfinite(X)).sum())}")
    names = cf.feature_names(params)
    val = df[df["season"].isin(va)]
    print("gate: decode-a-batch (val context vs raw):")
    print(cf.decode_sample(val, X[df["season"].isin(va).to_numpy()], names, params, n=6).to_string(index=False))
    print()


def run_screen(labeled_path, params_path, out_dir, splits, seed=0, max_rows=None):
    """Orchestrate all five heads; write per-head importance CSVs + the decision table."""
    params = cf.load_params(params_path)
    df, X = load_context(labeled_path, params)
    if max_rows is not None and len(df) > max_rows:  # smoke path only
        keep = np.random.default_rng(seed).permutation(len(df))[:max_rows]
        keep.sort()
        df, X = df.iloc[keep].reset_index(drop=True), X[keep]

    _run_gates(df, X, params, splits)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for head in HEADS:
        res = screen_head(df, X, params, splits, head, seed=seed)
        results.append(res)
        importance_frame(res).to_csv(out / f"b2_{head}_importance.csv", index=False)
        beat = "beats" if res["val_metric"] < res["base_rate_metric"] else "FAILS"
        print(f"{head:6s} n_tr={res['n_train']:>8} n_val={res['n_val']:>8} "
              f"val={res['val_metric']:.4f} base={res['base_rate_metric']:.4f} [{beat} base rate]")

    table = decision_table(results)
    table.to_csv(out / "b2_feature_decisions.csv", index=False)
    (out / "b2_screen_summary.json").write_text(json.dumps(
        {r["head"]: {k: r[k] for k in ("task", "n_train", "n_val", "val_metric",
                                       "base_rate_metric", "best_iteration")} for r in results},
        indent=2))
    print("\nfeature decisions (flag = near-zero everywhere, needs an argument to enter):")
    print(table[["group", "verdict", "frozen_in", "best_head", "best_perm_norm"]].to_string(index=False))
    return results, table


def main():
    import argparse

    from src.config.splits import load_splits

    parser = argparse.ArgumentParser(description="Phase B.2 GBM feature screen (XGBoost + permutation + SHAP).")
    parser.add_argument("--labeled", default="data/processed/pitch_events_labeled.parquet")
    parser.add_argument("--params", default="src/features/context_vectorizer_params.json")
    parser.add_argument("--out-dir", default="results/phase_b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="subsample for a fast smoke run; omit for the real screen")
    args = parser.parse_args()

    run_screen(args.labeled, args.params, args.out_dir, load_splits(),
               seed=args.seed, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
