import numpy as np
import pandas as pd

from src.analysis.embedding_norm_check import norm_exposure, reference_bar


def _stats(n):
    return pd.DataFrame({
        "embedding_index": np.arange(1, n + 1),
        "log_prior_pa": np.linspace(3, 8, n),
        "stratum": np.repeat(["low", "medium", "high"], n // 3)})


def test_norm_exposure_signs_follow_norm_trend():
    n = 60
    stats = _stats(n)
    rng = np.random.default_rng(0)
    up = np.zeros((n + 1, 4))
    up[1:, 0] = np.linspace(1, 3, n)  # norm rises with exposure
    up[1:, 1:] = rng.normal(0, 0.01, (n, 3))
    down = up.copy(); down[1:, 0] = up[1:, 0][::-1]
    a, b = norm_exposure(up, stats), norm_exposure(down, stats)
    assert a["slope"] > 0 and a["slope_ci95"][0] > 0
    assert b["slope"] < 0 and b["r"] < 0
    assert set(a["spread"]) == {"low", "medium", "high"} and a["n"] == n


def test_reference_bar_one_sided():
    rows = [{"stage": "rebuild", "config": "baseline", "seed": str(s), "status": "ok",
             "reference": str(1.0 + 0.001 * s)} for s in range(5)]
    rows += [{"stage": "x", "config": "good", "seed": "0", "status": "ok", "reference": "0.99"},
             {"stage": "x", "config": "bad", "seed": "0", "status": "ok", "reference": "1.2"}]
    good, bad = reference_bar(rows, "x", "good"), reference_bar(rows, "x", "bad")
    assert good["passes"] and good["advisory"] and good["margin"] > 0
    assert not bad["passes"]
    assert reference_bar(rows, "x", "none")["passes"] is None
