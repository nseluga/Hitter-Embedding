import numpy as np
import pandas as pd

from src.analysis.matchup_explorer_data import dequantize, double_center, extra_attributes, quantize


def test_extra_attributes_mirror_hands_and_skip_eval_season():
    # an RHP and an LHP with mirror-image pitches must get identical traits
    rows = []
    for hand, s in (("R", 1), ("L", -1)):
        for pt, px, pz in (("FF", -0.5, 3.3), ("SI", -0.5, 2.0), ("SL", 0.4, 1.0), ("CH", -0.3, 1.2)):
            rows += [dict(pitcher=1, p_throws=hand, season=2020, pitch_type=pt, release_speed=90.0 if pt in ("FF", "SI") else 82.0,
                          pfx_x=s * px, pfx_z=1.0, release_pos_x=-s * 2.0, release_pos_z=6.0, release_extension=6.5,
                          release_spin_rate=2300.0, plate_x=s * 0.5, plate_z=pz, sz_top=3.5, sz_bot=1.5, zone=5)] * 25
    rows.append(dict(rows[0], season=2025, release_speed=200.0))  # sealed season: must be ignored
    out = extra_attributes(pd.DataFrame(rows), [2020]).set_index("p_throws")
    r = out.loc["R"]
    assert np.allclose(out.loc["L", r.index].astype(float), r.astype(float))
    assert np.isclose(r.fb_run, 6.0) and np.isclose(r.brk_sweep, 4.8) and np.isclose(r.rel_side, 2.0)
    assert np.isclose(r.glove_side, 6.0) and np.isclose(r.velo_gap, 8.0) and np.isclose(r.mix_depth, 4.0)
    assert np.isclose(r.si_share, 0.5) and np.isclose(r.off_share, 0.25) and np.isclose(r.fb_ivb, 12.0)
    # location by pitch class, against the batter's zone: FF at 90% of zone height is up, SI at 25% is not
    assert np.isclose(r.fb_up, 0.5) and np.isclose(r.brk_down, 1.0) and np.isclose(r.off_down, 1.0) and np.isclose(r.brk_height, 1.0)


def test_quantize_round_trip():
    w = np.random.default_rng(0).uniform(0.03, 0.7, (50, 80))
    assert np.abs(dequantize(quantize(w)) - w).max() <= 1e-5


def test_double_center_removes_hitter_and_pitcher_means():
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 0.5, (30, 40))
    bf = rng.uniform(1, 100, 40)
    I = double_center(w, bf)
    assert np.allclose(I @ (bf / bf.sum()), 0)  # each hitter's BF-weighted effect is 0
    assert np.allclose(I.mean(0), I.mean(0)[0])  # pitcher columns share one offset
    # additive matrix has no matchup effect
    add = rng.normal(size=30)[:, None] + rng.normal(size=40)[None, :]
    assert np.allclose(double_center(add, bf), 0)
