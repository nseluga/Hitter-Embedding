"""Unit tests for the label-derivation pipeline. Synthetic fixtures only, no snapshot dependency."""

import numpy as np
import pandas as pd
import pytest

from src.data import labels


def frame(descriptions):
    """A frame carrying one row per given description string."""
    return pd.DataFrame({"description": list(descriptions)})


# expected swing/contact for every value in the cleaned snapshot's enum
EXPECTED = {
    "ball":                    (0, pd.NA),
    "called_strike":           (0, pd.NA),
    "blocked_ball":            (0, pd.NA),
    "hit_by_pitch":            (0, pd.NA),
    "intent_ball":             (0, pd.NA),
    "swinging_strike":         (1, 0),
    "swinging_strike_blocked": (1, 0),
    "foul":                    (1, 1),
    "foul_tip":                (1, 1),
    "hit_into_play":           (1, 1),
}


def test_swing_and_contact_map_every_description():
    df = frame(EXPECTED.keys())
    result = labels.add_swing_contact_labels(df)
    for _, row in result.iterrows():
        want_swing, want_contact = EXPECTED[row["description"]]
        assert row["swing"] == want_swing
        if pd.isna(want_contact):
            assert pd.isna(row["contact"])
        else:
            assert row["contact"] == want_contact


def test_swing_is_never_null():
    df = frame(EXPECTED.keys())
    result = labels.add_swing_contact_labels(df)
    assert result["swing"].notna().all()


def test_contact_is_null_exactly_on_takes():
    df = frame(EXPECTED.keys())
    result = labels.add_swing_contact_labels(df)
    swung = result["swing"].astype(bool)
    # contact is defined exactly where a swing occurred
    assert result.loc[~swung, "contact"].isna().all()
    assert result.loc[swung, "contact"].notna().all()


def test_unknown_description_raises():
    df = frame(["ball", "not_a_real_description"])
    with pytest.raises(ValueError, match="unknown pitch descriptions"):
        labels.add_swing_contact_labels(df)


def test_input_frame_not_mutated():
    df = frame(["hit_into_play"])
    labels.add_swing_contact_labels(df)
    assert "swing" not in df.columns


# ---- Block L2: contact-quality labels ----

def make_batted(**overrides):
    """A batted-ball row with valid coordinates and launch data, overridable."""
    row = {
        "description": "hit_into_play", "stand": "R",
        "hc_x": 125.42, "hc_y": 100.0, "launch_speed": 95.0, "launch_angle": 20.0,
    }
    row.update(overrides)
    return row


def test_spray_zero_up_the_middle():
    # a ball at the home-plate x with y in front of the plate points to center
    df = pd.DataFrame([make_batted(hc_x=labels.HOME_PLATE_HC_X, hc_y=100.0)])
    result = labels.add_contact_quality_labels(df)
    assert result["spray"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_pull_is_positive_for_both_hands():
    # RHB pulls to LF (hc_x left of home), LHB pulls to RF (hc_x right of home)
    rhb_pull = pd.DataFrame([make_batted(stand="R", hc_x=75.42, hc_y=100.0)])
    lhb_pull = pd.DataFrame([make_batted(stand="L", hc_x=175.42, hc_y=100.0)])
    assert labels.add_contact_quality_labels(rhb_pull)["spray"].iloc[0] > 0
    assert labels.add_contact_quality_labels(lhb_pull)["spray"].iloc[0] > 0


def test_mirroring_is_symmetric():
    # same field location gives equal-and-opposite spray for opposite hands
    loc = dict(hc_x=75.42, hc_y=100.0)
    rhb = labels.add_contact_quality_labels(pd.DataFrame([make_batted(stand="R", **loc)]))
    lhb = labels.add_contact_quality_labels(pd.DataFrame([make_batted(stand="L", **loc)]))
    assert rhb["spray"].iloc[0] == pytest.approx(-lhb["spray"].iloc[0])


def test_quality_null_off_the_in_play_domain():
    # a foul carries EV/LA but is not quality-labeled: all three must be null
    df = pd.DataFrame([make_batted(description="foul")])
    result = labels.add_contact_quality_labels(df)
    assert result[["ev", "la", "spray"]].iloc[0].isna().all()


def test_quality_present_on_in_play():
    df = pd.DataFrame([make_batted(launch_speed=102.3, launch_angle=15.0)])
    result = labels.add_contact_quality_labels(df)
    assert result["ev"].iloc[0] == pytest.approx(102.3)
    assert result["la"].iloc[0] == pytest.approx(15.0)
    assert result["spray"].notna().iloc[0]


def test_in_play_without_coordinates_has_null_spray_but_keeps_ev():
    # spray can be null inside the domain when hit coordinates are missing
    df = pd.DataFrame([make_batted(hc_x=np.nan, hc_y=np.nan)])
    result = labels.add_contact_quality_labels(df)
    assert pd.isna(result["spray"].iloc[0])
    assert result["ev"].notna().iloc[0]


def test_near_plate_spray_artifact_is_clipped_but_keeps_ev():
    # hit coords just behind the plate origin (hc_y > HOME_PLATE_HC_Y) make the
    # angle formula blow past 90 deg; spray is nulled, EV survives (launch tracking)
    df = pd.DataFrame([make_batted(hc_x=175.0, hc_y=labels.HOME_PLATE_HC_Y + 5.0,
                                   launch_speed=98.0)])
    raw = labels.field_side_angle(df).iloc[0]
    assert abs(raw) > labels.SPRAY_ABS_MAX  # precondition: this row is the artifact
    result = labels.add_contact_quality_labels(df)
    assert pd.isna(result["spray"].iloc[0])
    assert result["ev"].iloc[0] == pytest.approx(98.0)


def test_reconciliation_reports_and_clears_spray_clip():
    rows = [make_batted(hc_x=75.42, hc_y=100.0),                       # normal pull, kept
            make_batted(hc_x=175.0, hc_y=labels.HOME_PLATE_HC_Y + 5.0)]  # near-plate artifact
    report = labels.reconcile_labels(labels.derive_labels(pd.DataFrame(rows)))
    assert report["n_spray_clipped"] == 1
    assert report["n_extreme_spray_gt90"] == 0
    assert report["n_spray"] == 1


# ---- Block L3: orchestrator, reconciliation, decode gate ----

def mixed_frame():
    """One row per description, batted rows carrying valid launch/coordinate data."""
    rows = []
    for description in EXPECTED:
        rows.append(make_batted(description=description))
    return pd.DataFrame(rows)


def test_derive_labels_produces_all_label_columns():
    result = labels.derive_labels(mixed_frame())
    assert all(col in result.columns for col in labels.LABEL_COLUMNS)


def test_reconciliation_identities_close():
    result = labels.derive_labels(mixed_frame())
    report = labels.reconcile_labels(result)
    assert report["n_swing"] == report["n_whiff"] + report["n_contact"]
    assert report["n_contact"] == sum(report["contact_breakdown"].values())
    assert report["n_pitches"] == report["n_swing"] + report["n_take"]


def test_reconciliation_counts_in_play_without_coordinates():
    rows = [make_batted(), make_batted(hc_x=np.nan, hc_y=np.nan)]  # one in-play ball has no coords
    report = labels.reconcile_labels(labels.derive_labels(pd.DataFrame(rows)))
    assert report["n_in_play"] == 2
    assert report["n_spray"] == 1
    assert report["n_in_play_without_coords"] == 1


def test_nesting_guard_fires_on_leaked_quality_label(monkeypatch):
    # simulate a regression where the quality head is labeled off the in-play domain
    def leaky_quality(df):
        df = df.copy()
        df["ev"], df["la"], df["spray"] = 95.0, 20.0, 5.0  # set on every row, takes included
        return df

    monkeypatch.setattr(labels, "add_contact_quality_labels", leaky_quality)
    with pytest.raises(AssertionError, match="in-play domain"):
        labels.derive_labels(frame(["called_strike"]))


def test_decode_sample_shape_and_columns():
    result = labels.derive_labels(mixed_frame())
    sample = labels.decode_sample(result, n=5, seed=0)
    assert list(sample.columns) == ["description"] + labels.LABEL_COLUMNS
    assert len(sample) == 5
