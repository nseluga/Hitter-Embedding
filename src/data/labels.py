"""
Derive the factorized process-head labels from the modeling pitch table.

The conditional model factorizes each pitch outcome into a nested chain (§1.2 of
the architecture plan): p(swing) -> p(contact | swing) -> p(quality | contact).
This module produces those targets. Every label is a pure function of the
pitch's OWN row (its `description` and its own batted-ball measurement); nothing
here reads another row, so "no future-dated features" holds by construction.

Labels are TRAINING targets derived from the (deliberately filtered) modeling
table. They are not evaluation ground truth: claim-1 targets are aggregated from
a complete outcome source, never from this table (two-table principle).

Block L1: swing and contact labels.
Block L2: spray angle + the in-play-only contact-quality labels.
Block L3 (this commit): orchestrator, reconciliation report, and CLI entrypoint.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# a swing that misses: bat-on-ball did not happen
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
# a swing that touches the ball (fouls included; a foul is real bat-on-ball)
CONTACT_DESCRIPTIONS = {"foul", "foul_tip", "hit_into_play"}
# any swing is a whiff or a contact; derived so the two can never drift apart
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | CONTACT_DESCRIPTIONS

# no bat sent at the pitch
TAKE_DESCRIPTIONS = {"ball", "called_strike", "blocked_ball", "hit_by_pitch", "intent_ball"}

# the full enum present in the cleaned snapshot; a value outside this set is
# schema drift and must not be silently misclassified as a take
KNOWN_DESCRIPTIONS = SWING_DESCRIPTIONS | TAKE_DESCRIPTIONS

# home-plate origin of the Statcast hit-coordinate frame, in the hc_x/hc_y units.
# MLB does not publish these; the values are the established community convention,
# agreed by abdwr3e (App. C), BGSU's Statcast reference, and Weise's savant writeup.
# The L2 regression-guard (mean spray ~ 0, field thirds) checks our export matches.
HOME_PLATE_HC_X = 125.42
HOME_PLATE_HC_Y = 198.27

# the only description that yields a batted ball in play; the contact-quality head
# is defined on exactly this set. Fouls are contact but carry no in-play outcome
# (and no spray), so they are NOT quality-labeled even when EV/LA are present.
IN_PLAY_DESCRIPTION = "hit_into_play"

# a fair ball lies within the ~90-deg foul-line wedge, so |spray| > 90 is
# physically impossible. Those rows are a near-plate coordinate artifact: when hit
# coords sit near the plate origin the angle formula blows up. The spray label is
# unreliable there and is nulled; EV/LA come from launch tracking (not hit coords)
# and stay valid. Effect measured in results/phase_b/spray_clipping.csv (VC n*
# 82->73); the physical argument, not the small n* gain, is the reason.
# (2026-07-22 decision, Phase B.)
SPRAY_ABS_MAX = 90.0


def add_swing_contact_labels(df):
    """
    Add the `swing` and `contact` labels from `description`.

    `swing` is 0/1 over every pitch and is never null. `contact` is defined only
    on swings: 1 when the bat touched the ball, 0 on a whiff, and null on takes
    (the whiff model is not asked about pitches the batter never offered at).
    Raises if `description` holds a value outside the known Statcast enum.
    """
    unknown = set(df["description"].dropna().unique()) - KNOWN_DESCRIPTIONS
    if unknown:
        raise ValueError(f"unknown pitch descriptions, cannot label: {sorted(unknown)}")

    df = df.copy()
    swing = df["description"].isin(SWING_DESCRIPTIONS)
    made_contact = df["description"].isin(CONTACT_DESCRIPTIONS)

    df["swing"] = swing.astype("int8")
    # 1/0 on swings, <NA> on takes
    df["contact"] = made_contact.astype("Int8").where(swing, other=pd.NA)
    return df


def field_side_angle(df):
    """
    Raw horizontal launch angle in degrees from hit coordinates, before mirroring.
    0 = up the middle, negative = left field (3B side), positive = right field.
    Null wherever hit coordinates are missing.
    """
    return np.degrees(np.arctan2(
        df["hc_x"] - HOME_PLATE_HC_X,
        HOME_PLATE_HC_Y - df["hc_y"],
    ))


def add_contact_quality_labels(df):
    """
    Add the `ev`, `la`, and `spray` contact-quality labels.

    All three are defined only on balls in play (`hit_into_play`) and are null
    everywhere else. `ev`/`la` are the raw Statcast launch measurements masked to
    that domain; masking matters because launch_speed/launch_angle are present on
    many fouls, which are contact but not quality-labeled. `spray` is the
    horizontal launch direction in degrees, mirrored so positive = pull for both
    hands. Within the in-play domain a value may still be null if that specific
    measurement is missing (e.g. an in-play ball with no hit coordinates).
    """
    df = df.copy()
    in_play = df["description"] == IN_PLAY_DESCRIPTION

    # mirror to a batter-intrinsic axis: RHB pulls to LF (negative field-side
    # angle), so flip its sign; positive spray then means pull for both hands
    pull_sign = np.where(df["stand"] == "R", -1.0, 1.0)
    spray = pd.Series(field_side_angle(df) * pull_sign, index=df.index)

    df["ev"] = df["launch_speed"].where(in_play, other=np.nan)
    df["la"] = df["launch_angle"].where(in_play, other=np.nan)
    # in-play only, then null the near-plate |spray| > 90 artifact (see SPRAY_ABS_MAX)
    spray = spray.where(in_play, other=np.nan)
    df["spray"] = spray.where(spray.abs() <= SPRAY_ABS_MAX, other=np.nan)
    return df


# label columns produced by the full pipeline, in dependency order
LABEL_COLUMNS = ["swing", "contact", "ev", "la", "spray"]


def derive_labels(df):
    """
    Run the full label pipeline (L1 swing/contact, L2 contact quality) and
    return the labeled frame. Asserts the factorization nesting holds: contact
    is defined only on swings, and any quality label implies a ball in play.
    """
    df = add_swing_contact_labels(df)
    df = add_contact_quality_labels(df)

    # nesting invariants: a broken mask here silently poisons a downstream head
    assert (df.loc[df["contact"].notna(), "swing"] == 1).all(), "contact set on a non-swing"
    has_quality = df[["ev", "la", "spray"]].notna().any(axis=1)
    assert (df.loc[has_quality, "description"] == IN_PLAY_DESCRIPTION).all(), \
        "quality label set off the in-play domain"
    return df


def reconcile_labels(df):
    """
    Build the label reconciliation report: counts that must close, missingness
    accounting, and the spray regression-guard. A masking bug shows up as an
    identity that fails to balance rather than as a silently wrong training set.
    Expects a frame already passed through `derive_labels`.
    """
    swing = df["swing"].astype(bool)
    contact = df["contact"]
    in_play = df["description"] == IN_PLAY_DESCRIPTION
    spray = df["spray"]
    field_side = field_side_angle(df)  # raw angle; NaN iff hit coords missing

    report = {
        "n_pitches": int(len(df)),
        "n_swing": int(swing.sum()),
        "n_take": int((~swing).sum()),
        "n_whiff": int((contact == 0).sum()),
        "n_contact": int((contact == 1).sum()),
        "contact_breakdown": {
            d: int((df["description"] == d).sum()) for d in sorted(CONTACT_DESCRIPTIONS)
        },
        "n_in_play": int(in_play.sum()),
        "n_ev": int(df["ev"].notna().sum()),
        "n_la": int(df["la"].notna().sum()),
        "n_spray": int(spray.notna().sum()),
        # truly missing hit coordinates (angle is NaN); disjoint from the clip below
        "n_in_play_without_coords": int((in_play & field_side.isna()).sum()),
        # in-play sprays nulled by the |spray| > 90 clip; coords present, angle valid
        # (raw angle, pre-mirror; abs is sign-invariant so mirroring does not matter)
        "n_spray_clipped": int((in_play & (field_side.abs() > SPRAY_ABS_MAX)).sum()),
        # post-clip this must be 0: the label carries no |spray| > 90 survivors
        "n_extreme_spray_gt90": int((spray.abs() > SPRAY_ABS_MAX).sum()),
        "spray_field_side_mean": _round(field_side[in_play].mean()),
        "spray_pull_mean": _round(spray.mean()),
    }

    # identities that must hold exactly; failing loud beats a poisoned ablation
    assert report["n_swing"] == report["n_whiff"] + report["n_contact"]
    assert report["n_contact"] == sum(report["contact_breakdown"].values())
    assert report["n_extreme_spray_gt90"] == 0, "spray clip left |spray| > 90 survivors"
    # every in-play ball is exactly one of: sprayed, coord-missing, or clipped
    assert report["n_in_play"] == (
        report["n_spray"] + report["n_in_play_without_coords"] + report["n_spray_clipped"]
    ), "in-play spray partition does not close"
    return report


def decode_sample(df, n=12, seed=0):
    """Return a small labeled sample for the eyeball gate: raw description beside its labels."""
    cols = ["description"] + LABEL_COLUMNS
    return df.sample(min(n, len(df)), random_state=seed)[cols].reset_index(drop=True)


def _round(value):
    return None if pd.isna(value) else round(float(value), 3)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Derive process-head labels for the modeling pitch table.")
    parser.add_argument("--in-path", default="data/processed/pitch_events.parquet")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    df = derive_labels(df)
    report = reconcile_labels(df)

    for key, value in report.items():
        print(f"{key:26s} {value}")
    print("\ndecode sample (eyeball gate):")
    print(decode_sample(df).to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pitch_events_labeled.parquet"
    df.to_parquet(out_path, index=False)
    (out_dir / "label_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(df)} labeled rows to {out_path}")


if __name__ == "__main__":
    main()
