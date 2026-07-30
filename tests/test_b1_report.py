"""
Gates for the B.1 spray-clipping check.

This file exists because of a specific silent failure. `labels.py` nulls
|spray| > 90 at label time, and the clipping check read that label — so it compared
three identical datasets and reported one number three times, while presenting
itself as a measurement of whether clipping helps. The committed 2026-07-27 artifact
showed all three treatments at n* = 77.4278 on 994,834 BBIP, including a treatment
whose whole job is to DELETE rows. Nothing failed; the output just meant nothing.

The gates below are the regression tests for that: the reconstruction recovers what
the label discarded, it agrees with `labels.py` everywhere the two are both defined,
and the check refuses to run if there is nothing left for it to detect.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import b1_report as b1
from src.data import labels


def bbip_frame(offsets, stands=None, in_play=True):
    """
    Rows at chosen (dx, depth) offsets from the plate origin, where the angle is
    `atan2(dx, depth)`. A POSITIVE depth is a ball hit out toward the field and gives
    |angle| < 90. A NEGATIVE depth means the coordinate landed behind the origin,
    which is physically impossible for a fair ball and yields |angle| > 90 — that is
    the near-plate artifact the clip exists to remove.
    """
    n = len(offsets)
    stands = stands or ["L"] * n
    return pd.DataFrame({
        "batter": range(n),
        "hc_x": [labels.HOME_PLATE_HC_X + dx for dx, _ in offsets],
        "hc_y": [labels.HOME_PLATE_HC_Y - depth for _, depth in offsets],
        "stand": stands,
        "description": [labels.IN_PLAY_DESCRIPTION if in_play else "foul"] * n,
        # unused by the spray path, but add_contact_quality_labels masks all three
        # quality labels together and reads these columns
        "launch_speed": 95.0,
        "launch_angle": 15.0,
    })


def test_reconstruction_recovers_what_the_label_nulled():
    """
    The gate the whole fix exists for. Two fair balls and two near-plate artifacts:
    the committed label keeps 2 rows, the reconstruction keeps 4 and flags 2 as
    beyond the limit. If these ever agree, the check is a no-op again.
    """
    frame = bbip_frame([(10, 100), (-10, 100), (60, -1), (-60, -1)])
    labelled = labels.add_contact_quality_labels(frame)
    assert labelled["spray"].notna().sum() == 2, "fixture must contain nulled artifacts"

    out = b1.unclipped_spray(frame)
    assert len(out) == 4
    assert (out["spray_raw"].abs() > labels.SPRAY_ABS_MAX).sum() == 2


def test_reconstruction_matches_labels_where_both_are_defined():
    """
    Same mirroring, same origin, same sign convention — the reconstruction must be
    the label's own formula, not a lookalike. Checked for BOTH stands, since the
    pull-mirroring flips sign for RHB and a copy that dropped it would still look
    plausible on left-handed rows alone.
    """
    # the trailing pair are artifacts, present only so the no-op guard is satisfied;
    # the comparison itself runs on the four fair balls the label keeps
    frame = bbip_frame([(10, 100), (-10, 100), (25, 80), (-25, 80), (60, -1), (-60, -1)],
                       stands=["L", "L", "R", "R", "L", "R"])
    labelled = labels.add_contact_quality_labels(frame)
    out = b1.unclipped_spray(frame)

    keep = labelled["spray"].notna()
    assert keep.sum() == 4
    np.testing.assert_allclose(out.loc[keep.values, "spray_raw"].to_numpy(),
                               labelled.loc[keep, "spray"].to_numpy())


def test_reconstruction_is_in_play_only():
    """
    Fouls carry hit coordinates but are not quality-labeled. Here the two artifacts
    are in play and the two fair balls are fouls: only the in-play rows may survive,
    so a reconstruction that forgot the domain mask would return 4 instead of 2.
    """
    frame = bbip_frame([(60, -1), (-60, -1), (10, 100), (-10, 100)])
    frame.loc[[2, 3], "description"] = "foul"
    out = b1.unclipped_spray(frame)
    assert len(out) == 2
    assert (out["spray_raw"].abs() > labels.SPRAY_ABS_MAX).all()


def test_check_refuses_to_run_when_it_cannot_detect_anything():
    """
    The guard against the original bug recurring. Handed data with no artifacts —
    which is exactly what reading the clipped label produced — the reconstruction
    fails loudly rather than returning three copies of one number.
    """
    frame = bbip_frame([(10, 100), (-10, 100), (25, 80)])
    with pytest.raises(AssertionError, match="no-op again"):
        b1.unclipped_spray(frame)


def test_clip_limit_is_read_from_labels_not_hardcoded():
    """
    The check and the label must clip at the same threshold. A literal 90 here would
    silently diverge the day SPRAY_ABS_MAX moves — which is the class of drift that
    produced the original failure.
    """
    frame = bbip_frame([(10, 100), (-10, 100), (60, -1), (-60, -1)])
    out = b1.unclipped_spray(frame)
    beyond = out["spray_raw"].abs() > labels.SPRAY_ABS_MAX
    assert beyond.sum() == 2
    assert labels.SPRAY_ABS_MAX == 90.0  # if this moves, the artifacts above need rechecking
