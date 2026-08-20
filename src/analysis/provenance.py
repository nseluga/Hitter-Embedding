"""
Phase F.1 -- what produced an artifact, recorded in the artifact.

Every audit module reads a built tensor directory and a set of checkpoints, and until now
almost none of them said which. `results/phase_e` carries ~20 JSON artifacts and three
record a `data_dir`. The 2026-08-19 decision-log entry established by a field-by-field
audit that `data/processed/phase_d` and `phase_d5` differ ONLY in `quality_bin_edges` and
the `ev`/`la`/`spray` index arrays, so today's numbers are unaffected by the stale defaults.
That finding lives in prose. This module puts it in the artifacts, so the next reader does
not have to take the log's word for it.

`stamp` is descriptive and never raises on a legitimate build. `assert_quality_bins` is the
one that blocks, and it exists to discharge that entry's own Revisit-if condition: a new
module reading the quality-bin arrays must default to `phase_d5` or assert against the
shipped constants. F.3 and F.4 read those arrays, so they call it.

Read-only. Nothing trains, nothing is scored here.
"""

import hashlib
import json
import subprocess
from pathlib import Path

# the build D.10 trained on; the arrays whose contents differ between builds
CANONICAL_DATA_DIR = "data/processed/phase_d5"
QUALITY_ARRAYS = ("ev", "la", "spray")


def manifest_digest(data_dir):
    """SHA-256 of a build's manifest.json bytes. Identifies the build without reading 1.7GB."""
    path = Path(data_dir) / "manifest.json"
    assert path.exists(), f"no manifest at {path} -- is {data_dir} a built tensor directory?"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision():
    """Current commit, or None outside a work tree. Never raises -- provenance is not a gate."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def stamp(data_dir, arm=None, seeds=None, eval_season=None, **extra):
    """
    The provenance block every audit artifact carries.
    data_dir: built tensor directory; arm/seeds: which ensemble was scored.
    Returns a JSON-serializable dict for the artifact's `provenance` key.
    """
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    block = {
        "data_dir": str(data_dir),
        "manifest_sha256": manifest_digest(data_dir),
        "train_seasons": manifest.get("train_seasons"),
        "quality_bin_edges_present": sorted(manifest.get("quality_bin_edges", {})),
        "arm": arm,
        "seeds": list(seeds) if seeds is not None else None,
        "eval_season": eval_season,
        "git_revision": git_revision(),
    }
    block.update(extra)
    return block


def assert_quality_bins(data_dir, reference_data_dir=CANONICAL_DATA_DIR):
    """
    Blocks a module that reads ev/la/spray from a build whose bin edges are not the trained
    build's. Discharges the 2026-08-19 Revisit-if condition in code rather than in a comment.
    Returns the verified edges.
    """
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    edges = manifest.get("quality_bin_edges")
    assert edges, f"{data_dir}/manifest.json carries no quality_bin_edges"
    assert set(edges) == set(QUALITY_ARRAYS), \
        f"expected edges for {QUALITY_ARRAYS}, found {sorted(edges)}"
    reference = json.loads(
        (Path(reference_data_dir) / "manifest.json").read_text())["quality_bin_edges"]
    for name in QUALITY_ARRAYS:
        assert edges[name] == reference[name], (
            f"{data_dir} has different {name} bin edges than {reference_data_dir}, which is "
            f"the build D.10 trained on. Scoring a quality head against these indices would "
            f"mis-bin every batted ball. Pass --data-dir {reference_data_dir}.")
    return edges
