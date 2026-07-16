# Decision Log — Hitter Embedding

Append-only. Format fixed by `~/os/knowledge/frameworks/research-standards.md`
§4 — `/research-partner` and `/research-review` both parse against it.

---

## <YYYY-MM-DD> — <decision title>
- **Decision:** what was chosen
- **Alternatives:** what was considered and rejected
- **Rationale:** why, in terms a skeptical reviewer would accept
- **Revisit if:** the condition under which this should be reopened

---

## 2026-07-14 — Statcast raw-snapshot storage design
- **Decision:** Raw Statcast frozen to `data/raw/statcast/snapshot_<date>/season=<YYYY>.parquet` — per-season, all columns, immutable; committed `manifest.json` holds pull metadata. Data gitignored; script + manifest committed.
- **Alternatives:** Monolithic parquet (forces full-memory loads); column pruning at pull time (unrecoverable without re-pull).
- **Rationale:** Date-stamped snapshots handle Statcast's retroactive revisions; immutable raw lets the pitch-event table re-derive without re-fetching.
- **Revisit if:** snapshot outgrows local disk.

---

## 2026-07-16 — Fix `.gitignore` so snapshot manifests are actually committed
- **Decision:** Replace the blanket `data/raw/` ignore with per-level `/*` ignores plus negations, so `data/raw/statcast/snapshot_*/manifest.json` is tracked while all `*.parquet` snapshot files stay ignored.
- **Alternatives:** Move manifests out of `data/raw/` (breaks the self-describing snapshot layout); `!` negation on top of the blanket `data/raw/` ignore (does not work — git will not re-include a child of a fully-excluded directory).
- **Rationale:** The 2026-07-14 storage decision and the pull-script docstring both state the manifest ships with the public repo for reproducibility, but `git check-ignore` confirmed the manifest was excluded and untracked — the reproducibility guarantee was silently broken. This corrects the implementation to match the decision; no design change.
- **Revisit if:** the snapshot directory layout changes (the negation patterns are path-shaped).
