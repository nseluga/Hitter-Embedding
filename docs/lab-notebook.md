# Lab Notebook — Hitter Embedding

One entry per working session. Format fixed by
`~/os/knowledge/frameworks/research-standards.md` §5 — `/research-partner`
appends to it every session.

Write entries for a stranger reading them: honest about
mistakes and negative results, but in a semi-formal register,
not raw stream-of-consciousness.

---

## <YYYY-MM-DD> — <session focus>
- **Did:** what was built/run, and where it lives
- **Why:** the reasoning, at explain-it-in-an-interview depth
- **Learned:** new concepts introduced this session (these mark the concept as "explained" — see teaching rule)
- **Next:** the concrete next step

---

## 2026-07-16 — Progress review against Architecture Plan v2; manifest-commit fix
- **Did:** Reviewed the repo state against the Layer 1 Architecture Plan v2 (Phase A–G build order).
  Found Phase A only partially done: the Statcast pull script (`src/data/pull_statcast.py`) is
  written and a pull was run (per commit `dbd57ce`), but A.2 (batted-ball spin-field verification +
  spray-angle derivation), A.3 (pitch-event table + leakage unit tests), and A.4 (frozen
  walk-forward split config) are all not started. Fixed a reproducibility defect in `.gitignore`:
  the decision log and pull-script docstring both promise `manifest.json` ships with the public
  repo, but the blanket `data/raw/` ignore was silently excluding it (`git check-ignore` confirmed
  it was untracked). Rewrote the rules with the per-level `/*` idiom so manifests are committable
  while parquet snapshots stay ignored; verified with `git check-ignore`.
- **Why:** A blanket `dir/` ignore fully excludes the directory and git will not descend into it, so
  a later `!dir/file` negation can never re-include a child — the canonical fix is to ignore
  contents per level (`data/raw/*`, `data/raw/statcast/*`, …) and carve the wanted file back with a
  negation at each level. The manifest is the committed record that makes a snapshot reproducible;
  losing it silently defeats the whole snapshot-versioning design (decision 2026-07-14).
- **Learned:** gitignore child re-inclusion rule (no re-include under a fully-excluded parent dir).
  Note: the `os` research-partner / ml-engineer skills are not present in the remote container
  (`~/os` not cloned here); their intent is applied via the frozen rules in `CLAUDE.md` instead.
- **Next:** Freeze `src/config/split_config.yaml` (A.4 — needs Nate's sign-off on fold boundaries
  since split config is a frozen non-negotiable), then run A.2 spin-field verification against the
  snapshot and build the A.3 pitch-event table with its three leakage unit tests.
