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

## 2026-07-15 — Phase A data foundation: pull, profile, clean
- **Did:** Pulled 2015–2025 Statcast to a versioned parquet snapshot of 7.80M pitches via src/data/pull_statcast.py. Verified batted-ball spin is unavailable, so the contact-quality space is EV, LA, and spray. Built and executed the profiling notebook. Wrote and tested the cleaning pipeline in src/data/clean.py with 9 tests, producing the 7.35M-pitch modeling table.
- **Why:** The snapshot is versioned because Statcast revises retroactively. Profiling before coding caught a DH-era confound in position-player detection, since NL pitchers batted before 2022 and batting-PA alone misfires while batters-faced is era-robust. The core-versus-optional missingness split avoids imputing context the conditional query depends on. The two-table principle keeps model filters from biasing evaluation targets.
- **Learned:** batters-faced as an era-robust role discriminator; spin-versus-movement redundancy via active Magnus spin; the two-table modeling and target separation.
- **Next:** Freeze the walk-forward split config, then build the label and feature-derivation module.
