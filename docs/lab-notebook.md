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

---

## 2026-07-17 — Phase A completion: labels and frozen splits
- **Did:** Built src/data/labels.py (swing/contact/quality labels + spray angle, 16 tests) producing pitch_events_labeled.parquet with a reconciliation report; froze the walk-forward split in src/config/split_config.json with a validating loader (src/config/splits.py, 8 tests). Both verified on the real 7.35M table. Phase A steps 3-4 complete.
- **Why:** Labels follow the §1.2 factorization (swing -> contact | swing -> quality | contact); contact-quality is in-play only because fouls carry EV but no batted-ball outcome, and the masked columns stop foul EV from poisoning the quality head. Spray uses a three-source-corroborated formula with a real-data guard since MLB does not publish the coordinate origin. The split is contiguous walk-forward, frozen before any comparison, because random CV would leak a hitter's future into his own ID embedding and manufacture the small-sample result we mean to test.
- **Learned:** the process-head factorization and its nesting invariants; in-play-only quality masking; the spray-angle derivation and pull-mirroring; walk-forward-as-forecasting and why random splits contaminate the small-sample claim.
- **Next:** Run /research-review on Phase A, then start Phase B (feature-value stage: stabilization, GBM screening, bat-tracking placement, outcome-dimension ablations).

---

## 2026-07-20 — Phase B step 1: eval targets and stabilization
- **Did:** Started Phase B (bat-tracking excluded from v1, Nate's call). Cleaned tracking-artifact pitch_type codes (UN/AB/PO, 36 rows) at source in src/data/clean.py and regenerated both Phase A parquets (7.35M rows). Built the context vectorizer (src/features/context_features.py, 48-dim, train-only fit) as the shared GBM/DL input. Built the wOBA eval-target aggregation (src/data/eval_targets.py + src/config/woba_weights.json) from the complete raw source. Built the split-half stabilization estimator (src/analysis/stabilization.py) and ran the full process-vs-outcome panel. 64 tests pass.
- **Why:** Eval targets come from the complete source (two-table principle) so side-specific wOBA is unbiased ground truth; the FanGraphs wOBA weights were sourced and formula-verified, then validated by reproducing published league wOBA to ±0.0005 rather than recalled from memory. Stabilization (split-half + Spearman-Brown) quantifies signal-per-PA, the small-sample currency, and B.1 exists to test the process-beats-outcome premise before building anything.
- **Learned:** feature-selection taxonomy (filter/wrapper/embedded plus permutation-importance and SHAP); split-half reliability + Spearman-Brown + the stabilization point; the wOBA linear-weights formula and the complete-source eval-target build; train-only fit as the standardization leakage boundary; the closed-form synthetic gate for verifying a reliability estimator.
- **Found:** process stabilizes an order of magnitude faster than outcome. Swing ~97 pitches, whiff ~34 swings, EV ~34 BBIP, LA ~21 BBIP, spray ~77 BBIP (all tens of PA-equivalent) vs side-specific wOBA ~407 PA. Spray is the slowest process signal (location-driven noise); its near-plate artifact (clipping deferred) may depress it. The exposure asymmetry is the bite: at 800 PA only 143 hitters qualify vs LHP vs 476 vs RHP.
- **Next:** Scaffold notebooks/02_feature_value.ipynb (B.1 writeup: common PA axis, clipped-vs-unclipped spray, signal-per-PA ranking), then B.2 GBM screening. Phase A /research-review is still deferred; run it before any Phase B result is promoted to a logged finding.
