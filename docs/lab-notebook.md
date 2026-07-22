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

---

## 2026-07-21 — Phase B.1 methodology hardening and results
- **Did:** Hardened the stabilization methodology and generated the B.1 artifacts. Added a variance-components (one-way random-effects) estimator with bootstrap CIs and a sequential-split mode to src/analysis/stabilization.py (6 new tests, 70 pass); added src/analysis/b1_report.py, which writes the panel, common-PA-axis ranking, spray-clipping check, and wOBA survivorship decomposition to results/phase_b/ (5 CSVs + 3 figures). Logged three decisions in the decision log. Decided B.1 does not warrant its own notebook: the logic lives in the tested module and the numbers are one command, so a notebook would only re-paste tested code; results/ + this entry are the record, a thin reporting notebook is deferred to paper time.
- **Why:** Two weaknesses surfaced in the 07-20 numbers. (1) The single r=0.5 threshold was unsourced; literature (Carleton, Baseball Prospectus; FanGraphs reliability updates) reports r=0.7 as "reliable" and frames 0.5 as the equal-weight-with-prior point, so we now report both. (2) The pooled-process-vs-side-specific-outcome comparison was asymmetric, and split-half at large n only uses hitters who reach that n, biasing the outcome number. The variance-components estimator (equivalent to Cronbach's alpha / KR-21) decomposes signal and noise over all hitters, giving an analytic reliability(n), CIs, and the shrinkage constant; the sequential split gives the across-circumstance number the projection task actually faces; matched side-specific process cuts kill the asymmetry.
- **Found:** The survivorship bias was real and material. Split-half put wOBA vs LHP at ~435 PA; variance-components on all 2142 hitters puts it at ~190. The gap is entirely population, not estimator: the two agree on a fixed population, but restricting to durable regulars halves the between-hitter signal variance (0.00137 to 0.00073) and so doubles n* (190 to ~385). On a common PA axis the honest process-beats-outcome gap is several-fold, not an order of magnitude: whiff ~28 PA-equiv (~7x), swing ~31, launch angle ~62, exit velocity ~63 (~3x), spray ~122 (~1.6x) vs wOBA ~190-198. Matched side-specific process stays fast (whiff vs LHP 45, vs RHP 50, ~ pooled 51), so the gap is not an artifact of the outcome being split by hand. Spray-clipping barely helps: dropping the ~1% |spray|>90 near-plate artifact moves n* from 82 to 73 (VC), so spray remains the slowest process signal regardless. The premise holds robustly under every estimator and split; only the magnitude shrank from the survivorship-inflated 07-20 figure.
- **Learned:** variance-components / one-way-random-effects reliability and its equivalence to Cronbach's alpha and KR-21; survivorship bias in split-half-at-each-n and how to detect it (n* stable when the estimator is restricted to the survivor subpopulation); the r=0.5 vs r=0.7 threshold distinction and the regression-to-the-mean / shrinkage reading of n* = noise/signal; random vs sequential (across-circumstance) split-half.
- **Next:** Run the deferred Phase A /research-review before promoting B.1 to a logged finding, then start B.2 GBM feature screening (XGBoost + SHAP on the 48 context features against the process outcomes, through the frozen split, tuned on val never test) in a fresh session. Open: the spray label decision (drop |spray|>90 vs keep) now has evidence — the gain is small and would require a labels.py change plus a parquet regen, so it is Nate's call, not yet made.
