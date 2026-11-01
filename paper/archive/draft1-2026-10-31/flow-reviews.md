# Flow / structure reviews of main.tex (2026-10-31)

Three independent Opus reviewers, fresh context, none applied. Each read the paper from a different reader's position:

- **FO**: MLB front-office analyst
- **ML**: machine-learning researcher
- **ST**: skeptical sabermetric statistician

They were given the Alcorn 2018 and Melville 2023/2024 papers for comparison. My recommendation is the last column, and it is for discussion only.

## Where they agree

| # | Change | FO | ML | ST | Their core argument | My take |
|---|---|---|---|---|---|---|
| A | **Name the two targets early, side by side.** The registered test scores side-specific wOBA (level). The split is the gap between sides, measured on 2024 / 545 hitters / min-exposure strata. Say this in the abstract, intro ¶1 and §3.7. Give the split evidence its own home in Results. | 1 | 1 | 1 | The paper is motivated by the split but the test scores level. The reader first learns this at §4.3 (p.11). ST: "ranking hitters by wOBA vs LHP mostly means ranking good hitters", so a sabermetrician reads bait-and-switch unless told up front. | **Do it.** All three ranked it #1, and the fact-check pass kept hitting the same confusion. |
| B | **Retitle around the finding.** "Conditional Queries…" sells a general interface. §5.2 says the other queries are unvalidated and run on an older model. Move "a split is a query" from the abstract's middle sentence to a clause. | 2 | 4 | 6 | A headline the evaluation does not test gets punished. The interface stays as contribution 1. | **Do it.** Keep the method in the title as well, e.g. "Ordering Hitters Against a Pitcher Hand from a Pitch-Level Model, and How Much of a Platoon Split One Season Can Show". Your call on wording. |
| C | **Move §3.6 Model checks out of Methods.** It becomes a short labelled "the model reproduces the league" opener for Results, or goes to an appendix. Move the 87% coverage sentence there from §4.3. | 4 | 3 | 4, 7 | §3.6 reports held-out 2025 numbers before §3.7 defines the evaluation. Coverage has nothing to do with the ceiling paragraph it sits in. | **Do it.** It is a cheap move. |
| D | **Shrink Methods with an appendix.** Candidates: pitcher-batter rule, effective-speed note, spray formula, take-surface detail, six-config screen + LR replay. Leave a one-line pointer for each. | 3 | 8 | 7 | Results start on p.9 of 16, and Methods ≈ 6 pp against Results ≈ 4. | **Do it if SSAC allows appendices.** Check the SSAC27 template and page cap first (still open decision #1). |
| E | **Make the abstract and conclusion end on the takeaway**, not the in-sample embedding 73–79% result. The conclusion should stop re-listing the abstract's numbers. | 2 | 4 | 5 | Readers remember the last sentence. ST: ending on in-sample structure after a failed test reads as spin. | **Do it.** (I already restored the "if the 2026 test confirms it" condition in the conclusion's last sentence, which ST flagged.) |
| F | **Move "Platoon direction" out of §4.4 into the split section.** | 1, 7 | 1 | 4 | It scores the predicted split, not the embedding. | **Do it** as part of A. |
| G | **Figure 1 = the whole pipeline.** Heads → 12-state count chain → BF-weighted sum over pitchers, with the network as an inset or panel (a). FO wants it cited from intro ¶4. | 5 | 7 | – | Composition is the one original modelling step and it has no picture. Readers already know what an embedding + MLP looks like. | **Do it.** Figure 1 is not drawn yet, so this costs nothing extra. |
| H | **Cut intro / Related Work overlap.** Tango + Klaassen appear twice. The Melville 2024 factorization contrast appears in both RW and §3.3. FO adds Alcorn, also twice. | 8 | 8 | 8 | Each source is met twice within two pages. | **Do it.** |
| I | **Tell the "scored twice / selection" story once.** It is told now in §3.1, the §4 preamble and §4.2. ML wants one "selection and hold-out" paragraph or timeline at the end of §3.7. FO wants an evidence-status table heading §4. Either way, keep a short tag at each point of use. | 6 | 2 | 2 (part) | Retold three times, while the fact that matters most (the medium band was seen on the first 2025 read) appears once, late. | **Do it, ML's version.** Put the protocol in §3.7 and a one-clause back-reference at §4.2. |

## Where they differ or stand alone

| # | Change | Who | Argument | My take |
|---|---|---|---|---|
| J | **Intro order.** ML: move the result detail after the method (problem → shrinkage → scout → model → results). ST: keep results first but split ¶1 into three labelled sentences (registered failed / exploratory / post-selection) and drop the 113/452 cutoffs. | ML 5, ST 8 | ¶1 uses terms not yet defined. | **ST's version.** Leading with the null is the draft's most honest move, and the problem is density, not position. |
| K | **Treat the pooled rank win as exploratory everywhere.** Put §4.1 = the registered low-stratum verdict only, plus a new "Exploratory cells" subsection (6 of 16, double-read caution, medium caution). Add a forest plot of all 16 cells. | ST 2 | The paper's own rule says anything outside the registered cells is exploratory, but the abstract, §5.1 and the conclusion present the pooled rank win as a main result. | **Needs your answer first.** Were the pooled cells part of the registration, or only low? If only low, ST is right and the abstract wording must change. |
| L | **Power next to the verdict.** Move the "needed ≈5,158 batters" paragraph into §4.1 and restate it as the minimum detectable effect at 246. Say whether it was computed before or after registration. Add the underpowered stratum, the double scoring and the level/split mismatch to §5.3. | ST 3 | "Could this test have passed?" is the first question a reader asks about a null. | **Do it.** The §5.3 additions are clearly right. Moving the power paragraph is optional. |
| M | **Rename the 2024 min-exposure strata** (e.g. "min-exposure terciles"). Report baseline level-ceiling fractions, not just the model's. | ST 4 | Two different "medium" bands sit in neighbouring sections and will be conflated. The level ceiling only explains the level test if baselines are on it too. | **Rename: do it. Baseline level ceiling:** needs a new computation (model only exists now). Your call. |
| N | **Order §4.2 wins first, then losses** (to match its title). Call pooled ordering the claim and medium its sharpest instance. | ML 6 | The title says "wins and loses" but the section opens with losses. The headline is never settled in one place. | Conflicts with K. Settle K first. |
| O | **Cut §4.4 to the attributes result + one figure.** Neighbours and seed stability go to the appendix. | FO 7 | It is a third kind of result on a third model, the neighbours paragraph disclaims itself, and the space is needed elsewhere. | **Lean yes.** Keep the attributes figure as the Alcorn-lineage hook. |
| P | **Figure 3 (one hitter's EV/LA/spray vs LHP/RHP).** FO: move it to the intro beside the scout paragraph. ML: move it to §3.5 and carry the hitter through the chain to W per hand (Melville's worked example). ST: cut it, or put it by Fig 8. | FO 4, ML 3, ST 7 | Everyone agrees it is misplaced in Model checks. | **ML's version.** Combined with G it gives the composition a worked instance. |
| Q | **Shorten §5.2** (other queries / explorer) to two sentences inside §5.3. | ST 6 | It is unvalidated and runs on the old build. | Lean yes, if B is done. |

## Verdicts in one line each

- **FO:** The order is honest and the pre-registration story is clear. The reader has to assemble the finding themselves: A–C/D fix that.
- **ML:** The section order is sound. The blockers are the title, the unnamed level/split split, and the selection protocol being scattered across five places.
- **ST:** "More honest than most SSAC papers." Three things pull the other way: the level-vs-split framing, the pooled win labelled inconsistently, and an ending on in-sample results without the 2026 condition. Fix A, K and L and it reads as a clean negative result with a measured exploratory lead.
