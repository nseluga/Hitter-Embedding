# Final judge review: draft4/main.tex (SSAC 2027, Baseball track)

Reviewer: independent subagent, research-review method, 2026-11-17.
Read: main.tex (compiled to 14 pages, about 5,250 words of body text and a 300-word abstract), docs/decision-log.md, docs/research-manifest.md, figs/, the `% Source:` files under results/, and the four SSAC exemplars in ~/os/knowledge/library/baseball-research/.
I did not edit the tex. Below are problems and the smallest fix for each. Replacement prose is left to the author.

---

## 1. Verdict

**Not a finalist as written. It is a credible accepted paper, and its rigor is above all four exemplars.**

A judge scoring novelty, rigor, and industry impact would probably give:

- **Rigor: high.** Pre-registration, batter-clustered bootstrap, a measurement ceiling, calibration by stratum, and honest negative results. None of the four exemplars does more than one of these.
- **Novelty: medium.** The pitch-level hitter process model with the count-chain query, where "a new split is a new weight vector", is a real step past Alcorn 2018. The paper does not say so in one place, and it never cites the two SSAC embedding papers by Heaton that a judge will have in mind.
- **Impact: low.** This is what holds the paper back. The test it set for itself failed. The only clear win is one exploratory cell out of sixteen. The one decision demo (Table 3) gives the same top nine as the generic vs-LHP ranking and as the manager's real lineup, so it shows the query runs but not that it adds anything. The paper has no outside, industry-grade comparison like Heaton 2023's sportsbook table, and it never puts the gain into runs or decisions.

**What holds it back most:** a reader finishes without one result showing that a team would decide differently, and better, with this model. Three changes would move the impact score the most, and none needs new training:
1. Lead with the use case and state the contribution plainly (A2, A3).
2. Show The Book's platoon regression, which the industry already uses, beside EB. It is already computed (A6).
3. Make Table 3 a comparison against the baseline ranking instead of against the manager (A1, B1).

Two rigor items could hurt badly if a judge or discussant finds them before the author discloses them: 2025 was scored twice (A4), and the medium win is one cell of sixteen (A5).

Count: 12 mistakes and 9 questions, listed below.

---

## 2. Ranked findings

Rank is by how much the fix would change a judge's score. Type follows research-standards §2: **M** is a mistake, **Q** is a question.

### (a) Fixable by rewording or restructuring

**A1 (M). Table 3 and §4.4 present a demo that shows no added value. Biggest impact item.**
*Location:* main.tex §4.4 "Other queries", the paragraph on the Phillies–Braves game and Table 3 (`tab:staff`, about lines 330–360).
*Problem:*
- The evidence given is that "the model's top nine are eight of the nine Phillies who started that day."
- The same table shows that the generic vs-LHP column has the same top nine (Bohm, Harper, Turner, Realmuto, Castellanos, Schwarber, Kepler, Bader, Wilson), with only three adjacent swaps.
- So the staff weighting changed no lineup decision. Agreeing with the manager is not evidence of value either.
- A judge will read the paper's one industry demonstration as "reproduces what the team already did."
*Smallest fix:*
- Say what the demo shows: the query runs end to end on a real game's staff, and it moves every projection by 7 to 35 points. Do not imply that matching the started lineup validates it.
- B1 below turns this into evidence.

**A2 (M). The abstract and intro bury the contribution.**
*Location:* abstract (lines about 50–55) and intro last paragraph (about line 80).
*Problem:*
- The abstract is about 300 words in one block. It opens with a general claim ("understanding how to use great hitters is often a simple task").
- It reports the failed test before the result that matters to a team.
- It ends with the use case. A judge skimming 100 abstracts reads the first three sentences.
- Heaton 2023 (14 pp, abstract about 230 words) puts the industry users in its first sentence. Alcorn 2018 (about 150 words) states the method's claim and one headline result.
*Smallest fix:*
- Reorder, keeping the author's own sentences: problem (bench/platoon decisions with thin data) → method (pitch-level model, query by reweighting pitchers) → what a team gets → results, with the failed test stated once.
- Cut to about 200–250 words.

**A3 (M). No related-work positioning, and the two most relevant SSAC papers are not cited.**
*Location:* intro ¶3–4 (about lines 72–78) and references.
*Problem:*
- The only prior embedding work cited is Alcorn 2018.
- Heaton 2022 (player form embeddings, SSAC finalist) and Heaton 2023 (contextual event embeddings, SSAC) are the papers an SSAC baseball judge will compare this one to. Neither is cited.
- Public projection systems' platoon handling (Steamer/ZiPS/THE BAT split projections) is also not mentioned. The paper therefore never says why a team should use this instead of what it has.
- Without these, the novelty claim rests on "Alcorn suggested Statcast, so we did it", which reads as incremental.
*Smallest fix:*
- Add a short paragraph or a footnote-length Related Work: Heaton 2022/2023 embed event sequences and outcomes. This model embeds the hitter against measured pitch physics and composes a query through the count. State that difference in one sentence.
- Check each new citation against the shelved PDFs.

**A4 (M). The paper does not disclose that 2025 was scored twice.**
*Location:* abstract ("sealed 2025 data"), §2.1 (about line 92), §3 ¶1.
*Problem:*
- The 2026-09-09 decision-log entry records that 2025 was scored on the old build (embedding_sgd_sgd_lr1) and again on the clean build after the pitcher-batter contamination fix. The verdict was the same both times: low failed, medium passed.
- The 2026-08-18 entry itself calls a second report "test-set reuse regardless of labeling."
- The paper says "sealed" and reports only the second pass. A discussant who asks would find an undisclosed second look. The honest version costs one sentence and strengthens the paper, because the verdict did not change.
*Smallest fix:* one sentence in §2.1 or at the start of §3. Say that 2025 was first scored with a build that included pitchers batting, that a corrected build was scored once more, and that both passes gave the same verdict.

**A5 (M). The medium-stratum win is reported without its multiplicity.**
*Location:* abstract ("an exploratory result"), intro last ¶, §3 ¶2, §4 ¶2, §5.
*Problem:*
- The 2026-09-08 log registers medium as "one of 16 stratum×opponent×gate cells". The 2026-09-20 entry says the abstract would report it "as one exploratory cell of sixteen."
- The paper never states the 16. The medium rank interval against GBM is 0.013 to 0.267 (paired CSV), close to zero. Any multiplicity correction would remove it.
- Without the count, "exploratory" reads as a hedge rather than a number.
*Smallest fix:* state "one of 16 cells" once where the medium result first appears in §3, and once in the abstract.

**A6 (M). The industry-standard baseline is already computed and not shown.**
*Location:* intro ¶3 (cites Tango et al. p.157 for the 2,200/1,000 constants), Table 1, §2.4.
*Problem:*
- Frozen rule 1 names "empirical-Bayes platoon regression (The Book)" as a required baseline. The intro presents The Book as "the usual way."
- model_v1_claim1_scores_min_pa_sweep_clean_dim64_final.csv already holds an `eb_book_` row on 2025: pooled RMSE 0.0460, weighted rank 0.402, and medium RMSE 0.0494 with rank 0.220. These are almost identical to the bivariate EB.
- Saying "results against The Book's constants are the same as against the fitted EB" ties the comparison to a method front offices actually use. That is the cheapest impact gain in the paper.
*Smallest fix:* one sentence or footnote in §3 stating that The Book's fixed constants score the same as EB, with the pooled numbers.

**A7 (M). §3.1 sets 60% against 49%/48% as if the model ranks the split better. The evidence does not show that.**
*Location:* line 222.
*Problem:*
- Paired split-rank differences in differential_exhibit.csv: EB − model −0.10 [−0.36, +0.17], and GBM − model −0.11 [−0.41, +0.20]. Both cross zero.
- The fraction interval 26–91% comes from differential_fraction_of_ceiling.csv, where the file's own `interval_note` says it holds the ceiling fixed and "understates total uncertainty."
- The sensitivity route (A) puts the ceiling at 0.167, below the model's 0.177, so the fraction would be 106%.
- The provenance comment at line 220 says the CI comes from routes.json. It comes from differential_fraction_of_ceiling.csv.
*Smallest fix:*
- State that the three methods' split rankings are not distinguishable on 2024, and that the interval treats the ceiling as known.
- Fix the source comment.

**A8 (M). §3.1 omits the 2024 strata where the baselines lead.**
*Location:* line 222, Figure 4.
*Problem:*
- The 2024 exhibit has split-rank fractions by stratum.
- Low stratum (n = 207): model 0.156, EB 0.197, GBM 0.236. The baselines rank the low-exposure split better.
- Medium: model 0.181, EB 0.036, GBM 0.062.
- The paper reports only the pooled bar. The pattern actually helps its thesis, because the model is useful at moderate exposure on both seasons. Leaving out the low row while the abstract motivates the work with "less data than standard projections need" is selective.
*Smallest fix:* one sentence giving the low and medium 2024 split figures. These are post-selection descriptives, so label them as such.

**A9 (M). The abstract's "60% of the rank-order signal and 64% of the level signal" is ambiguous and has no comparator.**
*Location:* abstract, §3.1.
*Problem:*
- Both numbers are rank-correlation fractions. One is for the split and one for the level, but "rank-order" vs "level" reads as rank vs magnitude.
- The 64% has no baseline. level_ceiling_level_ceiling.json reports only the model.
*Smallest fix:* name them as "the split" and "the overall level". B2 supplies the baseline level fractions. Until then, drop the 64% from the abstract or say that it has no comparator.

**A10 (Q). The selection and schedule confounds are not stated in Methods.**
*Location:* §2.3 eq. (3), §2.4, Discussion ¶2.
*Problem:*
- Projections use training-season pitchers weighted by BF, not the pitchers each hitter faced in 2025. This appears only late in the Discussion.
- Managers also shield platoon hitters from same-hand pitchers, so the low stratum against the same hand is a selected group, and the 10-PA floor adds more selection.
- The manifest's own domain sanity question asks whether the effect "survive[s] the way the game is actually managed." A baseball judge will ask it too.
*Smallest fix:* one or two sentences in §2.4 naming both effects and saying that they apply to all three methods equally (or saying why they do not). B4 can size the schedule part.

**A11 (M). Housekeeping that a judge will notice.**
*Location:* line 17 macro, and 8 captions (Figures 1, 2, 3, 4, 6, 7, 8 and Table 3); title block line 45–46; line 129; footer.
*Problem:*
- "[TO REVIEW]" is printed in red in 8 captions.
- "Paper ID" is a placeholder.
- The code URL github.com/nseluga/hitter-embedding identifies the author. The SSAC copy of Alcorn 2018 is anonymized ("<reveals identity>").
- The footer logo should be checked against the official 2027 template.
*Smallest fix:* remove the macro uses, fill in the ID, replace the URL with the anonymized placeholder until acceptance if 2027 review is blind, and confirm the footer.

**A12 (M). Interpretation sits in Results.**
*Location:* §3 ¶4 (line about 203: "so most of the low stratum's miscalibration comes from hitters the model had never seen"); §3 ¶1 "Further analysis was exploratory" is fine.
*Problem:* this is a causal reading of a post-hoc split. The Discussion already says the same thing with the right caveat.
*Smallest fix:* keep the numbers in §3 and move the "so ..." clause to the Discussion.

**A13 (Q). The Tango citation points to a page the project never held.**
*Location:* intro ¶3, "Tango et al. [14, p. 157]".
*Problem:* the shelved source (tango-2009-the-book-p157-platoon-constants.md) is Tango's 2009 blog comment quoting Dolphin's p.157 text. The note says "the book page itself has never been held." The constants are corroborated (Klaassen 2010), so the number is safe. The page citation is second-hand.
*Smallest fix:* keep the citation and add "as quoted in" the insidethebook.com comment, or confirm against a physical copy.

**A14 (Q). Calibration coverage is reported on 2024 while every other calibration number is 2025.**
*Location:* §3 ¶4, "On 2024, the nominal 95% intervals covered ... 87% of 1,149 units".
*Problem:* the reader cannot tell why the season changes mid-paragraph. The use case in §5 ("a band around each projection") depends on this number.
*Smallest fix:* say why 2024 was used, or see B3.

### (b) Needs a re-run of existing analysis code (no training)

**B1 (M). Add baseline rankings to Table 3. Highest-value re-run.**
*Problem:* see A1. The decision the query supports is a ranking, but Table 3 shows only the model.
*Fix:*
- Add EB's (and/or GBM's) vs-LHP projection and rank for the same 13 Phillies. The case-study code already produces per-hitter EB splits.
- If EB gives the same top nine, say so. If it differs, the rows where it differs are the paper's industry story. Either way the table then tests something.
- This uses 2025 projections only, not outcomes, so it does not touch the sealed outcomes again.

**B2 (M). Baseline fractions of the level ceiling.**
*Problem:* 64% has no comparator (A9).
*Fix:* run the level-ceiling script for EB and GBM against the 0.742 ceiling on the same 545 hitters. This is 2024, post-selection, and costs nothing on 2025.

**B3 (Q). Interval coverage on 2025.**
*Fix:* calibration.csv has slopes by stratum on 2025 but no coverage. Compute coverage of the nominal 95% intervals on 2025 from the existing interval outputs, or state in the text why 2024 is the right season.

**B4 (Q). Size of the schedule confound, on 2024 only.**
*Fix:*
- Re-score the handedness query with S set to the pitchers each hitter actually faced, weighted by PA. The query code takes any weights.
- Do it on 2024, the selection season, not 2025. A third look at 2025 would spend the sealed season again (2026-09-09 log).
- Report how much RMSE and rank change. If the change is small, A10 closes with a number.

**B5 (Q). A decision-level number for impact.**
*Fix:*
- From the existing 2024 projections, take every same-team, same-side pair of hitters in the medium stratum. Score how often each method orders the pair the way their observed split ordered them, and the mean observed wOBA gap gained by following each method.
- Convert to runs per 100 decisions with standard wOBA-to-runs scaling.
- This turns "rank correlation 0.42 vs 0.28" into the unit a front office uses. Heaton 2023 and Melville 2023 both end on a quantity a practitioner can price.
- Do it on 2024 for the reason in B4, or label it clearly if done on 2025.

**B6 (Q). Table 1 at the 50-PA floor.**
*Problem:* the 2026-09-08 log says the Claim-1 headline is quoted at the 50-PA floor. The paper uses 10 PA in Table 1 and shows 50 PA only in Figure 2.
*Fix:* either add a 50-PA column group or log a superseding decision for the 10-PA headline. Figure 2 already shows the verdict does not change, so this is a consistency fix.

### (c) Needs new experiments: out of reach before the deadline

These are listed so the author can name them in §5 as future work. None is a must-do.

- **C1. Confirm the medium stratum on 2026.** The season is sealed, and §5's own power analysis says 377 batters are needed against 236 available, so seasons must be pooled. §5 already says this.
- **C2. An outside, industry-grade benchmark,** for example historical Steamer/ZiPS/THE BAT platoon-split projections for 2025 (the equivalent of Heaton 2023's sportsbook Table 3). It needs data the project does not have. If the author can get one public system's 2025 split projections without training, it would be the strongest single addition to the paper, but plan for it to be unavailable.
- **C3. A cold-start prior for unseen hitters** (minor-league data or a Statcast-derived prior). Seen-only results suggest this is where the low stratum loses. It needs retraining.
- **C4. Pitcher-identity residual and other query sets** (arm angle, pitch mix, single reliever) with their own held-out targets. These need new targets and baselines.
- **C5. A retrain that fixes the high-stratum level bias** (projections about 9 points high) and the low-stratum slope of 0.59.

---

## 3. Comparison with the SSAC-selected papers

| | Alcorn 2018 | Heaton 2022 (finalist) | Heaton 2023 | Melville 2023 | This draft |
|---|---|---|---|---|---|
| Length | 12 pp, about 3,900 words | 22 pp, about 6,800 words | 14 pp, about 6,500 words | 26 pp, about 10,700 words | 14 pp, about 5,250 words |
| Abstract | about 150 words, claim + one result | thesis in its first lines | about 230 words, opens with sportsbooks and teams | problem → equilibrium approach | about 300 words, one block, failed test before use case |
| Related work | inline in intro (critique of WAR, PECOTA, DRA) | full section 2.1–2.5 | full section | woven into the intro | Alcorn only, no SSAC peers cited |
| Conclusion | Future directions | Conclusions and Future Work | Conclusion | Conclusion with industry claim | Future directions only |
| Headline evidence | one table vs baselines, p < 0.001 | game-winner prediction 59.39% | external sportsbook benchmark (Table 3) with task-mismatch caveats | worked at-bat example across three game models | Table 1 vs internal baselines, failed pre-registered test, exploratory win |

**Framing of the contribution.**
- Alcorn's intro argues against something: ad hoc statistics (WAR's formula, PECOTA being proprietary, DRA being linear). It offers the scout analogy, so the reader knows the stakes before any method appears.
- Heaton 2023 frames a clear move: from bag-of-events to contextual embeddings, shown in Figure 1.
- Melville opens with a concrete strategy problem, pitch-sequencing heuristics, and gives a roadmap paragraph.
- This draft's intro has the right ingredients: thin data for bench hitters, luck entering after contact, and Alcorn's suggestion. It never states "the contribution is X, Y, Z" and it has no roadmap.
- The paper's distinctive idea, that one trained model answers any split by reweighting pitchers, appears in mid-abstract and in the last intro paragraph, and is not shown as the organizing claim.

**How results are presented.**
- The exemplars each have one headline exhibit a reader can quote: Alcorn's Table 2, Heaton 2023's sportsbook table, and Melville's Jansen/Phillips at-bat walk-through.
- This draft's headline exhibit (Table 1) shows a tie in the pooled row and a non-pass in the pre-registered row.
- Its strengths in rigor (clustered bootstrap, pre-registration, ceiling, calibration by stratum) go beyond all four exemplars. Alcorn reports one p-value. Heaton 2022 reports no intervals for its game-winner number.
- The draft should sell that rigor as a contribution, not only report it.

**How industry use is shown.**
- Heaton 2023 benchmarks against three sportsbooks and states a coverage advantage (4,856 predictions, 50% more than the closest book) that a book could act on.
- Melville checks empirically whether real players behave as the model says (batters 70% and pitchers 48% optimal).
- Heaton 2022 uses a game-winner probe.
- This draft has Table 3, which matches the manager's choice and the generic ranking (A1), and a "matchup card" in §5 that it admits needs recalibrated bands first.
- The gap to the exemplars is largest here. B1 and B5 are the cheapest ways to narrow it.

**Length and structure.** At 14 pages the draft is in the exemplar range; there is room to add Related Work and a short Conclusion. Check the 2027 call for a page limit. The Discussion holds the embedding analysis (§4.1–4.2), which Alcorn presents as Results 3.1–3.3. Either placement is defensible. Alcorn's "Results" heading for embedding structure makes it read as a finding, not an interpretation.

---

## 4. Claims stated more strongly or more weakly than the evidence

**Stated more strongly than shown**
1. "sealed 2025 data" (abstract). 2025 was scored twice (A4).
2. "against 49% for empirical Bayes and 48% for the boosted baseline" (§3.1). This implies the model ranks splits better, but the paired differences cross zero, and the fraction interval holds an uncertain ceiling fixed (A7).
3. "64% of the level signal" (abstract). It has no comparator, so a reader cannot tell whether it is good (A9, B2).
4. "most useful for ranking platoon options among hitters with a moderate track record" (abstract, Discussion). It rests on one exploratory cell out of 16. It is consistent with the 2024 medium split (0.181 vs 0.036/0.062), but that is the selection season (A5, A8).
5. "once they are set aside the model's error is clearly lower than both baselines'" (Discussion ¶2). This is a post-hoc subgroup, and the interval against GBM ends at −0.0005. "Clearly" is too strong for a cut made after the results were known, even though the next clause says so.
6. "which decreases the impact of outcome luck" (abstract). This is asserted as fact. The paper motivates it (intro ¶2) but does not measure it. Phrase it as the motivation, not as a result.
7. "so the embedding's similarity is not a restatement of season statistics" (§4.2). An overlap of 7 of 40 shows the lists differ, not that the difference is signal. The power purity (0.57 vs 0.33) is the evidence that the neighborhoods mean something. Lead with that.
8. Table 3 / "eight of the nine Phillies who started" (§4.4). This is offered as support, but the generic ranking gives the same set (A1).

**Stated more weakly than the evidence allows**
1. The Book's constants are computed and match EB (A6). Leaving them out loses the paper's best link to industry practice.
2. Rigor is reported but not claimed: pre-registration, batter-clustered intervals, and a measurement ceiling. None of the four SSAC exemplars has these, and the paper never says so.
3. The 2024 by-stratum split results support the moderate-exposure thesis on a second season (A8). They are omitted.
4. Seen-only RMSE at a 50-PA floor (−0.0041 [−0.0072, −0.0012]) is a stronger interval than the 10-PA figure, and it gets one clause.

**Checked and supported**
- Table 1 numbers match the paired and score CSVs.
- 60% = 0.177/0.296 and 64% = 0.475/0.742.
- Pooled slope 0.81 ± 0.05, so it is more than 3 SE below one.
- Case studies: EB closer for all five, as stated in the text.
- Calibration slopes match calibration.csv.
- Spearman 1904 is correctly cited for attenuation.
