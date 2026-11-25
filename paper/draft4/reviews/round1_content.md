# Round 1 review, content focus

Lens: does the draft spend its words the way Alcorn (2018) and the author's NBA shot-value writeup do, on how the model works, what it learns, how a team uses it, and why that matters, with process kept light.

## Word allocation (body prose only, no abstract, captions, tables, or comments; rough)

| Category | Sections | Words | Share |
|---|---|---|---|
| How it works | 2.1, 2.2, 2.3 | ~1,250 | 25% |
| Testing and statistical caveats | 2.4, 3.1, 3.2 | ~1,140 | 22% |
| What it learns | 3.3, 4.4, 4.5, 4.6 | ~1,000 | 20% |
| Case studies and disagreements | 4.2, 4.3 | ~430 | 8% |
| Motivation | 1 | ~530 | 10% |
| How a team uses it, why it matters | 4.1, 5, last paragraph of 2.3 | ~650 | 13% |

The how-it-works and what-it-learns shares are close to Alcorn's. The draft runs heavier than Alcorn on testing caveats (Alcorn has one test paragraph and one table) and lighter on use. Most of the caveat weight is earned, because the pre-registered test failed and the paper has to say so plainly. The fixes below shorten bookkeeping that does not carry a weak result, and they add interpretation built only from facts already in the draft. Nothing below softens the low-stratum failure, the exploratory status of the medium stratum, or the narrow intervals.

Answers in brief.
- Q1. Mostly yes. The input/output statement exists but sits at the end of 2.3, after the Markov chain. Two facts an analyst needs before using it are buried in Results. Hitters absent from training get a generic projection, and the output is valued by league-average contact value (items 2, 6, 15).
- Q2. Embedding alignment bookkeeping in 3.3, the floor sweep in 3.2, and the embedding-size numbers in 2.2 (items 7, 8, 13).
- Q3. Medium-stratum RMSE, the calibration slope and coverage, the 539-hitter MAE, the split ceiling, and the absent-hitter share all lack a decision-level reading (items 4, 5, 6, 10, 11).
- Q4. The stabilization and Tango numbers are as concrete as Alcorn. The intro never names a concrete decision and never says why the second baseline's kind of method falls short (items 1, 3).
- Q5. One body overreach ("skills emerge ... without being defined"), a few positive-leaning phrasings in 4.1 and 5, and repeated definitions (items 9, 12, 14).

---

## Ranked items

### 1. Intro names no concrete decision (Q4, Q1)
**Location.** Section 1, paragraph 1 (part1.tex line 24), after "...how a hitter performs against one kind of pitcher."
**Problem.** Alcorn's intro grounds the method in a scout's observation. This intro says "lineup decisions" and "a split" but never shows one. The concrete decisions already exist in the draft but appear only in 2.3 (pinch hitter against the reliever warming up) and 4.1 (which of two bench hitters starts). A front-office reader should meet the decision before the model.
**Proposed insertion** (after the sentence ending "against one kind of pitcher."):
> Choosing which of two bench hitters starts against a left-handed pitcher, or which pinch hitter to send up against the reliever warming up, are decisions of this kind.

4.1 can then keep its bench-hitter sentence as the payoff. If the repetition reads heavy, shorten 4.1's sentence to "Choosing between the two bench hitters of Section 1 is a ranking question, and a few points of error in both projections do not change the call if the two are in the right order."

### 2. Lead 2.3 with what a query takes and returns, and point to a worked output (Q1)
**Location.** Section 2.3, last paragraph (part1.tex line 102), "In use, a query takes a hitter, a set of pitchers, and a weight for each pitcher..."
**Problem.** This is the paragraph a team analyst most needs, and it comes after three steps and an equation. A reader skimming 2.3 learns the mechanics before learning the interface. The draft also never points the reader to a concrete output, although Table 2 (tab:cases) is exactly that.
**Proposal.** Move the paragraph's first sentence to open 2.3, directly after "A query connects the two in three steps", and add a pointer to the table.
> A query takes a hitter, a set of pitchers, and a weight for each pitcher, and returns a projected wOBA per plate appearance. Table~\ref{tab:cases} shows the output for eight named hitters as projected platoon splits.

Keep the rest of the paragraph (72 sampled pitches, series weighting, single reliever, only handedness tested) where it is.

### 3. Motivation says why empirical Bayes falls short but not why the second kind of method does (Q4)
**Location.** Section 1, paragraph 3 (part1.tex line 30).
**Problem.** Alcorn names each existing approach and its specific shortfall (WAR ad hoc, PECOTA proprietary, DRA linear). The intro does this for empirical Bayes but stops at "for most hitters the estimate stays close to the league-average split" without the consequence for a team. The boosted baseline, which already uses pitch data, is not motivated at all until 2.4, so the reader cannot see why pitch-level modeling should beat pitch-level summaries.
**Proposed text.** Replace "...so for most hitters the estimate stays close to the league-average split." with
> ...so for most hitters the estimate stays close to the league-average split, and two part-time hitters from the same side receive nearly the same projected split.

Then replace "This empirical Bayes shrinkage \citep{efron1972} uses only a hitter's outcomes in the split being projected, and it is one of the two baselines in this paper." with
> This empirical Bayes shrinkage \citep{efron1972} uses only a hitter's outcomes in the split being projected, and it is one of the two baselines in this paper. The other adds summaries of the hitter's own pitches, such as swing, whiff, and exit velocity rates by pitcher hand, in gradient-boosted trees \citep{chen2016}. It uses pitch data only as season rates and does not see individual pitches or the pitchers who threw them.

All facts are from 2.4 (line 113). 2.4 can then drop "It does not see individual pitches or pitchers" if the repetition grates, keeping "the count enters only as a two-strike indicator".

### 4. Medium-stratum gain is given as RMSE and percent with no sense of size, and repeats Table 1 (Q3, Q5)
**Location.** Section 3.2, paragraph 2 (part2.tex line 57), "...with rank correlation 0.42 against 0.28 and 0.22 and RMSE 0.0455 against 0.0479 and 0.0492 (5\% and 7\% lower), and all four intervals exclude zero."
**Problem.** The six numbers are already in Table 1. "5% and 7% lower" does not tell a team how much a projection moves. The draft's own theme (4.1, abstract) is ordering over sizing, and this is the place the numbers show it.
**Proposed text.**
> There the model beats both baselines on both measures, and all four intervals exclude zero. The error gain is small in wOBA terms, about 2 points below the boosted baseline and 4 below empirical Bayes against a typical error near 45 points, while the rank correlation rises from 0.28 and 0.22 to 0.42. The gain is in ordering these hitters more than in sizing their wOBA.

(0.0479 minus 0.0455 is 0.0024, 0.0492 minus 0.0455 is 0.0037, RMSE 0.0455 is about 45 points.) Keep "The clearest signal, an exploratory result," as the lead-in, unchanged.

### 5. Calibration slope and coverage stated without what they mean for a card (Q3)
**Location.** Section 3.2, calibration paragraph (part2.tex line 67).
**Problem.** "Slope is 0.59" and "covered ... 87%" are honest but leave the reader to work out the consequence. Future directions (part5 line 9) says the bands need recalibration, but the reader never sees how far off they are in usable terms.
**Proposed insertions.** After "...spread more widely than their outcomes support.":
> On average, a 10-point gap between two low-stratum projections corresponded to about 6 points of observed gap.

After "...covered the observed value for 87\% of 1,149 units.":
> A band shown as 95\% would therefore miss about one unit in eight rather than one in twenty.

Both follow from the slope and coverage already reported.

### 6. Absent hitters: the consequence is in Results, the input rule is in Data, and neither says what it means for use (Q1, Q3)
**Location.** Section 2.1 (part1.tex line 45, "hitters absent from them share one reserved embedding") and Section 3.2 (part2.tex line 57, "164 (43\%) belong to hitters absent from the training data, and each of these receives one of four projections").
**Problem.** An analyst reading 2.3 would assume any hitter can be queried. The consequence, one generic projection per batting side and pitcher hand, appears only in Results and Future. In 3.2 the fact is stated without its reading. These units cannot be told apart at all, which is part of why the pre-registered stratum is hard.
**Proposals.** (a) Add to the end of the 2.3 "In use" paragraph:
> A hitter absent from the training seasons has only the shared reserved embedding, so every such hitter of a batting side receives the same projection against a pitcher hand.

(b) In 3.2, after "...one of four projections set by batting side and pitcher hand.":
> The model cannot order these hitters among themselves, which leaves 220 of the stratum's 384 units where a hitter's own pitches inform the projection.

(384 minus 164 is 220.)

### 7. Embedding alignment bookkeeping opens 3.3 (Q2)
**Location.** Section 3.3, paragraph 1 (part2.tex line 87), from "Its five seeds give similar vectors..." to "...nearly unrelated to playing time (0.09)."
**Problem.** Alcorn's results open on what the embedding shows. This paragraph spends about 80 words on seed rotation, shuffled nulls, and norm scaling before any finding. These are checks that the analysis is sound, not results, and nothing weak hides in them (0.83 against 0.12 and 0.09 are reassuring).
**Proposed text.** Keep the first two sentences, then replace the rest with
> Each hitter's embedding is averaged over the five seeds and scaled to unit length, since an embedding's length grows with the hitter's exposure.\footnote{The five seeds give similar vectors for the same hitter, with a median cosine similarity of 0.83 after rotating each onto the first, against 0.12 when hitters are shuffled before the rotation is fit. Length correlates at 0.93 with log prior PA, and after scaling the main direction of variation correlates with it at only 0.09.}

### 8. Evaluation-floor sweep can move into the Figure caption (Q2)
**Location.** Section 3.2, last sentence of paragraph 2 (part2.tex line 57), "Raising the evaluation floor from 10 to 25 or 50 PA changes none of these outcomes, though at 50 PA the model's low-stratum RMSE is nominally above the boosted baseline's."
**Problem.** A robustness sweep at the end of the paragraph that carries the main verdict. The figure (fig:strata) already plots the 10 and 50 PA floors, so the caption is the natural home. The caveat about 50 PA must stay, because it is the one place the low-stratum point estimate flips, and the caption keeps it.
**Proposal.** Cut the sentence from the text. Change the fig:strata caption to
> The model's 2025 margin over the boosted baseline on error and on rank correlation, by prior-exposure stratum, at evaluation floors of 10 and 50 PA. Raising the floor to 25 or 50 PA changes no stratum's outcome, though at 50 PA the model's low-stratum RMSE is nominally above the boosted baseline's.

### 9. Overreach in the 3.3 conclusion about emergent skills (Q5)
**Location.** Section 3.3, paragraph 2, last sentence (part2.tex line 89), "Hitter-level skills therefore emerge from pitch-level training without being defined for the model."
**Problem.** Power is 90th-percentile exit velocity and contact is contact rate on swings. The exit velocity head and the contact head are trained on exactly those pitch outcomes (2.2). The recovery shows the embedding organizes hitters by these summaries, which matters. But "emerge without being defined" reads like Alcorn's finding that his embeddings learned qualities nobody fed them, and here the per-pitch versions of the labels are the training targets. The abstract is fixed, so the body should be the precise version.
**Proposed text.**
> The model was trained on single swings and balls in play, never on these hitter-level summaries, so the embedding organizes hitters by power and contact without being given either statistic.

In-zone swing and chase are the more striking recoveries, because no head separates in-zone from out-of-zone swings by name. That would be worth a clause only if the author confirms the swing head has no zone-specific target. Do not add it otherwise.

### 10. The 539-hitter split comparison needs a size reading (Q3)
**Location.** Section 3.2, last paragraph (part2.tex line 78), "...a difference of 0.0014 (interval 0.0002 to 0.0026)."
**Problem.** This is the comparison 4.3 calls "the evidence on accuracy" for splits, and it is the number most directly about a platoon decision. The reader is not told that 0.0014 is about 1 point of wOBA against errors near 45.
**Proposed text.** Replace "...a difference of 0.0014 (interval 0.0002 to 0.0026)." with
> ...a difference of about 1.4 points of wOBA (interval 0.2 to 2.6) against mean errors near 45 points.

The following sentence (519 of 539 reversible by noise) then reads as the natural consequence and needs no change.

### 11. Split ceiling and level query lack a one-line reading (Q3)
**Location.** (a) Section 3.1 (part2.tex line 24), after "...would reach a rank correlation of only about 0.30 with the observed split." (b) Section 4.6 (part3.tex line 105), after the level-query correlation.
**Problem.** (a) The ceiling is the paper's best tool for reading every later split result, including the case studies, but its practical meaning is left implicit. (b) "0.38 after removing log prior PA" does not say what removing prior PA buys.
**Proposed insertions.**
(a)
> One season of a hitter's observed split therefore says little about whether any split projection for that hitter was right.

(b)
> The level query therefore carries information about a hitter's 2024 wOBA beyond how much the hitter had played.

### 12. Positive-leaning phrasings in 4.1 and 5 (Q5)
**Location and proposals.**
- 4.1, paragraph 2 (part3.tex line 11), "With little exposure, the model's point estimates still rank hitters ahead of both baselines". Change to "With little exposure, the model's point estimates nominally rank hitters ahead of both baselines, with every interval spanning zero". The rest of the sentence stays.
- 5, paragraph 1 (part5.tex line 7), "...ordered hitters better than both, most clearly for hitters with a moderate track record against a hand." Change to "...ordered hitters better than both, most clearly, in an exploratory comparison, for hitters with a moderate track record against a hand."
- 5, paragraph 1, "The same query supports roster questions." Only the handedness query is tested (2.3, 4.6), and a roster comparison over expected opponents is a different pitcher set. Change to "The same approach could support roster questions."
- 5, paragraph 2 (part5.tex line 9), "The low-exposure margin is positive but underpowered, and confirming it at its observed size would take far more batters than one season holds." Change to "The low-exposure margin is nominally positive but underpowered, and confirming it at its observed size would take about 5,200 batters, roughly 21 times the stratum's 246." This also replaces the vague "far more" with the draft's own number (3.2).

### 13. Embedding-size selection numbers in 2.2 (Q2)
**Location.** Section 2.2, last paragraph (part1.tex line 78), "Embedding sizes of 32 and 64 were compared, and 64 was chosen because it gave the lower 2024 log loss averaged over five seeds (1.02327 against 1.02378)."
**Problem.** A selection-procedure detail with five-decimal log losses that no later argument uses. Alcorn reports only the chosen dimension. The 32-dimension variant does matter later (footnote in 2.4), so keep the comparison but drop the numbers.
**Proposed text.**
> An embedding size of 64 was chosen over 32 for its lower 2024 log loss averaged over five seeds.

Keep the SGD rationale for the embedding table, which is content (how the model treats a rarely seen hitter), and the 25-minute CPU time, which tells a team the model is cheap to retrain.

### 14. Repeated definitions and an ambiguous footnote (Q5, Q2)
- **226 PA defined three times.** Intro (line 24), 2.4 (line 109, "the 226 PA at which half the variation in wOBA against left-handed pitchers reflects skill"), and 3.1 (line 10). 3.1 also repeats that "the strata in Section 2.4 are set from this value", which 2.4 already says. Proposal: in 2.4 shorten to "The cuts are half and twice the 226 PA of Section~\ref{sec:intro}, a simple convention rather than a tuned choice." In 3.1 cut ", and the strata in Section~\ref{sec:eval} are set from this value". The Figure (fig:reliability) caption already shows the strata lines.
- **"Empirical Bayes sits just as close to the error floor"** appears in 3.1 and again in 4.1. 4.1 cites 3.1, so this is acceptable. Leave it.
- **Footnote in 2.4** (line 115), "the low-stratum error differences moved from nominally favoring the baselines to nominally favoring the model". The reader cannot tell which version favored which. Table 1 shows the final model nominally ahead in the low stratum (0.0605 against 0.0609 and 0.0620), so the 32-dimension version is the one that favored the baselines. Proposed: "No comparison changed whether its interval excluded zero, although its low-stratum error differences nominally favored the baselines where the final model's nominally favor the model." This keeps the honest point that the low-stratum sign is fragile.

### 15. Say what the projected value leaves out (Q1)
**Location.** Section 2.3, paragraph 2 (part1.tex line 86), after "...times the league's average wOBA for it."
**Problem.** The intro argues that fielders, parks, and where a ball lands add noise. 2.3 then values each ball in play by the league average for its exit velocity, launch angle, and spray bins, which is how the model sets that noise aside. The draft never connects the two, so an analyst cannot tell whether the output is a park- and defense-dependent forecast or a hitter-skill estimate. That difference matters when comparing it to a projection system.
**Proposed insertion.**
> Each ball in play is thus valued by the league's average result for how it was hit, not by where it landed, which sets aside the fielder and park effects described in Section~\ref{sec:intro}.

---

## Left alone on purpose
- The low-stratum verdict sentence, the 5,200-batter power figure in 3.2, the 43% absent share, the 26% to 91% interval on the split share, and the calibration paragraph. These carry the weak results and should stay in the text, not in footnotes.
- The 4.2 Bohm footnote. It is a pre-registration deviation and belongs where it is.
- 4.4 to 4.6 (surfaces, pitch-family directions, neighbors). These are the closest analogue to Alcorn's Section 3 and are the right kind of content. No use claim is added to them because the draft states none, and inventing one would break the no-new-facts rule.
