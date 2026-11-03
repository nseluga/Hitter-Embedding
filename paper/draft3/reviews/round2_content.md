# Round 2 review, content focus

Lens: does the draft spend its words the way Alcorn (2018) and the author's NBA shot-value writeup do, on how the model works, what it learns, how a team would use it, and why that matters, with process kept light. Round 1 fixes are taken as applied. Items below are new, or check a round-1 addition.

## Balance after round 1 (body prose, rough word counts)

| Category | Sections | Words | Share |
|---|---|---|---|
| Motivation | 1 | ~630 | 11% |
| How it works | 2.1 to 2.3 | ~1,220 | 22% |
| Testing and its caveats | 2.4, 3.1, 3.2 | ~1,260 | 23% |
| What it learns | 3.3, 3.5, 3.6 | ~1,070 | 19% |
| Case studies, other queries | 3.4, 3.7 | ~660 | 12% |
| Discussion, future, use | 4, 5 | ~650 | 12% |

The balance now matches Alcorn closely enough. Testing still runs heavier than Alcorn, and that weight is earned by the failed pre-registered test. The remaining problems are less about word share than about four things. (a) One honesty gap the draft's own rules expose (item 1). (b) Two round-1 additions that say more than the numbers do (items 2, 8). (c) A front-office reader still cannot tell what population and time window a projection describes, or how a band is produced (items 4, 5). (d) A few numbers still lack a decision-level reading (items 3, 6, 7, 9).

Answers in brief.
- **Q1.** Inputs and the query interface are now clear. Still unclear are which pitchers and which seasons a projection is drawn over, and how a projection gets an interval (items 4, 5).
- **Q2.** Sampling fallbacks in 2.3, the cosine in 3.3, and two small bookkeeping sentences (item 11). None carries a weak result.
- **Q3.** The 539-hitter split errors, the alternative split ceiling, the case-study "four of five", and the Pederson head result (items 3, 6, 7, 9).
- **Q4.** Already more concrete than Alcorn's. One number the argument rests on (about four pitches per PA) sits in the Discussion instead of the intro (item 10).
- **Q5.** Every derived figure checked out except "7%" (item 12). Overreaches are in the pooled result's status (item 1), the pitch-family closing sentence (item 2), and "not by where it landed" (item 8).

Arithmetic checked and correct: about 2 and 4 points (0.0479 minus 0.0455 is 0.0024, 0.0492 minus 0.0455 is 0.0037). 1.4 points (0.0014). Roughly 21 times (5,200 / 246 = 21.1). One in eight (87% coverage misses 13%, 1 in 7.7). About 6 points per 10 (slope 0.59). 220 units (384 minus 164). 43% (164/384). 55% (296/539). 64% (0.475/0.742). 60%, 49%, 48% (0.177, 0.1458, 0.1413 over 0.2965). The rank-difference intervals are centered on 0.05 and 0.07. Every z in Table 2 reproduces from the midpoint rule. "Empirical Bayes was closer for all five, by 0.002 to 0.014" reproduces (Trout 0.002, Soto 0.003, Pederson 0.008, Bohm 0.011, Schwarber 0.014).

---

## Ranked items

### 1. The pooled 2025 result is exploratory by the draft's own rule, but the text treats only the medium stratum that way (Q5, honesty)
**Location.** 2.4 (part1.tex line 127) says "Before the 2025 model was trained, the low stratum was specified as the primary test ... All other comparisons are exploratory." Then 3.2 paragraph 1 (part2.tex line 44) reports the pooled rank result with no label. The 3.2 closing sentence (line 48) puts "only in exploratory analysis" on the medium clause alone. Future directions sentence 2 (part5.tex line 7) does the same with "most clearly, in an exploratory comparison, for hitters with a moderate track record".
**Problem.** A reader comes away thinking the pooled ordering advantage is confirmed and only the medium-stratum result is exploratory. Under 2.4, both are exploratory. The Discussion's thesis (ordering over sizing) rests on the pooled result, so the label matters. The abstract is fixed, but it labels nothing as confirmatory, so the body can be exact without contradicting it.
**Proposed text.**
- 3.2 paragraph 1, append after "...both exclude zero.":
> Like every comparison outside the low stratum, these pooled comparisons are exploratory.
- 3.2 paragraph 2, change "although this comparison is exploratory" to "although this comparison is also exploratory".
- 3.2 closing sentence, replace with:
> Taken together, the one pre-registered test was not passed, and in exploratory comparisons the model ties both baselines on error and orders hitters better than both, most clearly for hitters with a moderate track record.
- Future directions sentence 2, replace with:
> On 2025 handedness splits it matched gradient boosting and empirical Bayes on error and, in exploratory comparisons, ordered hitters better than both, most clearly for hitters with a moderate track record against a hand.

### 2. Round-1 closing sentence of the pitch-family paragraph claims the embedding drops a distinction that the data mostly lacks (Q5, round-1 overreach)
**Location.** 3.5, paragraph 3 (part3.tex line 65), last three sentences, from "Both correlations in the embedding run higher..." to "...that it largely does not keep for power."
**Problem.** The observed correlations are 0.53 (whiff) and 0.83 (power). The embedding's are 0.64 and 0.94. Both embedding values sit exactly 0.11 above the observed ones. The gap between the two skills is already in the data, and the embedding mirrors it. "The embedding ... largely does not keep [a distinction] for power" reads as if the data holds a power distinction that the model lost. The label-count sentence explains a difference the data already shows. It also points the wrong way, since fewer labels would make the observed correlation lower, not the embedding's higher.
**Proposed text.** Replace the last three sentences with:
> Both correlations in the embedding run about 0.11 higher than the observed ones, partly because the two directions share the embedding's main direction of variation. The embedding therefore follows the data on both skills. Whiff rate differs by pitch family in the observed values and in the embedding, and power is close to a single skill in both.

### 3. The case studies credit the model with "four of the five" that empirical Bayes also matches (Q3, Q5)
**Location.** 3.4, paragraph after Table 2 (part3.tex line 30), after "...and Soto's observed 2025 split of $-0.074$ runs the same way."
**Problem.** From Table 2, empirical Bayes matches the same four expectations and also projects Soto worse against left-handed pitchers ($-0.044$, against the model's $-0.041$). Without saying so, "The projections match four of the five expectations" reads as evidence for the model. In fact it separates nothing.
**Proposed insertion.**
> Empirical Bayes matches the same four expectations and also projects Soto worse against left-handed pitchers ($-0.044$), so the case studies do not separate the two methods on direction.

### 4. A reader cannot tell which pitchers and which seasons a projection describes (Q1)
**Location.** (a) 2.3, after "A platoon split takes $S$ as all left-handed or all right-handed pitchers." (part1.tex line 105). (b) 2.2, after "...a learned vector that is the only input identifying the hitter." (part1.tex line 62).
**Problem.** An analyst using a projection for a 2025 decision needs two facts. Both follow from the draft, but neither is stated. First, the pitchers in $S$ and their weights come from the training seasons (step one samples training-season pitches, and $w^S_j \propto \text{BF}_j$ counts training-season batters faced). So a platoon projection is against the 2015 through 2024 pool of that hand, not the 2025 staffs. Second, each hitter has one embedding, and the context vector lists no season. So the projection pools a hitter's training seasons. The baselines, by contrast, use the trailing three seasons (2.4). Future directions' "weight the opposing staff" use only makes sense once (a) is said.
**Proposed text.**
(a)
> A platoon projection is therefore against the pitchers of that hand in the training seasons, weighted by how many batters each faced, not against the pitchers the hitter will meet in the projected season.

(b) The code agrees. `src/features/context_features.py` lists no season or age field, and its categorical inputs are pitch type, both hands, balls, and strikes.
> Each hitter has one embedding for all training seasons, so a projection describes the hitter across those seasons together.

### 5. The Methods never say how a projection gets an interval, and 2.2 reads as if it cannot have one (Q1)
**Location.** 2.2, last paragraph (part1.tex line 83): "each head's predicted distribution is averaged across the five before any query". 3.2 calibration paragraph (part2.tex line 58): "nominal 95\% intervals, built from the spread of the five seeds and each unit's sampling noise".
**Problem.** If the five copies are averaged before any query, a seed spread is not available at query time. The reader first learns that projections carry intervals in Results, but the matchup card in Future directions depends on them. One of the two descriptions needs to say per-seed queries are also run.
**Proposed text.** The code agrees. `src/model/query.py` (line 89) stores per-seed compositions because of the between-seed spread, so the 2.2 clause is the part that misleads. Replace the 2.2 clause with
> ...and each head's predicted distribution is averaged across the five for the point projection.

and append to the last paragraph of 2.3:
> Each projection also carries a nominal 95\% interval built from the spread of the five seeds and the sampling noise of the plate appearances being projected (Section~\ref{sec:test}).

### 6. The 539-hitter split errors need a size reading (Q3)
**Location.** 3.2, last paragraph (part2.tex line 69), after "...is about 1.4 points of wOBA."
**Problem.** The 1.4-point difference now has a reading, but the errors themselves do not. Both methods miss a hitter's 2025 split by about 45 points on average. That is as large as the largest projected split in Table 2 (Pederson, $-0.047$). This is the clearest number in the paper for why neither method's split size should drive a single hitter's platoon call. It also connects to the noise floor in 3.1.
**Proposed insertion.**
> Both mean errors, near 45 points, are about as large as the largest projected split in Table~\ref{tab:cases}, which is consistent with the noise in one season's observed split described in Section~\ref{sec:ceiling}.

The existing "For 519 of the 539 hitters..." sentence then follows naturally.

### 7. The alternative split ceiling sits below the model's own correlation, and the text does not say so (Q3, Q5)
**Location.** 3.1, paragraph 2 (part2.tex line 13), after "...and depends on how the variance of true splits is estimated."
**Problem.** The figure caption gives the second ceiling as 0.167, below the model's 0.177. A reader who checks the figure sees the model above its own ceiling and cannot tell what that means. Saying so makes "60%" honest. It is a rough share, not a measurement. It also explains why the measurement ceiling needs recomputing (Future directions).
**Proposed insertion.**
> With that variance taken from 2024 alone, the ceiling falls to 0.167, below the model's 0.177, so the share is a rough guide rather than a measurement.

### 8. Round-1 sentence "not by where it landed" conflicts with how spray is measured (Q5, round-1 overreach)
**Location.** 2.3, step one (part1.tex line 91), "Each ball in play is thus valued by the league's average result for how it was hit, not by where it landed, ..."
**Problem.** 2.1 says the spray angle is "computed from where it was fielded", and spray is one of the three bins being valued. So the value does depend on where the ball went. What the valuation sets aside is the actual result of the play.
**Proposed text.**
> Each ball in play is thus valued by the league's average result for its exit velocity, launch angle, and spray bins, not by its actual result, which sets aside the fielder and park effects described in Section~\ref{sec:intro}.

### 9. The Pederson head result is the paper's clearest "why it matters" point, and the draft does not say why it matters (Q3)
**Location.** 3.5, paragraph 2 (part3.tex line 54), after "...so the gap lies more in swing decisions than in the quality of contact."
**Problem.** This is the Alcorn-style payoff, showing what pitch-level structure lets a team see that an outcome-only method cannot. The intro already establishes that empirical Bayes "uses only a hitter's outcomes in the split being projected". The draft never connects the two.
**Proposed insertion.**
> Because each head covers one part of the plate appearance, the model can show where a projected split comes from, which a projection built only from outcomes, such as empirical Bayes, cannot.

### 10. Move the "about four pitches per plate appearance" figure into the intro (Q4)
**Location.** Intro paragraph 2 (part1.tex line 27), "A plate appearance yields one wOBA value but several of these pitch-level outcomes". Discussion paragraph 2 (part3.tex line 100), "(about four pitches per plate appearance)".
**Problem.** The intro's core argument is that pitch outcomes accumulate faster than PA outcomes, and it says "several". The draft's own number for this appears only in the Discussion. Alcorn's intro motivates with concrete cases, and this intro already does that with 226, 2,200, and 1,000 PA. This last one belongs there too.
**Proposed text.**
- Intro:
> A plate appearance lasts about four pitches and yields one wOBA value, but it yields several of these pitch-level outcomes, and they carry less of the luck that follows contact.
- Discussion, replace "(about four pitches per plate appearance)" with "(Section~\ref{sec:intro})".

Optional, to size the thin-data problem with numbers already in Table 1. Append to intro paragraph 1:
> Of the 1,157 pairings of hitter and pitcher hand scored in 2025 (Section~\ref{sec:test}), 664 had fewer than 452 earlier PA against that hand, twice that mark.

(384 + 280 = 664, 452 = 2 × 226.)

### 11. Process detail that can move to a footnote or caption (Q2)
None of these carries a weak result.
- **2.3 step one** (part1.tex line 91). Move "Sparse counts fall back to pitches with the same number of strikes, then to all of that pitcher's pitches to that side." and "Balls in play that Statcast did not measure, about 7\%, take the league's average value for such balls." into one footnote. Step one then reads as sample, score, value, which is what an analyst needs.
- **3.3 head paragraph** (part2.tex line 88). Move "A head's hitter-specific output is its average prediction for the hitter over a fixed set of 2,000 training-season pitches." into the fig:headpartial caption, as "A head's hitter-specific output is its average prediction over a fixed set of 2,000 training-season pitches."
- **3.3 platoon paragraph** (part2.tex line 100). Optional. Move the two cosine sentences ("Perturbing each embedding dimension ... far from the same.") to a footnote. The paragraph's evidence and verdict (57% against 50%, the stratum partials, "more weakly than power or contact") stay in the text, so the weak result stays visible.
- **2.1** (part1.tex line 48). "Feature scaling was fit on 2015 through 2023 only" can become a footnote. Keep "the baselines saw only seasons before the one being scored" in the text.

### 12. Small fixes and repetition (Q5)
- **"5\% and 7\%" (3.2, part2.tex line 46).** From Table 1, 0.0037/0.0492 is 7.5%, which rounds to 8%. The sentence also gives the same gain twice, as percent and as points. Replace "and its RMSE is 5\% and 7\% lower than theirs. In wOBA terms the error gain is small, about 2 and 4 points against a typical error near 45, so" with
  > and its RMSE is about 2 and 4 points of wOBA lower than theirs, against a typical error near 45 points. The error gain is therefore small, and

  so the sentence continues "the gain for these hitters lies mostly in ordering them."
- **"held-out 2024" (3.7, part3.tex line 89).** 2024 is the selection season (2.1, 3.1). Change "Its correlation with held-out 2024 wOBA" to "Its correlation with 2024 wOBA, the selection season,".
- **"The evidence on accuracy" (3.4, part3.tex line 37).** The 539-hitter comparison is evidence on split accuracy, and Table 1 is the evidence on side-specific accuracy. Change to "The evidence on split accuracy remains the 539-hitter comparison in Section~\ref{sec:test}."
- **Absent hitters are explained four times** (2.1 line 46, 2.3 line 107, 3.2 line 48, Future line 11). Cut 2.1's ", and hitters absent from them share one reserved embedding", since 2.3 now covers it. In 3.2, shorten "and each of these receives one of four projections set by batting side and pitcher hand" to "and these share the generic projections of Section~\ref{sec:query}".
- **Check, no text change proposed.** 3.3 (part2.tex line 88) says the heads are correlated with "training-season statistics" for the selection model (trained through 2023), including bat speed for 570 hitters. Confirm which season the bat-tracking data comes from. If it is 2024, write "(0.61, for the 570 hitters with 2024 bat tracking)".

---

## Left alone on purpose
- The 32 vs 64 comparison, the Bohm footnote, and the 5,200-batter figure in both 3.2 and Section 5. The last is a repeat, but it serves a different purpose in each place.
- The Discussion's "better read as a curve" and "which would explain". Both are hedged, and neither claims more than the three strata show.
- Uses for the location surfaces and neighbors. The draft states none, and proposing one would invent a rationale. Item 9 adds the one "why it matters" sentence the draft already supports.
