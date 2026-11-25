# Round 2 review: paragraph format and structure (Alcorn 2018 lens)

Scope: section order, paragraph shape, topic and closing sentences, figure lead-ins, number
density, float count, and anything the round-1 moves broke. The abstract, the Table 1 caption and
the Figure 1 (fig:powercontact) caption are untouched by every item. Every proposed sentence uses
only facts already in the draft, and none uses first person, dashes, colons or semicolons in prose.

## Where the draft stands after round 1

| | Alcorn (2018) | Draft 3, round 1 applied |
|---|---|---|
| Body length | about 9 pages | 16 pages |
| Floats | 6 figures, 2 tables | 10 figures, 2 tables |
| Results | 4 subsections, probes then the test | 7 subsections, ceiling and test then probes |
| Discussion | none, Future directions opens on one interpretive line | 2 paragraphs, the second ends on a number |
| Subsection endings | interpretive sentence ("thus appears to be exceptional") | mostly interpretive now, three exceptions below |

The round-1 moves mostly landed cleanly. All `\ref`s resolve. The three things they broke are a
scope sentence in 3.3 that now covers the wrong subsections (item 4), a pitch-family probe left in
the case-study subsection after the embedding probes were regrouped (item 3), and the same
one-sentence summary now appearing at the end of 3.2, near the end of Discussion, and at the
start of Future directions (items 2 and 5).

---

## 1. In 3.2, report the pre-registered low stratum before the exploratory medium stratum

**Location.** 3.2, paragraphs 2 and 3, "The margin depends on prior exposure..." and "In the low
stratum, the pre-registered test...".

**Problem.** Section 2.4 tells the reader the low stratum is the primary test, and the abstract
gives the failed test before the medium result. In 3.2 the reader instead meets the exploratory
medium win first, with four intervals excluding zero, and only then the primary verdict. Alcorn
states the test that was set up and reports its result in that order. Putting the pre-registered
verdict first is also the more honest order. With the swap, the existing "Taken together"
sentence closes the strata discussion naturally. The medium paragraph also gives the error gain
twice ("5% and 7% lower" and "about 2 and 4 points"), so one can go.

**Pattern.** Alcorn 3.4, "To test this hypothesis, at-bat outcomes were collected from the 2016
season" comes before the result.

**Proposed replacement.** Swap the two paragraphs and place the fig:strata float after the second.
New paragraph 2:

> The margin depends on prior exposure (Figure~\ref{fig:strata}). In the low stratum, the
> pre-registered test, the model's error is lower and its rank correlation higher than both
> baselines, but every interval spans zero, so the test was not passed. Of the stratum's 384
> units, 164 (43\%) belong to hitters absent from the training data, and each of these receives
> one of four projections set by batting side and pitcher hand. The model cannot order these
> hitters among themselves, which leaves 220 of the 384 units where a hitter's own pitches inform
> the projection. The stratum holds 246 batters, and detecting the observed rank margin over the
> boosted baseline with 80\% power would take about 5,200.

New paragraph 3:

> The clearest signal is in the medium stratum, hitters with 113 to 451 prior PA against a hand,
> although this comparison is exploratory. There the model beats both baselines on both measures,
> and all four intervals exclude zero. Its rank correlation is 0.42 against 0.28 for the boosted
> baseline and 0.22 for empirical Bayes. Its error is about 2 and 4 points of wOBA lower than
> theirs, against a typical error near 45, so the gain for these hitters lies mostly in ordering
> them. Among high-exposure hitters the three methods are even. Taken together, the model ties
> both baselines on error and orders hitters better than both, most clearly, and only in
> exploratory analysis, for hitters with a moderate track record.

---

## 2. End the Discussion on an interpretive sentence, not on a repeated result

**Location.** Discussion, paragraph 2, last two sentences, "With little exposure, the model's
point estimates still order hitters..." and "Its projections there are also spread more widely
than the outcomes support (calibration slope 0.59)."

**Problem.** The paper's interpretive section now closes on a parenthetical number that repeats
3.2 (the 0.59 slope, every interval spanning zero, too few batters). Alcorn closes every results
unit with what it means, and the author's writeup closes Results the same way. The paragraph
also explains why the medium range shows the clearest gain without saying that gain is
exploratory, which is the one place the body drops that flag. The closing sentence the
Discussion needs already exists, in the abstract's last line.

**Pattern.** Alcorn, "(batter|pitcher)2vec thus appears to be exceptional in its ability to
model". Writeup, "paint a clear picture of the shots that teams should try to create."

**Proposed replacement.** Replace the paragraph's last three sentences, from "Pitch-level
outcomes accumulate...", with

> Pitch-level outcomes accumulate several times faster than plate appearance results (about four
> pitches per plate appearance), which would explain why this range shows the clearest gain,
> although that gain was found in exploratory analysis. With little exposure, the model's point
> estimates still order hitters better than both baselines do, but every interval spans zero and
> the projections are spread more widely than the outcomes support. For a team, the model is
> therefore most useful for ranking platoon options among hitters with a moderate track record,
> where the decision is actually in question, not as a replacement for a projection system on
> raw accuracy.

The 0.59 slope stays in 3.2 and the 5,200-batter power figure stays in 3.2 and Future directions.

---

## 3. Move the pitch-family probe from 3.5 into 3.3, next to the power and contact probe

**Location.** 3.5, paragraph 3, "The embedding separates some skills by pitch type and not
others." and Figure fig:missdamage (part3.tex lines 63 to 72).

**Problem.** After round 1, 3.5 "What the model sees in a hitter" is about the case-study
hitters' location surfaces, which follows 3.4 well. The pitch-family paragraph is a different
kind of probe. It fits directions in the embedding across 1,335 hitters, the same method as the
power and contact directions in 3.3, and it splits those same two skills by pitch family. It also
relies on "the embedding's main direction of variation", which is defined in the 3.3 footnote,
two subsections away. Alcorn gives each probe of the embedding one home ("3.1 Visual
inspection", "3.2 Nearest neighbors") rather than splitting one method across subsections.

**Pattern.** Alcorn 3.2, "Probing the neighborhoods of individual embeddings can also yield
deeper insights".

**Proposed move.** Cut the comment block, paragraph and fig:missdamage float from part3.tex and
paste them into part2.tex directly after the fig:powercontact float, before "Each head draws on
the embedding...". No wording change is needed. The order of 3.3 becomes power and contact,
pitch family, heads, platoon, and the platoon paragraph still follows the heads figure it cites.
3.5 is left with its two surface paragraphs, the second of which already ends on an interpretive
sentence.

---

## 4. Fix the scope sentence in 3.3 that the reorder broke

**Location.** 3.3, paragraph 1, "The analyses below use the selection model, trained through
2023. For the classifiers and neighbors, ..."

**Problem.** After the round-1 moves, 3.4 follows 3.3,
and its Table 2 projections are 2025 projections from the final model, so "below" is wrong for
at least part of what follows. "Neighbors" also now points three subsections ahead without a
reference. Alcorn names the data each analysis uses where it is used ("collected from the 2016
season for previously unseen matchups").

**Proposed replacement.**

> The analyses in this subsection and in Sections~\ref{sec:sees} and~\ref{sec:neighbors} use the
> selection model, trained through 2023. For the classifiers here and the neighbors in
> Section~\ref{sec:neighbors}, each hitter's embedding is averaged over the five seeds and scaled
> to unit length.

The source comments confirm the selection model for 3.3, 3.5 (surfaces) and 3.6 (neighbors).
Confirm which model produced the 2024 disagreement ranks in 3.4 and the held-out 2024 level query
in 3.7 before adding either to this sentence.

---

## 5. Open Future directions on the use, not on a third copy of the result summary

**Location.** Future directions, paragraph 1, second sentence, "On 2025 handedness splits it
matched gradient boosting and empirical Bayes on error and ordered hitters better than both..."

**Problem.** This sentence restates the "Taken together" sentence of 3.2 (and the abstract)
almost word for word, and after item 2 the Discussion ends on the same point. Alcorn's Future
directions spends one sentence on what the results mean and moves straight to a concrete use
("For example, by swapping the embeddings of players in a proposed trade"). The exploratory flag
is not lost, because paragraph 2 says "The moderate-exposure advantage was found in exploratory
analysis and needs a confirmatory test."

**Pattern.** Alcorn, "These results prove neural embedding algorithms offer a principled means
of modeling talent", followed directly by "For example".

**Proposed cut.** Delete the second sentence. The paragraph then reads "The model turns per-pitch
hitter outcomes into a projected wOBA against any chosen set of pitchers. The most direct
application is a matchup card..." with everything after unchanged.

---

## 6. Split 2.4 paragraph 1 so the reliability figure gets its own topic paragraph

**Location.** 2.4, paragraph 1, "The handedness query, Equation~\eqref{eq:query} with $S$ set to
all pitchers of one hand, is scored on units..."

**Problem.** Round 1 moved the reliability material and Figure 2 here, but it went into the
middle of the unit-definition paragraph. The paragraph now runs unit, exposure, reliability,
strata, cuts, then back to the 10 PA floor and the level and split definitions. Figure 2 is cited
in a parenthesis in sentence 3. Alcorn introduces each figure in a topic sentence and keeps one
idea per paragraph.

**Pattern.** Alcorn, "The (batter|pitcher)2vec model architecture can be seen in Figure 2".

**Proposed replacement.**

> The handedness query, Equation~\eqref{eq:query} with $S$ set to all pitchers of one hand, is
> scored on units, where a unit is one hitter's wOBA against one pitcher hand in 2025. A unit
> needs at least 10 PA in 2025. The same projections also give a hitter's overall level (wOBA
> pooled over both hands) and platoon split (wOBA against left-handed pitchers minus wOBA against
> right-handed pitchers). Results are reported for all units and by prior exposure, the hitter's
> PA against that hand in all earlier seasons since 2015.
>
> The exposure strata are set from the reliability of wOBA against left-handed pitchers, which
> can be seen in Figure~\ref{fig:reliability}. Reliability after $n$ PA, $n/(n+n^*)$, reaches one
> half at $n^* = 226$ PA in the training seasons. Units fall into three strata, low (fewer than
> 113 prior PA), medium (113 to 451), and high (452 or more). The cuts are half and twice $n^*$,
> a simple convention rather than a tuned choice.

---

## 7. Give interval coverage its own paragraph in 3.2

**Location.** 3.2, calibration paragraph, from "The model's prediction intervals are also too
narrow." to the end.

**Problem.** The paragraph carries two findings on two seasons, the 2025 calibration slopes and
the 2024 interval coverage. Each has its own consequence sentence, so each can stand alone. At
eight sentences it is one of the longest paragraphs in Results. Alcorn starts a new paragraph when
the object changes, even inside one probe.

**Pattern.** Alcorn 3.2 starts a new paragraph at "For pitchers, Craig Kimbrel is Aroldis
Chapman's nearest neighbor".

**Proposed split.** Break the paragraph before "The model's prediction intervals are also too
narrow." The new paragraph keeps its three sentences unchanged and already opens on a topic
sentence and closes on "A band shown as 95\% would therefore miss about one unit in eight rather
than one in twenty."

---

## 8. Stop describing the boosted baseline twice

**Location.** 2.4, paragraph on baselines, "The boosted baseline is gradient-boosted trees
\citep{chen2016} fit on one row per hitter, season, and pitcher hand, using the same trailing
outcomes plus summaries of the hitter's own pitches, such as swing, whiff, and exit velocity rates
by pitcher hand and pitch group."

**Problem.** Round 1 added the baseline description to the Introduction, which now says "summaries
of the hitter's own pitches, such as swing, whiff, and exit velocity rates by pitcher hand, in
gradient-boosted trees \citep{chen2016}". 2.4 repeats the same example list and citation. Alcorn
names prior approaches in the Introduction and defines the baseline once, where it is used.

**Pattern.** Alcorn, "A naïve strategy was used as a baseline for performance comparisons."

**Proposed replacement.**

> The boosted baseline is the gradient-boosted model described in Section~\ref{sec:intro}, fit on
> one row per hitter, season, and pitcher hand, using the same trailing outcomes plus the
> hitter's pitch summaries by pitcher hand and pitch group.

The next sentence ("The count enters it only as a two-strike indicator.") stays.

---

## 9. Consider cutting Figure fig:cases, which repeats Table 2 for eight of its fourteen hitters

**Location.** 3.4, last paragraph, "Ozuna, Judge, and Altuve are graded against 2025 in
Figure~\ref{fig:cases}, along with six medium-stratum hitters..." and the fig:cases float.

**Problem.** The draft has 10 figures against Alcorn's 6. Table 2 already gives the model, empirical
Bayes, observed split, standard error and $z$ for the five case-study hitters and the three
high-exposure disagreements. The figure redraws those eight and adds six medium-stratum hitters
that the text never names and says show only "how the two methods differ rather than which is
right". Alcorn's test rests on one table and lets it carry the numbers.

**Pattern.** Alcorn, "a 0.94\% improvement (Table 2)", with no figure duplicating the table.

**Proposed replacement** (if cut). Delete the fig:cases float and replace the paragraph's last
four sentences, from "Ozuna, Judge, and Altuve are graded...", with

> Ozuna, Judge, and Altuve are graded against 2025 in the bottom rows of Table~\ref{tab:cases}.
> Their observed splits have standard errors near 0.05, at least the size of the gaps between the
> two projections, so the cases show how the two methods differ rather than which is right. The
> evidence on accuracy remains the 539-hitter comparison in Section~\ref{sec:test}.

(Table 2 gaps are 0.017, 0.005 and 0.018 against standard errors of 0.049 to 0.051.) Before
cutting, check the six medium-stratum hitters in the figure. If they lean clearly toward one
method, keep the figure, so the cut cannot look selective.

---

## 10. Move two sampling details in 2.3 to a footnote so step one ends on its point

**Location.** 2.3, paragraph 2, "The first step fixes one pitcher.", sentences "Sparse counts fall
back to..." and "Balls in play that Statcast did not measure, about 7\%, ..."

**Problem.** At eight sentences this is the densest Methods paragraph. Its best sentence, that
each ball in play is valued by how it was hit and not where it landed, sits second to last and is
followed by a bookkeeping detail. Alcorn keeps the Model section to what the model does. The
author's writeup ends method paragraphs on the reason for a step ("This ensured that each model
was evaluated under well-calibrated and comparable settings").

**Pattern.** Writeup, "This ensured that each model was evaluated under well-calibrated and
comparable settings."

**Proposed replacement.** Cut both sentences from the text and attach them as a footnote at the
end of the second sentence ("...run through the network for hitter $h$ and averaged.").

> \footnote{Sparse counts fall back to pitches with the same number of strikes, then to all of
> that pitcher's pitches to that side. Balls in play that Statcast did not measure, about 7\%,
> take the league's average value for such balls.}

The paragraph then ends on "...which sets aside the fielder and park effects described in
Section~\ref{sec:intro}."

---

## 11. Close the heads paragraph in 3.3 with an interpretive sentence

**Location.** 3.3, "Each head draws on the embedding for a different part of the plate
appearance."

**Problem.** The paragraph ends on a list of five correlations. Every other 3.3 paragraph closes
on a "therefore" sentence. Alcorn's first probe ends on what the numbers add up to.

**Pattern.** Alcorn, "At the least, (batter|pitcher)2vec appears to be capable of capturing the
information".

**Proposed addition** (after "...average launch angle (0.49) and pull rate (0.34)."):

> Each head's output therefore tracks the standard statistic for its own outcome, after playing
> time is removed.

---

## Checked and left alone

- 3.3 paragraph 1 is long (eleven sentences, figure cited in sentence ten), but round 1 merged
  its opener with the first probe on purpose. Splitting it again would recreate a thin setup
  paragraph.
- 3.6 nearest neighbors still lists eight names twice. Round 1 item 12 already proposed thinning
  them, so it is not re-proposed here.
- 3.7 Other queries could sit after 3.2 with the other query results, but at the end of Results it
  leads into the Discussion cleanly and 2.3 points to it by label. No move proposed.
- 3.1 is still number dense, but the 26\% to 91\% interval carries the non-significance of the
  split share and the 0.0633 floor is cited by the Discussion, so nothing there can be cut honestly.
- The Introduction's bench-hitter example reappears in Discussion paragraph 1 with a pointer to
  Section 1. That is a deliberate callback, not a repetition from the moves.
- All `\ref` and `\eqref` targets resolve. Unreferenced labels (sec:ordering, sec:anchors,
  sec:disagree, sec:embedding, sec:future) are harmless.
