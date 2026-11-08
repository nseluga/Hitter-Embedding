# Round 3 review: structure and paragraph format (Alcorn 2018 lens, final round)

Scope: section order, paragraph boundaries, topic and closing sentences, float placement and
lead-ins, repetition. Rounds 1 and 2 are taken as applied or deliberately skipped, and none of
their items is re-proposed. The abstract (part1.tex 1 to 17), the Table 1 caption and the
fig:powercontact caption are untouched. Every proposed sentence uses only facts already in the
draft, with no first person, dashes, colons or semicolons in prose. Six items, ranked by impact.

## Where the draft stands

Each Results subsection now has one topic, and most end on an interpretive sentence the way
Alcorn's do. Three structural problems remain. Results alternate between query results and
embedding probes (test, embedding, case studies, surfaces, neighbors), which forces a scope
exception and a forward reference to Table 2. The subsection carrying the main verdict (3.2) puts
its summary in the middle and ends on a caveat. 3.1 ends on a bridge, while 3.2 opens with a
bare table under its heading.

---

## 1. Move "Case studies and disagreements" up so it directly follows the 2025 test

**Location.** part3.tex lines 1 to 45 (the subsection through the fig:cases float), moved into
part2.tex between line 71 (end of the 539-hitter paragraph) and line 73
(`\subsection{What the embedding learned}`).

**Current.** Results run 3.1 ceiling, 3.2 test, 3.3 embedding, 3.4 case studies, 3.5 surfaces,
3.6 neighbors, 3.7 other queries. 3.4 goes back to 2025 split projections after 3.3 has moved
on to the embedding. That order forces the exception in part2.tex:80 ("Except for the 2025
projections in Section 3.4, ..."). It also makes the 539-hitter paragraph cite Table 2
(part2.tex:71) about four pages before the table is introduced.

**Proposed.** After the move, Results run ceiling, test, case studies (query results), then
embedding, surfaces, neighbors (embedding probes), then other queries. Alcorn keeps his probes
together (3.1 to 3.3) and his test in one place (3.4). The case-study opener already works as a
topic sentence after the split paragraph, and the subsection's last sentence ("The evidence on
split accuracy remains the 539-hitter comparison in Section 3.2") now points to the subsection
just before it. The surfaces subsection still follows the heads it relies on. Two wording edits
go with the move.

- part2.tex:80, sentence 2.
  Current: "Except for the 2025 projections in Section~\ref{sec:anchors}, the analyses in this
  section and those that follow use the selection model, trained through 2023."
  Proposed: "The analyses in this section and those that follow use the selection model, trained
  through 2023."
- part3.tex:35, after the topic sentence, so the moved 2024 ranks keep the model attribution the
  old scope sentence gave them (confirm against disagreement_summary.json).
  Proposed insertion: "The model's projections here come from the selection model, trained
  through 2023."

The Table 2 citation in part2.tex:71 can stay as written, because the table now follows within
a page.

**Why.** It removes the scope exception and the long forward reference, and it gives the reader
all query results first and then all embedding probes, the way Alcorn groups his probes.

---

## 2. Close 3.2 on its verdict, not on a caveat in the middle

**Location.** part2.tex:48, last two sentences, and the end of part2.tex:71.

**Current.** "Taken together, the one pre-registered test was not passed. In exploratory
comparisons the model ties both baselines on error and orders hitters better than both, most
clearly for hitters with a moderate track record." sits at the end of the third of six
paragraphs. Calibration, interval coverage and split grading come after it, and the subsection
ends on "...so the comparison is informative only in aggregate."

**Proposed.** Delete the two sentences from part2.tex:48, so that paragraph ends on "Among
high-exposure hitters the three methods are even." Add a short closing paragraph after
part2.tex:71:

> Taken together, the one pre-registered test was not passed. In exploratory comparisons the
> model ties both baselines on error and orders hitters better than both, most clearly for
> hitters with a moderate track record. Its projected platoon splits are also slightly closer to
> the observed splits than those of empirical Bayes, though only in aggregate. For little-seen
> hitters its projections are spread too widely, and its intervals are too narrow.

**Why.** Alcorn's test subsection ends on one summary paragraph ("(batter|pitcher)2vec thus
appears to be exceptional..."). Placed last, the summary covers all six findings, including the
two weak ones that now come after it.

---

## 3. End 3.1 on its interpretation and move the bridge to open 3.2

**Location.** part2.tex:13 (3.1 paragraph 2), part2.tex:26 to 44 (Table 1 float and the first
3.2 paragraph).

**Current.** 3.1's interpretive sentence ("One season of a hitter's observed split therefore says
little about whether any projection of that split was right.") is third of ten. The subsection
ends on the bridge "The 2025 test in the next section scores each side's wOBA separately. This
target falls between the other two, ...". 3.2 then opens with the Table 1 float directly under
its heading, before any text (rendered p.7).

**Proposed.** part2.tex:13 becomes

> The platoon split is much harder to measure. True splits vary little across hitters compared
> with the noise in one season's observed split, so even a perfect projection would reach a rank
> correlation of only about 0.30 with it. After removing each batting side's average split, the
> model reaches 0.177, 60\% of that ceiling, compared with 49\% for empirical Bayes and 48\% for
> the boosted baseline. The model's share is imprecise (interval 26\% to 91\%, wide enough to hold
> both baselines' shares) and depends on how the variance of true splits is estimated. With that
> variance taken from 2024 alone, the ceiling falls to 0.167, below the model's 0.177, so the
> share is a rough guide rather than a measurement. The model's RMSE for the split is 0.0633,
> close to the 0.0639 floor that sampling noise sets, and empirical Bayes is about as close. One
> season of a hitter's observed split says little about whether any projection of that split was
> right.

3.2 paragraph 1 (part2.tex:44) opens with the bridge as its topic sentence:

> The 2025 test scores each side's wOBA separately, a target that falls between the level and the
> split, with more real variation across hitters than the split and fewer PA behind each value
> than the level. The results by stratum and pooled can be seen in Table~\ref{tab:test}. Across
> all 1,157 units ... [rest unchanged]

Move the Table 1 float (lines 26 to 41, caption untouched) to after this paragraph.

**Why.** Each subsection then ends on its point and opens on its setup, as in Alcorn 3.4 ("To test
this hypothesis, at-bat outcomes were collected..."). The heading is no longer followed by a bare
table. "therefore" is dropped because the sentence now closes the paragraph, and round 2 voice
flagged "therefore" closers.

---

## 4. Cut the third restatement in the pitch-family paragraph

**Location.** part2.tex:91, last sentence.

**Current.** "...The embedding follows the data on both skills. Whiff rate differs by pitch family
in the observed values and in the embedding, and power is close to a single skill in both."

**Proposed.** Delete "Whiff rate differs by pitch family in the observed values and in the
embedding, and power is close to a single skill in both." The paragraph ends on "The embedding
follows the data on both skills."

**Why.** Both halves of that sentence already appear mid-paragraph ("so the embedding keeps a
distinction that is present in the data", "so power is close to a single direction in the
embedding"). Alcorn closes on one interpretive sentence, not two.

---

## 5. Give the embedding subsection an interpretive last sentence

**Location.** part2.tex:113, end of paragraph (last paragraph of "What the embedding learned").

**Current.** Ends on "...at 0.07 in the low stratum, 0.14 in the medium, and 0.13 in the high,
each with an interval above zero." Round 1 placed a "therefore" closer here, and it was moved up
to the topic sentence. The subsection now ends on three numbers.

**Proposed addition.**

> The split is a faint direction in the embedding, and it is faintest in the low stratum, where
> the pre-registered 2025 test was not passed.

**Why.** Alcorn ends every probe on what the numbers mean ("At the least, ... appears to be
capable of capturing"). This sentence adds a reading the topic sentence does not have, because
it links the weak low-stratum signal to the failed test. It only places the two facts side by
side and claims no cause.

---

## 6. Keep the query-flexibility point together in the last 2.3 paragraph (optional)

**Location.** part1.tex:107.

**Current.** The absent-hitter sentence sits between "the weights need not be batters faced" and
"Only the handedness query is tested against outcomes here...", which breaks one line of
thought (any pitcher set can be queried, only one is tested, the others are in 3.7).

**Proposed.**

> Each $W_{hj}$ is built from sampled pitches, so any set of pitchers the model has seen can be
> queried without retraining, and the weights need not be batters faced. Only the handedness
> query is tested against outcomes here, and Section~\ref{sec:otherqueries} shows what two other
> queries return. A hitter absent from the training seasons has only the shared reserved
> embedding, so all such hitters who bat from the same side receive the same projection against a
> given pitcher hand. Each projection also carries a nominal 95\% interval built from the spread
> of the five seeds and the sampling noise of the plate appearances being projected
> (Section~\ref{sec:test}).

**Why.** This is a pure reorder. Each point about what a query can do is followed by where it is
tested, and the two caveats (absent hitters, intervals) come last.

---

## Checked and left alone

- The fig:cases cut (round 2 item 9), the 0.59 parenthetical at the end of the Discussion (round 2
  item 2), the neighbor name lists (round 1 item 12) and the length of 3.3 paragraph 1 were all
  kept on purpose in earlier rounds. None is re-proposed.
- Figure and table lead-ins already follow Alcorn ("can be seen in Figure 1a / Figure 2 /
  Table 1 / Table 2", "Figure 8 correlates"). The parenthetical citations are also Alcorn's
  pattern ("(Figure 4)", "(Table 2)").
- Discussion and Future directions. Both paragraphs of the Discussion open on topic sentences and
  close on interpretation. Future directions has its use, limits and extensions paragraphs, and
  its repeats of the 5,200 figure and the interval problem are next steps, not summary.
- 3.5, 3.6 and 3.7 each end on an interpretive sentence already.
- No hardcoded section numbers in the .tex, so item 1's renumbering is safe. Every cross
  reference uses `\ref`.
