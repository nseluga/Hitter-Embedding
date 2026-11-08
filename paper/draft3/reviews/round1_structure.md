# Round 1 review: paragraph format and structure (Alcorn 2018 lens)

Scope: section order, paragraph shape, topic and closing sentences, figure and equation lead-ins,
number density, and float count. Wording-level style, grammar and fact checks are out of scope.
The abstract and the Table 1 caption are untouched by every item below.

## How the two papers compare at a glance

| | Alcorn (2018) | Draft 3 |
|---|---|---|
| Body length | about 9 pages | 16 pages |
| Floats | 6 figures, 2 tables | 12 figures, 2 tables |
| Top-level order | Intro, Methods (Data, Model), Results, Future directions | Intro, Methods, Results, Discussion, Future directions |
| Results subsections | 4, each one probe or test, each 1 to 3 paragraphs | 3, the last one 4 dense paragraphs |
| Where probes live | All in Results (3.1 to 3.3), nearest neighbors is 3.2 | Split across Results 3.3 and Discussion 4.2 to 4.6 |
| Subsection endings | One interpretive sentence ("At the least, ... appears to be capable of capturing") | Often end on a number or a caveat |
| Concrete uses | Only in Future directions ("by swapping the embeddings of players in a proposed trade") | Split between 2.3 and Future directions |

Alcorn puts the probes before the test (3.1 to 3.3, then 3.4). The draft should not copy that
order, because the verbatim abstract leads with the 2025 test and the body should follow the
abstract. Everything else in Alcorn's shape carries over.

---

## 1. Move the probe subsections from Discussion into Results and leave Discussion as two paragraphs

**Location.** Section 4, subsections 4.2 Case studies through 4.6 Other queries.

**Problem.** Subsections 4.2 to 4.6 report new numbers, tables and figures (Table 2, Figures 8 to
12). They are results, not discussion. Nearest neighbors in particular sits in Discussion while
the neighbor-purity sentence that belongs with it sits in 3.3. A reader looking for "what the
embedding learned" has to read two sections. Alcorn keeps every probe inside Results and has no
Discussion at all, and the author's writeup goes straight from Results to a short Conclusion.

**Pattern.** Alcorn 3.2 opens "Probing the neighborhoods of individual embeddings can also yield
deeper insights about the model." It is a Results subsection.

**Proposed move.**

```
3 Results
  3.1 How much signal one season holds        (ceiling only, see item 2)
  3.2 Handedness splits on 2025
  3.3 What the embedding learned
  3.4 Case studies and disagreements          (old 4.2 + 4.3, see item 3)
  3.5 What the model sees in a hitter         (old 4.4)
  3.6 Nearest neighbors                       (old 4.5, see item 12)
  3.7 Other queries                           (old 4.6)
4 Discussion                                  (old 4.1 text, no \subsection heading)
5 Future directions
```

Mechanics. Cut the `\section{Discussion}` line from the top of part3.tex and paste it, together
with the two paragraphs of old 4.1 (from "The model is better at saying which of two hitters"
through "too few batters to confirm the margin."), after the last float of old 4.6. Delete
`\subsection{Ordering more than sizing}`. Keep `\label{sec:ordering}` on the new `\section`
line. The order of old 4.2 to 4.6 stays the same, because 4.4 and 4.5 rely on the case-study
hitters being introduced first.

---

## 2. Move the reliability paragraph and Figure 2 into 2.4 Evaluation, where the strata are defined

**Location.** 3.1, first paragraph, "One season of outcomes limits how well any projection can
score..." and Figure 2 (fig:reliability). Also the Introduction, first paragraph, "(Section 3.1)".

**Problem.** The value 226 PA defines the strata in 2.4, but it is measured in 3.1. The
Introduction and 2.4 both point forward to Results for a number they use as a method input.
Alcorn's Methods carries everything needed to read the evaluation, such as the 90% frequency
filter in 2.1. The second sentence of 3.1 also exists only to connect back to 2.4 ("and the
strata in Section 2.4 are set from this value").

**Pattern.** Alcorn 2.1 states its filter where it is applied: "only the most frequent batters
and pitchers involved in 90% of at-bats".

**Proposed move and replacement.**

In 2.4, first paragraph, replace

> Units fall into three strata, low (fewer than 113 prior PA), medium (113 to 451), and high (452
> or more). The cuts are half and twice the 226 PA at which half the variation in wOBA against
> left-handed pitchers reflects skill, a simple convention rather than a tuned choice.

with

> The reliability of wOBA against left-handed pitchers, $n/(n+n^*)$, reaches one half at
> $n^* = 226$ PA in the training seasons (Figure~\ref{fig:reliability}). Units fall into three
> strata, low (fewer than 113 prior PA), medium (113 to 451), and high (452 or more). The cuts
> are half and twice $n^*$, a simple convention rather than a tuned choice.

Move the Figure 2 float into 2.4, after that paragraph. Its caption is unchanged.

In the Introduction, change `(Section~\ref{sec:ceiling})` to `(Section~\ref{sec:eval})`.

3.1 then opens with

> One season of outcomes limits how well any projection can score, so the size of that limit is
> measured before the projections are compared. Comparing the spread of hitters' observed values
> with the spread that sampling noise alone would produce gives the share of the spread that is
> real, and from that share the highest rank correlation any method could reach against those
> outcomes. The ceilings were computed on 2024, the selection season, for the 545 hitters with at
> least 10 PA against each hand.

(The sentences "The reliability of wOBA ... set from this value." and "The same idea gives a
ceiling for each target." are deleted.)

---

## 3. Cut two figures and merge the case-study and disagreement subsections

**Location.** Figure 9 (fig:disagreement), Figure 12 (fig:levelquery), and subsections 4.2 Case
studies and 4.3 Disagreements with empirical Bayes.

**Problem.** The draft has 12 figures against Alcorn's 6, and two of them only repeat what the
text already states. Every number the argument uses from Figure 9 (0.77 overall, 0.80, 0.79,
0.69 by stratum) and Figure 12 (0.38, interval 0.30 to 0.45, 0.17, 0.38, 0.52) is already in
the prose. Alcorn only adds a figure when it shows something the text cannot, such as a t-SNE map
or an architecture. Separately, Table 2 already holds both the case-study hitters and the
disagreement hitters, and Figure 8 plots both groups, so 4.2 and 4.3 are one probe split across
two headings.

**Pattern.** Alcorn 3.4 runs its whole test on one table, "a 0.94% improvement (Table 2)".

**Proposed cuts.**

- Delete the Figure 9 float and the parenthetical `(Figure~\ref{fig:disagreement})` in "the two
  methods' projected splits have a Spearman correlation of 0.77, ...".
- Delete the Figure 12 float and the parenthetical `(Figure~\ref{fig:levelquery})` at the end of
  the level-query paragraph.
- Merge 4.2 and 4.3 under one heading, `\subsection{Case studies and disagreements}`, keeping
  `\label{sec:anchors}` and adding `\label{sec:disagree}` on the same line so other references
  still resolve. Paragraph order stays as it is now.
- Optional, if more length has to go. Figure 2 (reliability) can also be cut once item 2 puts
  $n^*$ and the strata in the 2.4 text. In that case move its 95% interval into the sentence,
  "reaches one half at $n^* = 226$ PA in the training seasons (95\% interval 198 to 265)". That
  would bring the count to 9.

---

## 4. Split the dense strata paragraph in 3.2, move the floor check to a footnote, and add a closing sentence

**Location.** 3.2, second paragraph, "The margin depends on prior exposure (Figure 4)..."

**Problem.** One paragraph covers three strata, the pre-registered verdict, the cold-start share,
a power calculation and a sensitivity check. It holds about 20 numbers. The pre-registered result,
which is the one the paper has to be most careful about, sits in the middle of the paragraph.
Alcorn's test subsection ends on one interpretive sentence, but 3.2 has no sentence that says
what the test found as a whole.

**Pattern.** Alcorn closes 3.4 with "(batter|pitcher)2vec thus appears to be exceptional in its
ability to model". The writeup does the same: "Table 1 highlights that performance remains
relatively consistent".

**Proposed replacement** (two paragraphs and a footnote. Every number is kept, and the "not
passed" verdict stays next to its intervals.)

> The margin depends on prior exposure (Figure~\ref{fig:strata}). The clearest signal, an
> exploratory result, is in the medium stratum, hitters with 113 to 451 prior PA against a hand.
> There the model beats both baselines on both measures, with rank correlation 0.42 against 0.28
> and 0.22 and RMSE 0.0455 against 0.0479 and 0.0492 (5\% and 7\% lower), and all four
> intervals exclude zero. Among high-exposure hitters the three methods are even.
>
> In the low stratum, the pre-registered test, the model's error is lower and its rank
> correlation higher than both baselines, but every interval spans zero, so the test was not
> passed.\footnote{Raising the evaluation floor from 10 to 25 or 50 PA changes none of these
> outcomes, though at 50 PA the model's low-stratum RMSE is nominally above the boosted
> baseline's.} Of the stratum's 384 units, 164 (43\%) belong to hitters absent from the training
> data, and each of these receives one of four projections set by batting side and pitcher hand.
> The stratum holds 246 batters, and detecting the observed rank margin over the boosted baseline
> with 80\% power would take about 5,200. Taken together, the model ties both baselines on error
> and orders hitters better than both, most clearly, and only in exploratory analysis, for
> hitters with a moderate track record.

---

## 5. Move the use examples out of 2.3 and into Future directions, and cut the repeated sampling detail

**Location.** 2.3, last paragraph, "In use, a query takes a hitter, a set of pitchers..."

**Problem.** The paragraph repeats the "72 sampled pitches, six in each of 12 counts" detail from
step one and the "without retraining" point from the Introduction and the abstract. Its two use
cases (series weighting, a single reliever) involve queries the paper does not test. Alcorn puts
every concrete use in Future directions, and Methods only says what the model computes. In this
draft, putting the untested uses next to the untested-query caveat in Section 5 is also the more
honest placement.

**Pattern.** Alcorn Future directions, "For example, by swapping the embeddings of players in a
proposed trade".

**Proposed replacement for 2.3, last paragraph.**

> In use, a query takes a hitter, a set of pitchers, and a weight for each pitcher, and returns a
> projected wOBA per plate appearance. Any set of pitchers the model has seen can be queried
> without retraining, and the weights need not be batters faced. Only the handedness query is
> tested against outcomes here, and Section~\ref{sec:otherqueries} shows what two other queries
> return.

**Insertion in Future directions, first paragraph**, after "...with a band around each
projection.":

> Other weightings serve other decisions. A team preparing for a series could weight the opposing
> staff by the innings each pitcher is expected to throw, and a team choosing a pinch hitter could
> set $S$ to the single reliever warming up.

---

## 6. Split the second Future directions paragraph into limits and extensions

**Location.** Section 5, second paragraph, "Two limits stand between the model and those uses..."

**Problem.** The paragraph opens on two limits but goes on to list five items, three of which are
extensions and not limits ("Beyond these", "Finally"). Alcorn's second Future directions paragraph
is only about extensions and opens on a topic sentence saying so. The writeup's Conclusion also
puts the limit first and then the future work ("Our work is limited, as our model does not take
into account").

**Pattern.** Alcorn, "Because the neural embedding template is so flexible, it can easily be
adapted".

**Proposed replacement** (two paragraphs. The text is unchanged except for the new topic sentence
of the third paragraph.)

> Two limits stand between the model and those uses. The prediction intervals are too narrow and
> need recalibration before a card can show a band, and the moderate-exposure advantage was found
> in exploratory analysis and needs a confirmatory test, which may need to pool 2026 with later
> seasons to reach 80\% power. The low-exposure margin is positive but underpowered, and
> confirming it at its observed size would take far more batters than one season holds.
>
> Three extensions would widen what the model can answer. The velocity, location, and
> single-pitcher queries need held-out targets and baselines of their own before they can inform
> decisions. Hitters absent from training, who now receive one of four generic projections, need
> a better starting point, either a prior built from what is known about them or, as
> \citet{alcorn2018} suggested, minor league data mapped into the same space. Finally, the count
> is the only game state the model sees, and the score, inning, and runners could be added.

---

## 7. Split the ceiling paragraph in 3.1 at the level/split boundary

**Location.** 3.1, third paragraph, "For a hitter's level, wOBA pooled over both hands, the
ceiling is 0.742..."

**Problem.** The paragraph carries 11 numbers across two different targets and then a bridge to
3.2. The level result is one sentence and could join the method paragraph before it, and the
split result needs its own paragraph. Alcorn gives each comparison its own paragraph and keeps
the dotted/dashed ceiling detail in the caption, which the draft already does well.

**Pattern.** Alcorn 3.2 gives batters and pitchers separate paragraphs, "For pitchers, Craig
Kimbrel is Aroldis Chapman's nearest neighbor".

**Proposed move.** Join "For a hitter's level, ... or 64\% of it (Figure~\ref{fig:ceiling})." onto
the end of the method paragraph from item 2 ("The ceilings were computed on 2024, ..."). Start a
new paragraph at "The platoon split is much harder to measure." and keep it to the end, including
the bridge sentence "The 2025 test in the next section ...". No wording changes are needed.
(If item 1's order is used, "next section" still holds, because 3.2 follows 3.1.)

---

## 8. Move the embedding preprocessing in 3.3 into a footnote and merge the opener with the first probe

**Location.** 3.3, first paragraph, "The loss scores single pitches and never sees a hitter-level
statistic..."

**Problem.** After a good topic sentence, the paragraph spends four sentences on seed alignment,
averaging and length scaling before any probe starts. Alcorn mentions preprocessing in one clause
("The t-SNE algorithm [14] was used to map ...") and then moves straight to the finding. The
sentence naming the selection model has to stay in the main text, because it tells the reader
which model the probes describe.

**Pattern.** Alcorn 3.1, "Visually inspecting low-dimensional approximations of neural network
representations can often provide some intuition".

**Proposed replacement** for paragraph 1, which then runs straight into the current paragraph 2
("Hitters were split into thirds ...") as one paragraph:

> The loss scores single pitches and never sees a hitter-level statistic such as a contact rate
> or a 90th-percentile exit velocity, so any such summary present in the embedding was assembled
> by the model. The analyses below use the selection model, trained through 2023. For the
> classifiers and neighbors, each hitter's embedding is averaged over the five seeds and scaled to
> unit length.\footnote{The five seeds give similar vectors for the same hitter, with a median
> cosine similarity of 0.83 after rotating each onto the first, against 0.12 when hitters are
> shuffled before the rotation is fit. Embeddings are averaged after this alignment. They are
> scaled to unit length because an embedding's length grows with the hitter's exposure
> (correlation 0.93 with log prior PA), and after scaling the main direction of variation is
> nearly unrelated to playing time (0.09).} Hitters were split into thirds on attributes measured
> from their own pitches, ...

(The rest of the paragraph is unchanged, except that the neighbor-purity sentence moves to item 12.)

---

## 9. Give the platoon-direction paragraph its evidence up front and a closing sentence

**Location.** 3.3, last paragraph, "The embedding holds the platoon split more weakly." Also the
last sentence of the head paragraph before it, "No head tracks a hitter's observed platoon split,
with every correlation at 0.05 or below."

**Problem.** The head paragraph ends on a platoon result that belongs in the next paragraph. The
platoon paragraph itself ends on three partial correlations with nothing to interpret them. Since
this is the last paragraph of the section, the reader leaves on numbers. Alcorn's probe
subsections each end on a sentence saying what the numbers mean.

**Pattern.** Alcorn 3.2 ends "could potentially reveal surprising similarities between players".

**Proposed replacement.** Delete the "No head tracks ..." sentence from the head paragraph. The
platoon paragraph becomes

> The embedding holds the platoon split more weakly. No head's output tracks a hitter's observed
> platoon split (Figure~\ref{fig:headpartial}), with every correlation at 0.05 or below. After
> removing each batting side's average split, ... [unchanged] ... at 0.07 in the low stratum,
> 0.14 in the medium, and 0.13 in the high, each with an interval above zero. The platoon split is
> therefore present in the embedding, but more weakly than power or contact.

---

## 10. Split the location-surface paragraph so the Pederson explanation gets its own paragraph

**Location.** Old 4.4 (new 3.5), first paragraph, "Because the network scores single pitches, it
can be read one pitch location at a time."

**Problem.** One paragraph describes four surfaces and then changes topic to Pederson's head
outputs, which is a different analysis (swing decisions, not contact quality) and has its own
conclusion. Alcorn starts a new paragraph when the probe changes, as when 3.2 moves from the map
to the computed neighbors.

**Pattern.** Alcorn 3.2's second paragraph opens on the new computation, "When calculating
nearest neighbors in the learned embedding space".

**Proposed move.** Start a new paragraph at "Each hitter's surface also changes little between
pitcher hands." and run it to the end ("...than in the quality of his contact."). Optionally, end
the first paragraph with an interpretive sentence that only restates the four descriptions above
it:

> These surfaces therefore differ in shape, not only in level.

(The sentence says "These" and not "The five" because the paragraph describes four of the five
surfaces.)

---

## 11. End the pitch-family paragraph on its finding, not on a caveat

**Location.** Old 4.4 (new 3.5), second paragraph, "The embedding separates some skills by pitch
type and not others."

**Problem.** The paragraph ends on a caveat ("Both correlations in the embedding run higher than
the observed ones, partly because ..."), so the reader has to go back to the topic sentence to
find the result. In Alcorn the qualification comes inside the paragraph and the closing sentence
gives the interpretation.

**Pattern.** Alcorn 3.1, "At the least, (batter|pitcher)2vec appears to be capable of capturing
the information".

**Proposed replacement** for everything after "...the two directions correlate at 0.94 and the
observed values at 0.83, close to a single direction.":

> Both correlations in the embedding run higher than the observed ones, partly because the two
> directions share the space's dominant variation. Exit velocity is observed only on balls in
> play, which leaves fewer labels per pitch family than swings and contact have and may account
> for the difference between the two skills. The embedding therefore keeps a pitch-family
> distinction for whiffs that it largely does not keep for power.

---

## 12. Thin the nearest-neighbor name lists and give the subsection the purity result

**Location.** Old 4.5 (new 3.6), both paragraphs. Also 3.3, the sentence "A hitter's ten nearest
neighbors share the hitter's power third 57% of the time, against 33% with shuffled labels."

**Problem.** The first paragraph lists 24 names in three lists of eight, and only one overlap
(Acuña) carries the argument. Alcorn names two or three players per point. The purity statistic
is a nearest-neighbor result, but it sits in 3.3, away from this subsection.

**Pattern.** Alcorn 3.2, "with close pairs including: Mike Trout/Paul Goldschmidt, Dee
Gordon/Ichiro Suzuki". It names three pairs, not lists.

**Proposed replacement.** In the first paragraph, replace the Trout statistical list sentence with

> Ranked by ten observed statistics instead, covering plate discipline, contact quality, and wOBA,
> Trout's eight nearest hitters include Yordan Álvarez, Paul Goldschmidt, Juan Soto, and Mookie
> Betts, and only Ronald Acuña appears on both lists.

Keep the two embedding lists in full, because "mostly power hitters" and "only Acuña appears on
both" have to be checkable against them. Delete the 57% sentence from 3.3 and replace the second
paragraph with

> Neighborhoods are similar in skill but do not form separate groups. A hitter's ten nearest
> neighbors share the hitter's power third 57\% of the time, against 33\% with shuffled labels.
> The silhouette score, near zero when groups overlap, is at most 0.054 for any number of
> clusters from two to ten, so the space is better read as neighborhoods of similar hitters than
> as a set of hitter types.

---

## 13. Split the training paragraph in 2.2 the way Alcorn separates the loss from the implementation

**Location.** 2.2, last paragraph, "The network has 266,317 parameters..."

**Problem.** Optimizer choices, the size comparison, regularization, the ensemble, runtime and the
code link are all in one paragraph. The five-decimal log losses are the only selection numbers in
Methods. Alcorn's last Model paragraph is only about implementation ("implemented in Keras 2.0 ...
trained on a laptop ... code ... can be found at") and gives no comparison numbers.

**Pattern.** Alcorn, "The model was implemented in Keras 2.0 [13] and trained on a laptop".

**Proposed replacement** (two paragraphs and a footnote)

> The network has 266,317 parameters. Weights outside the embedding table are trained with AdamW
> \citep{loshchilov2019} at a learning rate of $10^{-3}$ and batches of 8,192 pitches. The
> embedding table is trained with plain stochastic gradient descent at a learning rate of 1.0,
> because an adaptive optimizer would give a rarely seen hitter steps as large as a regular's,
> while plain gradient descent moves each embedding in proportion to the hitter's data. Embedding
> sizes of 32 and 64 were compared, and 64 was chosen because it gave the lower 2024 log loss
> averaged over five seeds.\footnote{1.02327 for 64 dimensions against 1.02378 for 32.} Weight
> decay of $10^{-2}$ applies to all weights and dropout of 0.1 to the trunk.
>
> Five copies are trained from different seeds as a deep ensemble \citep{lakshminarayanan2017},
> and each head's predicted distribution is averaged across the five before any query. The model
> was implemented in PyTorch, and each copy trains on a CPU in about 25 minutes. The code is
> available at \url{https://github.com/nseluga/hitter-embedding}.

---

## 14. Give the taken-pitch table its own short paragraph in 2.2

**Location.** 2.2, second paragraph, "The context passes through two layers of 128 units..."

**Problem.** The paragraph describes the network trunk and all six heads, and then changes topic
to a separate, non-network component (the taken-pitch table). Alcorn gives each model component
its own paragraph, each led into in words ("The batter and pitcher embeddings are then
concatenated together (4) and fed into...").

**Pattern.** Alcorn, "The batter and pitcher embeddings are then concatenated together (4)".

**Proposed move.** Start a new paragraph at "Taken pitches depend mostly on the pitcher and the
umpire." No wording changes.

---

## 15. Refer to equations by number where they are used later

**Location.** 2.4, first sentence, "The handedness query is scored on a hitter's wOBA against one
pitcher hand in 2025, called a unit." Also old 4.6 (new 3.7), "The level query averages a
hitter's projections..."

**Problem.** Equations 1 to 3 are never cited after they appear. Alcorn ties later prose back to
numbered equations ("the WAR formula (1)", "a standard softmax layer (5) and (6)"), which lets a
reader jump straight to the definition.

**Pattern.** Alcorn, "fed into a standard softmax layer (5) and (6)".

**Proposed replacement.** In 2.4

> The handedness query, Equation~\eqref{eq:query} with $S$ set to all pitchers of one hand, is
> scored on a hitter's wOBA against that hand in 2025, called a unit.

In 3.7, change "The level query averages a hitter's projections against left- and right-handed
pitchers" to "The level query averages a hitter's two handedness projections from
Equation~\eqref{eq:query}".

---

## Checked and left alone

- Introduction. Its order (problem, pitch-level idea, usual method, Alcorn, "This paper
  introduces") already follows Alcorn's arc, and each paragraph opens on a topic sentence.
- The 2.3 three-step structure ("The first step ...", "The second step ...", "The third step
  ...") and the way the equations are led into in words match Alcorn's Model section closely.
- The "can be seen in Table 1 / Figure 1a" lead-ins already follow Alcorn's phrasing.
- Old 4.1 "Ordering more than sizing". After item 1 it is the whole Discussion, and both of its
  paragraphs open on clear topic sentences.
- The footnote in 2.4 about the 32-dimension rerun. A footnote is the right place for it.
