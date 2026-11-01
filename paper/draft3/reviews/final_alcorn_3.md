# Alcorn comparability, content focus (reviewer 3)

## Word allocation (body prose, excluding abstract, captions, tables, references; rough)

| Category | Alcorn (~2,800 words) | Draft 3 (~5,000 words) |
|---|---|---|
| (a) How it works | ~32% (word2vec framing, data, model, loss) | ~21% (data, network, loss, query) |
| (b) How to use it / what it is for | ~15% (Future directions written as uses: trade back-simulation, prospect mapping, defense input) | ~6% (one "matchup card" sentence, one bench-bat sentence, the other-queries paragraph) |
| (c) Why it matters | ~18% (stats are ad hoc, scouts think in contexts) | ~7% (Intro paragraphs 1, 2, 4, one Discussion paragraph) |
| (d1) Accuracy testing | ~15% (Section 3.4) | ~34% (Evaluation, 3.1, 3.2, 4.1, most of 4.2, half of 4.4, half of 5) |
| (d2) What it learned, interpretive | ~20% (PCA, neighbors, doppelgängers) | ~18% (3.3, surfaces, pitch-family split, neighbors) |
| (e) Research process | ~0% | ~11% (optimizer and dimension screens, weight decay screen, step replay, earlier-build footnote, floor sweep, power calc, anchor replacement, registered 2026 rule) |

The draft matches Alcorn on (d2) already. The gap is that (d1) and (e) together take ~45% against Alcorn's ~15%, and (b) and (c) are under half Alcorn's share. Alcorn tells the reader once how the test came out and spends the saved space on what the embeddings let a team do. The draft grades the model again in 4.1, 4.2, 4.4 and 5, after 3.2 has already done it.

## 1. Top changes, ranked

**1. Add a "Using the query" paragraph at the end of Section 2.3 (b).** Anchor: after "Section~\ref{sec:otherqueries} shows what two other queries return." Right now the reader learns the math of Eq. 3 but never what a user supplies, what comes back, or what it costs. That is the "how to use mine in your work" content the author asked for, and it is the paper's actual contribution (one weight vector per question). Without it, "no retraining" stays a slogan. Proposed text:

> In use, a query takes a hitter, a set of pitchers, and a weight for each pitcher, and returns a projected wOBA per plate appearance. Each $W_{hj}$ needs only the 72 sampled pitches of one hitter and pitcher, six in each of 12 counts, so a full query over every pitcher of one hand runs in seconds on a CPU. The weights need not be batters faced. A team preparing for a series can weight the opposing staff by the innings each pitcher is expected to throw, and a team choosing a pinch hitter can set $S$ to the single reliever warming up. Hitters absent from training receive the generic projection for their batting side, so the query adds nothing for them until they accumulate pitches.

(Confirm the "seconds" claim against `src/model/query.py` timings before keeping it, or drop that clause.)

**2. Add a worked decision example to Section 4.1 (b, c).** Anchor: "Choosing which of two bench bats starts against a left-handed pitcher is a ranking question". This sentence is the paper's thesis and is never shown. Add a small table of one real 2025 team's three or four non-everyday hitters, their medium-or-low-stratum projections against left-handed pitchers from the model and empirical Bayes, the ranking each gives, and the observed 2025 order. It needs no new modeling, only a lookup in the existing 2025 projection file. It is the one artifact that shows a reader what the abstract's conclusion ("ranking platoon options among hitters with a moderate track record") means. Lead-in text:

> Table X gives one such decision. For the four hitters [team] used most often in a platoon role in 2025, the model and empirical Bayes each rank the hitters against left-handed pitchers, and the observed 2025 order is shown beside them. [One sentence on where the two rankings differ and why, from the head outputs.]

Also add one (c) sentence after it, using Table 1 counts: "Medium-exposure units are 280 of the 1,157 scored in 2025, about one in four, so the range where the model helps most covers a large share of real platoon choices." Losing this keeps the medium-stratum result as a statistic rather than a use.

**3. Cut the hyperparameter paragraph in 2.2 to its reasons (e).** Anchor: "The network has 266,317 parameters. ..." (287 words). The author's own instruction is "sizes of 32 and 64 were tested because x, y was chosen because z". Keep exactly that and drop the screens, the earlier-build loss comparison, the step-replay detail, and the seed-cosine statistic (move the 0.83 vs 0.12 to the first paragraph of 3.3 if wanted, where it supports reading the embedding). Replacement (~150 words):

> The network has 266,317 parameters. Weights outside the embedding table are trained with AdamW \citep{loshchilov2019} at a learning rate of $10^{-3}$ and batches of 8,192 pitches. The embedding table is trained with plain stochastic gradient descent at a learning rate of 1.0. An adaptive optimizer would give a rarely seen hitter steps as large as a regular's, while plain gradient descent moves each embedding in proportion to the hitter's data. Embedding sizes of 32 and 64 were compared, and 64 was chosen because it gave the lower 2024 log loss averaged over five seeds. Weight decay of $10^{-2}$ applies to all weights and dropout of 0.1 to the trunk. Five copies are trained from different seeds as a deep ensemble \citep{lakshminarayanan2017}, and each head's predicted distribution is averaged across the five before any query. Each copy trains on a CPU in about 25 minutes. The code is available at \url{https://github.com/nseluga/hitter-embedding}.

**4. Rewrite Future directions paragraph 1 as uses, Alcorn style (b, c), and halve paragraph 2 (e).** Alcorn's Section 4 is almost entirely "teams would be able to". The draft's is two-thirds a to-do list. Replace the "Several extensions" paragraph with:

> Two limits stand between the model and those uses. The prediction intervals are too narrow and need recalibration before a card can show a band, and the moderate-exposure advantage was found in exploratory analysis and needs a confirmatory test on 2026 and later seasons. Beyond that, the velocity, location, and single-pitcher queries need held-out targets of their own, hitters absent from training need a better starting point, such as minor league data mapped into the same space \citep{alcorn2018}, and game state beyond the count could be added.

Then add one use sentence to paragraph 1: "Because the pitcher enters only through the pitches thrown, the same query can project a hitter against a pitcher acquired at the deadline, using that pitcher's pitches from another team." Cut "a fixed rule for 2026 is already registered", the 60% power figure, and "The measurement ceiling should be recomputed" (process).

**5. Make 4.2 and 4.3 one interpretive walk-through per hitter (d2 over d1).** Anchors: "In 2025, empirical Bayes was closer for all five" and "Figure~\ref{fig:surfaces} shows". The case studies currently read as a second accuracy test, with $z$ scores, while the surfaces that explain each projection sit a section later. Lead each hitter with what the model sees (surface, head outputs), then its projected split, with one closing sentence on grading. This turns the case studies into Alcorn's 3.2: concrete demonstrations of how the method reads a hitter. Drop the $z$ column from Table 3 and keep its caption's last sentence.

**6. Cut repeated grading from 4.1 and 4.4 (d1).** 4.1's first paragraph restates 3.1 and 3.2 (error floor, narrow intervals, single-hitter noise). Cut to: "The model is better at saying which of two hitters will do better against a hand than by how many points, and Section 3 shows that one season of outcomes cannot reward sizing for any method." In 4.4, cut "Checking it is harder than checking the projection" through the reliability numbers to one sentence: "One season of outcomes cannot separate these matchup effects from noise (split-half reliability 0.08), so these projections remain untested."

**7. Strip remaining process asides (e).** Delete the Evaluation footnote on the earlier 32-dimension build. Move "Raising the evaluation floor from 10 to 25 or 50 PA changes none of these outcomes..." to the Figure 5 caption, which already shows both floors. In 4.2, compress the Bohm replacement into a footnote ("Alec Bohm replaced a pre-registered hitter whose training data showed no split.") so the disclosure stays but stops leading the section.

**8. One (a) sentence of intuition before Eq. 3.** Anchor: "The third step chooses the pitchers." Add: "In effect, the projection is the hitter's expected wOBA if every plate appearance came against the chosen pitchers in proportion to their workload." Alcorn spends words making the mechanism graspable; the draft's query is described accurately but only in mechanics.

## 2. What should not change

- Section 2.3's three-step structure and Eq. 2 are the clearest how-it-works writing in the paper, and they match Alcorn's equation-then-gloss style.
- Section 3.3 and the surfaces, pitch-family, and neighbor material are the right analog of Alcorn 3.1 to 3.3, including the stats-based neighbor contrast (Alcorn's Goldschmidt move).
- The Alcorn paragraph in the Introduction, which positions the model as the PITCHf/x extension Alcorn proposed.
- The honest null on the pre-registered test. Alcorn is unhedged because the result was clean. This one is not, and the draft states it plainly once in 3.2. Keep it there, just stop restating it.

## 3. Verdict

The draft now sounds like Alcorn but does not yet spend its words like Alcorn. Its interpretive material is at the right share, its methods are accurate, and the process noise is down from earlier drafts. But accuracy testing and process still take nearly half the body against Alcorn's sixth, while "how to use it" sits at 6% against 15%. The paper's selling point, a new question costs a new weight vector, is never shown being used. Adding a "Using the query" paragraph and one worked platoon decision, cutting the hyperparameter paragraph and the repeated grading in 4.1, 4.4 and 5, and rewriting Future directions as uses would move about 500 words from (d1) and (e) to (a) through (c). That gives a shorter paper whose balance is close to Alcorn's.
