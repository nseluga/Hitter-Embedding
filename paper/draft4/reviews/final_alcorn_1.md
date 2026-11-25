# Final review, lens: Alcorn comparability, structure and format

Alcorn source: `~/personal-os/knowledge/library/baseball-research/alcorn-2018-batter-pitcher2vec.pdf` (12 pp). Draft: `draft3/main.tex`, rendered `build9/main.pdf` (renders at **18** pages, not 17).

## Side-by-side map

| Element | Alcorn 2018 | Draft 3 |
|---|---|---|
| Pages | 12 (body ~9.5, refs ~2.5) | 18 (abstract 2, body ~14.5, refs from p18) |
| Abstract | 1 paragraph, ~140 w, no headings or floats | 4 headings, ~430 w, Table 1 + Fig 1 (fixed, out of scope) |
| 1 Introduction | 7 paras, ~650 w, Eq 1, Fig 1 | 5 paras, ~500 w, no floats |
| 2 Methods | 2.1 Data, 2.2 Model. ~550 w, Table 1, Fig 2, Eqs 2-7 | 2.1 Data, 2.2 Model, 2.3 Querying, 2.4 Evaluation. ~1,550 w, Fig 2, Eqs 1-3, 1 footnote |
| 3 Results | 3.1 PCA, 3.2 Nearest neighbors, 3.3 Doppelgängers, 3.4 Unseen matchups. ~900 w, Figs 3-5, Table 2, Eqs 8-9 | 3.1 Signal ceiling, 3.2 2025 test, 3.3 Embedding. ~1,120 w, Figs 3-8 |
| Discussion | none | 4.1-4.4, ~1,710 w, Figs 9-13, Table 2 |
| Future directions | 2 paras, ~420 w, Fig 6, opens with one-sentence summary | 2 paras + lone URL line, ~300 w |
| Closing | Acknowledgments, refs follow on same page | URL line alone on p17, `\clearpage`, refs on p18 |
| Paragraph length | median ~90 w, max ~170 | median ~130 w, max 287, seven over 185 |
| Floats / equations | 6 figs, 2 tables / 9 eqs | 13 figs, 2 tables / 3 eqs |
| Code link | p1 footnote + end of 2.2 | abstract, end of 2.2, end of 5 |
| Captions | "Figure 1." label, 1-2 sentences, table caption above table | "Figure 1:" label, up to ~80 w, table caption below |

**Justified divergences (keep):** the Discussion section (Alcorn has nothing to interpret beyond one test, this paper has a mixed verdict and the author asked for it); 2.3 Querying (the query is the contribution, Alcorn's paper has no analogue); 2.4 Evaluation in Methods (a pre-registered test must be specified before its result, Alcorn's single naive baseline fit inside 3.4); Results order (ceiling, test, embedding is the author's order and leads with the headline); more figures than Alcorn (author's direction); fewer equations (Alcorn spends six on a standard softmax model, the draft spends its three on the novel chain and query). No intro or future-directions figure is fine.

**Divergences to bring in line:** total length and paragraph size, process history in Methods, the closing, caption style, figure count where a figure carries no reader-facing finding.

## 1. Top changes

**1. Fix the closing so the paper ends like Alcorn's.** Where: end of Section 5, "The code is available at \url{...}." plus the `\clearpage` before `\bibliographystyle`. Why: this third copy of the URL is the only text pushed onto p17, which is otherwise blank, and the `\clearpage` then starts references on p18. Alcorn ends Future directions, then Acknowledgments, then references on the same page. The URL already sits at the end of 2.2, exactly where Alcorn puts it. Delete the line and the `\clearpage`. Saves one to two pages at zero cost. Consider a one-line Acknowledgments if anyone is owed one, matching Alcorn's closing.

**2. Cut the training paragraph from 287 to ~150 words.** Where: 2.2, "The network has 266,317 parameters..." Why: it is twice Alcorn's longest paragraph and three times his training paragraph, and it narrates process (an earlier data build, a single-seed screen, replayed learning-rate reductions) that the author explicitly asked to drop. Keep the "tested because x, chosen because z" sentence for embedding size. Seed stability (0.83 vs 0.12) belongs with the embedding analysis in 3.3, where alignment over seeds is already mentioned. Replacement:

> The network has 266,317 parameters. Weights outside the embedding table are trained with AdamW \citep{loshchilov2019} at a learning rate of $10^{-3}$ and batches of 8,192 pitches, with weight decay of $10^{-2}$ and dropout of 0.1 in the trunk. The embedding table is trained with plain stochastic gradient descent at a learning rate of 1.0, which moves each embedding in proportion to the hitter's data rather than giving a rarely seen hitter steps as large as a regular's. Embedding sizes of 32 and 64 were compared, and 64 was chosen because it gave the lower 2024 log loss averaged over five seeds (1.02327 against 1.02378, a gap about four times the seed-to-seed standard deviation). Five copies are trained from different seeds as a deep ensemble \citep{lakshminarayanan2017}, and each head's predicted distribution is averaged across the five before any query. The model was implemented in PyTorch and trained on a CPU in about 25 minutes per seed. The code is available at \url{https://github.com/nseluga/hitter-embedding}.

**3. Fix Figure 10's placement.** Where: 4.2, `fig_disagreement` environment. Why: in the render it floats past the 4.3 heading to p14 and splits the 4.3 pitch-family paragraph mid-sentence ("present in the | [Fig 10, Fig 11] | data."). A Case studies figure that appears inside "What the model sees" breaks the section mapping Alcorn keeps clean (every figure sits in the subsection that cites it). Move the `fig_disagreement` block above `fig_cases` in the source, or use `[H]`. Re-check p13 to p15 after change 5.

**4. Drop Figure 8 (platoon direction).** Where: end of 3.3, `fig_platoon_direction.png`. Why: it shows 64 bar heights for two raw vectors. No SSAC reader can read a finding off it, and Alcorn never shows raw dimensions, only projections (PCA, t-SNE) and named neighbors. The paragraph's numbers (310 of 545, cosine 0.14, partials 0.07/0.14/0.13) carry the result alone. Results 3.3 then ends on the head-correlation figure, which is the cleaner "what the embedding learned" close. Lost if kept: a page of float for a weak, uninterpretable picture.

**5. Give nearest neighbors its own subsection, titled as Alcorn titles it.** Where: 4.3, last paragraph, "Nearest neighbors in the embedding show what the model treats as similar." Why: this is the single most Alcorn-parallel analysis in the paper (embedding neighbors against stat-based neighbors, with Trout as the example, as in Alcorn 3.2), and it is buried as the fifth paragraph of a subsection about location surfaces. Make it "4.4 Nearest neighbors" (Other queries becomes 4.5) and split the 196 words into two paragraphs at "The neighbor search also uses direction only". Optional, cheap: a two-row table (Pederson, Trout × embedding neighbors, stat neighbors) as the Alcorn-style visual.

**6. Shorten captions and match Alcorn's caption format.** Where: all captions, worst are Fig 4 ("Rank correlation with 2024 outcomes against the ceiling...", ~80 w), Fig 11 (~75 w), Fig 9, Fig 10. Why: Alcorn's captions are one or two sentences that say what is plotted, and interpretation lives in the text. The draft's captions repeat text numbers and argue ("which the model exceeds", "Every bar holds both projections"). Also set `\usepackage[labelsep=period]{caption}` for "Figure 1." and put `\caption` above `tabular` for tables, as Alcorn and the SSAC template do. Fig 4 replacement:

> Rank correlation with 2024 outcomes against the ceiling one season allows, for the level of wOBA (left) and the platoon split after removing each batting side's average (right). Dashed and dotted lines are the main and alternative ceilings.

**7. Recast the second Future directions paragraph as outward-facing.** Where: Section 5, "Several extensions would make these uses more reliable..." (189 w, seven items). Why: Alcorn's Future directions are uses and extensions for the reader's own work. The draft's paragraph reads as an internal to-do list ("should be recomputed on a season not used for model selection"). Replacement:

> Several extensions would make these uses more reliable. The prediction intervals need recalibration before a card can show a band. The advantage at moderate exposure was exploratory, and a confirmatory rule for 2026 is registered, although one season holds only about 60\% of the batters needed to detect that margin, so the test may pool 2026 with later seasons. Hitters absent from training, who now share one of four generic projections, need a better starting point, such as minor league data mapped into the same space as \citet{alcorn2018} suggested. The pitcher-trait and single-pitcher queries need held-out targets built for them. Finally, the score, inning, and runners could join the count as game state.

**8. Remove the 2.4 footnote.** Where: "2025 was also scored with an earlier build that used a 32-dimension embedding and kept pitchers batting..." Why: Alcorn has no footnotes, and this one is pure process history the author asked to drop. It also invites a reader to wonder which build is real. If a robustness note must survive, it belongs in the repo README.

## 2. Things that should NOT change

- Section skeleton numbering and depth (1, 2.1 to 2.4, 3.1 to 3.3, 4.x, 5) mirrors Alcorn's two-level scheme.
- 2.1 Data reads like Alcorn 2.1: source, cleaning, counts, split, in one paragraph.
- The intro's arc (problem, prior approach via Alcorn, alternative via Tango, "This paper introduces...") matches Alcorn's intro move for move.
- Eq 1 with its "where for pitch $i$..." gloss is exactly Alcorn's equation style.
- Future directions opening (one-sentence summary, then applications, then the Alcorn counterfactual roster idea) is the right Alcorn-style close.
- 3.1 to 3.3 order follows the author's stated Results plan.

## 3. Verdict

Structurally the draft is recognizably an Alcorn-shaped paper: same section skeleton, same intro arc, same Methods style, same Future-directions move, and the added Querying, Evaluation, and Discussion sections are earned by a paper that tests a query rather than only presenting an embedding. Where it departs from Alcorn without cause is mass. The body runs about 5,800 words against Alcorn's ~3,000, its paragraphs run about 40% longer, and the overflow is concentrated in process narrative (training paragraph, footnote, to-do-list future work) and argumentative captions, which is what the author asked to remove. Changes 1, 2, 4, 7, and 8 together cut roughly 450 words, one figure, and two pages with no loss of findings. Changes 3 and 5 fix the one real layout defect and surface the analysis most directly comparable to Alcorn. Expect the paper at about 15 to 16 pages afterward. Further length cuts should come from the 180 to 230 word paragraphs in 2.4, 4.1, and 4.2, but those are content calls for other reviewers.
