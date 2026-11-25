# Draft 3 final reviews (2026-11-01)

Six independent opus reviewers read the assembled `draft3/main.tex` and the rendered PDF.
- Three reviewed **flow**. Their lenses were argument arc (F1), reader and figures (F2), and length and claims (F3).
- Three compared the draft **to Alcorn 2018**, reading the full PDF. Their lenses were structure (A1), tone (A2), and content focus (A3).

Facts were already checked section by section, so these reviews cover flow and comparability only. Full reviews, including replacement prose, are in `reviews/final_*.md`.

Priority: M = must, S = should, C = could, x = raised (unranked), blank = not raised.

## Already applied (objective defects, not editorial)

| Fix | Raised by |
|---|---|
| The float barrier never ran because `titlesec` overrode the `\subsection` redefinition. Figure 4 had landed inside 3.2 and Figure 10 inside 4.3. The redefinition now comes after the `titlesec` setup, and every figure stays in its own subsection. | F1, F2, A1 |
| The Fig 2(b) caption said "he threw" (a gendered pronoun). It now reads "the pitcher's real pitches". | F2 |
| Deleted the third "code is available" line and the `\clearpage` before the references. Page 17 had been nearly blank. The paper is now 17 pages, not 18. | F2, F3, A1, A2 |

## Where reviewers agree (2+ of 6)

| # | Change | F1 | F2 | F3 | A1 | A2 | A3 | Argument | My take |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cut the 2.2 training paragraph from 287 to about 150 words. Drop the earlier-build loss, the single-seed screen, the replayed steps, and "as the architecture plan called for". Move the seed-stability cosine (0.83 vs 0.12) to 3.3 | | | M | S | M | S | This is the paper's longest paragraph and the last real process narration. "As the plan called for" is not a reason to test 32 vs 64 | **Yes, must.** It needs **your** actual reason for testing 32 and 64. A1 and A2 both left a placeholder for it |
| 2 | Keep the Pederson thread in one place, 4.3. Its payoff is that the split comes from swing decisions, not contact quality | S | S | S | | | S | The swing half is in 4.2 and the contact half in 4.3, so the main finding never gets stated | **Yes.** F2 would also replace the near-duplicate bottom row of Fig 11 (vs RHP) with a swing-probability L−R map from `surfaces_location.csv`. That is a cheap figure and a better use of the slot |
| 3 | Drop Fig 8 (64-bar platoon-direction chart) | x | M | | S | | | It is a gradient over arbitrary dimensions with no visual takeaway, and the text already carries the cosine 0.14 | **Yes.** It also leaves Results ending on the head-partial figure, which is stronger |
| 4 | Rewrite §5 as uses rather than a to-do list. Echo the abstract takeaways. "Against a given pitcher" becomes "against a pitcher hand". "Far from significant" becomes "underpowered" | S | | M | S | | S | The single-pitcher query is untested, so the matchup card oversells it. "Far from significant" reads as a negative result | **Yes.** A3 also adds a trade-deadline acquisition use. I like it, since it is Alcorn's counterfactual move |
| 5 | Bridge 3.1 to 3.2. The ceilings are for level and split on 2024, and the test is side-specific wOBA on 2025, but nothing links them | S | S | | | | | The reader can't tell how the 0.475 and the 0.47 relate | **Yes**, one sentence. F2 also flags that Fig 4's level panel shows only the model. Add the baselines or say why in the caption |
| 6 | Repetition sweep. The length-vs-exposure 0.93 appears 3×, the 43% unseen hitters 3×, the "noise can flip which is closer" point 4×, and "no retraining" 5× | x | | S | | x | S | Together these cost about 120 words | **Yes.** It is mechanical |
| 7 | Trim 4.1 paragraph 2. It repeats the 2.4 design-intent statement and the 43% fact, and states the pitch-accumulation explanation as fact | | | M | | S | S | It repeats Methods, and the mechanism is untested | **Yes** to trimming it to "would explain". F3's addition fits your framing: the low stratum ranks well but is overspread (slope 0.59). Note that F1 wants the opposite (see disagreement B) |
| 8 | Shorten captions to describe, not argue. A1 adds Alcorn's format: "Figure 1." with a period and table captions above | | S | | S | x | | Several captions restate the result | **Yes** to shortening. The caption format is **your call**, since the SSAC template may differ |
| 9 | Bohm replacement into a footnote | | | | | S | S | It keeps the disclosure without halting the case-study paragraph | **Your call.** The disclosure has to stay. A footnote is fine. A2 asks you to confirm the swap happened before you saw any projections |
| 10 | Table 2: drop it (F2: Fig 9 plots all 8 of its hitters) or drop the z column (A3) | x | S | | | | C | Fig 9 duplicates it. F1 adds that its bottom rows (Judge, Altuve, Ozuna) appear before the disagreement idea is introduced | **Lean drop the table** and keep Fig 9. That moves the z-scores into the Fig 9 caption range |

## Raised by one reviewer, but strong

| # | Change | Who | Argument | My take |
|---|---|---|---|---|
| 11 | Fig 1 image (the abstract figure) is raw notebook output. Panel titles read "Power tercile (ev_p90)" and "Density-binned … -- grey <5", the colorbar reads "woba_level", and the low third washes out | F2 (M) | It is the first figure anyone sees. The caption is verbatim but the image is not | **Yes, must.** Relabel only, with no data change |
| 12 | Fig 5 (strata margins) reads as a debug plot. Its title is "Claim 1: model vs gbm_full … favour". It shows only the GBM, while the text claims all 4 medium intervals exclude zero | F2 (S) | The headline test's intervals appear only here | **Yes.** Add the EB rows and a pooled row, drop the 50-PA series (it's in the text), and relabel |
| 13 | Add a "Using the query" paragraph at the end of 2.3. It covers what you supply, what comes back, the cost (72 pitches per hitter and pitcher), and other weightings (a series staff by expected innings, a single reliever for a pinch hitter) | A3 (S) | Alcorn spends about 15% on how to use it, and this draft about 6%. The selling point (a new question is a new weight vector) is never shown in use | **Yes.** The paragraph says a query runs "in seconds on a CPU", which is **unverified**. I'd check it or drop it |
| 14 | Worked platoon decision in 4.1: one real 2025 team's 3–4 platoon hitters, with the model's ranking, the EB ranking, and the observed order vs LHP | A3 (S) | This is "here's why it matters" in concrete form. It is a lookup in the existing 2025 projections | **Your call.** It's cheap and very Alcorn, but it's a hand-picked example. It needs a stated rule for choosing the team |
| 15 | §4.4: the "level query" is the same pooled projection 3.1 already scored. The abstract's single-pitcher query never appears, and "location" shows up in §5 unannounced | F1 (S) | 2.3 promises more than 4.4 delivers | **Yes** to saying the level query equals the pooled projection. For the single pitcher, either add one example query or soften 2.3. F3 says leave 4.4 alone otherwise |
| 16 | The calibration paragraph opener "follows the same pattern" is wrong. Calibration is poor only in the low stratum, which is not the margin pattern. Move the 87% coverage sentence to 4.1 | F1 (S) | The 4.1 text already repeats it | **Yes** |
| 17 | The 3.1 ceiling-share comparison (60% vs 49%/48%) implies a ranking the 26–91% interval can't support. Add "wide enough to hold both baselines' shares" | F3 (S) | It is an overclaim | **Yes** |
| 18 | Intro: add one clause of design intent (built mainly for low exposure, meant to be accurate for all). Label the medium-stratum result "exploratory" once in 3.2 | F3 (S) | It matches your framing and is honest about what was registered | **Yes.** One label, not a hedge on every sentence |
| 19 | Tone cleanups. Hedge stacks: 3.3 "only weakly … positive but far from parallel" and 4.3 "One possible reason … not tested". Workflow jargon: "selection model/runs" (define once in Data as "model trained through 2023"). Informal phrases: "do their damage", "power bats", "shade into one another", "the statistic it should" | A2 (S) | These are the last places the register slips from Alcorn's | **Yes.** It is mechanical |
| 20 | Layout. Eq 1 runs past the right margin (p5). Footnote 1 splits across pages. The abstract's code link sits alone under Fig 1 | F2 (C) | Polish | **Yes** when we finalize. Recheck after the cuts, since pages will move |
| 21 | One sentence of plain intuition before Eq 3 (the count chain) | A3 (C) | Alcorn introduces each equation in words first | **Yes** |

## Where reviewers disagree

**A. Cut or add?** A1 and F3 want 450–900 words cut, which brings the paper to about 15 pages. A3 wants about 500 words moved, from testing and process to usage and why-it-matters (#13, #14). These fit together: cut the process (#1), the repetition (#6), and the re-grading in 4.1/4.4, then spend part of the savings on #13. **Recommend both.**

**B. How to restructure Section 4.**
- F1: split the disagreement analysis out of 4.2 into its own subsection, name the six medium-stratum disagreement hitters, and give the exposure argument (4.1 para 2) its own subsection.
- A3: merge 4.2 and 4.3 into one walk-through per hitter (surface, then heads, then split, then grade).
- A1: give the nearest-neighbor paragraph its own subsection, "Nearest neighbors", like Alcorn's 3.2.

**My take:** F1's split. Grading five high-exposure hitters right after arguing for the medium stratum reads as evidence against the argument, and naming the medium hitters fixes that. A3's per-hitter merge fights your Results/Discussion split. Neighbors as a short subsection (A1) is compatible with F1. **Your call.**

**C. The 2.4 footnote on the earlier 32-dimension build.** A1 and A3 want it deleted as process history. A2 wants it reworded so it doesn't read as history. F3 says keep it. **Lean keep, reworded (A2).** It is the one disclosure that 2025 was scored twice, and cutting it would hide that.

**D. 4.1 exposure argument.** F1 wants it expanded into its own subsection and tied to the 3.3 platoon-direction r by stratum (0.07/0.14/0.13). F3 and A2 want it trimmed. **Lean trim (#7).** Add F1's tie-in as one sentence only if the direction figure goes (#3), since the numbers would otherwise be lost.

## Overall verdicts
- F1: the arc works and Results is in the right order. The Discussion thread breaks. Every fix is a heading, a move, or a bridge.
- F2: the prose now matches Alcorn's register, but the figures haven't caught up. Figs 1, 5 and 8 read as notebook output.
- F3: close. About 600–900 words of excess and three calibration misses (#4, #7, #17).
- A1: already shaped like Alcorn. It departs mainly on length (5,800 vs 3,000 words) and in captions that argue.
- A2: the voice is mostly impersonal and close to Alcorn. A few process asides and hedge stacks remain.
- A3: it sounds like Alcorn but doesn't spend its words like Alcorn. Testing and process take about 45% against his 15%, and usage and why-it-matters are under half his share.
