# Final review, draft 3: flow and argument arc

## 1. Top changes (ranked)

**1. Give the disagreement analysis its own subsection, and say plainly that the case studies sit in the regime where the model does not help.**
*Where:* 4.2, Table 2 bottom rows, and the paragraph "Hitters on whom the model and empirical Bayes disagree...".
*Why:* 4.1 has just argued that the model's edge is in the medium stratum. 4.2 then presents five high-exposure stars, and empirical Bayes is closer for all five. A reader who is following the argument reads this as evidence against it. The six medium-stratum hitters, who belong to the regime the paper argues for, appear only in the bottom panel of Figure 9 and are never named. Table 2 also shows Judge, Altuve and Ozuna before the paper has introduced the idea of a disagreement, so a result arrives before its setup. To fix both, keep 4.2 for the five pre-registered hitters plus the top of Table 2, and start a new subsection, "Where the model and empirical Bayes disagree", that holds the disagreement paragraph, Figure 10, the bottom rows of Table 2 (or fold them into Figure 9 only), and the Figure 9 grading. After "All five are high-exposure hitters, so they show how the model describes hitters it knows well." add:
> "This is the stratum where the 2025 test found no advantage, so the cases illustrate the model's description of hitters rather than its accuracy."

If you skip this, the Discussion's strongest qualitative evidence sits next to its weakest regime, and the reader is left to work out why.

**2. Move the exposure argument out of "Ordering more than sizing" into its own subsection.**
*Where:* 4.1, second paragraph, "The model was built to be accurate for every hitter...".
*Why:* This paragraph is the paper's central explanation. It pays off Intro paragraph 2, where pitch outcomes carry less luck, and the medium-stratum result in the abstract. Right now it sits under a heading about something else. Give it its own heading, `\subsection{Why the advantage peaks at moderate exposure}`. It can also pick up evidence the paper already has but never connects. Add at the end:
> "The platoon direction in Section 3.3 points the same way, tracking the observed split less closely in the low stratum (0.07) than in the medium and high (0.14 and 0.13)."

**3. Connect the ceiling targets to the tested target, and announce the ceiling in the Intro.**
*Where:* 3.1, after "...for the 545 hitters with at least 10 PA against each hand." Also the last sentence of the Intro.
*Why:* 3.1 sets ceilings for *level* and *split*, and 3.2 then tests *side-specific wOBA*. Nothing says how the two relate, so a reader holding Table 1's rank correlation of 0.47 looks for its ceiling and can't find one. Insert:
> "The tested target, a hitter's wOBA against one hand, combines two parts, the hitter's overall level and the platoon split. The two are bounded separately because a projection can rank hitters well on level while saying little about the split, which is the part a platoon decision turns on."

The Intro roadmap also never mentions the ceiling, so Results opens on a step nobody announced. Replace the last sentence of the Intro with:
> "After measuring how much of each target one season of outcomes can reveal, the query is tested on 2025 handedness splits against empirical Bayes and gradient-boosted baselines, and the embedding is examined for what it learned from per-pitch outcomes alone."

**4. Fix "Other queries" so that it isn't a repeat and covers what the paper promises.**
*Where:* 4.4 first half, 2.3 "Section~\ref{sec:otherqueries} shows what two other queries return", and the abstract's "or a specific pitcher".
*Why:* The "level query" in 4.4 uses the same pooled projection that 3.1 already scored (0.475 against a 0.742 ceiling). Without saying so, it reads like a new query. Meanwhile the single-pitcher query, which the abstract names as a use case, never appears outside Future directions, and a "location" query shows up there that was never introduced. Rewrite the opening of 4.4:
> "Any set of pitchers defines a query, and only the handedness query was tested against baselines. Taking $S$ as a single pitcher gives $W_{hj}$ directly, the matchup form named in the abstract. The level query is the pooled projection scored against its ceiling in Section 3.1, here checked against playing time (Figure 13). Its correlation with held-out 2024 wOBA, after removing log prior PA, is 0.38 across 521 hitters (interval 0.30 to 0.45), so its ranking is not a proxy for exposure. Unlike the handedness margin, which peaks in the medium stratum, it rises steadily with exposure, from 0.17 in the low stratum to 0.38 in the medium and 0.52 in the high."

In 2.3, change "shows what two other queries return" to "discusses two others". In Section 5, drop "location" or introduce it in 4.4.

**5. Keep the Pederson thread in one place.**
*Where:* 4.2, "Pederson's head outputs differ by pitcher hand... against left-handed pitchers." and the end of the first paragraph of 4.3, "...differs little between hands (Section~\ref{sec:anchors})."
*Why:* This is the paper's best interpretive finding. Pederson's split comes from swing decisions, not from contact quality. It is split across two subsections with a back-reference, and the swing half sits in the case-study section before the reader has seen what a head output looks like. Delete the sentences from 4.2 and end the 4.3 paragraph with:
> "...so the quality of contact differs little between hands. The split comes from the swing head instead. Pederson swings less than an average hitter against both hands, and the shortfall is larger against left-handed pitchers."

**6. Bridge the heads paragraph to the platoon direction, and don't end Results on the weakest figure.**
*Where:* 3.3, "The embedding also holds a weak platoon direction."
*Why:* The paragraph before ends "No head tracks a hitter's observed platoon split", and the next one opens by saying the embedding does hold one. That reads as a contradiction. Replace the opening with:
> "No head tracks the split on its own, but the composed query recovers a weak platoon direction."

Figure 8, a 64-bar gradient chart with a cosine of 0.14, is also the last thing a reader sees before the Discussion. It is hard to read and makes a weak close. The text carries the result. Cut the figure, or move the paragraph so the power and contact result (Figure 1 and the classifiers) closes Results. In either case, write "(Figure 1, shown with the abstract)" so the reader knows to flip back to page 2.

**7. Calibration paragraph: correct the claim of a shared pattern, and move the interval coverage.**
*Where:* 3.2, "Calibration follows the same pattern" and "The model's prediction intervals are too narrow. On 2024..."
*Why:* The stratum margin peaks in the medium stratum and falls to even in the high one. Calibration is good in both medium and high and poor only in low, so it does not follow "the same pattern". The coverage result also comes from 2024, a different season and a different concept, so dropping it into the middle of the 2025 test breaks the thread. 4.1 repeats it anyway. Replace the opener with "Calibration separates the low stratum from the other two (Figure 6)." and move the 87% coverage sentence into 4.1, replacing "(Section~\ref{sec:test})".

**8. Float placement, the Section 5 opener, and repeated facts.**
- *Floats escape their subsections.* In the PDF, Figure 4 (the ceiling) lands on page 8 in the middle of 3.2, splitting the sentence "empirical Bayes's 0.40". Figure 10 (disagreement) lands in 4.3, the surfaces subsection. The likely cause: `\let\oldsubsection...\renewcommand\subsection{\FloatBarrier...}` (lines 12 to 13) runs before `\usepackage{titlesec}`, which redefines `\subsection` when it loads and discards the barrier. Move those two lines after the `titlesec` setup.
- *Section 5 opener.* It drops the paper's two takeaways. Replace the second sentence with:
  > "On 2025 handedness splits it matched gradient boosting and empirical Bayes on error and ordered hitters better than both, with the clearest margin for hitters with a moderate history against a hand."
- *Repeats to cut.* The code URL appears three times (abstract, end of 2.2, end of Section 5), so cut the last one. The fact that embedding length tracks exposure (0.93) appears in 2.2, 3.3 and 4.3, so cut the 4.3 one. The point that noise can flip which projection is closer appears in the 3.2 text, the Table 2 caption, the 4.2 text and the Figure 9 caption, so keep it in one caption and one place in the text.

## 2. What should not change

- The Results order, from signal ceiling to test to what the embedding learned, is right. The ceiling primes the reader to read a tie on RMSE as a property of the data, not a failure of the model. Keep it.
- 3.2 leads with the pooled result, then names the failed pre-registered test plainly before the medium-stratum finding. The paper is honest and still makes its case.
- The opening of 4.1 ("better at saying which of two hitters... than by how many points", plus the bench-bat example) is the clearest statement of the thesis and echoes the abstract's conclusion well.
- The Methods sequence (data, network, three-step query, evaluation) reads as one pipeline. The three steps of the query are cleanly separated.
- Alcorn appears at both ends: as lineage in the Intro, and in Section 5 through the counterfactual roster comparison and minor-league mapping. That closes the frame.
- 4.4 ends with an honest "remain untested", which leads naturally into Future directions.

## 3. Overall verdict

The macro arc works. Intro (noise lives at the end of the PA) leads to Methods (model the pitch, compose to wOBA), then Results (ceiling, test, learned structure), then Discussion (why). The Results order the author wanted is in place and earns its keep. The thread breaks in the Discussion. The paper's main explanation, the exposure curve, sits under a heading about something else. The case studies then show only the regime where the model has no edge, and they introduce the disagreement hitters before the concept that explains them. "Other queries" repeats a level result from 3.1 and never shows the single-pitcher query the abstract promises. Each fix is a heading, a moved paragraph or a bridge sentence, not new analysis. Together they would make the Discussion read as one argument: the model orders better than it sizes, it orders best at moderate exposure because pitch outcomes accumulate faster than results, and here is what that looks like in actual hitters. Fix the float placement too, since a figure landing mid-sentence in the wrong subsection undercuts the polish elsewhere.
