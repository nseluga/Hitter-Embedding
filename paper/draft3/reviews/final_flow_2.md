# Final review, lens: the reader and the figures (draft 3, build9 PDF)

Read as an MLB front-office analyst at SSAC. Pages are from the build9 PDF, which renders at **18 pages**, not 17.

## 1. Top changes (ranked)

**1. Rebuild the Figure 1 image. It is the first visual a judge sees, and it is raw notebook output.** (Abstract, p2, "Hitters placed by the embedding on fitted power and contact directions")
The panel titles are code: "Power tercile (ev_p90) held-out CV accuracy 0.791", "Density-binned wOBA gradient (12x12) -- grey <5 hitters / x spans ev_p90 in [82.46, 114.20] mph". The colorbar reads "woba_level", both axes say "LDA score", and the light-gray "low" tercile almost disappears on white. The abstract caption is fixed, so the image has to explain itself.
- Axes: "Power direction (SD)" and "Contact direction (SD)".
- Panel titles: "Observed power third" and "Mean observed wOBA (cells with 5+ hitters)".
- Legend: "Low / Middle / High". Use a darker gray for Low.

If the image stays as it is, an analyst's first impression is that the paper is a code dump, and the abstract's strongest qualitative claim has no readable evidence.

**2. Cut Figure 8, the platoon-direction bar chart.** (§3.3, p11)
It shows 64 bars over unlabeled embedding dimensions, titled "finite-difference gradient (unit norm), cosine = 0.142" and "E.14 ridge weight". To a reader, embedding dimensions are arbitrary, so the bars carry no meaning. The one message is the cosine, and the text already states it. Cutting the figure loses nothing and also fixes the half-empty top of p11. Replace the paragraph's second sentence with:
> The direction along which the projected left-minus-right gap grows is only weakly aligned with a direction fit to decode the empirical Bayes split from the embedding (cosine 0.14, interval 0.08 to 0.20).

**3. Drop Table 2. Figure 9 already shows every row of it.** (§4.2, p12 to p13)
Figure 9 plots the same eight hitters' model, empirical Bayes, and observed splits with error bars, plus six more hitters. The table's only extra content is the $z$ column, and the Figure 9 caption already says "Every bar holds both projections." Back to back, the pair shows the same thing twice and pushes the interesting material (Figures 10 and 11) a page later. Two text changes are needed.
- §4.2 intro: change "Table~\ref{tab:cases} gives each projected 2025 split…" to "Figure~\ref{fig:cases} (top) gives each projected 2025 split beside the empirical Bayes projection and the observed split."
- Figure 9 caption: change "the three high-exposure disagreements in Table~\ref{tab:cases}" to "three high-exposure hitters among the twenty largest 2024 disagreements (Ozuna, Judge, Altuve)".

**4. Make Figure 5 earn its space next to Table 1.** (§3.2, p9)
This is the only place the headline test's intervals appear, so it is the key results figure. Right now it reads as a debug plot.
- The suptitle is "Claim 1: model vs gbm_full by exposure stratum, filled marker = CI excludes zero in model's favour". That is internal naming and British spelling.
- The strata are lowercase, and the axis labels say "vs gbm_full".
- It shows only the boosted baseline, but the text claims "all four intervals exclude zero" for the medium stratum against both baselines. The reader cannot see two of those four intervals.

Fixes:
- Plot both baselines as two marker colors.
- Add an "All" row. The pooled rank-correlation win is the abstract's main positive result and has no visual.
- Drop the 50-PA series, since the text says it "changes none of these outcomes". Leave that to one clause in the caption.
- Relabel the panels "Rank correlation, model minus baseline (higher favors model)" and "RMSE, model minus baseline (lower favors model)".

**5. Rework Figure 11 so it explains a platoon split, not just contact quality.** (§4.3, p14)
The bottom row (against right-handed pitchers) looks almost the same as the top row. The text's point is exactly that "each hitter's surface also changes little between pitcher hands", which one sentence can carry. Meanwhile, the most interesting mechanism in the paper has no visual. §4.2 says Pederson's split comes from swing decisions ("the swing shortfall is larger… against left-handed pitchers"), and §4.3 then refers back to §4.2. `results/v_chain_clean/surfaces_location.csv` already contains `p_swing` and `p_contact` by location and hand. Replace the bottom row with swing probability against left-handed pitchers minus against right-handed pitchers, by location. Then move the Pederson head-output sentences from §4.2 into §4.3 beside the figure.

A scout reads this kind of chart without help ("Pederson chases less but also offers less at lefties' pitches over the plate"). It is the clearest answer to "what does the model see that empirical Bayes does not", and it costs no new modeling.

**6. Fix the page-layout defects.**
- **p17 is about 95% empty and the references sit alone on p18.** Delete the last line, "The code is available at \url{…}", and the `\clearpage` before `\bibliography`. The URL already appears in the abstract and in §2.2, so this would be its third appearance. The references then fit on p17, which gives the expected 17 pages.
- **p2: the abstract's URL is cut off from its sentence.** "Code and processed data:" ends p1, and the URL appears alone below Figure 1 on p2. Move the Table 1 and Figure 1 source blocks after the Conclusion paragraph. They already render after it, so the abstract's order does not change, and the URL then continues its sentence at the top of p2.
- **Equation 1 runs past the right margin** (p5), and its number drops to the next line. Wrap it with `split` or `multline` after the $\log p(c_i)$ term.
- **Footnote 1 splits across pp. 6 and 7.** Shortening it to one sentence would fix that.

**7. Give the reader one bridge sentence in §3.1, and fix Figure 4's lone bar.** (p7)
§3.1 moves between three targets. The 2025 test scores side-specific wOBA, while the ceilings are computed for "level" and "split" on 2024, and an analyst gets lost there. After "The same idea gives a ceiling for each target." insert:
> Side-specific wOBA, the quantity tested on 2025, combines a hitter's overall level with the hitter's platoon split, and the two have very different ceilings, so each is measured separately.

In Figure 4 the left panel shows only a Model bar, while the right panel shows all three methods. The reader will ask how the baselines did on level. Either add the empirical Bayes and boosted bars, or add "Baselines are shown only for the split." to the caption.

**8. Caption and label fixes.**
- **Figure 2(b)** says "sample 6 real pitches **he** threw". That is a gendered pronoun, which the writing rules forbid. Change it to "the pitcher threw".
- **Figure 3**: the dashed curve ("Rank-correlation ceiling, √reliability") is never explained. Add: "The dashed curve, the square root of reliability, is the highest rank correlation any projection could reach at that PA."
- **Figure 7**: say that the values are partial correlations after removing log prior PA, and give the number of hitters. A caption should stand alone.
- **Figure 10**: the subtitle "Spearman ρ by prior-PA stratum" is fine, but it uses "LHP" while the body spells out "left-handed pitchers". "LHP" is acceptable in a figure, but use the same form in Figures 9, 10 and 11.

## 2. What should not change

- **Figure 6 (calibration)** is the cleanest figure in the paper, and every analyst understands its slope-by-stratum layout.
- **Figure 7 (head heatmap)**: its diagonal shows at a glance that each head learned its job.
- **Figure 12 (whiff vs. power by pitch family)**: one comparison of two clean scatters, and the contrast between r = 0.64 and r = 0.94 is easy to see.
- **Figure 13 (level query)**: it shows the result rising with exposure in a form the reader can see.
- **§4.1 "Ordering more than sizing"**: the bench-bat example is the best translation into front-office terms in the paper. Keep it first in the Discussion.
- **The case-study framing**: expectations were fixed in advance, and noise is stated plainly ("noise alone could flip which projection is closer"). That candor builds trust with a skeptical analyst.
- **Results order** (ceiling, then test, then what the embedding learned) follows the author's stated plan, and it works.

## 3. Verdict

The prose is now in Alcorn's register, and the Results and Discussion carry far more visuals than before. The figure set has not caught up with the writing, though. Three figures still look like analysis-notebook output, with code-variable names, "Claim 1" titles and British spelling: Figure 1 (the abstract figure), Figure 5 (the headline test) and Figure 8. Two pairs repeat each other: Table 2 with Figure 9, and the two rows of Figure 11. An SSAC analyst skims the figures before the text. As drafted, the skim gives an impression of an unfinished internal report on exactly the two figures that carry the main claim. The work is mostly cutting and relabeling, not new analysis. Cut Figure 8 and Table 2, rebuild Figures 1 and 5 with plain labels and both baselines, and turn Figure 11's redundant row into a swing-by-location platoon panel. Fix the empty p17, and the paper reads as finished at 17 pages with a visual story an analyst can follow without the text.
