# Round 3 review: content accuracy and claims (final round)

Scope: claims that overreach, numbers that disagree with each other or with the result files,
context an Alcorn-style reader needs, and inaccuracies introduced by rounds 1 and 2. Rounds 1 and
2 are taken as applied and none of their items is re-proposed. The abstract (part1.tex 1 to 17),
the tab:test caption and the fig:powercontact caption are untouched. Every new number in a
proposed sentence comes from the result file named under Evidence. Proposed text has no first
person, no gendered pronouns, and no dashes, colons or semicolons in prose. Nine items, ranked by
impact.

## Open question settled: which embedding the named neighbors use

The named neighbor lists in Section 3.6 come from the **seed-0 checkpoint of the selection model**
(`clean_clean_dim64_s0`), not the five-seed mean. Three independent checks agree.

- Code. `src/analysis/model_visualization_anchors.py` loads one checkpoint
  (`saved["model"]["embedding.weight"]`, default `..._s0.pt`), row-normalizes it, and writes
  `neighbours.csv` and `neighbours_high.csv`. It never calls `seed_mean_embedding`.
- Log. `docs/decision-log.md`, entry 2026-10-31, says V.8 neighbours, V.9 surfaces and V.10
  disagreement were re-run on `clean_clean_dim64` "(s0 checkpoint ...)".
- Recomputation. Rebuilding `neighbours_high` from `clean_clean_dim64_s0.pt` reproduces the
  published lists exactly for all five anchors. Rebuilding from the Procrustes-aligned seed mean
  does not. Overlap with the published lists is Trout 6/8, Pederson 2/8, Schwarber 6/8, Bohm 6/8,
  Soto 5/8, and the embedding/statistics overlap drops from 9 of 40 to 7 of 40. (Scratch script
  `/tmp/r3_nb.py`, not a result file, so none of these overlaps should appear in the paper.)

The purity (57%) and silhouette (0.054) figures in 3.6 and the classifiers in 3.3 **do** use the
seed mean (`src/analysis/embedding_structure.py` calls `seed_mean_embedding`). The head partials
(`model_visualization_heads.py`, "frozen seed-0 trunk") and the surfaces are also seed 0. The
draft currently says the opposite for the neighbors, which is item 1.

---

## 1. The draft says the neighbors use the seed-averaged embedding. They use seed 0.

**Location.** part2.tex:80 (third sentence and the scope sentence before it), part3.tex:68.

**Problem.** part2.tex:80 says "For the classifiers here and the neighbors in
Section~\ref{sec:neighbors}, each hitter's embedding is averaged over the five seeds". That is
false for the named lists, which are one seed's vectors. Pederson's list in particular is
seed-specific, since only 2 of its 8 names survive averaging. The same scope sentence also hides
that the head outputs and surfaces are seed 0, so a reader assumes everything after 3.3 is the
ensemble.

**Evidence.** `src/analysis/model_visualization_anchors.py` (single-checkpoint load,
K_NEIGHBOURS=8), `docs/decision-log.md` 2026-10-31,
`src/analysis/model_visualization_heads.py` header, `src/analysis/embedding_structure.py`,
`results/v_chain_clean/surfaces_summary.csv`.

**Proposed text.** part2.tex:80, replace the two sentences beginning "Except for the 2025
projections" with

> Except for the 2025 projections in Section~\ref{sec:anchors}, the analyses in this section and
> those that follow use the selection model, trained through 2023. The head outputs, the response
> surfaces and the named nearest neighbors come from its first seed alone. For the classifiers
> here and the neighborhood summaries in Section~\ref{sec:neighbors}, each hitter's embedding is
> averaged over the five seeds and scaled to unit length.

part3.tex:68, insert after the first sentence ("Nearest neighbors in the embedding show what the
model treats as similar.")

> The named lists below use the first seed's embedding, scaled to unit length, and averaging over
> the five seeds changes several of the names, so the lists illustrate the space rather than
> define it.

The "9 of 40" figure is a seed-0 number and is consistent with the lists, so it can stay.
Alternative fix if the author prefers seed-mean names: add a seed-mean option to `run_v8`, save
the CSV under `results/v_chain_clean/`, and rewrite both lists and the 9 of 40 from it.

---

## 2. The medium-stratum result travels without its multiplicity

**Location.** part2.tex:48 (first sentence and the "Taken together" sentence), part5.tex:9.

**Problem.** The medium band is the paper's headline positive finding. The decision log is
explicit that it is "one exploratory cell of sixteen" and that the multiplicity must travel with
the number. The draft calls it exploratory but never says it was picked out of sixteen
comparisons, the only row of Table 1 where all four intervals exclude zero. part5.tex:9 gives the
power cost for the failed low-stratum test (5,200) but not for the medium result, which is the
one actually registered for 2026 and is much cheaper to confirm.

**Evidence.** `results/model_evaluation_final_clean/model_v1_claim1_verdict_min_pa_sweep_clean_dim64_final.json`
(medium is the only stratum passing both gates),
`results/model_evaluation_final_clean/model_v1_power_restatement_min_pa_sweep_clean_dim64_final.csv`
(medium needs 377 batters, multiplier 1.60),
`results/model_evaluation_final_clean/model_v1_claim1_paired_min_pa_sweep_clean_dim64_final.csv`
(medium has 236 batters), `docs/decision-log.md` 2026-09-08 and 2026-09-20.

**Proposed text.** part2.tex:48, replace the first sentence with

> The clearest signal is in the medium stratum, hitters with 113 to 451 prior PA against a hand.
> This comparison is exploratory. It is one of sixteen in Table~\ref{tab:test} (four rows, two
> baselines, two measures) and the only row where all four intervals exclude zero.

part5.tex:9, replace "needs a confirmatory test, which may have to pool 2026 with later seasons to
reach 80\% power." with

> is registered as a secondary hypothesis for 2026. Detecting it at its 2025 size with 80\% power
> would take about 377 batters, against 236 in the 2025 medium stratum, so 2026 may need to be
> pooled with later seasons.

---

## 3. The Pederson head reading claims a decomposition that was never run

**Location.** part3.tex:54, sentences 3 to 5.

**Problem.** "so the gap lies more in swing decisions than in the quality of contact" compares a
1 percentage point difference in swing rate with a 4-point difference in wOBA on contact. Those
are on different scales and no analysis converts them into shares of the projected gap. The
contact head also leans slightly against left-handed pitchers (contact rate 1.15 points above
average against them, 1.35 against right-handed pitchers), so all three heads move the same small
way. In addition, the surfaces are the seed-0 selection model while the projected gap in Table 2
is the 2025 final model. "the model can show where a projected split comes from" overstates what
three small, unscaled differences support.

**Evidence.** `results/v_chain_clean/surfaces_summary.csv` (Pederson p_swing -0.0225 L, -0.0122 R,
p_contact +0.0115 L, +0.0135 R, q +0.0439 L, +0.0482 R). No decomposition output exists under
`results/`.

**Proposed text.** Replace from "The head outputs place the projected platoon gap elsewhere." to
the end of the paragraph with

> The swing head shows a larger difference. Pederson swings less often than an average hitter
> against both hands, by about 2.2 percentage points against left-handed pitchers and 1.2 against
> right-handed pitchers, while making contact slightly more often than average against both. These
> differences are small and on different scales, so they do not say which part of the plate
> appearance produces the projected gap. Because each head covers one part of the plate
> appearance, the heads can point to where a projected split may come from, which a projection
> built only from outcomes, such as empirical Bayes, cannot.

---

## 4. The interval paragraph implies the seeds matter to the width. They barely do.

**Location.** part2.tex:67, second sentence (and by extension part1.tex:107, added in round 2).

**Problem.** Both sentences name "the spread across the five seeds" first, as if it were a
meaningful part of the band. At 95%, the seed spread alone has a mean half-width under 6 points of
wOBA and covers 8% of units. Adding sampling noise gives about 105 points and 87%. So the
undercoverage is a property of the noise term, and a reader planning recalibration (part5.tex:9)
needs to know that.

**Evidence.** `results/measurement_ceiling_clean/coverage_labels_coverage_labels.csv` (n=1,149,
95% level, seed only coverage 0.083, half-width 0.0056, seed plus target noise coverage 0.869,
half-width 0.105).

**Proposed text.** part2.tex:67, insert after "The intervals are built from the spread across the
five seeds and each unit's sampling noise."

> Nearly all of the width comes from sampling noise. The seed spread alone gives a mean half-width
> of about 6 points of wOBA and covers 8\% of units, against about 105 points once sampling noise
> is added.

part1.tex:107 can stay as written once this sentence is in 2.3.

---

## 5. The power and contact directions exaggerate the tradeoff the figure shows

**Location.** part2.tex:80, after "Each hitter can be placed on the fitted power and contact
directions (Figure~\ref{fig:powercontact})". Text only, the caption is untouched.

**Problem.** The decision log records that the fitted directions share dominant variance, so
positions on them correlate about twice as strongly as the observed statistics do. It also says
any figure showing the anti-diagonal band must carry the hitter-level number. The text currently
reads the figure without it, and a reader will take the band as a real power and contact
tradeoff of that size.

**Evidence.** `results/v_chain_clean/embedding_structure/embedding_alt_views.json`
(power_contact axis_correlation lda_projected -0.438, raw_metric -0.224), `docs/decision-log.md`
line 1652.

**Proposed text.** Insert after the figure reference sentence

> Positions on the two fitted directions correlate at $-0.44$, against $-0.22$ for the observed
> statistics, so the figure overstates how much power and contact trade off across hitters.

---

## 6. The ceiling share's interval is narrower than the real uncertainty

**Location.** part2.tex:13, sentences 5 and 6.

**Problem.** The 26% to 91% interval resamples only the numerator with the ceiling held fixed,
which the result file itself flags as understating uncertainty. The 2024-only ceiling (0.167) is
also not just a lower alternative. It comes from a route whose variance estimate is highly
sensitive to the noise model. Without these two facts a reader may treat 60% as a measurement
with a known interval.

**Evidence.** `results/measurement_ceiling_clean/differential_fraction_of_ceiling.csv`
(interval_note, numerator-only bootstrap), `results/measurement_ceiling_clean/routes.json`
(Route A, "3% noise-model error swings tau2 by ~100%").

**Proposed text.** Replace "The model's share is imprecise (interval 26\% to 91\%, wide enough to
hold both baselines' shares) and depends on how the variance of true splits is estimated." with

> The model's share is imprecise (interval 26\% to 91\%, wide enough to hold both baselines'
> shares), and that interval holds the ceiling fixed, so the true uncertainty is larger. The share
> also depends on how the variance of true splits is estimated.

and replace "With that variance taken from 2024 alone," with

> With that variance taken from 2024 alone, an estimate that small errors in the noise model can
> move by about its own size,

---

## 7. The 539-hitter comparison omits that both methods get the sign right equally often

**Location.** part2.tex:71, after the sentence ending "about 1.4 points of wOBA."

**Problem.** The paragraph reports only the measures where the model edges ahead. On the measure
closest to a platoon decision, whether the projected split points the right way, the two methods
are level. Leaving that out makes the 55% read stronger than it is.

**Evidence.** `results/case_studies_2025/case_studies_2025.json` (sign right on 2025, model 338,
empirical Bayes 337, of 539).

**Proposed text.** Insert

> The two projections got the direction of the observed split right about equally often, for 338
> and 337 of the 539 hitters.

---

## 8. Calibration reports slopes but not the level miss or the pooled slope

**Location.** part2.tex:58.

**Problem.** The paragraph says the medium and high slopes are near one, which reads as
well-calibrated. The same file shows the high-stratum projections sit about 9 points of wOBA above
observed on average, the low-stratum ones about 7 points below, and the pooled slope is 0.81 with
an interval that excludes one. The low-stratum slope of 0.59 also has a standard error of 0.15,
which a reader needs before reading "6 points of observed gap" as precise.

**Evidence.** `results/model_evaluation_final_clean/calibration.csv` (low mean_pred 0.287,
mean_obs 0.294, slope 0.59, se 0.15; high 0.330 and 0.321; all slope 0.806, z -3.77).

**Proposed text.** Append to the paragraph

> Across all strata the slope is 0.81, below one by more than three standard errors. The
> projections are also off in level. High-stratum projections averaged about 9 points of wOBA
> above the observed values and low-stratum projections about 7 points below.

---

## 9. "Every correlation at 0.05 or below" sits on an interval that excludes zero

**Location.** part2.tex:113, second sentence.

**Problem.** Minor. The exit velocity head's platoon correlation is exactly 0.05 with an interval
of 0.013 to 0.089, so "No head's output tracks a hitter's observed platoon split" is slightly
stronger than the data. The size is negligible, but a reader checking the figure will see one
interval clear of zero.

**Evidence.** `results/v_chain_clean/head_partial_loadings.csv` (platoon row, EV 0.050,
[0.013, 0.089]).

**Proposed text.** Replace the sentence with

> No head's output tracks a hitter's observed platoon split closely (Figure~\ref{fig:headpartial}).
> Every correlation is 0.05 or below in size, and only the exit velocity head's interval excludes
> zero.

---

## Checked and consistent (no change needed)

Alcorn facts (part1.tex:33), data counts (part1.tex:46 to 48 against manifest.json), every Table 1
value and interval, pooled rank intervals, 164 of 384 (43%), 246 batters and 5,200 (21 times),
calibration slopes, 87% of 1,149, the 539, 296, 243, 519 and 0.0448 vs 0.0462 figures, Table 2,
the 16 disagreements (9 vs 7), disagreement agreement rates, the level query (0.378, n=521), the
platoon partials (0.07, 0.14, 0.13), the cosine (0.14), 310 of 545, classifier accuracies, seed
stability (0.83 vs 0.12), norm and PC1 (0.93, 0.09), purity and silhouette, and bat speed
(n=570, training seasons). No round 1 or round 2 change introduced a wrong number. The one round 2
sentence with a content problem is the interval sentence at part1.tex:107, handled by item 4.
