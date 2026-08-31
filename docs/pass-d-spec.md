# Pass D — final claim-1 table

Prompt for a fresh opus high-effort window. First action: invoke `/research-partner hitter-embedding` and confirm the standup block. Requires: the 2025 refit run complete (launched overnight with the command Pass B printed), Phase V reviewed and closed.

## Scope

1. **Score the 2025 out-of-time eval** with the refit model. This is the only unread number in the project. §5.7: the model is frozen the moment this is scored — no reruns, no tweaks, whatever it says.
2. **Regenerate the main exhibit** with the Pass A2 pipeline on the 2025 eval: model / C.2 / C.3-full × pooled + three strata, RMSE-vs-noise-floor and rank-vs-Spearman-ceiling, bootstrap intervals. Drop the pre-tuning stamp; this is the final table.
3. **Report the 2024 → 2025 comparison** in one short artifact: did the fractions and orderings hold out of time. A contradiction is reported as a contradiction.

## Forbidden

Everything that changes the model. No new analyses — the abstract session with Nate decides what, if anything, gets added.

## Done when

- Final table committed to `results/` from the one pipeline command.
- All tests pass, exit 0. `Finding:` entry in the decision log with the headline fractions; lab-notebook entry.
- The abstract session is unblocked: every number it will cite exists in a committed artifact.
