# Pass B — close tuning + citations

Prompt for a fresh opus high-effort window. First action: invoke `/research-partner hitter-embedding` and confirm the standup block. Requires: warmup-arm runs complete (5 seeds), Pass A1 merged.

## Scope

1. **`notebooks/07_tuning`.** Reads the o1 factorial artifacts plus the warmup arm (lr 1e-3, warmup 2000, seeds 0–4). Selection metric is `reference` per the Phase O spec and nothing else. Pre-registered expectation: the warmup arm does not clear 2 SE; a contradiction is reported as a contradiction. This notebook must close before anything touches 2025.
2. **Citations.**
   - The Book p.157: at most one more hour. On failure, execute the pre-decided fallback — cite empirical-Bayes platoon regression to a verifiable source and report the fitted ρ (0.652/0.719) as the values used. Do not return to Nate for this.
   - Brown (2008) via Project Euclid; drop the "ceiling as estimand" characterization if the full text does not support it.
   - Spearman via archive.org.
3. **If the warmup arm wins** (clears 2 SE on `reference`): adopt it as the final config, rerun the Pass A2 exhibit command, and log the supersession.
4. **Prepare the 2025 refit launch**: one command training the final config on the frozen split for the 2025 out-of-time eval (~2 h GPU). Print it; do not launch. §5.7: nothing may change the model after this run is scored.

## Forbidden

No arm selection — Phase O selects no arms. No 2025 data read anywhere in this pass.

## Done when

- 07 closed with the final config named; citation edits committed; refit command printed for Nate.
- All tests pass, exit 0. Decision-log entries in five-field format; findings marked `Finding:`.
