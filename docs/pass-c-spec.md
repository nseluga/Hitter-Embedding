# Pass C — Phase V

Prompt for a fresh opus high-effort window. First action: invoke `/research-partner hitter-embedding` and confirm the standup block. Requires Pass B merged (model final for 2024). Runs **before** the 2025 refit, per §5.7's out-of-time promotion rule.

## Shape

This pass starts with a spec, not a build. Grill Nate (`/grill-me`) into a `docs/phase-v-spec.md` listing every visualization and probe, each with a pre-registered expectation, then build, then Nate reviews.

## Known probes to seed the grilling

1. **Embedding structure** — what the learned hitter table organizes on (§5.1 probe machinery exists).
2. **The platoon-skill direction** — is there a direction in embedding space that carries the platoon differential, and how it relates to exposure.
3. **Disagreement cases** — where the model and C.2/C.3-full rank hitters differently, and on whom: named hitters, their exposure stratum, and whether the disagreements concentrate where the ceiling says signal exists.

## Forbidden

No model changes — V reads the trained artifacts. No 2025 data. No probe promoted to a claim without its pre-registered expectation; V is descriptive unless the spec says otherwise.

## Done when

- `phase-v-spec.md` approved by Nate before building.
- Every figure regenerates from a committed script; outputs in `results/`.
- All tests pass, exit 0. Notebook + decision-log entries in five-field format.
- Verdict from `/research-review` on the phase recorded before the refit launches.
