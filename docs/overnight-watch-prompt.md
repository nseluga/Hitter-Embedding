You are the overnight watchdog for `scripts/overnight.sh`, already running under nohup (log: `/tmp/overnight.out`, per-stage logs: `/tmp/hitter-overnight/`). Read `docs/handoff-overnight-run.md` first; it is binding.

Your job: make sure the run finishes by morning, then do the handoff's end-of-run steps.

Loop until `/tmp/overnight.out` shows the final stage done or the script has exited:
1. Wait with `timeout 590 tail -n 30 -f /tmp/overnight.out` (Bash tool timeout 600000). Do not use bare `sleep`.
2. Check `pgrep -f scripts/overnight.sh`. Alive and log advancing → keep waiting. Alive and log silent > 90 min on one stage → note it, keep waiting (training stages are ~25–40 min per seed; the final-run tensor build can be long).
3. Exited non-zero → read the failing stage log, diagnose, fix ONLY script/path/flag/env problems, then resume with `FROM=<stage> nohup ./scripts/overnight.sh > /tmp/overnight.out 2>&1 &` (append the old log to `/tmp/overnight.prev.out` first). Max two resume attempts on the same stage; after that stop and write what you ruled out.
4. Replay gate (stage 3) fails → STOP. Do not set `GATE_OVERRIDE`. Record the reference value and the tolerance miss in `docs/run-notes-overnight.md` and end.

Hard rules (from the handoff): never edit `src/model/`, loss code, or the data pipeline; never rerun any 2025 stage (4–6) a second time; never make a Tier 3 decision; never launch a second heavy job while one is running (8 GB machine).

When the script completes (stage 6 done, exit 0), wrap up in this order — this is the "After all stages" block of the handoff's Run section:
1. Numbers from every stage log to `docs/run-notes-overnight.md` (nodecay slope vs +0.438; replay gate reference value; refit 2025 claim-1 by stratum vs `eb_bivariate`/`gbm_full`; prior-on 2024 chain deltas; fidelity bb/k).
2. Spawn one Sonnet verifier (Agent tool, model sonnet, read-only) to check: stale arm names in results, pins reproduce, 2025 read only by stages 4–6, row counts. Append its findings to the run notes.
3. `pytest -q` — record pass/fail count.
4. `git add` results/, docs/run-notes-overnight.md, the pins entry, and a lab-notebook entry (Did/Why/Found/Learned/Next) in `docs/lab-notebook.md`; one commit. No push.
5. Last lines of `docs/run-notes-overnight.md`: the "Report to Nate (morning)" shape from the handoff, then `WATCHDOG DONE <date>`.
If any wrap-up step fails, write what failed to the run notes and stop; do not retry stages.
