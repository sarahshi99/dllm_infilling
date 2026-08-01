# Current Paper-Agent Action

Timestamp: 2026-08-01 UTC

Research decision: `iterate`.

Action: implement and evaluate only `v2_hard_v2_boundary`, then stop for independent review. Do not resume V2-OpenTail protocol-v1 and do not start V2-OpenTail-v2 or Joint-OpenTail-v2 in this round.

Branch/worktree:

- branch: `codex/dreamon-progressive-v2-slots`
- worktree: `/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v2-slots`
- iteration starting HEAD: `0cf2e9bd47590629fdde31ab1028b59044e63cef`

Frozen evidence:

- manifest SHA256: `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`
- population: 642 rows, 642 unique task IDs, 115 base problems
- role: exact-three-line development/mechanism population, not held-out test
- model snapshot: `8ccc74750e43177327f29dab9e91882ba759e194`

Superseded protocol-v1 evidence:

- V2-Hard-v1 remains immutable at 642/642, Pass@1 `7/642`, compile `46/642`, and 421 forward-cap failures.
- Required label: `V2-Hard-v1 decoder/protocol failure diagnostic; not a clean test of progressive-slot efficacy`.
- V2-OpenTail-v1 partial357 is abandoned with `resumable:false` at `repro_results/abandoned/dreamon_v2_opentail_protocol_v1_partial357/`.
- Joint-OpenTail-v1 was never started.

Protocol-v2 decoder revisions:

1. newline-containing hard-slot proposals become online `line_boundary` actions;
2. EOS deletes selected/right unresolved masks only within the active region;
3. a repeated exact full transition terminates immediately as `exact_deterministic_cycle`.

Execution gate:

1. focused tests, runner tests, py_compile, shell and source scan;
2. Smoke 5 requires 5/5 completed and zero error/cycle/invariant/cross-region-delete rows;
3. Pilot 30 requires 30/30 completed, compile at least 24/30, Pass@1 at least 10/30, and complete discard/reference diagnostics;
4. Full 642 runs only if the pilot gate passes, with preregistered non-completion early stops;
5. stop after V2-Hard-v2 result or any failed gate.

Protocol: `repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json`.

Command: `bash repro_scripts/run_dreamon_progressive_v2_hard_v2.sh {verify|smoke|pilot|full}`.
