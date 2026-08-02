# Current Paper-Agent Action

Timestamp: 2026-08-02 UTC

Research decision: `reframe`.

Action status: DreamOn V3 A/B execution is complete. Stop after independent review; do not start OpenTail, Joint-OpenTail, BoundaryShift, AST repair, or another progressive variant.

Branch: `codex/dreamon-progressive-v3-budgeted`

Frozen identity: 642 rows / 115 base problems, exact-three-line development/mechanism population; not held-out. Manifest SHA256 `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`; model snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.

Completed outcome:

- A `v3_hard_budgeted`: Pilot30 14/30 Pass and 23/30 compile; engineering gate passed, Full threshold failed, so A Full was not run.
- B `v3_hard_budgeted_nonempty_oracle`: Pilot30 18/30 and Full642 authorized. Full completed 642/642 with 280 Pass, 471 compile, 106 exact, task-macro 42.46%, and zero blank/cycle/forward-cap/runtime/protocol failures.
- B is an oracle structural diagnostic, not deployable and not held-out. Full paired net is `-21` versus one-shot and `-36` versus historical V1 progressive.

Primary artifacts:

- `docs/paper_agent/experiments/20260802_dreamon_v3_hard_budgeted.md`
- `docs/paper_agent/experiments/20260802_dreamon_v3_hard_budgeted_nonempty_oracle.md`
- `repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642/full_analysis.json`

Next action: independent review and claim reframing only.
