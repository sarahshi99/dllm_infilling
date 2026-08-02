# Current Paper-Agent Action

Timestamp: 2026-08-02 UTC

Research decision: `reframe`.

Action status: DreamOn V3 A/B execution is complete. Stop after independent review; do not start OpenTail, Joint-OpenTail, BoundaryShift, AST repair, or another progressive variant.

Branch: `codex/dreamon-progressive-v3-budgeted`

Frozen identity: 642 rows / 115 base problems, exact-three-line development/mechanism population; not held-out. Manifest SHA256 `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`; model snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.

Completed outcome:

- A `v3_hard_budgeted`: Pilot30 14/30 Pass and 23/30 compile; the original Full threshold failed. A Full was later run only by explicit user authorization as a post-hoc mechanism diagnostic: 281/642 Pass, 538/642 compile, 97/642 exact, task-macro 41.17%, 137 blank-slot rows, and zero terminal/protocol failures.
- B `v3_hard_budgeted_nonempty_oracle`: Pilot30 18/30 and Full642 authorized. Full completed 642/642 with 280 Pass, 471 compile, 106 exact, task-macro 42.46%, and zero blank/cycle/forward-cap/runtime/protocol failures.
- B vs post-hoc A Full: 15 wins / 16 losses, McNemar `p=1.0`, clustered CI `[-2.60pp,+2.09pp]`; compile help/harm `6/73`, exact help/harm `9/0`. The nonempty guard removes all A blank slots but does not improve functional Pass and uses 25.04% more token-forwards.
- B is an oracle structural diagnostic, not deployable and not held-out. Full paired net is `-21` versus one-shot and `-36` versus historical V1 progressive.

Primary artifacts:

- `docs/paper_agent/experiments/20260802_dreamon_v3_hard_budgeted.md`
- `docs/paper_agent/experiments/20260802_dreamon_v3_hard_budgeted_nonempty_oracle.md`
- `repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642/full_analysis.json`
- `repro_results/dreamon_progressive_v3_hard_budgeted_ab_comparison_all642/report.zh.md`

Next action: independent review and claim reframing only.
