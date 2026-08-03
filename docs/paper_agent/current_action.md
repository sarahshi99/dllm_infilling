# Current Paper-Agent Action

Timestamp: 2026-08-03 UTC

Research decision: `iterate_v3c_targeted`.

Action status: execute only the frozen V3-C0/V3-C pure-newline targeted protocol. Do not start compile gate, general nonempty guard, OpenTail, Joint-OpenTail, BoundaryShift, carry-and-remask, AST repair, random sampling, or held-out evaluation.

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
- `docs/paper_agent/experiments/20260803_dreamon_v3c_targeted_protocol.md`
- `manifests/v3_pure_newline_blank30.meta.json`

Frozen V3-C target:

- Pure30 is sealed from A Full trace only: 30 rows / 21 base problems / slot0-1-2 distribution 6/12/12; A baseline Pass is 3/30.
- C0 vetoes one exact leading pure-newline blank termination without canvas mutation.
- C inserts one locked physical blank newline without consuming the content slot.
- Each method must pass Smoke5, Pure30, and Pilot30 engineering isolation; Full642 is authorized only when Pure30 Pass is at least 6/30.

Next action: implement and execute only this frozen targeted protocol, then stop for independent review.
