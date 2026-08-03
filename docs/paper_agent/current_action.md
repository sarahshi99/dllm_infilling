# Current Paper-Agent Action

Timestamp: 2026-08-03 UTC

Research decision: `advance`.

Action status: V3-C targeted execution is complete. Stop all DreamOn experiment execution and wait for independent review. The only supported next step is a separately preregistered held-out validation of V3-C; do not run it automatically.

Branch: `codex/dreamon-progressive-v3-budgeted`

Frozen identity: 642 rows / 115 base problems, exact-three-line development/mechanism population; not held-out. Manifest SHA256 `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`; model snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.

Completed V3 outcomes:

- A `v3_hard_budgeted`: post-hoc explicitly authorized Full642, 281 Pass / 538 compile / 97 exact, 137 blank-slot rows.
- B `v3_hard_budgeted_nonempty_oracle`: Full642, 280 Pass / 471 compile / 106 exact. Blanket nonempty removes blanks but does not improve Pass and substantially harms compile; retain as negative oracle diagnostic.
- C0 `v3_c0_budgeted_oneshot_pure_newline_veto`: Smoke5 passed; Pure30 was 5/30 Pass and failed the frozen 6/30 Full gate. C0 Full was not run.
- C `v3_c_budgeted_nonconsuming_blankline`: Smoke5 passed; Pure30 14/30; Pilot30 17/30 with all engineering/isolation checks passing; Full642 completed 642/642 at 292 Pass / 541 compile / 97 exact / task-macro 43.54%.
- C vs A Full: 12 wins / 1 loss / 280 both-pass / 349 both-fail; McNemar `p=0.00341797`; 115-base-problem cluster bootstrap CI `[+0.50pp,+3.17pp]`; compile help/harm 4/1. No-trigger isolation passed for all 612 rows.
- C vs one-shot is net -9 and C vs historical V1 is net -24, with clustered intervals including zero. This development population does not establish superiority to those controls.
- C vs C0 on Pure30: 10 wins / 1 loss; cluster CI `[+8.33pp,+53.57pp]`. The evidence favors physical blank-line reconditioning over a simple one-shot veto.
- Frozen-future normal confidence was higher in only 2/31 trigger events by either entropy or top1-probability comparison. This does not support a general future-slot-confidence explanation.

Primary artifacts:

- `docs/paper_agent/experiments/20260803_dreamon_v3c_targeted_protocol.md`
- `docs/paper_agent/experiments/20260803_dreamon_v3c_targeted_results.md`
- `repro_results/dreamon_v3c_targeted_analysis/report.zh.md`
- `repro_results/dreamon_v3c_targeted_analysis/analysis.json`
- `repro_results/dreamon_v3_c_nonconsuming_blankline_all642/`

Next action: independent review. If accepted, preregister one held-out V3-C validation with no method changes. Do not restart C0, blanket nonempty, compile gate, BoundaryShift/carry, AST repair, OpenTail, Joint-OpenTail, random retries, or another development-population variant.
