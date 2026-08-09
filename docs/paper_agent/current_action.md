# Current Paper-Agent Action

Timestamp: 2026-08-09 UTC

Status: `completed`.

Branch: `codex/frontier-gated-dreamon-64`

Experiment: independent `Frontier-Gated DreamOn V0`.

Frozen boundary: historical DreamOn V1/V2/V3, Hard3, OpenTail, Nonempty, compile-gate, slot, line-wise, BoundaryShift, and V3 expansion-budget code/manifests/results remain read-only frozen/superseded experiments. Their negative/mixed evidence is retained.

Implementation result:

- One continuous `prefix + dynamic_middle + suffix` state; step 0 has exactly 64 masks.
- Only unresolved-mask position eligibility changes: `w in {1,4,8,16,infinity}`.
- Full-sequence official DreamOn forward, native expand/delete/broadcast/stop semantics, and ordinary newline tokens are preserved.
- `w=infinity` matched unmodified native DreamOn on 55 real samples for final tokens/text, every selected position/action, and stop reason, with normal/newline-after-continue/expand/delete coverage.

Population result:

- Pilot-30 development/mechanism: `w=1 28/30`; `w=4,8,16,infinity 27/30`; all five automatically advanced at the frozen `16/30` gate.
- Fixed-full-1000 development/validation: `w=1 555/1000`, `w=4 555/1000`, `w=8 554/1000`, `w=16 553/1000`, `w=infinity 553/1000`; all configurations completed 1000/1000 with zero frontier violations and zero exceptions.
- Paired vs `w=infinity`: `w=1` help/harm `24/22`, `w=4` `4/2`, `w=8` `1/0`, `w=16` `1/1`. Cluster-bootstrap intervals do not support a robust superiority claim; `w=8` lower bound is exactly zero.
- No finite window shows systematic broadcast-delete harm. The result is near-tie/mixed development evidence, not an official 5815 full and not a frozen test.

Primary artifacts:

- `experiments/frontier_gated_dreamon/report.zh.md`
- `experiments/frontier_gated_dreamon/pilot30_summary.json`
- `experiments/frontier_gated_dreamon/fixed_full_1000_summary.json`
- `experiments/frontier_gated_dreamon/review_manifest.latest.json`
- `docs/paper_agent/codex_handoff_frontier_gated_dreamon_20260809.zh.md`

Decision: `complete_near_tie_no_5815_expansion`. Do not tune windows, add mechanisms, run 5815, or call this frozen-test evidence. Any held-out follow-up requires a separately preregistered decision.
