# Current Paper-Agent Action

Timestamp: 2026-06-17 18:20 CST

## Action Name

Discovery V4 signal-model literature brainstorming and executable plan.

## Current Phase

Route2 error analysis Discovery V3 is complete:

- output: `analysis_outputs/route2_error_analysis_20260617_165806`
- result: `1033` joined rows, pairwise `6/0/795/232`, `33` triggered failed-long rows, `56` missed failed-long rows, `31/33` triggered failed-long rows with rescue length >= oracle
- decision: `mixed_rescue_quality_and_gate_recall`

The next step is not a GPU experiment. It is a CPU-first Discovery V4 design that searches for useful signals and mechanisms in the three-layer stack:

1. Error and action anatomy.
2. Discovery model layer.
3. Policy distillation layer.

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback. The brainstorm compares risk-controlled selection, slice discovery, rule mining, trace-shape discovery, calibration/OOD signals, weak supervision, and counterfactual/uplift diagnostics.
- `superpowers:using-git-worktrees`: checked. Current branch is the dedicated `paper-agent-overnight`; this planning action is narrow documentation and does not require a new worktree.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`.

The current tool environment does not expose callable `superpowers:*` skill files, so this action follows the local project protocol as fallback.

## New Planning Outputs

- Design spec: `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`
- Executable plan: `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`
- Literature brainstorm brief: `docs/paper_agent/experiments/20260617_discovery_v4_literature_brainstorm.md`

## Key Design Decision

V4 should not treat "find a feature" as single-feature enumeration. It should treat the problem as risk-controlled action selection:

- `MissedLongHead`: find inference-visible signals for the `56` missed failed-long rows.
- `RescueQualityHead`: explain the `33` triggered failed-long rows, especially because `31/33` already have rescue length >= oracle.
- `PolicyDistillation`: convert any useful discovery model into a small, reviewer-readable, training-free gate/action rule.

## Literature-Inspired Method Families

- Risk-controlled selection: optimize coverage under short/current-pass risk constraints.
- Slice/subgroup discovery: find local regions of model failure or rescue success.
- Rule extraction: use shallow trees, sparse scores, and rule ensembles as microscopes.
- Time-series trace shape: search late plateau, high-confidence stagnation, early collapse, progress-then-stall.
- Calibration/OOD residuals: normalize trace confidence by selected length, stop reason, and probe disagreement.
- Weak supervision: combine noisy heuristics before distilling a rule.
- Counterfactual/uplift diagnostics: treat rescue as an action, but keep causal claims conservative because current action logs are biased.

## GPU Policy

No GPU experiment is running or should be launched by this action. Before any future GPU work:

- write a new action brief with success/kill criteria;
- check `nvidia-smi`;
- do not use GPU `2/3` while other users' tasks are present;
- require a CPU candidate that passes held-out risk gates.

Recent check showed GPU2 and GPU3 occupied at roughly `40GB` used each.

## Next Implementation Target

Implement CPU-only:

- `analysis/discovery_v4_signal_audit.py`
- `tests/test_discovery_v4_signal_audit.py`

Expected output:

- `analysis_outputs/discovery_v4_signal_audit_TIMESTAMP/row_action_table.csv`
- `slice_candidates.csv`
- `rule_candidates.csv`
- `trace_shape_candidates.csv`
- `calibration_residuals.csv`
- `uplift_diagnostics.csv`
- `policy_shortlist.md`
- `report.md`
