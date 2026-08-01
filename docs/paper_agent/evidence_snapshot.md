# Paper-Agent Evidence Snapshot

DreamOn decoder iteration note, 2026-08-01 UTC: V2-Hard-v1 remains immutable at Pass@1 `7/642`, compile `46/642`, with `421` forward-cap failures, but is now explicitly classified as `V2-Hard-v1 decoder/protocol failure diagnostic; not a clean test of progressive-slot efficacy`. V2-OpenTail-v1 partial357 is abandoned with `resumable:false`, and Joint-v1 was never started. The only active DreamOn experiment is protocol-v2 method `v2_hard_v2_boundary`; it must pass strict 5-row and 30-row preregistered gates before any 642-row run. No protocol-v2 empirical result exists yet.

Phase 5 update, 2026-07-12 UTC: authoritative starting HEAD for the resumed correction is `3315ed82d42770e3ed7d8ae20e9d5f1570940ff6`. The original pass-trained F3 OOF score is now explicitly `supervised_probe_diagnostic` and cannot authorize or enter deployable selection. The current deployable mechanism is `AST/def-use bridge proxy V0`, using a fixed deterministic formula and a corrected within-task/cross-canvas gate. It is not full Semantic Bridge Projection. Focused correction tests currently pass (`18` tests before final verification). No new empirical candidate/pass evidence exists yet; frozen test remains sealed and `test_evaluation_count=0`.

Generated from existing local `results.jsonl` files. Raw outputs are not copied here.

Update: 2026-06-01 01:52 CST. This snapshot remains the compact evidence anchor for the current A6000 checkpoint; the later probe-curve audits are tracked separately in `docs/paper_agent/probe_curve_signal_audit.md`, `docs/paper_agent/probe_curve_signal_audit.json`, `docs/paper_agent/probe_curve_split_score_audit.md`, and `docs/paper_agent/probe_curve_split_score_audit.json`.

H200 migration note, updated 2026-07-07 UTC: `analysis_outputs/h200_bootstrap_20260705_103617/` records the new-server bootstrap audit. Bootstrap verdict is `host_h200_available_sandbox_gpu_hidden`: default sandbox GPU checks fail because `/dev/nvidia*` is hidden, but approved host/unsandboxed checks show H200 is healthy and `dllm_env` sees CUDA. GitHub SSH auth and remote branch freshness are verified. Tier 1 H200 core reruns are complete; compact audit `analysis_outputs/h200_repro_audit_20260707_tier1_v2/` reports `h200_material_outcome_drift` (Control `787`, Midcons `794`, Route2 `795`, V6 `796`, Local CAL `769`). H200 action-bank rebuild `analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/` covers train/calibration/validation only (`927` tasks, `4635` rows, no test rows), and Controller V1 replay `analysis_outputs/controller_validation_h200_20260707_v1_replay/` again selects zero intervention with validation `89/127`; comparison audit is `analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`. CPU-only material drift triage `analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/` is retained as a pre-acceptance audit artifact, writes no row-level test details, and keeps frozen test sealed with evaluation count `0`. Researcher decision `docs/paper_agent/h200_evidence_base_decision.zh.md` accepts H200 reruns as the new evidence base; A6000 results below are historical reference and must not be silently mixed with H200 evidence.

H200 Controller V2 note, updated 2026-07-07 UTC: feasibility audit `analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility/` gives `mixed_controller_failure`; validation has `17/127` recoverable rows and `67/127` harmable rows. Controller V2 validation `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/` gives `risk_certification_limited_test_sealed`. The only nonzero V2 signal is `ordinal_only + probe_trace_fused`: validation `90/127`, wins/losses `5/4`, interventions `41`, population harm upper95 `7.06%`; gate fails and frozen test remains sealed.

H200 Controller V3 note, updated 2026-07-08 UTC: candidate screen `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/` completes Family A/B/C on validation and writes `topk_policy_curves.csv`, `validation_predictions.csv`, `validation_action_selection.csv`, and `validation_summary.json`. Best pass-count Family A point is `probe_trace_fused, k=15`: `91/127`, wins/losses `3/1`, net `+2`, population harm upper95 `3.68%`, but it has one `<=8` bucket loss. The selected exploratory route is the conservative Family A `probe_only, k=5`: `90/127`, wins/losses `1/0`, population harm upper95 `2.33%`. Route decision is `weak_validation_signal_test_sealed`; frozen-test gate fails and frozen test remains sealed with evaluation count `0`.

Phase 4 generalization note, updated 2026-07-08 UTC: second-backbone feasibility audit `analysis_outputs/second_backbone_feasibility_20260708_phase4_v4/` recommends `Dream-org/Dream-Coder-v0-Base-7B` as the most feasible second backbone because local cache, official-canvas runner, evaluator compatibility, and historical full SingleLine results exist. Fresh Dream-Coder oracle diagnostic `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/` completes the 15-case subset: primary/control `7/15`, simple length policy `7/15`, oracle-sufficient canvas `14/15`, missed-long oracle recoveries `3`, triggered-long oracle recoveries `3`, short regressions `0`. This is mixed cross-backbone evidence rather than clean model-agnostic confirmation. Second-regime feasibility `analysis_outputs/second_regime_feasibility_20260708_phase4_v4/` finds MultiLine/RandomSpan aliases but missing local JSONL files under `data/`; the continuation constructs `synthetic_second_regime_minimal` and fresh diagnostic `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/` gives control/deployable/oracle `18/18`, so it is an unblock/sanity result, not an official benchmark. LR-DLLM final attempt `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/` gives `blocked_missing_algorithmic_detail`. Paper skeleton lives at `paper/diagnostic_mixed_draft/`.

Codex audit note, 2026-07-02: no raw outputs were copied or regenerated in this audit. The repository/evidence audit is tracked in `docs/paper_agent/codex_repository_audit.zh.md`; the current handoff is `docs/paper_agent/codex_handoff.latest.zh.md`.

Action-ceiling dry-run note, 2026-07-02: `analysis_outputs/action_ceiling_20260702_dryrun/` contains compact case/action manifests only. It does not add a new pass-rate result and did not launch GPU.

Action-ceiling 3-case pilot note, 2026-07-02: `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/` contains the strict 3-case GPU pilot requested after the dry-run scaffold. Verdict is `positive_control_only`: `116/L0` replayed A=fail/B=pass and C/D also pass, while `85/L0` and `113/L3` fail under A/B/C/D. This is a compact diagnostic result, not a new full-run pass-rate claim, and it does not justify automatic expansion to 9 cases.

Action-equivalence audit note, 2026-07-02: `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/action_equivalence.{json,md}` shows that the old D `steps96` action was output-equivalent to C in all three pilot cases. The audit distinguishes `pass_level_canvas_effects` from `candidate_level_canvas_effects`; absence of pass improvement must not be written as absence of generation change.

Distinct-candidate ceiling pilot note, 2026-07-02: `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/` contains the Phase 1b strict 3-case diagnostic. Verdict is `candidate_diversity_without_correctness`: action-distinctness gate passed via auditable trajectory differences, `113/L3` produced a distinct no-early-commit hash, but no hard-case correct candidate appeared. `85/L0` remained a single-hash `SyntaxError`; `113/L3` remained `UnitTestFailure`; only the positive control passed. This is candidate-existence evidence, not a deployable Pass@1 claim.

Phase 2 frozen-controller note, 2026-07-03: attribution output `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/` shows C oracle-sufficient canvas recovers `29/89` hard cases and E/F/G add only `2` incremental hard recoveries; triggered failed-long remains `0/33`. The deployable action bank `analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/` covers train/calibration/validation only (`927` rows × `5` actions = `4635` action rows). Controller validation `analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/` fails the preregistered risk gate: no nonzero-intervention calibration operating point satisfies the 5% harm upper-confidence budget. Frozen test remains sealed with `test_evaluation_count=0`. Verdict: `no_validation_signal_test_sealed`.

## Runs

| Run | Rows | Pass | Rate | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `a6000_control` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `midcons` | 1033 | 795 | 76.96% | 89.97% | 78.45% | 58.89% | 20.73% | 16.13% |
| `mid_precision` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `true_long` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `combined` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |

## Pairwise Against A6000 Control

| Candidate | Common Rows | Wins | Losses | Net | Wins By Bucket | Losses By Bucket |
|---|---:|---:|---:|---:|---|---|
| `midcons` | 1033 | 8 | 0 | 8 | `{'<=8': 1, '9-12': 3, '13-16': 4}` | `{}` |
| `mid_precision` | 1033 | 0 | 0 | 0 | `{}` | `{}` |
| `true_long` | 1033 | 0 | 0 | 0 | `{}` | `{}` |
| `combined` | 1033 | 0 | 0 | 0 | `{}` | `{}` |

## Long-Failure Summary For `midcons`

- long_total: `113`
- failed_long_total: `91`
- underselected_failed_long: `90` (98.90%)
- failed_long_base_source: `71`
- failed_long_selected_length_histogram: `{'3': 39, '4': 5, '5': 2, '6': 8, '7': 6, '8': 3, '9': 7, '10': 1, '11': 1, '12': 4, '13': 5, '14': 3, '15': 2, '16': 3, '20': 1, '21': 1}`

## Long-Underestimate Sweep

- evaluated_rules: `16776`
- strict_viable_rules: `0`
- best_rule: `sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.4|supp>=0|src=base`
- best_true_long_precision: `35.48%`
- best_failed_long_recall: `36.26%`
- best_short_risk_rate: `40.86%`
- best_current_pass_risk_rate: `27.96%`

## Probe-Curve Audit Pointer

- audit file: `docs/paper_agent/probe_curve_signal_audit.md`
- rows_with_probe_curve_features: `1033/1033`
- rows_with_stopping_trace: `0/1033`
- evaluated single-feature thresholds: `4106`
- strict_viable_thresholds: `0`
- best threshold: `long_score_max <= 0.229253`
- best threshold risk: `8.70%` short-risk, above the `5%` GPU gate

Interpretation: the current evidence supports using `midcons` as a short/medium checkpoint and rejects single-feature probe-curve thresholds as a direct GPU policy. The next offline step should be strict-split multivariate or learned probe scoring.

## Strict-Split Probe-Score Audit Pointer

- audit file: `docs/paper_agent/probe_curve_split_score_audit.md`
- split discipline: `5` deterministic SHA256 task-id folds, train-thresholds only
- rows: `1033`
- feature_count: `24`
- aggregate held-out trigger_count: `63`
- aggregate held-out true_long_precision: `47.62%`
- aggregate held-out failed_long_recall: `32.97%`
- aggregate held-out short_risk_rate: `22.22%`
- aggregate held-out current_pass_risk_rate: `7.94%`
- strict_heldout_pass: `False`

Interpretation: the simple dependency-free multivariate score does not pass the offline GPU gate. GPU work remains blocked until a safer signal, trace-enabled evidence, dynamic canvas control, or length-regularized modeling plan is justified.
