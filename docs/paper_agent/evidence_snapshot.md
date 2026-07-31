# Paper-Agent Evidence Snapshot

DreamOn min16 versus DreamCoder Fixed16 paired completion, 2026-07-31 UTC: row Pass@1 is `90.7228%` versus `73.1392%`; equal-weight 148-cluster macro is `83.1788%` versus `69.2358%`. DreamOn minus Fixed16 has row help/harm `194/31`, cluster help/harm/tie `81/9/58`, and macro delta `+13.943pp` with 95% CI `[+9.618,+18.405]pp`. This is a full-system training/checkpoint plus dynamic-canvas contrast, not an isolated dynamic-length or compute-matched effect. Frozen test remains sealed with `test_evaluation_count=0`.

DreamOn min8 versus DreamCoder Fixed8 paired completion, 2026-07-31 UTC: row Pass@1 is `90.3991%` versus `60.7335%`; equal-weight 148-cluster macro is `81.3313%` versus `48.0879%`. DreamOn minus Fixed8 has row help/harm `286/11`, cluster help/harm/tie `111/1/36`, and macro delta `+33.243pp` with 95% CI `[+28.390,+38.417]pp`. The two arms share population, pinned source decoder, sampling, and row seed keys, but differ in checkpoint/training class; this is a full-system contrast, not an isolated dynamic-length or compute-matched effect. Frozen test remains sealed with `test_evaluation_count=0`.

LR-DLLM DreamCoder RandomSpan paired completion, 2026-07-31 UTC: the `paper-guided, author-unverified reimplementation of LR-DLLM` has row/task-macro Pass@1 `18.3784%`, versus `33.5135%` for the identical-key Fixed64 control. Primary minus Fixed64 has row help/harm `104/328`, cluster help/harm/tie `19/107/22`, and task-macro delta `-15.135pp` with 95% CI `[-17.905,-12.365]pp`. This is a clear negative RandomSpan result; datasets remain separate and no combined Mean is reported. Frozen test remains sealed with `test_evaluation_count=0`.

DreamOn official-source SingleLine completion, 2026-07-31 UTC: all five released-source min4/8/16/32/64 max64 arms completed `927/927` on the project non-frozen population. Row Pass@1 is `88.4574%/90.3991%/90.7228%/91.2621%/91.6936%`; equal-weight 148-cluster macro is `78.1111%/81.3313%/83.1788%/83.0372%/85.1299%`. Every canonical audit has zero missing/duplicate/error/accounting defects. min32/min64 each retain one recovered resource-OOM failure journal; append-only resume/dedup completed. No DreamCoder DreamOn-matched fixed controls are complete, so no paired delta is claimed. Frozen test remains sealed with `test_evaluation_count=0`.

Completed baseline pairing snapshot, 2026-07-31 UTC: CAL SingleLine primary minus official_fixed32 has row help/harm `130/96`, cluster help/harm/tie `47/31/65`, and task-macro delta `+1.898pp` with 95% CI `[-2.233,+5.759]pp`. CAL authors’ DAEDAL FIM adaptation dynamic minus Fixed8 is `+0.410pp` CI `[-2.062,+2.734]pp` on SingleLine and `+0.455pp` CI `[-1.375,+1.791]pp` on MultiLine. The paper-guided, author-unverified LR-DLLM reimplementation minus Fixed64 on DreamCoder SingleLine is `+0.796pp` CI `[-6.746,+8.178]pp`; RandomSpan primary row/macro is `18.3784%`, and MultiLine primary row/macro is `31.9945%/37.1082%`, with their Fixed64 controls still running. All completed paired cluster CIs cross zero. DreamOn official-source min4/8/16/32/64 SingleLine fulls are complete; matched DreamCoder fixed controls remain pending. Frozen test remains sealed with `test_evaluation_count=0`.

CAL SingleLine completion, 2026-07-31 UTC: **official-source CAL, initial length 32, on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset** completed `838/838` with zero failure/error. Row Pass@1 is `52.3866%`; equal-weight 143-cluster macro is `41.2711%`, 10,000-bootstrap 95% CI `[36.3798%,46.0499%]`; total forwards/token-forwards are `38730/10548607`. The identical-key official_fixed32 arm is still running, so no paired delta/help-harm is reported. Project Fixed64 sensitivity smoke began only after action commit `5bc4df8` was pushed and is not called equal-compute. Frozen test remains sealed with `test_evaluation_count=0`.

LR-DLLM SingleLine completion, 2026-07-31 UTC: the **paper-guided, author-unverified reimplementation of LR-DLLM** completed DreamCoder SingleLine `927/927` with zero failure/error. Row Pass@1 is `72.7077%`; equal-weight 148-cluster macro is `60.5238%`, 10,000-bootstrap 95% CI `[54.9216%,65.8574%]`; total forwards/token-forwards are `44952/11288608`. The identical-key Fixed64 arm is still running, so no paired delta/help-harm is reported. Its released slot triggered the queued DAEDAL MultiLine Fixed8 full at `2026-07-31T10:56:09Z`. Frozen test remains sealed with `test_evaluation_count=0`.

Concurrent external-baseline execution snapshot, 2026-07-31 UTC: CAL SingleLine primary and official_fixed32 have passed 12-case smoke plus resume-noop and both 838-row full runs are active. The **paper-guided, author-unverified reimplementation of LR-DLLM** passed its 12-case technical and 64-case mechanism gates; DreamCoder SingleLine/RandomSpan/MultiLine primary runs and the SingleLine Fixed64 control are active. **CAL authors’ DAEDAL FIM adaptation** has an active SingleLine dynamic run and a completed SingleLine Fixed8 control: row Pass@1 `50.8353%`, equal-weight 143-cluster macro `37.5859%`, 10,000-bootstrap 95% CI `[32.6317%,42.6749%]`, total forwards/token-forwards `3817/928151`. DAEDAL MultiLine dynamic is active and its Fixed8 full is resource-queued after successful smoke/resume gates. DreamOn checkpoint `8ccc7475…` is now cached under Apache-2.0, but no frozen local adapter/manifest exists yet. Frozen test remains sealed with `test_evaluation_count=0`; no partial accuracy was inspected.

Baseline execution checkpoint, 2026-07-31 UTC: CAL SingleLine 838 is technically ready but remains `0/12` because physical GPU0 is occupied by unrelated PIDs `755980/810890/818373`; no project GPU process or output was created. LR-DLLM has no located author code and is frozen as a **paper-guided, author-unverified reimplementation of LR-DLLM** with DreamCoder 927/1480/5079 manifests and CPU preflight complete, but no GPU outcome. **CAL authors’ DAEDAL FIM adaptation** has SingleLine/MultiLine adapters and same-decoder Fixed8 controls preflighted, also with no GPU outcome. Frozen test remains sealed with `test_evaluation_count=0`.

External baseline closure update, 2026-07-31 UTC: official-source CAL on the `4,990`-row / `143`-cluster project-non-frozen MultiLine CAL-Rest common subset is now unblinded after analyzer commit/push. Row Pass@1 is `1643/4990 = 32.9259%`; equal-weight task-macro is `27.8471%` with 10,000 cluster-bootstrap 95% CI `[24.3313%,31.2715%]`. Total forwards/token-forwards are `233,648/63,545,173`, mean wall is `3.8762s`, and peak H200 memory is `15.42 GiB`. No identical-key official_fixed32 result exists, so no paired delta/help-harm is claimed. Frozen test remains sealed with `test_evaluation_count=0`.

Canonical reconciliation, 2026-07-17 UTC: `docs/paper_agent/ccfa_master_roadmap.zh.md` and `method_portfolio.current.json` are the current route/status sources. M1/M2/M3 are formally reviewed and not promoted; no paper primary is selected. M1's fair full-vs-generic result is negative/flat with sparse cone activation, M2 has activated constraints but no reliable grouped advantage, and M3 is lower-token but lower-accuracy than uniform. Historical 5079 M1 remains safely paused/resumable. official CAL has passed its 12-case technical smoke and is running the corrected `4990` CAL-Rest non-frozen common full with integrity-only progress; partial CAL accuracy is not read. M4 remains independent and is supervisor-queued for a strict safe second GPU slot. The historical `40632/40632` bank remains complete/read-only, and frozen test remains sealed at count zero.

Phase 5 completed update, 2026-07-12 UTC: sole baseline `45bead22e3d21daa707be724cf2bdcbbf776592a`. H200 full bank passed integrity with `1332/1332` base rows over `148` tasks and `728/728` alpha auxiliary rows over `91` verified tasks; zero missing/duplicate/extra/error/frozen rows. F2 equivariance delta AUC was `0.0143`, 95% CI `[-0.0550,0.0829]`. The deterministic combined proxy reached cross-canvas within-task pairwise accuracy `0.6273` and positive selection nets, but failed the preregistered all-baseline grouped-bootstrap lower-bound condition. Conditional V0 verdict is `killed_corrected_within_task_gate_failed`; no supervised fallback or extra generation ran. Fresh verification passed `18` tests plus compile/artifact/schema/diff checks. Frozen test remains sealed with `test_evaluation_count=0`. Phase 5 decision: `iterate`.

Generated from existing local `results.jsonl` files. Raw outputs are not copied here.

Update: 2026-06-01 01:52 CST. This snapshot remains the compact evidence anchor for the current A6000 checkpoint; the later probe-curve audits are tracked separately in `docs/paper_agent/probe_curve_signal_audit.md`, `docs/paper_agent/probe_curve_signal_audit.json`, `docs/paper_agent/probe_curve_split_score_audit.md`, and `docs/paper_agent/probe_curve_split_score_audit.json`.

H200 migration note, updated 2026-07-07 UTC: `analysis_outputs/h200_bootstrap_20260705_103617/` records the new-server bootstrap audit. Bootstrap verdict is `host_h200_available_sandbox_gpu_hidden`: default sandbox GPU checks fail because `/dev/nvidia*` is hidden, but approved host/unsandboxed checks show H200 is healthy and `dllm_env` sees CUDA. GitHub SSH auth and remote branch freshness are verified. Tier 1 H200 core reruns are complete; compact audit `analysis_outputs/h200_repro_audit_20260707_tier1_v2/` reports `h200_material_outcome_drift` (Control `787`, Midcons `794`, Route2 `795`, V6 `796`, Local CAL `769`). H200 action-bank rebuild `analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/` covers train/calibration/validation only (`927` tasks, `4635` rows, no test rows), and Controller V1 replay `analysis_outputs/controller_validation_h200_20260707_v1_replay/` again selects zero intervention with validation `89/127`; comparison audit is `analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`. CPU-only material drift triage `analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/` is retained as a pre-acceptance audit artifact, writes no row-level test details, and keeps frozen test sealed with evaluation count `0`. Researcher decision `docs/paper_agent/h200_evidence_base_decision.zh.md` accepts H200 reruns as the new evidence base; A6000 results below are historical reference and must not be silently mixed with H200 evidence.

H200 Controller V2 note, updated 2026-07-07 UTC: feasibility audit `analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility/` gives `mixed_controller_failure`; validation has `17/127` recoverable rows and `67/127` harmable rows. Controller V2 validation `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/` gives `risk_certification_limited_test_sealed`. The only nonzero V2 signal is `ordinal_only + probe_trace_fused`: validation `90/127`, wins/losses `5/4`, interventions `41`, population harm upper95 `7.06%`; gate fails and frozen test remains sealed.

H200 Controller V3 note, updated 2026-07-08 UTC: candidate screen `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/` completes Family A/B/C on validation and writes `topk_policy_curves.csv`, `validation_predictions.csv`, `validation_action_selection.csv`, and `validation_summary.json`. Best pass-count Family A point is `probe_trace_fused, k=15`: `91/127`, wins/losses `3/1`, net `+2`, population harm upper95 `3.68%`, but it has one `<=8` bucket loss. The selected exploratory route is the conservative Family A `probe_only, k=5`: `90/127`, wins/losses `1/0`, population harm upper95 `2.33%`. Route decision is `weak_validation_signal_test_sealed`; frozen-test gate fails and frozen test remains sealed with evaluation count `0`.

Historical Phase 4 feasibility note, updated 2026-07-08 UTC: second-backbone feasibility audit `analysis_outputs/second_backbone_feasibility_20260708_phase4_v4/` recommended `Dream-org/Dream-Coder-v0-Base-7B`. The then-missing local MultiLine/RandomSpan JSONL blocker was subsequently resolved and is superseded by the completed official first-pass/full-allowed runs described in the canonical reconciliation above. The synthetic 18/18 run remains unblock/sanity evidence only. LR-DLLM final attempt `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/` gives `blocked_missing_algorithmic_detail`.

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

## Execution Sprint V1 evidence anchor

- P2.1 remains complete with 148 task-group cluster inference; frozen test count remains `0`.
- Historical Phase6 candidate bank is complete (`40632/40632`) and its fixed score-only precursor analysis covers `5079` spans / `148` groups. It finds ordinary confidence materially below fixed64 (macro delta `-0.0620`, 95% CI `[-0.0944,-0.0285]`); Phase5 combined deterministic proxy is positive (`+0.0531`, CI `[+0.0148,+0.0957]`); fixed abductive score-only is weak (`+0.0200`, CI crosses zero). These are historical selector observations, not M1 full outcomes.
- 2026-07-16 cost-accounting correction regenerated a separate immutable compact report at `analysis_outputs/phase6_score_only_candidate_selection_20260716_cost_accounting_v2/`. Accuracy fields are byte-for-byte unchanged versus the 2026-07-15 report; only cost accounting changed: fixed64 is `64` forwards / `4096` token-forwards with its own wall time, while every eight-candidate selector is `512` forwards / `30720` token-forwards with the sum of all eight candidate wall times. Generic/M1 full remains contractually `576` forwards in the independent M1 analyzer.
- official CAL corrected provenance is pinned to CAL `741e8418` and HumanEval-Infilling `88062ff`; 4990 CAL-Rest non-frozen common population is frozen in the audit artifact. A documented one-line runtime evaluator enablement overlay leaves the pinned checkout unchanged. Smoke=`12/12` integrity pass and full is running; no partial accuracy is exposed in this snapshot.
