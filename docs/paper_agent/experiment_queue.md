# Experiment Queue

Updated: 2026-07-11 UTC

Only selected, executable experiments belong here. Frozen test remains sealed unless a validation gate explicitly passes.

## EXP-010: DreamOn V2-Hard-v2 boundary decoder diagnostic

Status: stopped_pilot_gate_failed

Decision: `iterate`.

Hypothesis: Correct online interpretation of newline proposals, region-local EOS broadcast deletion, and exact deterministic-cycle stopping removes the catastrophic termination/compile failure caused by V2-Hard-v1 decoder defects.

Inputs: frozen 642-row / 115-base-problem exact-three-line development/mechanism population; DreamOn-v0-7B snapshot `8ccc74750e43177327f29dab9e91882ba759e194`; protocol version 2.

Command: `bash repro_scripts/run_dreamon_progressive_v2_hard_v2.sh {verify|smoke|pilot|full}`.

Smoke gate: 5/5 completed; zero unresolved/runtime/protocol/cycle/invariant/cross-region-delete rows; sequential activation and direct extraction.

Pilot gate: 30/30 completed, compile at least 24/30, Pass@1 at least 10/30, zero errors/cycles, complete oracle discard/reference diagnostic.

Full early stop: at least 5 non-completed rows in the first 50, cumulative non-completed rate above 5% after 100, or any runtime/invariant damage.

Stop condition: stop after V2-Hard-v2 or any failed gate. Do not run V2-OpenTail-v2 or Joint-OpenTail-v2.

Expected outputs: `repro_results/dreamon_progressive_v2_hard_v2_{smoke5,pilot30,all642}/` and a protocol-v2 experiment report.

Result: verification passed and Smoke 5 passed. Pilot 30 produced 25 completed rows, 5 exact cycles, Pass@1 11/30, compile 19/30, exact 5/30. The gate failed completion/cycle/unresolved/protocol/compile checks, so full was not started. Decision: `stop_before_full`.

## EXP-007: Phase 5 full RandomSpanLight shared candidate bank

Linked idea: IDEA-011, IDEA-015
Status: blocked_not_started
Hypothesis: A full-first shared bank can falsify candidate-diversity, equivariance, semantic-bridge, and ranking premises without method-specific generation distributions.
Inputs: all `148` allowed non-frozen `HumanEval-RandomSpanInfillingLight` rows; LLaDA-8B-Base; canvas `16/32/64/128`; seeds `0/1`; fixed64 `(64,0)`; one diagnostic oracle ceiling.
Smoke/full rows: exactly `108` then `1332`; alpha-renaming mirrors are a separate auxiliary block after the full gate.
Command: exact approved H200 command in `docs/paper_agent/current_action.md`.
Success criterion: canvas-128 same-protocol validation, schema/evaluator/resume/row-count/duplicate audits pass, zero frozen rows, unchanged test lock; smoke automatically continues full.
Failure criterion: any substitution for 128, missing/duplicate/error row, frozen intersection, evaluator/schema failure, raw code in compact outputs, or changed test lock.
Current blocker: approval service rejected launch before process creation with `422 model not found: codex-auto-review`.
Expected report: `analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1/report.md`

## EXP-008: Phase 5 F1–F4 premise falsification

Linked idea: IDEA-011, IDEA-014, IDEA-015
Status: blocked_on_exp007
Hypothesis: One or more inference-visible premises add stable grouped ranking signal; failure is a valid kill result.
Inputs: EXP-007 full bank only.
Outputs: `analysis_outputs/phase5_premise_falsification_20260711_v1/`.
Success/failure: F1/F2/F4 always report. F3A supervised probes are diagnostic only. F3B passes only under the corrected deterministic within-task/cross-canvas, paired-selection, and short-safety gate.
Stop: no deployable method implementation if F3 fails.

## EXP-009: AST/def-use bridge proxy V0 conditional reranker

Linked idea: IDEA-011
Status: blocked_on_f3_gate
Hypothesis: If the corrected gate passes, a standalone deterministic AST/def-use bridge proxy improves Pass@1 relative to fixed64 and confidence reranking on the same eight candidates.
Inputs: EXP-007 full bank and EXP-008 grouped OOF combined scores.
Outputs: `analysis_outputs/phase5_semantic_bridge_v0_20260711_v1/`.
Kill: if the corrected gate fails, write `killed_corrected_within_task_gate_failed`; no supervised-score fallback, extra generation, homotopy, birth–death, particle assembly, or fusion.

## EXP-001: CPU claim-boundary consolidation for diagnostic mixed paper

Linked idea: IDEA-004, IDEA-010
Status: completed
Hypothesis: Existing compact artifacts already support a bounded diagnostic claim, but only if Dream-Coder, second-regime, controller, and LLaDA attribution evidence are separated by strength and status.
Inputs:
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/combined_results.csv`
- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/summary.json`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/topk_policy_curves.csv`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/validation_action_selection.csv`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/validation_summary.json`
- `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/results.csv`
Commands:
```bash
python - <<'PY'
# CPU-only consolidation executed by Codex on 2026-07-08.
# It reads the inputs above and writes:
# analysis_outputs/research_planning_20260708_cpu_claim_audit/{summary.json,report.md,dreamcoder_case_taxonomy.csv}
PY
```
Outputs:
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/report.md`
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/summary.json`
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/dreamcoder_case_taxonomy.csv`
Success criterion: Report cleanly separates strong LLaDA attribution evidence, mixed Dream-Coder evidence, weak/negative controller evidence, and insufficient synthetic second-regime evidence without opening frozen test.
Failure criterion: Any output mixes historical A6000 with H200 evidence, treats synthetic second-regime as official benchmark, or uses frozen-test rows/results.
Stop condition: Stop after compact report and taxonomy are written; no GPU and no raw generated code.
Expected report path: `analysis_outputs/research_planning_20260708_cpu_claim_audit/report.md`

## EXP-002: Dream-Coder expanded oracle-sufficient diagnostic on bounded non-test manifest

Linked idea: IDEA-002, IDEA-005
Status: completed
Hypothesis: Dream-Coder's 15-case mixed result will remain oracle-canvas recoverable on a larger non-test manifest, but its missed-vs-triggered split may differ from LLaDA H200.
Inputs:
- `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/case_manifest.csv`
- `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/summary.json`
- Existing Dream-Coder runner in `experiments/phase4_continuation.py`
- Local cached model `Dream-org/Dream-Coder-v0-Base-7B`
Commands:
```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase4_continuation.py \
  --mode dream-oracle \
  --timestamp 20260708_dreamcoder_expanded37_v1 \
  --manifest analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/case_manifest.csv
```
Outputs:
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/summary.json`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/combined_results.csv`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/stratum_summary.csv`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/stratum_taxonomy.csv`
Success criterion: Expanded manifest yields nontrivial oracle-canvas recovery, no uncontrolled short-case regression, and a case taxonomy that clarifies whether Dream-Coder's triggered/missed behavior is model-dependent.
Failure criterion: Oracle-canvas recovery disappears, report is all-pass/all-fail without diagnostic separation, or output cannot be produced from cached model.
Stop condition: Stop after one bounded 37-case oracle-sufficient run; do not add E/F/G actions because Dream-Coder has no trace-remasking adapter; do not use frozen test.
Expected report path: `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md`
Result: Completed on 2026-07-08. Primary/control `11/37`, best simple `11/37`, oracle-sufficient canvas `26/37`. Stratum oracle pass: missed_failed_long `3/12`, triggered_failed_long `4/4`, medium_near_long_underselection `6/8`, short_primary_pass_harmable `6/6`, positive_control_recoverable `7/7`. Interpretation remains second-backbone diagnostic evidence, not model-agnostic confirmation.

## EXP-003: Second-regime stress gate before any benchmark claim

Linked idea: IDEA-001, IDEA-009
Status: completed_full_allowed_done
Hypothesis: A useful second-regime diagnostic requires source-labeled official data with genuine control/deployable failures; if oracle-sufficient canvas recovers a nonzero subset, full allowed official population plus fixed hard-tail diagnostics can support mixed-stress taxonomy without becoming a positive deployable controller claim.
Inputs:
- `analysis_outputs/second_regime_feasibility_20260708_phase4_v4/compatibility_report.md`
- `data/HumanEval-MultiLineInfilling.jsonl`
- `data/HumanEval-RandomSpanInfilling.jsonl`
- `data/HumanEval-RandomSpanInfillingLight.jsonl`
- `analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- `analysis_outputs/grouped_split_20260702_accel2/test_tasks.json`
Commands:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/official_second_regime_diagnostic.py --mode manifest

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/official_second_regime_diagnostic.py --mode diagnostic
```
Outputs:
- CPU manifest gate: `analysis_outputs/second_regime_official_manifest_20260708_v1/report.md`
- Official bounded GPU diagnostic: `analysis_outputs/second_regime_official_diagnostic_20260708_v1/report.md`
- CPU hard-tail manifest: `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/report.md`
- Bounded 48-case hard-tail diagnostic: `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/report.md`
- Supplemental fixed full104 hard-tail stress diagnostic: `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/report.md`
- Full allowed official second-regime diagnostic: `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/report.md`
Success criterion: The selected manifest has nontrivial failures under control/deployable policy and a measurable oracle-canvas recoverability or rescue-limited fraction.
Failure criterion: Official files remain missing and synthetic stress is all-pass, too small, or too artificial; the paper must keep second-regime as unresolved.
Stop condition: Stop after the full allowed official second-regime diagnostic unless a concrete bug appears. Do not run synthetic stress, add policies, tune cal-lite, open frozen test, or start Controller V4.
Expected report path: `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/report.md`
Result: Completed on 2026-07-10. Manifest gate produced `120` official cases, `40` per config and `10` per bucket, with frozen-controller-test rows `0` and evaluator smoke `12/12`. First-pass GPU results: control fixed64 `34/120`, best deployable cal-lite `36/120`, oracle-sufficient canvas `49/120`, oracle gain vs control `26`, deployable harm vs control `15`, oracle harm vs control `11`. The approved 48-case hard-tail diagnostic used balanced sample groups (`12/12/12/12`), source configs `16/16/16`, buckets `12/12/12/12`, frozen rows `0`, and did not alter first-pass labels. 48-case result: control `12/48`, deployable `16/48`, oracle `28/48`, genuine canvas-recoverable `24`, rescue/non-canvas `12`, deployable help `13`, deployable harm `9`, oracle harm vs control `8`, first-pass label changes `0`. Reviewer-requested full104 fixed hard-tail stress result: control `18/104`, deployable `20/104`, oracle `33/104`, genuine canvas-recoverable `26`, rescue/non-canvas `60`, deployable help `17`, deployable harm `15`, oracle harm vs control `11`, first-pass label changes `0`. Full allowed official population result: `6707` rows, frozen rows `0`, control fixed64 `2019/6707`, best deployable cal-lite `1464/6707`, oracle-sufficient canvas `3180/6707`, oracle gain vs control `1633`, deployable help `620`, deployable harm `1175`, oracle harm vs control `472`, rescue/non-canvas-limited `3055`. Interpretation: official second-regime is mixed stress evidence, not a positive deployable controller claim; the full allowed diagnostic strengthens the mixed diagnostic claim, the 120-case first pass remains the preregistered/unbiased diagnostic estimate, and full104 remains supplemental failure taxonomy/robustness evidence. Second-regime GPU work is stopped.

## EXP-004: Controller V1/V2/V3 route-closure consolidation

Linked idea: IDEA-004, IDEA-010
Status: completed
Hypothesis: The controller route can be closed as validation-only weak/negative evidence because oracle action-bank headroom does not translate into a deployable policy that passes harm/frozen-test gates.
Inputs:
- `analysis_outputs/controller_validation_h200_20260707_v1_replay/report.md`
- `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/report.md`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/report.md`
Commands:
```bash
# CPU-only consolidation generated by Codex on 2026-07-08.
```
Outputs:
- `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`
- `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.csv`
- `analysis_outputs/controller_route_closure_20260708_v1/summary.json`
Success criterion: One table records oracle upper bound, selected policy, interventions, validation wins/losses, validation pass, harm upper95, and frozen-test decision for V1/V2/V3.
Failure criterion: Any row implies frozen-test authorization or positive deployable controller evidence.
Stop condition: Do not add Controller V4 unless a new evidence source appears outside repeated validation tuning.
Expected report path: `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`
Result: Completed. V1 selects zero intervention; V2 best nonzero has `5/4` wins/losses and harm upper95 `7.06%`; V3 exploratory top-k has `1/0` wins/losses but fails frozen-test gate. Frozen test remains sealed with evaluation count `0`.

## EXP-005: Paper evidence consolidation for diagnostic mixed paper

Linked idea: IDEA-010
Status: completed
Hypothesis: The completed evidence base can support a CCF-A diagnostic mixed paper if every table row explicitly separates support, limits, allowed claims, forbidden claims, and paper placement.
Inputs:
- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/summary.json`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/summary.json`
- `analysis_outputs/second_regime_official_diagnostic_20260708_v1/summary.json`
- `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/summary.json`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/summary.json`
- `analysis_outputs/controller_route_closure_20260708_v1/summary.json`
- `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/summary.json`
Commands:
```bash
python experiments/paper_evidence_consolidation.py
```
Outputs:
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/claim_matrix.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/claim_matrix.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/main_results_table.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/main_results_table.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/official_second_regime_writeup.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/failure_taxonomy_table.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/failure_taxonomy_table.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/paper_claim_rewrite.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/summary.json`
Success criterion: The consolidation names the final central claim, three contributions, claim matrix, main results table, official second-regime write-up, failure taxonomy, and forbidden claims without opening frozen test or upgrading negative evidence into positive controller claims.
Failure criterion: Any artifact mixes H200 with historical A6000 evidence, treats hard-tail full104 as an unbiased benchmark, or claims deployable controller success.
Stop condition: Stop after compact Markdown/CSV/JSON outputs and web-review-ready handoff updates; no generation or GPU needed for this experiment.
Expected report path: `analysis_outputs/paper_evidence_consolidation_20260710_v1/paper_claim_rewrite.md`
Result: Completed on 2026-07-10. Produced `7` claim-matrix rows, `6` main-results rows, `8` failure-taxonomy rows, and a paper claim rewrite. Verdict: `paper_evidence_consolidation_completed`.

## EXP-006: Dream-Coder full allowed SingleLine optional diagnostic

Linked idea: IDEA-002, IDEA-005
Status: completed
Hypothesis: A full non-frozen Dream-Coder SingleLine diagnostic can reduce sampling noise from expanded37 while preserving frozen-test discipline, but it should remain second-backbone evidence rather than model-agnostic confirmation.
Inputs:
- Local allowed non-frozen `HumanEval-SingleLineInfilling` rows
- Existing Dream-Coder runner and cached `Dream-org/Dream-Coder-v0-Base-7B`
- Frozen-controller-test split lock for exclusion only
Commands:
```bash
# Optional full-run branch executed on 2026-07-10 with fixed policies:
# primary/control, best simple length policy, oracle-sufficient canvas.
```
Outputs:
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/report.md`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/summary.json`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/manifest.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/results.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/stratum_summary.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/taxonomy_summary.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/comparison_vs_expanded37.csv`
Success criterion: Manifest includes only non-frozen allowed SingleLine rows; all three fixed policies complete; results clarify whether oracle canvas recovery persists at full allowed scale.
Failure criterion: Any frozen-controller-test row appears, policies are tuned after results, E/F/G actions are added for Dream-Coder, or outputs require raw generated-code dumps.
Stop condition: Stop after one full allowed Dream-Coder SingleLine run; do not add Dream-Coder E/F/G actions or tune policy after results.
Expected report path: `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/report.md`
Result: Completed on 2026-07-10. Manifest has `927` cases, frozen rows `0`, and `2781` policy rows. Primary/control `735/927`, best simple length policy `744/927`, oracle-sufficient canvas `858/927`; oracle gain vs primary `137`, simple help/harm `25/16`, oracle harm vs primary `14`, rescue/non-canvas-limited `55`. Interpretation: optional full allowed second-backbone SingleLine diagnostic, not model-agnostic confirmation.

## EXP-007: DreamOn V3 cumulative budget and nonempty oracle

Status: completed. A Pilot30 stopped before Full at 14/30. B Pilot30 reached 18/30 and authorized B Full642; Full completed 642/642 with 280 Pass and 471 compile, but paired net is `-21` versus one-shot and `-36` versus historical V1. Decision `reframe`; no further DreamOn progressive experiment is queued.
