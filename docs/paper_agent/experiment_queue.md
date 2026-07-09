# Experiment Queue

Updated: 2026-07-08 UTC

Only selected, executable experiments belong here. Frozen test remains sealed unless a validation gate explicitly passes.

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
Status: completed_full104_supplement_done
Hypothesis: A useful second-regime diagnostic requires a source-labeled official manifest with genuine control/deployable failures; if oracle-sufficient canvas recovers a nonzero subset, bounded hard-tail diagnostics can complete taxonomy/robustness without becoming an unbiased benchmark aggregate or positive deployable controller claim.
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
Success criterion: The selected manifest has nontrivial failures under control/deployable policy and a measurable oracle-canvas recoverability or rescue-limited fraction.
Failure criterion: Official files remain missing and synthetic stress is all-pass, too small, or too artificial; the paper must keep second-regime as unresolved.
Stop condition: Stop after the reviewer-requested fixed full104 hard-tail stress/taxonomy diagnostic unless a concrete bug or clearly preregistered follow-up appears. Do not run synthetic stress, add policies, open frozen test, or start Controller V4.
Expected report path: `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/report.md`
Result: Completed on 2026-07-09. Manifest gate produced `120` official cases, `40` per config and `10` per bucket, with frozen-controller-test rows `0` and evaluator smoke `12/12`. First-pass GPU results: control fixed64 `34/120`, best deployable cal-lite `36/120`, oracle-sufficient canvas `49/120`, oracle gain vs control `26`, deployable harm vs control `15`, oracle harm vs control `11`. The approved 48-case hard-tail diagnostic used balanced sample groups (`12/12/12/12`), source configs `16/16/16`, buckets `12/12/12/12`, frozen rows `0`, and did not alter first-pass labels. 48-case result: control `12/48`, deployable `16/48`, oracle `28/48`, genuine canvas-recoverable `24`, rescue/non-canvas `12`, deployable help `13`, deployable harm `9`, oracle harm vs control `8`, first-pass label changes `0`. Reviewer-requested full104 fixed hard-tail stress result: control `18/104`, deployable `20/104`, oracle `33/104`, genuine canvas-recoverable `26`, rescue/non-canvas `60`, deployable help `17`, deployable harm `15`, oracle harm vs control `11`, first-pass label changes `0`. Interpretation: official second-regime is mixed stress evidence, not a positive deployable controller claim; the 120-case first pass remains the unbiased official diagnostic estimate, while full104 is supplemental failure taxonomy/robustness evidence. Second-regime GPU work is stopped.

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
