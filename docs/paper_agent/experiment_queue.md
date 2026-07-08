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
Status: completed_cpu_data_recovery_gate
Hypothesis: A useful second-regime diagnostic requires a manifest with genuine control/deployable failures; the current synthetic minimal set is too easy and should not be promoted.
Inputs:
- `analysis_outputs/second_regime_feasibility_20260708_phase4_v4/compatibility_report.md`
- `analysis_outputs/second_regime_unblock_20260708_phase4_continue_v3/second_regime_manifest.csv`
- `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/summary.json`
- Candidate official files, if recovered: `data/HumanEval-MultiLineInfilling.jsonl`, `data/HumanEval-RandomSpanInfilling.jsonl`, `data/HumanEval-RandomSpanInfillingLight.jsonl`
Commands:
```bash
# Gate A: verify whether official second-regime JSONL files now exist.
ls data/HumanEval-MultiLineInfilling.jsonl data/HumanEval-RandomSpanInfilling.jsonl data/HumanEval-RandomSpanInfillingLight.jsonl

# Gate B: if official files remain missing, do not claim benchmark status.
# Build only a clearly labeled stress manifest from non-test existing evidence, then run a bounded diagnostic.
# A dedicated stress manifest script/report should be added before GPU execution if web/user approve this route.
```
Outputs:
- Official route: a compact official second-regime diagnostic report under `analysis_outputs/second_regime_diagnostic_*`
- Synthetic stress route: a clearly labeled stress manifest and report under `analysis_outputs/second_regime_stress_*`
- CPU data recovery gate: `analysis_outputs/second_regime_official_data_recovery_20260708_cpu_v1/report.md`
Success criterion: The selected manifest has nontrivial failures under control/deployable policy and a measurable oracle-canvas recoverability or rescue-limited fraction.
Failure criterion: Official files remain missing and synthetic stress is all-pass, too small, or too artificial; the paper must keep second-regime as unresolved.
Stop condition: Do not run a GPU diagnostic until the manifest has source labels, no frozen-test rows, and expected nontrivial failure diversity. Stop immediately if the only available manifest is the 18/18 easy synthetic set.
Expected report path: `analysis_outputs/second_regime_stress_<timestamp>/report.md` or `analysis_outputs/second_regime_diagnostic_<timestamp>/report.md`
Result: CPU-only official data recovery succeeded on 2026-07-08. `HumanEval-MultiLineInfilling` (`5815` rows), `HumanEval-RandomSpanInfilling` (`1640` rows), and `HumanEval-RandomSpanInfillingLight` (`164` rows) were exported locally from `loubnabnl/humaneval_infilling`; first 3 tasks per config passed schema/evaluator smoke. GPU remains `not_run`; next gate is a labeled official second-regime manifest before execution.
