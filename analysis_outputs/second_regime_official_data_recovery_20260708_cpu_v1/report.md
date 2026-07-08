# Official Second-Regime Data Recovery Report

Verdict: `official_second_regime_data_recovered_cpu_only`.

Source dataset: `loubnabnl/humaneval_infilling`.
GPU status: `not_run`.
Frozen controller test: `sealed_not_touched`.

## Export Summary

| Config | Status | Rows | Export path | SHA256 |
|---|---|---:|---|---|
| `HumanEval-MultiLineInfilling` | `exported_official_jsonl` | 5815 | `/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl` | `3a1cb133597ff699fde9dba92788ef8dabf46f273f44477195b36639bd908306` |
| `HumanEval-RandomSpanInfilling` | `exported_official_jsonl` | 1640 | `/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfilling.jsonl` | `be10c57f855885910dd8baa21389b209001f386ac478912fa6ece9f0446b32fb` |
| `HumanEval-RandomSpanInfillingLight` | `exported_official_jsonl` | 164 | `/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl` | `3239733a4bf3a907b9dd47a0caf1a6497354603d9595a194bca433580270c73f` |

## Evaluator Smoke

First 3 tasks per recovered config were reconstructed from prefix + reference middle + suffix and checked with the existing verifier stack.

| Config | Smoke rows | All tier3 passed? |
|---|---:|---|
| `HumanEval-MultiLineInfilling` | 3 | `True` |
| `HumanEval-RandomSpanInfilling` | 3 | `True` |
| `HumanEval-RandomSpanInfillingLight` | 3 | `True` |

## Gate Decision

Official data recovery succeeded at CPU/schema level if all three configs are exported and evaluator smoke passes. No GPU diagnostic was run. The next step is to construct a labeled official second-regime manifest that records source config, task id, length bucket, and excludes controller frozen-test rows before any GPU execution.

## Compact Outputs

- `summary.json`
- `official_dataset_exports.csv`
- `evaluator_smoke.csv`
