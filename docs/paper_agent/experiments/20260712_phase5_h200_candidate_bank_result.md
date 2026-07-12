# Phase 5 H200 Candidate Bank And Premise Falsification Result

Timestamp: 2026-07-12 UTC

Baseline: `45bead22e3d21daa707be724cf2bdcbbf776592a`

## Execution

Exact launch command:

```bash
bash scripts/manual_launch_phase5_h200_candidate_bank.sh
```

Environment: H200, `CUDA_VISIBLE_DEVICES=0`, `/home/shx/miniconda3/envs/dllm_env/bin/python`, `GSAI-ML/LLaDA-8B-Base`, `64` denoising steps. Log: `logs/paper_agent/20260712_phase5_randomspanlight_candidate_bank.log`. Raw local output: `outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1/`. Compact output: `analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1/`.

The process was verified through its PID, H200 memory/utilization, log growth, and candidate JSONL growth. Existing external GPU processes were not interrupted. Individual-row continuation policy was active; no row failed.

## Candidate Bank

- Allowed non-frozen tasks: `148`.
- Base rows: `1332/1332`; `0` missing, duplicate, extra, error, frozen, or malformed rows.
- Deployable grid rows: `1184` (`16/32/64/128 × seeds 0/1`).
- Oracle-sufficient diagnostic-ceiling rows: `148`.
- F2 alpha auxiliary: `728/728` over `91` reference-verified tasks; `0` missing, duplicate, extra, or error rows.
- Full deployable-grid passes: `330/1184`; diagnostic-ceiling passes: `73/148`.
- Frozen test: `sealed`; `test_evaluation_count=0`.

## F1–F4

- F1: mean unique candidate hashes `4.0`; mean unique parsable AST hashes `3.3243`; `61` all-fail tasks, of which `6` contain complementary correct semantic units. This is diagnostic-only evidence.
- F2: equivariance-augmented AUC `0.5174` versus controlled baseline `0.5031`; delta `0.0143`, grouped-bootstrap 95% CI `[-0.0550, 0.0829]`. Verdict: `f2_equivariance_not_independently_predictive`.
- F3/F4 deterministic combined proxy: cross-canvas within-task pairwise accuracy `0.6273`; global AUC `0.5844` is secondary only. Selection versus fixed64 is `23` wins / `8` losses / net `+15`, with short net `+5`; versus confidence it is `27` wins / `13` losses / net `+14`, with short net `+3`.
- The combined proxy did not meet the preregistered requirement that every grouped-bootstrap primary delta lower bound exceed zero. Delta lower bounds were negative for prefix-only (`-0.0735`), token/canvas (`-0.0038`), and ordinary confidence (`-0.0516`); only suffix-only was positive (`0.0053`).

## Conditional V0 Decision

`AST/def-use bridge proxy V0` was killed by the corrected gate: `killed_corrected_within_task_gate_failed`. The failed condition was `positive_primary_delta_vs_all_baselines`. No deployable reranker, supervised-score fallback, new heuristic, method fusion, or extra generation was run. Pass-trained supervised probes remained diagnostic-only and were not supplied to selection.

Phase 5 decision: `iterate`.

The evidence supports continued investigation of program-structure signals, but not promotion of the current fixed proxy formula. A future iteration must be a separately preregistered mechanism with new independent structure evidence; it may not tune the current formula on these outcomes or open the frozen test.

## Verification

- Focused tests: `18` passed.
- `py_compile`: candidate-bank, premise-falsification, and V0 scripts passed.
- JSON/CSV parsing and row-count assertions passed.
- Forbidden-feature audit passed; supervised scores have no deployable role.
- Compact raw-schema scan found no generated/reference code fields.
- `git diff --check` passed.
- Frozen test remained sealed with `test_evaluation_count=0`.
