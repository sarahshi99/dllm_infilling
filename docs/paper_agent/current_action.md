# Current Paper-Agent Action

Timestamp: 2026-07-12 UTC

## Action Name

Phase 6 exploratory Abductive Program-State Bridge V1 on full allowed non-frozen HumanEval-MultiLineInfilling.

## Baseline And Prior Phase

- Starting commit: `52f07457a7974fafa69d59914bbc412a2d83e305`.
- Phase 5 is completed: full base bank `1332/1332`, alpha `728/728`, F1–F4 completed, V0 verdict `killed_corrected_within_task_gate_failed`, frozen test count `0`.
- The Phase 5 strict uncertainty gate was a paper-level promotion gate. It does not block exploratory Phase 6 implementation.
- Phase 5 fixed combined-proxy weights are frozen historical baselines and will not be adjusted.

## Method Scope

Implement standalone `Abductive Program-State Bridge V1`:

1. Extract suffix-derived backward obligations: variables that the middle must define, data dependencies that must be restored, control-flow requirements, and boundary requirements.
2. Compute the forward program state induced by prefix plus candidate middle.
3. Rank same-pool candidates by full-program parse/boundary compatibility, control-flow contradictions, unsatisfied obligations, def/use conflicts, undefined uses, and restored dependencies.
4. Use no reference code, unit tests, pass/fail labels, oracle length, fitted/supervised score, task/split identity, or frozen-test statistics for selection.
5. Do not fuse with homotopy, birth–death, particle assembly, controller, or cal-lite.

V1 uses a fixed lexicographic structural ranking key, not a fitted model or another hand-tuned weighted sum. Ordinary confidence is an independent baseline only.

## Dataset And Candidate Pool

- Dataset: `/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl`.
- Source rows: `5815`.
- Frozen exclusion: `16` HumanEval base-task groups, `736` rows.
- Allowed rows: `5079`.
- Model: `GSAI-ML/LLaDA-8B-Base`.
- Shared candidates per row: canvas `16,32,64,128` × seed `0,1`, `64` denoising steps.
- Expected full candidate rows: `40632`.
- Fixed64 is canvas `64`, seed `0`.
- No existing candidate bank matches dataset, model, prompt, canvas, steps, and seed exactly; Phase 6 generates a new bank.

## Technical Smoke And Auto-Full

Smoke is technical only:

- runner executes;
- evaluator and schema are valid;
- no frozen rows;
- append-only resume produces no duplicates;
- GPU memory is healthy.

There is no performance gate. If the technical smoke passes, the runner automatically continues the full `5079` rows. Individual candidate errors are recorded and the run continues; pause only for a real program error, resource failure, integrity violation, or changed test lock.

## Comparisons And Reporting

All methods select from the same eight-candidate pool:

- fixed64;
- confidence reranking;
- prefix-only Phase 5 proxy;
- Phase 5 fixed combined proxy;
- Abductive Program-State Bridge V1.

Report Pass@1, pairwise wins/losses/net, reference-length buckets for offline reporting only, help/harm, selection latency, candidate-generation latency, GPU decode cost, wall time, and peak CUDA memory. The result is exploratory/development evidence, not held-out SOTA.

## Exact Planned Commands

Tests and compile:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_phase6_abductive_bridge_v1.py tests/test_phase6_multiline_candidate_bank.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/phase6_abductive_bridge_v1.py experiments/phase6_multiline_candidate_bank.py
```

GPU smoke with automatic full continuation:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase6_multiline_candidate_bank.py run --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl --output-dir outputs_clean/phase6_multiline_candidate_bank_20260712_v1 --compact-dir analysis_outputs/phase6_multiline_candidate_bank_20260712_v1 --smoke-cases 8 --auto-full
```

Same-pool comparison:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/phase6_abductive_bridge_v1.py --bank-dir outputs_clean/phase6_multiline_candidate_bank_20260712_v1 --compact-bank-dir analysis_outputs/phase6_multiline_candidate_bank_20260712_v1 --output-dir analysis_outputs/phase6_abductive_bridge_v1_20260712_v1
```

## Paths And Safety

- tmux session: `phase6-multiline-bank`.
- log: `logs/paper_agent/20260712_phase6_multiline_candidate_bank.log`.
- raw local bank: `outputs_clean/phase6_multiline_candidate_bank_20260712_v1/`.
- compact bank: `analysis_outputs/phase6_multiline_candidate_bank_20260712_v1/`.
- comparison: `analysis_outputs/phase6_abductive_bridge_v1_20260712_v1/`.
- Frozen test remains `sealed`; `test_evaluation_count=0`.

## Expected Outcome And Stop Conditions

Expected outcome: a complete same-pool exploratory comparison that reveals whether explicit backward obligations plus forward program state improve selection on MultiLine.

Stop only for a real runner/program error, resource exhaustion, changed frozen lock, frozen-row intersection, non-resumable/duplicate output, evaluator/schema corruption, or unsafe raw-code publication. Do not stop because V1 underperforms a baseline; performance is measured after full completion.

Expected documentation outputs: Phase 6 result brief, checkpoint/dashboard/activity updates, compact bank audit, same-pool comparison report, and a post-full validation-plan decision.
