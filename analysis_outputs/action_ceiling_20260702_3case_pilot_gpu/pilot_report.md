# True-Long Action-Ceiling 3-Case Pilot

verdict: `positive_control_only`

## Execution Sanity

- branch: `codex/risk-controlled-dynamic-rescue`
- commit: `a9ba687e4158bee93799ab76f61f9c1be782454b`
- command: `CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 /home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/action_ceiling_matrix.py --timestamp 20260702_3case_pilot_gpu --task-ids-csv SingleLineInfilling/HumanEval/116/L0,SingleLineInfilling/HumanEval/85/L0,SingleLineInfilling/HumanEval/113/L3 --max-pilot-cases 3 --execute-pilot`
- output_dir: `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu`
- task_ids: `['SingleLineInfilling/HumanEval/116/L0', 'SingleLineInfilling/HumanEval/85/L0', 'SingleLineInfilling/HumanEval/113/L3']`
- experimental_seeds: `[0]`
- row_count: `12`

## Determinism Result

- status: `deterministic`
- check: `{'task_id': 'SingleLineInfilling/HumanEval/116/L0', 'action_id': 'A_primary', 'seed': 1070938084, 'first_generated_text_sha256': '1a26b0bfb1ab1ffa166dd00d968cc8533bb72db54c8ea09fa5eb6accee35f43d', 'second_generated_text_sha256': '1a26b0bfb1ab1ffa166dd00d968cc8533bb72db54c8ea09fa5eb6accee35f43d', 'first_passed': False, 'second_passed': False, 'deterministic': True}`

## Historical Replay Comparison

- replay_mismatches: `[]`

## Per-Task A/B/C/D Table

| Task | Pool | Seed | A primary | B Route2 | C oracle canvas | D oracle canvas steps96 |
|---|---|---:|---|---|---|---|
| `SingleLineInfilling/HumanEval/116/L0` | `positive_control_rescued` | 0 | FAIL | PASS | PASS | PASS |
| `SingleLineInfilling/HumanEval/85/L0` | `triggered_failed_long` | 0 | FAIL | FAIL | FAIL | FAIL |
| `SingleLineInfilling/HumanEval/113/L3` | `missed_failed_long` | 0 | FAIL | FAIL | FAIL | FAIL |

## Candidate Existence

- ceiling_candidates: `[{'task_id': 'SingleLineInfilling/HumanEval/116/L0', 'case_pool': 'positive_control_rescued', 'action_id': 'C_oracle_sufficient', 'experimental_seed': 0}, {'task_id': 'SingleLineInfilling/HumanEval/116/L0', 'case_pool': 'positive_control_rescued', 'action_id': 'D_oracle_sufficient_steps96', 'experimental_seed': 0}]`

## Canvas Effect

- canvas_effects: `[]`

## Steps96 Effect

- steps96_effects: `[]`

## Trigger Opportunity

- trigger_opportunities: `[]`

## Cost

- cost: `{'sum_total_sec_including_probe': 55.4909551065648, 'avg_total_sec_including_probe': 4.6242462588804, 'p95_total_sec_including_probe': 6.723802521824837}`

## Negative Results

- no C/D pass cases: `[{'task_id': 'SingleLineInfilling/HumanEval/85/L0', 'experimental_seed': 0}, {'task_id': 'SingleLineInfilling/HumanEval/113/L3', 'experimental_seed': 0}]`

## Next-Step Verdict

- verdict: `positive_control_only`
- stop_rule: do not expand beyond this 3-case pilot without researcher review.
