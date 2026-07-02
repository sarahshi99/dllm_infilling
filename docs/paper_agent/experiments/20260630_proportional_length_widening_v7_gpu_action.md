# Proportional Length Widening V7 GPU Action

Date: 2026-06-30 CST

## Action Name

Expanded-grid GPU smoke and full run for proportional length widening.

## Stage And Workflow

This follows the project Superpowers-style implementation path: brainstorming and CPU audit are complete; this action is the writing-plans / executing-plans GPU stage. Subagents remain disabled, and execution is serial: smoke first, then full on the same GPU if smoke exits cleanly.

## Reviewer Motivation

The CPU replay rejected the existing compact-grid version, but did not reject the user's actual idea because the stored grid only reached length `24`. A reviewer-relevant test needs real generation with an expanded base probe grid and the proportional rule active.

## Hypothesis

When the best base probe length is moderately long, selecting the longest near-best candidate by a proportional score threshold can reduce long under-selection while preserving the current training-free, inference-time framing.

## Baseline / Dataset / Model / Metric

- Model: `GSAI-ML/LLaDA-8B-Base`.
- Dataset: HumanEval single-line infilling.
- Baseline comparison: current `midcons` full trace, `795/1033 = 76.96%`:
  `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Primary metric: pass count / pass rate.
- Secondary metrics: proportional promoted count, true-long promoted count, short promoted count, pairwise wins/losses vs baseline, oracle bucket changes.
- Comparison type: same-backbone follow-up GPU experiment, not external SOTA.

## GPU / Environment

- GPU: `CUDA_VISIBLE_DEVICES=2` only.
- User explicitly approved running on GPU2 even while other users have processes there, as long as we do not kill or disturb them.
- HF mirror/cache:
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `HF_HUB_DISABLE_XET=1`
  - `HF_HOME=/home/shx/.cache/huggingface`

## Smoke Command

Targeted smoke uses seven known current-midcons failed true-long rows:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --model-path GSAI-ML/LLaDA-8B-Base \
  --base-probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,40,48 \
  --base-alpha 0.06 \
  --proportional-widening \
  --prop-base-threshold 0.985 \
  --prop-slope 0.02 \
  --prop-min-threshold 0.85 \
  --prop-min-base-length 12 \
  --prop-max-expansion 3.0 \
  --official-eval-max-s3-len 12 \
  --repair-max-s3-len 5 \
  --repair-min-official-len 6 \
  --repair-max-official-len 9 \
  --repair-min-delta 1 \
  --repair-max-delta 8 \
  --suspicion-max-s3-len 5 \
  --suspicion-min-official-len 16 \
  --suspicion-max-official-len 64 \
  --suspicion-min-delta 1 \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --task-ids-csv SingleLineInfilling/HumanEval/11/L6,SingleLineInfilling/HumanEval/32/L1,SingleLineInfilling/HumanEval/39/L5,SingleLineInfilling/HumanEval/50/L0,SingleLineInfilling/HumanEval/69/L0,SingleLineInfilling/HumanEval/71/L0,SingleLineInfilling/HumanEval/80/L5 \
  --experiment-name smoke_v7_prop_widen_expgrid_gpu2
```

Smoke log:

```text
logs/paper_agent/20260630_v7_prop_widen_smoke_gpu2.log
```

## Full Command

If smoke exits successfully and does not hit OOM or script-level corruption, run the same policy on the full dataset:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --model-path GSAI-ML/LLaDA-8B-Base \
  --base-probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,40,48 \
  --base-alpha 0.06 \
  --proportional-widening \
  --prop-base-threshold 0.985 \
  --prop-slope 0.02 \
  --prop-min-threshold 0.85 \
  --prop-min-base-length 12 \
  --prop-max-expansion 3.0 \
  --official-eval-max-s3-len 12 \
  --repair-max-s3-len 5 \
  --repair-min-official-len 6 \
  --repair-max-official-len 9 \
  --repair-min-delta 1 \
  --repair-max-delta 8 \
  --suspicion-max-s3-len 5 \
  --suspicion-min-official-len 16 \
  --suspicion-max-official-len 64 \
  --suspicion-min-delta 1 \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --experiment-name full_v7_prop_widen_expgrid_gpu2
```

Full log:

```text
logs/paper_agent/20260630_v7_prop_widen_full_gpu2.log
```

## Success Criteria

- Smoke writes a valid run directory, `results.jsonl`, and `summary.json`.
- Full writes `1033` rows and a valid `summary.json`.
- Report pass rate, promoted count, true-long promoted count, short promoted count, and pairwise wins/losses vs current midcons.
- If full pass count exceeds `795`, update `experiment_results.zh.md` as follow-up evidence.

## Kill Criteria

- Stop before full if smoke exits nonzero, OOMs, cannot load model/data, or writes invalid JSON.
- Stop and document if GPU2 memory pressure causes immediate allocation failure.
- Do not kill or interrupt other users' processes.

## Known Risks

- The expanded grid increases probe cost, so runtime may be slower than current midcons.
- Widening may promote short rows, causing losses despite improving some true-long rows.
- If most long failures are generation-quality failures, length widening alone may not help.

## Expected Documentation Outputs

- Update this action file with smoke/full status and output paths.
- Update `docs/paper_agent/current_action.md`.
- If full completes, update `docs/paper_agent/experiment_results.zh.md`, dashboard, and checkpoint with compact results.

## 2026-07-01 Status: First Full Run Failed Before Completion

The targeted smoke completed cleanly:

- output: `outputs_clean/smoke_v7_prop_widen_expgrid_gpu2_20260630_223710`
- rows: `7`
- pass: `0/7`
- proportional promoted rows: `0`
- exit: `0`

The first full run did not complete:

- output: `outputs_clean/full_v7_prop_widen_expgrid_gpu2_20260630_223900`
- log: `logs/paper_agent/20260630_v7_prop_widen_full_gpu2.log`
- exit: `1`
- completed rows before crash: `761/1033`
- partial pass: `584/761 = 76.74%`
- partial pairwise vs current midcons on common rows: `1` win, `2` losses
- proportional promoted rows in partial output: `6`

Crash reason:

```text
ValueError: Correction probe produced no usable candidates at or above max(base_len=48, strong_min_len=13)
```

Interpretation:

This was not GPU OOM and not another user's process interrupting the run. It was a runner boundary bug exposed by the expanded base grid: proportional/base selection can choose length `48`, while the existing strong correction grid only goes up to `40`. The fix is to skip correction when the correction grid is shorter than the already selected base length, keep the widened base selection, and record `correction_grid_too_short`.

Verification after fix:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_lcal_official_bounded_repair_proportional.py tests/test_proportional_length_widening_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_lcal_official_bounded_repair.py analysis/proportional_length_widening_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --help
git diff --check
```

Result: all passed (`10` tests).

Next action:

Run a 1-task guard smoke on `SingleLineInfilling/HumanEval/127/L10`, then restart a clean full run under a new experiment name if the guard smoke exits cleanly.

## 2026-07-01 Final Result: Clean Full Run Completed, Negative

After the grid-boundary fix, the one-task guard smoke exited cleanly and the clean full run completed normally.

Guard smoke:

- output: `outputs_clean/guard_v7_prop_widen_gridfix_gpu2_20260701_001308`
- task: `SingleLineInfilling/HumanEval/127/L10`
- result: `0/1`
- proportional/base selection: `16 -> 48`
- oracle length: `16`
- note: the guard verified the runner fix, but also exposed over-widening risk on a baseline-pass row.

Clean full run:

- output: `outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`
- log: `logs/paper_agent/20260701_v7_prop_widen_full_gridfix_gpu2.log`
- rows: `1033`
- pass: `792/1033 = 76.67%`
- current `midcons` baseline: `795/1033 = 76.96%`
- pairwise vs current `midcons`: `1` win, `4` losses, `791` tie-pass, `237` tie-fail

Key diagnostics:

| Metric | Value |
|---|---:|
| proportional promoted rows | `10/1033` |
| promoted true-long rows | `0` |
| promoted short rows | `1` |
| promoted current-pass rows | `8` |
| avg promoted delta length | `6.2` |
| avg original base abs error on promoted rows | `1.1` |
| avg new base abs error on promoted rows | `6.9` |
| correction grid too short | `1` |
| long-bucket wins vs `midcons` | `0` |
| short-bucket losses vs `midcons` | `2` |

Interpretation:

This V7 setting does not support continuing proportional widening as the current main policy. It did not promote any oracle `>=17` true-long rows, did not produce any long-bucket wins, and made the promoted-row length estimate substantially worse on average. The overall result is `-3` tasks versus current `midcons` and `-10` tasks versus the best current LLaDA-Base follow-up, V6 short override `802/1033 = 77.64%`.

The idea is not permanently ruled out, but the tested form is too permissive: widening by near-best proportional score with `max_expansion=3.0` mostly affects short/medium rows and can overshoot badly. Any future proportional variant should be redesigned CPU-first with strict caps and guard conditions before another GPU full run.
