# LLaDA-Instruct Midcons Full Cross-Model Run

Timestamp: 2026-06-04 20:22 CST

## Action Name

LLaDA-Instruct `midcons` full cross-model run on GPUs 2,3.

## Motivation

This run checks whether the current LLaDA-Base `midcons` bounded-repair policy transfers to `GSAI-ML/LLaDA-8B-Instruct`. It is the first same-family cross-model validation step after recognizing that different base models must be compared against their own baselines.

## Hypothesis

If the `midcons` policy captures a general LLaDA confidence/length-control signal, it should preserve or improve the historical LLaDA-Instruct LCAS-v3 result (`817/1033 = 79.09%`) while keeping short-bucket regressions controlled.

## Baseline And Metrics

- Baseline: `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/results.jsonl`.
- Dataset: `HumanEval-SingleLineInfilling`, test split, `1033` tasks.
- Model: `GSAI-ML/LLaDA-8B-Instruct`.
- Metrics: total pass rate, pairwise wins/losses, oracle-length bucket pass rates, short-bucket losses, long-bucket wins, length-selection metrics, runtime.
- Comparison type: local same-backbone cross-protocol candidate comparison.

## Command

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/results.jsonl \
  --output-dir /home/shx/projects/dllm_infilling/outputs_clean \
  --experiment-name full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23 \
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
  --mid-rescue-min-long-ratio 0.8
```

## Runtime Paths

- tmux session: `llada_instruct_midcons_20260604`
- GPU set: `2,3`
- Log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log`
- Output prefix: `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_`

## Success Criteria

- Full `1033` sample run exits `0`.
- `summary.json` reports `num_samples=1033`.
- Candidate is at least competitive with the historical LLaDA-Instruct baseline, or regressions are explained by bucket-level evidence.
- No claim is upgraded beyond local same-backbone evidence without Claim Readiness Gate.

## Kill Criteria

Investigate rather than continue if there are GPU OOM errors, malformed output rows, prompt/evaluation failures, or an output collision with historical runs.

## Known Risks

This run does not establish external SOTA by itself. It tests method transfer within the LLaDA family and may reveal threshold overfitting to LLaDA-Base.

## Startup Note

The initial launch used the default HuggingFace endpoint and failed before model loading because direct network access to `huggingface.co` was unavailable. The active launch uses `HF_ENDPOINT=https://hf-mirror.com`.

## Final Result

Output directory:

`/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`

Summary:

- Completed samples: `1033`.
- Candidate pass rate: `815/1033 = 78.90%`.
- Historical same-backbone baseline: `817/1033 = 79.09%`.
- Net delta: `-2` tasks, `-0.19` percentage points.
- Pairwise: `17` wins, `19` losses, `798` tie-pass, `199` tie-fail.
- Short-bucket loss: `17` losses in oracle `<=8`; `2` losses in oracle `9-12`.
- Long-bucket win: `4` wins among oracle `>=17`.
- Bucket pass rates by oracle length: `<=8 = 90.64%`, `9-12 = 82.76%`, `13-16 = 66.67%`, `17-24 = 19.51%`, `25+ = 16.13%`.
- Runtime: `avg_total_sec_including_probe = 4.1766`, faster than the historical baseline `6.8661`.

Interpretation:

This is useful negative transfer evidence. The current `midcons` bounded-repair thresholds improve the LLaDA-Base A6000 control but do not transfer cleanly to `GSAI-ML/LLaDA-8B-Instruct`. It should not be used to upgrade the paper claim. The next cross-model step should compare each literature backbone against its own local baseline and correct prompt/canvas runner.
