# True-Long Action-Ceiling Matrix Action Brief

日期：2026-07-02 CST

## Action Name

`true_long_action_ceiling_matrix_v0`

## Scope

先实现 dry-run 和 small pilot scaffold，不启动 full GPU。默认命令只生成 case/action manifest、summary 和 report。只有显式添加 `--execute-pilot` 才会加载模型，并且 pilot 有 `--max-pilot-cases` hard guard。

## Hypothesis

True-long 剩余失败不是单一 length-selection 问题。需要分开判断：

- current primary 是否 canvas 不足；
- current Route2 rescue 是否 generation/selection 不足；
- oracle-sufficient canvas 是否能产生正确候选；
- missed failed-long rows 是否存在 trigger recall ceiling。

## Interventions

预注册四个 action：

- A：当前 `midcons` primary。
- B：当前 Route2 precision `len32`。
- C：oracle-sufficient canvas，`max(primary_len, route2_len, oracle_len)`，offline ceiling only。
- D：与 C 相同 canvas，但固定 `96` decode steps，offline ceiling only。

## Control

当前 LLaDA-Base `midcons` trace run：

```text
/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl
```

当前 Route2 precision len32：

```text
/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl
```

## Expected Outcomes

- 如果 C/D 也无法产生正确候选：true-long 路线应转向 backbone/generation limitation 或 diagnostic paper。
- 如果 C/D 有正确候选但 B 没有：优先研究 rescue generation/action design。
- 如果 missed failed-long pool 中 C/D 有候选：优先研究 trigger recall。
- 如果 candidate pool 有正确答案但 deployable selector 选错：优先研究 selector。

## Success Criteria

Dry-run success:

- 生成 `case_manifest.csv`、`action_manifest.csv`、`summary.json`、`report.md`。
- case pool 覆盖 positive controls、triggered failed-long、missed failed-long。
- report 明确标注 C/D 是 `offline_ceiling_only`，不能作为部署方法。

Pilot success:

- 只跑预注册 small case set。
- 输出每个 action 的 pass/fail、canvas length、cost。
- 能回答至少一个瓶颈方向：canvas ceiling、generation ceiling、selector gap 或 trigger gap。

## Kill Criteria

- case join 不完整；
- oracle length 缺失；
- positive controls 不能复现；
- pilot 需要超过 hard guard 的 case count；
- C/D 被误写为 deployable action。

## Dry-Run Command

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/action_ceiling_matrix.py \
  --timestamp 20260702_dryrun \
  --max-cases-per-pool 3
```

## Pilot Command Pattern

只在研究者确认后运行：

```bash
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/action_ceiling_matrix.py \
  --timestamp 20260702_pilot_gpu2 \
  --task-ids-csv 'SingleLineInfilling/HumanEval/116/L0,SingleLineInfilling/HumanEval/108/L6,SingleLineInfilling/HumanEval/10/L5' \
  --execute-pilot \
  --max-pilot-cases 3
```

## No-GPU Statement

本 brief 和 scaffold 实现阶段不启动 GPU。dry-run 只读已有 compact/raw result paths 来生成 manifest。
