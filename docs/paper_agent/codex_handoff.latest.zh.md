# Codex Handoff Latest

更新日期：2026-07-03 CST

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 分支：`codex/risk-controlled-dynamic-rescue`
- 本轮起点 commit：`467734de306bedd81c217ea512d370e8cec3cf51`
- 本轮已提交 commits：`4e5c0fc`、`89e30c1`、`e11c18e`、`9eca940`、`7def5f7`、`2738b9b`
- 本轮最终 docs/results commit：见 push 后 branch HEAD。
- working tree：最终 push 前应仅包含本轮 docs/results metadata；push 后应为 clean。
- 当前阶段：accelerated Phase 1b correction completed: 95-case seed-0 long-failure generation screen, conditional multi-seed, grouped split, CAL same-protocol sanity, and CAL full comparison.
- GPU：95-case screen 使用 `CUDA_VISIBLE_DEVICES=2`；CAL full resume 按研究者要求停止旧 GPU3 run 后在 `CUDA_VISIBLE_DEVICES=1,0` 上完成。

当前最可信结论：

> `G_oracle_sufficient_trace_span_remask` 机制真实执行；95-case oracle-sufficient offline screen 显示 hard cases 中 `31/89` 出现新正确候选，但全部来自 missed failed-long，triggered failed-long 为 `0/33`。这支持存在 limited/mixed generation-ceiling signal，同时说明当前 E/F/G 对 triggered failed-long 的 rescue-generation ceiling 很弱。CAL same-protocol full run 完成后，local official CAL pass rate 为 `0.7493`，低于 control/midcons/Route2/V6。

## 2. 本轮完成内容

代码与 runner：

- `experiments/action_ceiling/long_failure_generation_screen.py`
  - 新增 95-case seed-0 screen runner，自动构建 33 triggered failed-long、56 missed failed-long、6 positive controls。
  - 执行 E/F/G 三个 generation actions，生成 compact CSV/JSON/Markdown。
  - 自动选择最多 30 个 case 做 conditional multi-seed diagnostic，严格区分 seed-0 broad screen 与条件性诊断。
- `experiments/action_ceiling/distinct_candidate_ceiling.py`
  - 新增 `G_oracle_sufficient_trace_span_remask`：基于 C trajectory 内部 trace，按 token flip count 与 final confidence 选最高不稳定位置，以其为中心 remask contiguous span。
- `experiments/action_ceiling/create_grouped_split.py`
  - 生成按 `HumanEval/<id>` 分组的 train/calibration/validation/test split，验证无 overlap。
- `experiments/action_ceiling/cal_same_protocol_sanity.py`
  - 完成 10-case CAL same-protocol sanity。
- `experiments/action_ceiling/cal_full_compare.py`
  - 对完成的 CAL full run 与 control/midcons/Route2/V6 做同协议 paired comparison。
- `experiments/action_ceiling/cal_official_lcas_v3_resume.py`
  - 新增 CAL full partial-run resume helper；本轮将 741-row partial run 补齐为 1033-row merged run。

结果与文档：

- 95-case screen：`analysis_outputs/long_failure_generation_screen_20260702_accel95/`
- grouped split：`analysis_outputs/grouped_split_20260702_accel2/`
- grouped split protocol：`docs/paper_agent/grouped_split_protocol.zh.md`
- CAL sanity：`analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10_v2/`
- CAL full resume manifest：`analysis_outputs/cal_full_resume_20260703_gpus10/`
- CAL full compact comparison：`analysis_outputs/cal_full_same_protocol_compare_20260703_gpus10_resumed_full/`
- review manifest：`docs/paper_agent/review_manifest.latest.json`
- run registry：`docs/results/run_registry.md`

未提交/不应提交：

- 失败的第一次 CAL sanity 目录 `analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10/` 只包含 incomplete attempt，不作为成功 artifact。
- Raw full CAL outputs 保留在 `outputs_clean/`，不提交 `results.jsonl`。

## 3. 精确运行方式

95-case seed-0 + conditional multi-seed:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/long_failure_generation_screen.py \
  --timestamp 20260702_accel95
```

Grouped split:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/create_grouped_split.py \
  --timestamp 20260702_accel2
```

CAL 10-case sanity:

```bash
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/cal_same_protocol_sanity.py \
  --timestamp 20260702_accel_cal10_v2
```

CAL full resume on GPU 1/0:

```bash
CUDA_VISIBLE_DEVICES=1,0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/cal_official_lcas_v3_resume.py \
  --partial-run-dir /home/shx/projects/dllm_infilling/outputs_clean/full_official_cal_lcas_v3b_accel_gpus10_20260702_20260702_230148 \
  --manifest-output-dir analysis_outputs/cal_full_resume_20260703_gpus10 \
  --supplement-experiment-name full_official_cal_lcas_v3b_accel_gpus10_20260702_supplement \
  --merged-experiment-name full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full
```

CAL full comparison:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/cal_full_compare.py \
  --cal-run-dir /home/shx/projects/dllm_infilling/outputs_clean/full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full_20260703_111615 \
  --timestamp 20260703_gpus10_resumed_full
```

## 4. 结果

95-case broad screen:

| Group | Cases | New Correct | Unique Candidate Cases | Compile Repairs | Unchanged By All |
|---|---:|---:|---:|---:|---:|
| `triggered_failed_long` | 33 | 0 | 3 | 3 | 30 |
| `missed_failed_long` | 56 | 31 | 7 | 6 | 46 |
| `positive_control_rescued` | 6 | 6 | 1 | 4 | 5 |

Action-level seed-0:

| Action | Rows | Hard Correct | Compile Success | Changed vs C | Mean Sec | P95 Sec |
|---|---:|---:|---:|---:|---:|---:|
| E no early commit | 95 | 31 | 73 | 13 | 4.042 | 6.208 |
| F trace token remask | 95 | 30 | 71 | 7 | 4.229 | 5.896 |
| G trace span remask | 95 | 30 | 71 | 7 | 4.725 | 6.668 |

Hard-case overlap:

- hard cases with any new correct candidate：`31/89`
- E correct hard cases：`31`
- F correct hard cases：`30`
- G correct hard cases：`30`
- hard compile repairs：`9/89`
- diversity-without-correctness：`6/89`
- all-actions-unchanged：`76/89`
- all-actions-distinct-but-wrong：`1/89`

Conditional multi-seed:

- selected cases：`30`
- actions：`180`
- seeds：`1,2`
- hard selected cases with any correct candidate under seeds 1/2：`24`
- hard cases corrected only by conditional seeds 1/2 but not seed 0：`0`
- This is conditional diagnostic only, not an unbiased Pass@3 benchmark.

Grouped split:

- split seed：`20260702`
- task groups：train `98`，calibration `25`，validation `25`，test `16`
- rows：train `645`，calibration `155`，validation `127`，test `106`
- verification：passed, no overlap by original `HumanEval/<id>`.

CAL:

- sanity verdict：`protocol_matched_cal`
- full merged run：`1033` rows, `774` pass, pass rate `0.7493`
- same-protocol paired against CAL:
  - control：CAL wins `25`, losses `38`
  - midcons：CAL wins `17`, losses `38`
  - Route2 len32：CAL wins `17`, losses `44`
  - V6 short override：CAL wins `17`, losses `45`

Final verdict：`mixed_generation_signal`。

## 5. 研究解释

数据直接支持的事实：

- 在 oracle-sufficient offline canvas 下，E/F/G 能为 missed failed-long 产生大量正确候选，但 seed-0 triggered failed-long 没有一个被修复。
- 多数 hard cases 的最终候选 hash 仍不变：`76/89` all-actions-unchanged。
- G 的 contiguous-span remask 真实执行，但整体恢复数没有超过 E；它主要作为 diagnostic action，而不是已证明的新方法贡献。
- CAL same-repository same-protocol full run 完成，且低于本地 control/midcons/Route2/V6。

合理推断：

- 当前 generation family 存在 limited/mixed ceiling signal，主要在 missed failed-long；triggered failed-long 仍更像 rescue-generation 或 candidate-exploration bottleneck。
- 后续若研究 deployable controller，重点应是 trigger/selector 与 grouped calibration/test split，而不是继续扩大 oracle canvas action sweep。

尚未验证：

- 这些 oracle-sufficient correct candidates 是否能由 deployable, non-oracle controller 找到。
- 当前 screen 是否能泛化到其他 backbone、其他 benchmark 或 held-out grouped test。
- CAL 外部 official-code reproduction 尚未完成；本轮只支持 same-repository same-protocol local comparison。

## 6. 阻塞和风险

- 95-case screen 使用 oracle/reference sufficient length，不能报告为 deployable Pass@1。
- Conditional multi-seed 是 seed-0 后选样诊断，不能混入无偏主指标。
- HumanEval 已被多轮探索使用，controller 必须使用 grouped split，最终 held-out 不能继续被调参污染。
- First failed CAL sanity attempt 未作为 artifact 使用。

## 7. 下一步建议

1. 在 grouped split 上定义 frozen controller protocol。
   - 科学问题：非 oracle trigger/selector 能否利用 missed failed-long candidate-existence signal？
   - 输出：calibration-only controller + validation/test locked report。

2. 对 triggered failed-long 做 compact error taxonomy，而不是继续扩 action。
   - 科学问题：`0/33` 是 syntax repair failure、semantic capability failure，还是 trajectory lock-in？
   - 输出：按 SyntaxError/UnitTestFailure 和 hash-change 分层的小报告。

3. 将 CAL comparison 纳入同协议 baseline 表。
   - 科学问题：local CAL 在同协议下不优于已有 control/Route2/V6，对论文定位意味着什么？
   - 输出：baseline table + caveat: same-repository local CAL, not external official reproduction.

## 8. 文件索引

优先读：

- `analysis_outputs/long_failure_generation_screen_20260702_accel95/report.md`
- `analysis_outputs/long_failure_generation_screen_20260702_accel95/seed0_summary.json`
- `analysis_outputs/long_failure_generation_screen_20260702_accel95/multiseed_summary.json`
- `analysis_outputs/grouped_split_20260702_accel2/split_manifest.json`
- `docs/paper_agent/grouped_split_protocol.zh.md`
- `analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10_v2/protocol_comparison.md`
- `analysis_outputs/cal_full_same_protocol_compare_20260703_gpus10_resumed_full/report.md`
- `docs/paper_agent/review_manifest.latest.json`

暂时不建议读：

- 全量 `outputs_clean/*/results.jsonl`
- 大 raw traces
- checkpoint/cache 目录
