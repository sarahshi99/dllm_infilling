# Baseline Closure Phase 0：Population / Protocol / Analyzer Freeze

日期：2026-07-31 UTC

状态：`verified_ready_for_focused_commit_outcome_blind`

## Scope

本阶段只冻结 external baseline 的 population、protocol、manifest 与统计实现。排除 M5、M1--M4、PPT、ExecRepoBench final benchmark 和 frozen controller test。official CAL 4,990 final outcome 在本阶段 commit/push 前保持未解盲。

## Reviewer motivation

外部论文数字、官方源码在项目非冻结子集上的复现、以及本地 common-protocol 比较必须分层。只有 candidate keys、backbone、manifest、decoder、sampling、seed、steps 和 evaluator 全部一致时才计算 paired delta/CI；否则只报告绝对指标。

## Inputs and immutable populations

- 官方 HumanEval-Infilling revision：`88062ff9859c875d04db115b698ed4b0f0395170`。
- official CAL revision：`741e8418a88a732b4c92812424d4f03cab1f7b1f`。
- MultiLine CAL-Rest non-frozen common：既有 `4990` rows / `143` clusters manifest，只登记、不读 outcome。
- SingleLine project non-frozen allowed population：既有 `927`-row manifest；Phase 0 只取其 `task_id/task_group` 与 frozen-exclusion flag，与 official seed-42 CAL Rest 相交，生成 `838` rows / `143` clusters manifest。
- 两份 sealed controller files 不读取；`test_evaluation_count` 保持 `0`。

## Files

- `docs/paper_agent/baseline_population_and_protocol_matrix.current.zh.md`
- `docs/paper_agent/baseline_population_and_protocol_matrix.current.json`
- `analysis/baseline_grouped_analyzer.py`
- `analysis/build_cal_singleline_rest_manifest.py`
- `tests/test_baseline_grouped_analyzer.py`
- `tests/test_build_cal_singleline_rest_manifest.py`
- `analysis_outputs/baseline_manifests_20260731_v1/`

## Exact commands

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_baseline_grouped_analyzer.py tests/test_build_cal_singleline_rest_manifest.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/baseline_grouped_analyzer.py analysis/build_cal_singleline_rest_manifest.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_cal_singleline_rest_manifest.py --official-singleline /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-SingleLineInfilling.jsonl.gz --official-multiline /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-MultiLineInfilling.jsonl.gz --multiline-common-manifest analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl --allowed-singleline-manifest analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/manifest.csv --output-dir analysis_outputs/baseline_manifests_20260731_v1 --smoke-cases 12
git diff --check
```

GPU=`none`。不启动模型或 evaluator，不产生 pass-rate outcome。

## Success gate

1. SingleLine full/demo/rest/non-frozen CAL-Rest=`1033/100/933/838`，clusters=`143`。
2. 新 manifest 不包含 canonical solution、test、oracle length、passed 或 evaluator outcome。
3. 12-case smoke manifest base-function 去重并覆盖长度 strata；strata 仅用于技术 smoke，长度字段不进入 deployable 方法。
4. Analyzer synthetic fixtures 覆盖 row Pass@1、equal-weight task macro、10,000 cluster bootstrap、paired row/group help-harm、成本账本和动态机制分布。
5. 不同 candidate-key 集合无法产生 paired result。
6. focused commit/push 后才允许 CAL 4,990 解盲。

## Kill criteria and risks

- population、source hash 或 revision 不一致；
- 需要读取 sealed frozen files 才能继续；
- manifest 泄漏 forbidden fields；
- paired-key guard、forward accounting 或 tests 失败。

`reviewer_gate_disabled`；本阶段使用 local diff review + fresh verification fallback。

## Verification result

- 相关 tests：`Ran 17 tests` / `OK`。
- `py_compile`：通过。
- matrix 与 manifest summary JSON parse：通过。
- immutable manifest：`838 rows/143 clusters`；smoke=`12 rows/12 clusters`；duplicate/forbidden fields=`0`。
- `git diff --check`：通过。
- official CAL 4,990 final outcome 仍未读取。
