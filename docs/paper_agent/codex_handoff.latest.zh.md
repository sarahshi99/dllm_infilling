# Codex Handoff Latest

更新日期：2026-07-02 CST
当前阶段：Phase 0 repository/evidence audit completed; Phase 1 action-ceiling dry-run scaffold completed; pilot not yet run.

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 本轮分支：`codex/risk-controlled-dynamic-rescue`
- 基线分支：`paper-agent-overnight`
- 基线 commit：`e2b20ae630f05c7d33a549252232b9a5c9db9045`
- 本轮 Phase 0 审计内容 commit：`7ee7224 docs: add codex phase 0 repository audit`
- 本轮 Phase 1 scaffold commit：待提交
- 默认远端分支：`origin/main @ 2209463`
- 最新活跃远端分支：`origin/paper-agent-overnight @ e2b20ae`，领先 `origin/main` 26 commits
- 审计前 working tree：clean
- 本轮是否运行 GPU：否

当前最可信结论：

> 保守推理时长度控制和 selective rescue 可以低风险修复部分 medium/near-long under-selection；但 true-long 剩余失败同时受 trigger recall、rescue generation quality 和 candidate selection 限制。更长 canvas 必要但经常不充分。

## 2. 本轮完成内容

代码修改：

- 新增 `experiments/action_ceiling/action_ceiling_matrix.py`
- 新增 `experiments/action_ceiling/__init__.py`
- 新增 `tests/test_action_ceiling_matrix.py`

文档修改：

- 新增 `docs/paper_agent/codex_repository_audit.zh.md`
- 新增本文件 `docs/paper_agent/codex_handoff.latest.zh.md`
- 在 dashboard/results/registry/snapshot 中添加本轮审计索引和 no-new-run 说明
- 新增 `docs/paper_agent/experiments/20260702_action_ceiling_matrix_action.md`
- 新增 compact dry-run 输出 `analysis_outputs/action_ceiling_20260702_dryrun/`

已完成审计：

- 确认 `paper-agent-overnight` 是最新研究分支，不是 `main`。
- 梳理 control、midcons、Route2、V4-V8 结果关系。
- 审计关键 runner 的 gate/action/selector。
- 识别可复现性缺口：缺少统一 manifest、commit、command、env、VRAM/P95/forward accounting。
- 识别 benchmark leakage 风险：现有多轮 V4-V8 都基于同一 `HumanEval-SingleLineInfilling/test`，且 fold 按 full task_id 而非 original HumanEval task group。
- 实现 action-ceiling dry-run scaffold，默认只读已有 result paths，生成 case/action manifest；只有显式 `--execute-pilot` 才会加载模型。

## 3. 精确运行方式

本轮审计命令核心如下：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
git fetch --all --prune
git status --short --branch
git branch -avv --sort=-committerdate
git log --all --date=iso --pretty=format:'%h %cd %d %s' -n 20
```

读取文档和 compact reports：

```bash
sed -n '1,260p' ccfa_readiness_assessment.zh.md
sed -n '1,180p' docs/paper_agent/paper_agent_dashboard.zh.md
sed -n '1,220p' docs/paper_agent/experiment_plan.current.zh.md
sed -n '1,620p' docs/paper_agent/experiment_results.zh.md
sed -n '1,180p' analysis_outputs/route2_error_analysis_20260617_165806/report.md
sed -n '1,180p' analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md
```

读取 runner 结构：

```bash
rg -n "POLICIES|add_argument|trigger|selector|rescue|summary|seed|task_ids" clean_scripts/run_route2_trace_rescue.py
rg -n "selector|candidate|oracle_upper_bound|add_argument|task_ids" clean_scripts/run_route2_rescue_quality_v5.py
rg -n "official|repair|proportional|add_argument|task_ids" clean_scripts/run_lcal_official_bounded_repair.py
```

本轮未运行 GPU/pilot 实验，因此没有新 checkpoint；已生成一个 compact dry-run 输出目录用于预注册 case/action manifest。

Phase 1 dry-run 命令：

```bash
python experiments/action_ceiling/action_ceiling_matrix.py \
  --timestamp 20260702_dryrun \
  --max-cases-per-pool 3
```

Dry-run 输出目录：

```text
analysis_outputs/action_ceiling_20260702_dryrun
```

验证命令：

```bash
python -m py_compile experiments/action_ceiling/action_ceiling_matrix.py tests/test_action_ceiling_matrix.py
python -m unittest tests/test_action_ceiling_matrix.py
git diff --check
```

## 4. 结果

本轮结果是审计结论，不是新 pass-rate。

关键指标复核：

- A6000 control：`787/1033 = 76.19%`
- `midcons`：`795/1033 = 76.96%`，vs control `+8/0`
- Route2 precision len32：`801/1033 = 77.54%`，vs `midcons` `+6/0`
- V6 short override：`802/1033 = 77.64%`，vs Route2 `+1/0`
- V7：`792/1033 = 76.67%`，negative
- V8a/b/c：`786/781/782`，均 negative
- Route2 triggered failed-long：`33`
- Triggered failed-long with rescue length `>= oracle`：`31/33`
- Missed failed-long：`56`

Action-ceiling dry-run:

- case_count: `9`
- action_count: `36`
- case_pool_counts: `{'missed_failed_long': 3, 'positive_control_rescued': 3, 'triggered_failed_long': 3}`
- oracle_bucket_counts: `{'17-24': 2, '25+': 6, '9-12': 1}`
- planned actions: A primary, B Route2 len32, C oracle-sufficient canvas, D oracle-sufficient canvas + 96-step schedule
- pilot status: not run

成本信息现状：

- Route2 precision len32 记录 `avg_total_sec_including_probe = 5.4622`
- V6 记录 `avg_total_sec_including_probe = 4.8768`
- 多数 summary 有平均时间，但缺少统一 forward calls、P95 wall-clock 和 VRAM。

## 5. 研究解释

数据直接支持的事实：

- 当前 LLaDA-Base 最好 full result 是 V6 `802/1033`。
- Route2/V6 是低风险小幅正收益，且没有观察到 pairwise loss。
- V7/V8 的全局比例放长路线失败。
- Route2 的 true-long 失败不能只归因于 rescue length 不足。

合理推断：

- 下一阶段应做 action-ceiling matrix，而不是继续 V8 beta/threshold sweep。
- Phase 1 应把 canvas adequacy、generation adequacy、selector gap 和 trigger gap 分开测。

尚未验证：

- oracle-sufficient canvas 是否能在 true-long failed rows 上产生正确候选。
- conservative schedule 或 local refinement 是否改善 rescue generation。
- deployable selector 是否能识别已有正确候选。

与原假设冲突：

- “更强比例长度奖励会系统改善 true-long”目前被 V7/V8 full runs 反驳。

## 6. 阻塞和风险

- 现有 1033 rows 被反复用于 error analysis 和 rule selection，存在测试集调参风险。
- 当前 strict-split 是 full task_id folds，不是 original HumanEval task grouped split。
- 旧 run 缺少统一 command/commit/env manifest。
- 部分 checkpoint 下载和缓存路径位于 `/tmp` 或依赖 HF mirror，不保证长期可复现。
- 外部 CAL/LR-DLLM/DreamOn 仍未做同协议本地 baseline。
- 不应把文献 reported number 与本地 results 混成 protocol-matched SOTA 表。

## 7. 下一步建议

1. 研究者确认是否运行 Phase 1 small pilot。
   - 科学问题：true-long 剩余失败到底受 canvas、generation、selector 还是 trigger 限制？
   - 所需代码：已新增 `experiments/action_ceiling/action_ceiling_matrix.py`。
   - 预计输出：pilot 会新增 `pilot_results.csv`，并保留 dry-run manifests。
   - 改变方向的结果：如果 oracle-sufficient C/D 仍不能产生正确候选，应停止 true-long length-control 主线。

2. 建立 grouped split 文件。
   - 科学问题：后续 controller 是否存在 benchmark leakage？
   - 所需代码：按 `HumanEval/<id>` group 划分 train/calibration/validation/test。
   - 预计输出：`docs/paper_agent/grouped_split_protocol.zh.md` 和 machine-readable split JSON。
   - 改变方向的结果：若 held-out 效果显著低于当前 test-set探索结果，应降级方法 claim。

3. 补 run manifest 机制。
   - 科学问题：结果能否由后续研究者/agent 复现？
   - 所需代码：logger 保存 `manifest.json`，包含 command、commit、env、checkpoint、GPU、seed。
   - 预计输出：新 run 都带完整 manifest。
   - 改变方向的结果：旧关键结果若无法复现，需要作为 historical evidence 而非 main table evidence。

## 8. 文件索引

后续最应优先阅读：

- `docs/paper_agent/codex_repository_audit.zh.md`
- `analysis_outputs/action_ceiling_20260702_dryrun/report.md`
- `docs/paper_agent/experiments/20260702_action_ceiling_matrix_action.md`
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/experiment_results.zh.md`
- `docs/paper_agent/evidence_snapshot.md`
- `analysis_outputs/route2_error_analysis_20260617_165806/report.md`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`
- `analysis_outputs/route2_v6_short_override_audit_20260620/report.md`
- `analysis_outputs/proportional_length_widening_v7_midcons_20260630/report.md`
- `docs/paper_agent/experiments/20260701_next_step_brainstorm_after_v8.md`
- `docs/superpowers/plans/2026-07-02-post-v8-rescue-quality-and-local-guard-plan.md`

暂时不建议读取：

- 全量 `outputs_clean/*/results.jsonl`
- 大量 raw traces
- checkpoint/cache 目录
