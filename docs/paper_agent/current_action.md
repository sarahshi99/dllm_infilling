# Current Paper-Agent Action

更新时间：2026-07-15 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`

起点证据：`8b348f979f09cda07811e04b7ad3dee60e56373b`

## Action Name

`FAST-SPRINT-01`: official CAL + P2.1 grouped statistics + M1.1 Abductive Program-State Bridge + ExecRepoBench preparation。

## 为什么现在做

Phase 5 已完整结束；本轮 operational decision 是 `iterate_and_execute`，不是 blocked。M1-D0 fixed proxy 的历史 kill 仍有效，但不阻止一个独立的 M1 full implementation。当前同时补齐 official baseline、cluster-aware statistics、M1 bridge 和 repository-level external evaluator；冻结 controller test 保持 sealed。

本动作不重写历史实验结论，不打开 frozen test（`test_evaluation_count=0`）。所有 full run 必须 resume/dedup，并在结束后审计 missing/duplicate/error。GPU 串行顺序固定为 official CAL full -> M1 MultiLine full -> DreamOn official full；rho-EOS 先 compatibility。

## 立即执行的并行工作包

### `P1.1-OFFICIAL-CAL`

已固定 official CAL `741e8418a88a732b4c92812424d4f03cab1f7b1f` 与 official HumanEval-Infilling `88062ff9859c875d04db115b698ed4b0f0395170`。source audit 验证 MultiLine `5815/5815` exact task-ID mapping，project non-frozen intersection=`5079`，并写出无 outcome 的 12-case smoke manifest。审计覆盖模型、prompt、动态 canvas、steps/forward contract、seed 缺口、evaluator、environment/license；GPU smoke 等现有 M1 作业不再被打断后执行。local CAL/CAL-lite 不得改名为 official CAL。

输出：

- fixed upstream commit / environment manifest、compact audit、smoke/full resume logs、missing/duplicate/error audit。

DreamOn 在 CAL full 后按 training-based stratum 运行；rho-EOS 仅在 faithful infilling compatibility 后运行；LR-DLLM 只保留 blocker audit。

### `P2.1-GROUPED-STATS`

输入 existing `6707`-span / `20121`-result files；验证 `148` 个 `task_group`。primary estimand 为每个 base task 内先计算 policy accuracy、再对 task 等权平均。span-micro totals 仅 descriptive。执行 `10000` 次 fixed-seed cluster bootstrap、task-level paired wins/losses、group-aware label-swap permutation；config/length/error strata 必须同时报告 group count 与 cluster CI。

- accuracy-cost frontier；
- control/CAL-lite/oracle 八格 intersection；
- compact CSV/JSON/Markdown 与 paper-ready figure data。

禁止行级独立显著性。

输出：`analysis_outputs/second_regime_grouped_statistics_<timestamp>/`。

### `M1.1-ABDUCTIVE-PROGRAM-STATE-BRIDGE`

代码与独立 brief 已完成于 `0d75c71`：stage one 是 `16/32/64/128 × seeds 0/1` 的共享 64-step bank；stage two 分别执行实际 64-forward generic low-confidence remask 与 dependency-cone targeted remask。deployable method 禁止 reference code、oracle length、unit-test outcome、task ID、split label 与 passed labels；无可映射 dependency cone 时才 safe fallback 到 fixed64。模型 `GSAI-ML/LLaDA-8B-Base`；existing evaluator。下一 GPU step 仍是 12-case MultiLine technical smoke；技术通过后直接跑全部 `5079` non-frozen spans，不设性能 gate。此 job 等待前序 official CAL full，不能抢占既定 GPU 顺序。

报告 fixed64、ordinary-confidence best-of-grid、equal-compute generic remask、M1 score-only、M1 full、oracle ceiling；task-macro delta、span-micro accuracy、help/harm、分桶、forward/token budget、wall time、显存。性能 gate 仅用于 paper promotion。

### `P4.1-EXECREPOBENCH`

直接选择 ExecRepoBench 作为首个 external benchmark：dataset `fa61028c` 与 Qwen evaluator `33bc6aa` 已固定。Qwen checkout fetch 在 2026-07-15 再次被 approval control plane `422` 拒绝；不绕过下载，也不误报 evaluator smoke。checkout 恢复后审计环境、许可、字段、repository grouping 和多 repository/六 fill_type smoke；M1 配置冻结前不得打开最终 external result。

## 执行边界

- M1 full implementation 立即开始；M2/M3 cheap diagnostics 不是前置条件；M4 降级、A1 停止；P3 等 official CAL 后再决定。
- 所有并行指独立作业，不使用 subagent；GPU full 必须维持既定顺序。
- historical `802/1033` 为 A6000 historical；`796/1033` 为 H200 evidence base；不可混为单一结果。

## 当前结论

- Operational status：`iterate_and_execute`。
- Scientific status：`diagnostic_mixed_not_submission_ready`。
- Frozen test：`sealed`，`test_evaluation_count=0`。
- Phase 5 historical result report：`docs/paper_agent/experiments/20260712_phase5_h200_candidate_bank_result.md`。
