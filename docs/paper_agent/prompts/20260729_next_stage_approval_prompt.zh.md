# 下一阶段待审批执行 Prompt（2026-07-29）

下面的 prompt 可在用户批准后直接用于下一轮长期 paper-agent 执行。它授权完成现有缺口，但不授权旧 M1--M4 outcome-driven 调参、融合、恢复 5,079 或打开 frozen test。

```text
这是 dllm_infilling 长期 paper-agent research session。不要调用 create_goal、update_goal 或 get_goal；不要使用 subagent/parallel-agent/reviewer discovery。工作区固定为：

/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1

权威分支：codex/ccfa-execution-sprint-v1。

先读取 AGENTS.md、docs/paper_agent/research_agent_protocol.md、docs/paper_agent/pause_checkpoint.current.md、docs/paper_agent/method_portfolio.current.json、docs/paper_agent/runtime_status.current.json，以及：

docs/paper_agent/experiments/20260729_experiment_completion_status_and_next_stage.zh.md

已批准目标按以下顺序执行，不等待手动选择；每个阶段都先写 action brief，代码/分析先测试和 commit，再读取 outcome 或启动 full。不得 reset/clean，不得覆盖 raw output，不得读取 frozen test；保留用户未跟踪文件。长任务必须 tmux、append-only raw、独立 failure journal、resume/dedup、progress manifest、fail-stop。

阶段 A：official CAL final outcome-blind analysis

1. 核对 official CAL `4990/4990` canonical raw、0 missing/duplicate/error/failure journal，来源固定为 CAL commit `741e8418a88a732b4c92812424d4f03cab1f7b1f`、HumanEval-Infilling evaluator commit `88062ff9859c875d04db115b698ed4b0f0395170`。
2. 在读取 passed/outcome 前，先审计同一 4,990 non-frozen common subset 上有哪些真实 protocol-matched controls。冻结并 commit analyzer、tests、统计单位、success 定义、length buckets、cost ledger、CI 和 comparison list。
3. 如果不存在同 subset、同 evaluator、可审计的 control，不得伪造 paired delta；只报告 CAL absolute accuracy、10,000 bootstrap CI、selected-length distribution、search/formal/total forwards、token-forward、wall time、显存、错误类型和分桶。
4. analyzer freeze commit 后才允许读取 final outcome。生成正式中文 result report，更新 runtime status/dashboard/checkpoint，并 focused commit/push。

阶段 B：ExecRepoBench evaluator technical smoke

1. 重试 pinned dataset `fa61028ce495c9ceff58398b8a7c47b5ae9f5276` 与 Qwen evaluator `33bc6aabd7791ad7b32f7e92104f11f2359ba890` 的 host checkout。
2. 完成 license、schema、field、repository grouping、six-fill manifest 和 evaluator dependency audit；运行不暴露 final benchmark outcome 的最小 technical smoke。
3. 网络、revision、dependency 或 evaluator 失败时写 exact blocker；不得用替代 repo/commit 或自造 evaluator。
4. 当前没有 paper primary，因此不要运行或读取 final ExecRepoBench benchmark outcome。

阶段 C：DreamOn official-source reproduction

1. 新建并 commit DreamOn 12-case smoke brief。固定 official source `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8`、released model、HumanEval-Infilling evaluator revision、manifest、seed、decode/dynamic-expansion配置、forward/token/wall ledger和准确标签。
2. 明确官方 `torchrun --nproc_per_node 8` 与本机单 H200 的区别。若单 H200 只改变样本并行度、不会改变每例算法/随机性/数据/evaluator，则将运行标记为 `DreamOn official-source single-H200 reproduction`；不得称为 exact official 8-GPU topology reproduction。
3. 先运行 12-case technical smoke；要求 12 unique、0 missing/duplicate/error、evaluator正常、resume no-op、frozen=0、OOM/ECC=0、forward/token ledger 可解释。
4. smoke 通过后，如果 source audit 证明算法等价、完整 manifest 已冻结、预计显存安全且磁盘充足，则自动在独立 tmux 启动 full。运行期间只汇报完整性/进度/ETA，不读 partial accuracy。若拓扑会改变算法或显存不安全，安全停止并记录 exact blocker，不自行改算法。
5. full 完成后，先冻结 analyzer 再读取 outcome，生成 formal report并更新 baseline matrix。

阶段 D：新方法 M5 预研，不自动启动 GPU full

1. 汇总 M1 稀疏 activation、M2 不稳定增益、M3 accuracy/token trade-off、M4 assembly harm、controller low precision和 candidate-existence不足，形成一个独立的 M5 brainstorming/experiment brief。
2. M5 必须同时说明如何改善 candidate existence 与 risk-controlled selection；不得只是修改旧 V0 阈值、轮次、canvas、seed或融合 M1--M4。
3. 固定 allowed/forbidden inputs、fair baseline、forward/token budget、activation audit、12→148→296→927 gate、新 outcome-blind manifest 与 kill criteria。
4. 本阶段只允许 CPU feasibility、tests、manifest 和 preregistration；完成后提交一份用户可审批的 GPU 12-case smoke方案，不自动启动 M5 GPU。

执行纪律：

- 最多一个 GPU research process；不得 kill 无关 PID。
- 每个实质阶段使用 focused commits；push 失败不是科学 blocker，但需区分 local/pushed commit。
- 运行 py_compile、targeted tests、相关 full unittest、bash -n、JSON/CSV parse、missing/duplicate/error、forbidden-input、forward/token accounting、frozen seal 和 git diff --check。
- 不得删除/覆盖 outputs_clean、logs 或用户未跟踪目录。
- 不要 promise-only stop。安全且已批准的阶段要继续执行；只有真实数据、算法、CUDA、磁盘、网络/evaluator blocker 才停，并写 checkpoint/dashboard。

最终回复必须给：official CAL final result；ExecRepoBench smoke状态；DreamOn smoke/full状态与准确标签；M5 brief链接；branch/HEAD/commits/push；GPU/磁盘/frozen seal；所有未完成项与 exact blocker。
```

## 建议的审批回复

如果接受上述边界，可直接回复：

> 批准 2026-07-29 下一阶段方案：执行 A、B、C，并完成 D 的 CPU-only preregistration；允许 DreamOn 在满足 prompt 内等价性、安全和 smoke gate 后自动进入 full；不授权旧 M1--M4 调参/融合/扩展，不打开 frozen test。

如果暂时不希望消耗 GPU，可回复：

> 只批准阶段 A、B、D；DreamOn 停在 launch-ready brief，不启动 GPU。
