# Current Paper-Agent Action

更新时间：2026-07-16 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`

起点证据：`8b348f979f09cda07811e04b7ad3dee60e56373b`

## Action Name

`FAST-SPRINT-01`: official baselines、P2.1 grouped statistics、四个独立候选方法 M1--M4、以及 ExecRepoBench preparation。

## 当前决定与边界

Operational decision 是 `iterate_and_execute`，不是 blocked。M1、M2、M3、M4 是平行的独立候选方法；当前**没有**论文主方法。M1 只因最先完成代码而先进入 smoke，不代表 M2--M4 被降级、放弃或融合。Phase 5 的 `M1-D0` fixed proxy kill 和 `M4-D0/F1` premise 结果保持为历史结论，不能被改写成对四个新 V0 的 kill。

冻结 controller test 始终 sealed，`test_evaluation_count=0`；不得读取 106 个 sealed SingleLine test rows。所有 full run 必须支持 resume/dedup，并在结束后审计 missing/duplicate/error。旧共享 MultiLine candidate bank 已自然完成 `40,632/40,632` 且 final audit 通过；不重启、不重建、不改写 raw。

## 立即执行的工作包

### P1/P2/P4

- `P1.1-OFFICIAL-CAL`：official upstream 已固定为 `NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`。checkout 后审计 model/prompt/dataset mapping/canvas/steps/forwards/seed/evaluator/wall/GPU/memory；12-case technical smoke 后自动 full。DreamOn official full 是后续 baseline；rho-EOS 仅 faithful infilling compatibility 后 smoke/full；LR-DLLM 只保留 blocker audit。local CAL/CAL-lite 不得叫 official CAL。
- `P2.1-GROUPED-STATS`：已在 `c66678a` 完成。`6707` spans / `20121` results 背后是 `148` base-task groups；primary 是 equal-weight base-task macro accuracy，span-micro 只 descriptive。10,000 次 cluster bootstrap、paired wins/losses、group-aware label-swap、分层 CI、cost frontier 和 8-cell intersection 已写入 compact CSV/JSON/Markdown；禁止行级独立显著性。
- `P4.1-EXECREPOBENCH`：首个外部 benchmark 已固定为 ExecRepoBench。pinned provenance 为 dataset `fa61028ce495c9ceff58398b8a7c47b5ae9f5276` 与 Qwen evaluator `33bc6aabd7791ad7b32f7e92104f11f2359ba890`；`experiments/p4_execrepobench_audit.py` 已准备只输出无代码的 repository-grouped six-fill smoke plan。实际数据/evaluator checkout 的 host 网络 fetch 在 2026-07-13 被 approval control plane `422` 阻断，故 evaluator smoke 尚未宣称完成；最终 external result 仍等方法配置冻结后才可开。

### M1.1 Abductive Program-State Bridge

两轮语义/完整性审计与修复已在 `1d9ef3f` 完成：AST/def-use dependency cone、seed-0 fixed64 检测、64-forward null fallback、等 canvas/forwards/cardinality generic、分离 raw outputs、forbidden-input/activation/determinism/resume tests 均已落地。**历史 5079-case MultiLine M1** 已在用户安全中断后成为 `safely_paused_resumable`：原三个 raw 目录 append-only 保留，不读取部分性能，未来仅胜出方法可用原目录与 `--auto-full --selected-method-only-5079` 的 existing-key dedup 恢复。

新的独立 **M1 RandomSpanLight** 已于 2026-07-16 实际启动，路径为 `12 technical smoke → automatic 148-case full`，绝不恢复或改写旧 5079 raw。smoke 只审计 schema/CUDA/evaluator/missing/duplicate/error/forward budget/resume/frozen seal，不含性能 gate。其固定 grouped analyzer 与成本合同已在 `9694b29` 推送，且在读取新 148-case outcome 前完成；live PID/tmux/吞吐只记录于 `runtime_status.current.json`，不在本路线文档中报告中途性能。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：CPU runner 已升级为固定路线 `12 smoke → 148 RandomSpanLight → 296 MultiLine-Core / 927 SingleLine → selected-method-only 5079 MultiLine`；M2 的 auto-full 被强制限制为 `148`。在新 M1 t+10 资源审计满足 `free ≥25 GiB`、OOM/ECC=0 且 M1 继续增长后，M2 已于 2026-07-16 在独立 tmux/log/output 中实际启动；不等待 M1 结束，也不触碰旧 5079 M1。
- `M3 Birth-Death Canvas Diffusion V0`：`a874c54` 已完成独立 runner/analysis/brief/tests/launcher；初始 canvas `16/32/64/128` particles，uniform 与 birth/death 各固定 256 forwards/task，只用 inference-visible confidence、syntax、prefix/suffix compatibility。M2 资源安全完成后再作 12-case→148-case full。
- `M4 Semantic Particle Assembly V0`：`ed94471` 已完成独立 runner/analysis/brief/tests/launcher；完整 148-case offline structural assembly audit 已实际完成于 `analysis_outputs/m4_semantic_particle_assembly_20260713_v0/offline_summary.json`（best/assembly 均 148/148、zero missing/duplicate/error、frozen sealed/count 0）。从 8 candidates 提取 AST statement/basic-block/def-use fragments，以 inference-visible obligations 选择；M3 资源安全完成后作 64-forward repair smoke/full。

四个 V0 的数据路线固定为：12 technical smoke → 148 RandomSpanLight → 296 MultiLine-Core / 927 non-frozen SingleLine development comparison → selected-method-only 5079 MultiLine → method freeze 后 ExecRepoBench。裸 `--auto-full` 不得直接进入 5079。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

旧共享候选库已完成；当前新 M1 与 M2 最多并行两个独立 GPU experiment process。M1 启动 t+10 后已审计 memory/utilization/power/OOM/ECC 与行增长，并据此启动 M2；M2 启动后再次审计两者吞吐。只有两个任务均持续增长、OOM/ECC=0、且仍保留至少 25 GiB 显存余量时，才允许考虑第三个进程；总吞吐显著下降则回到单研究进程，但不停止方法代码开发。每个方法、每个进程均有独立 output/log directory。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。

## 2026-07-16 Execution Sprint V1 current state

`codex/ccfa-execution-sprint-v1`（base `ce416c4`）是权威执行分支；`afd3c45` 已 superseded。M1 5079 MultiLine 已安全暂停、可恢复但不自动恢复；M1 RandomSpanLight 新 runner、M2/M3/M4 独立 runner、共同 grouped analyzer 与 CAL adapter 已在 `9335d84` 后继续实现。完整 CPU tests 已通过，frozen test 仍为 sealed/count=0。

真实 GPU 执行已开始：新 M1 RandomSpanLight 先启动，M2 随 M1 的 t+10 安全审计启动；M3、M4 依空出的研究 GPU 槽按顺序进入各自 `12→148`。SciPy 在 `dllm_env` 中仍缺失，网络安装请求由宿主审批服务返回 `422 model not found: codex-auto-review`；这只推迟 official CAL smoke，不改变 M1--M4 的执行队列，也不构成 scientific blocker。
