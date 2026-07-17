# Current Paper-Agent Action

更新时间：2026-07-17 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`；M1--M4 唯一当前状态登记：`docs/paper_agent/method_portfolio.current.json`。

起点证据：`8b348f979f09cda07811e04b7ad3dee60e56373b`

## Action Name

`FAST-SPRINT-01`: official baselines、P2.1 grouped statistics、四个独立候选方法 M1--M4、以及 ExecRepoBench preparation。

## 当前决定与边界

Operational decision 是 `execute_and_monitor`，不是 blocked。M1、M2、M3、M4 是平行的独立候选方法；当前**没有**论文主方法。M1 只因最先完成代码而先进入 smoke，不代表 M2--M4 被降级、放弃或融合。Phase 5 的 `M1-D0` fixed proxy kill 和 `M4-D0/F1` premise 结果保持为历史结论，不能被改写成对四个新 V0 的 kill。

冻结 controller test 始终 sealed，`test_evaluation_count=0`；不得读取 106 个 sealed SingleLine test rows。所有 full run 必须支持 resume/dedup，并在结束后审计 missing/duplicate/error。旧共享 MultiLine candidate bank 已自然完成 `40,632/40,632` 且 final audit 通过；不重启、不重建、不改写 raw。

## 立即执行的工作包

### P1/P2/P4

- `P1.1-OFFICIAL-CAL`：official upstream 已固定为 `NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`。checkout 后审计 model/prompt/dataset mapping/canvas/steps/forwards/seed/evaluator/wall/GPU/memory；12-case technical smoke 后自动 full。DreamOn official full 是后续 baseline；rho-EOS 仅 faithful infilling compatibility 后 smoke/full；LR-DLLM 只保留 blocker audit。local CAL/CAL-lite 不得叫 official CAL。
- `P2.1-GROUPED-STATS`：已在 `c66678a` 完成。`6707` spans / `20121` results 背后是 `148` base-task groups；primary 是 equal-weight base-task macro accuracy，span-micro 只 descriptive。10,000 次 cluster bootstrap、paired wins/losses、group-aware label-swap、分层 CI、cost frontier 和 8-cell intersection 已写入 compact CSV/JSON/Markdown；禁止行级独立显著性。
- `P4.1-EXECREPOBENCH`：首个外部 benchmark 已固定为 ExecRepoBench。pinned provenance 为 dataset `fa61028ce495c9ceff58398b8a7c47b5ae9f5276` 与 Qwen evaluator `33bc6aabd7791ad7b32f7e92104f11f2359ba890`；`experiments/p4_execrepobench_audit.py` 已准备只输出无代码的 repository-grouped six-fill smoke plan。实际数据/evaluator checkout 的 host 网络 fetch 在 2026-07-13 被 approval control plane `422` 阻断，故 evaluator smoke 尚未宣称完成；最终 external result 仍等方法配置冻结后才可开。

### M1.1 Abductive Program-State Bridge

两轮语义/完整性审计与修复已在 `1d9ef3f` 完成：AST/def-use dependency cone、seed-0 fixed64 检测、64-forward null fallback、等 canvas/forwards/cardinality generic、分离 raw outputs、forbidden-input/activation/determinism/resume tests 均已落地。**历史 5079-case MultiLine M1** 已在用户安全中断后成为 `safely_paused_resumable`：原三个 raw 目录 append-only 保留，不读取部分性能，未来仅胜出方法可用原目录与 `--auto-full --selected-method-only-5079` 的 existing-key dedup 恢复。

新的独立 **M1 RandomSpanLight** 已完成 raw 路线 `stage1=1332/1332`、generic=`148/148`、dependency-cone=`148/148`，各自 unique 且 0 duplicate/error；状态为 `randomspanlight_raw_complete_outcome_analysis_pending`。历史 5079 保持 `safely_paused_resumable`，不得 resume、重启或改写 raw。先 commit outcome-blind fair analyzer，再读固定 148-case result。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：已完成独立 `12→148`，full integrity audit 为三臂各 `148/148`、0 missing/extra/duplicate/error、64 forwards/4096 token-forwards、frozen sealed/count=0。正式 grouped result 为 `constraints_activated_no_reliable_grouped_advantage`：schedule 改变候选 hash，但没有可靠 task-group 优势；不自动进入 296/927/5079，也不阻止 M3。详见 `docs/paper_agent/experiments/m2_constraint_homotopy_20260716_randomspanlight_result.zh.md`。
- `M3 Birth-Death Canvas Diffusion V0`：行为保持 pre-launch 修正已在 `8440af1` 完成：删除无效 scalar token penalty，普通 confidence 只用于 within-particle remask，syntax/obligation/contradiction 只用于 particle ranking、birth/death 与 selection。M3 已自然完成独立 `12→148` technical full；两臂各 `148/148`、0 duplicate/error、256 forwards，且只声称 equal-forward，不声称 equal-token。状态为 `randomspanlight_generation_and_technical_audit_complete_outcome_analysis_pending`；先 commit fixed grouped analyzer，再读结果。
- `M4 Semantic Particle Assembly V0`：`ed94471` 已完成独立 runner/analysis/brief/tests/launcher；完整 148-case offline structural assembly audit 已实际完成于 `analysis_outputs/m4_semantic_particle_assembly_20260713_v0/offline_summary.json`（best/assembly 均 148/148、zero missing/duplicate/error、frozen sealed/count 0）。从 8 candidates 提取 AST statement/basic-block/def-use fragments，以 inference-visible obligations 选择；M3 资源安全完成后作 64-forward repair smoke/full。

四个 V0 的数据路线固定为：12 technical smoke → 148 RandomSpanLight → 296 MultiLine-Core / 927 non-frozen SingleLine development comparison → selected-method-only 5079 MultiLine → method freeze 后 ExecRepoBench。裸 `--auto-full` 不得直接进入 5079。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

旧共享候选库、M2 与 M3 已完成；本轮不启动 M4、official CAL 或 DreamOn。M3 启动期间曾短暂出现两次 `nvidia-smi` driver communication failure，随后恢复；最终 H200 空闲 `120054 MiB`、ECC=0，且三个新方法没有记录到 OOM。每个方法、每个进程均保持独立 output/log directory。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。

## 2026-07-17 Execution Sprint V1 current state

`codex/ccfa-execution-sprint-v1`（base `ce416c4`）是权威执行分支；`afd3c45` 已 superseded。M1 5079 MultiLine 已安全暂停、可恢复但不自动恢复；M1 RandomSpanLight 新 runner、M2/M3/M4 独立 runner、共同 grouped analyzer 与 CAL adapter 已在 `9335d84` 后继续实现。完整 CPU tests 已通过，frozen test 仍为 sealed/count=0。

真实 GPU 执行已完成 M2 与 M3 的 `12→148` 路线；M1 RandomSpanLight raw 也已完整写入但尚未分析 outcome。official CAL 正在按 4,990 common subset 的 fail-stop/resume hardening 后启动 smoke；SciPy 的审批失败只能触发 pinned import-closure 审计，不把研究判为 blocked。
