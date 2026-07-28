# Current Paper-Agent Action

更新时间：2026-07-28 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`；M1--M4 唯一当前状态登记：`docs/paper_agent/method_portfolio.current.json`。

<!-- method-portfolio-status: M1=reviewed_not_promoted_v0; M2=reviewed_not_promoted; M3=reviewed_not_promoted_v0; M4=offline_148_complete_gpu_repair_pending_launch_authorized -->

## 2026-07-28 当前最小动作（覆盖历史 action）

`M4 Semantic Particle Assembly` repair 已获启动授权。official CAL 的 4,990-case full 已完整性完成（`4990/4990`、missing/error=`0/0`；运行期间未读取 partial accuracy），故不再把其历史 t+10 条件误写为当前阻塞。H200 当前 ECC=`0`、仅一个无关 GPU process、空闲约 `130GiB`；以第二研究 GPU process 启动固定 launcher 的 `12-case smoke → automatic 148-case full`。M1/M2/M3 的 148-group 正式结论保持不晋级，M1 5079 保持暂停，当前没有 paper primary。若 M4 technical/full integrity 通过，立即运行预先提交的 grouped analyzer；若 primary `assembly_without_repair vs best_single` 不满足 point delta `>0` 且 help>harm，则不进入 296/927。

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

新的独立 **M1 RandomSpanLight** 已完成并正式复核：stage1=`1332/1332`、generic=`148/148`、dependency-cone=`148/148`，各自 unique 且 0 duplicate/error。预冻结 full-vs-generic fair delta=`-0.68pp`、help/harm=`0/1`，cone targeted activation=`17/148`；结论为 `reviewed_not_promoted_v0`，不得用本 outcome retune 或进入 296/927/5079。历史 5079 继续 `safely_paused_resumable`。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：已完成独立 `12→148`，full integrity audit 为三臂各 `148/148`、0 missing/extra/duplicate/error、64 forwards/4096 token-forwards、frozen sealed/count=0。正式 grouped result 为 `constraints_activated_no_reliable_grouped_advantage`：schedule 改变候选 hash，但没有可靠 task-group 优势；不自动进入 296/927/5079，也不阻止 M3。详见 `docs/paper_agent/experiments/m2_constraint_homotopy_20260716_randomspanlight_result.zh.md`。
- `M3 Birth-Death Canvas Diffusion V0`：行为保持 pre-launch 修正已在 `8440af1` 完成。M3 现在已正式复核：birth-death − uniform task-macro=`-4.05pp`、help/harm=`8/14`；birth/death mechanism 确实激活且 mean token-forward 更低，但 accuracy point estimate 也更低。结论为 `reviewed_not_promoted_v0`，不进入 296/927/5079；仅保留 equal-forward/non-equal-token 的成本观察。
- `M4 Semantic Particle Assembly V0`：offline 148 structural audit 已完成；cost/activation analyzer 已在 `267dda5` 加固。official CAL 已完成完整性 full；2026-07-28 H200 audit 满足 ECC=0、预计加载后 free>=25GiB、且 M4 是第二研究 GPU process，故启动独立 repair `12→148`。不覆盖既有 raw；输出保持独立 append-only。

四个 V0 的数据路线固定为：12 technical smoke → 148 RandomSpanLight → 296 MultiLine-Core / 927 non-frozen SingleLine development comparison → selected-method-only 5079 MultiLine → method freeze 后 ExecRepoBench。裸 `--auto-full` 不得直接进入 5079。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

旧共享候选库、M2 与 M3 已完成；official CAL full 也已完整性结束。本轮启动 M4，DreamOn 仍不在本轮范围。H200 当前空闲约 `130GiB`、ECC=0；一个无关训练作业不中断且 M4 仅作为第二研究 process。每个方法、每个进程均保持独立 output/log directory。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。

## 2026-07-17 Execution Sprint V1 current state

`codex/ccfa-execution-sprint-v1`（base `ce416c4`）是权威执行分支；`afd3c45` 已 superseded。M1 5079 MultiLine 已安全暂停、可恢复但不自动恢复；M1 RandomSpanLight 新 runner、M2/M3/M4 独立 runner、共同 grouped analyzer 与 CAL adapter 已在 `9335d84` 后继续实现。完整 CPU tests 已通过，frozen test 仍为 sealed/count=0。

真实 GPU 执行已完成 M1/M2/M3 的 148 路线并正式复核；M1/M2/M3 当前都不晋级。official CAL 的 first import fail-stop 发现 pinned evaluator README 明示的 commented execution call；在不修改 checkout 的 documented one-line runtime overlay 后，12-case smoke=`12/12` integrity pass 并自动进入 4,990-case full。partial CAL accuracy 保持未读取；runtime/ETA/GPU 完整性在 `runtime_status.current.json` 记录。SciPy 的审批失败只触发 pinned import-closure 审计，不把研究判为 blocked。
