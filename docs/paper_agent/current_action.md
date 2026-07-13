# Current Paper-Agent Action

更新时间：2026-07-13 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`

起点证据：`8b348f979f09cda07811e04b7ad3dee60e56373b`

## Action Name

`FAST-SPRINT-01`: official baselines、P2.1 grouped statistics、四个独立候选方法 M1--M4、以及 ExecRepoBench preparation。

## 当前决定与边界

Operational decision 是 `iterate_and_execute`，不是 blocked。M1、M2、M3、M4 是平行的独立候选方法；当前**没有**论文主方法。M1 只因最先完成代码而先进入 smoke，不代表 M2--M4 被降级、放弃或融合。Phase 5 的 `M1-D0` fixed proxy kill 和 `M4-D0/F1` premise 结果保持为历史结论，不能被改写成对四个新 V0 的 kill。

冻结 controller test 始终 sealed，`test_evaluation_count=0`；不得读取 106 个 sealed SingleLine test rows。所有 full run 必须支持 resume/dedup，并在结束后审计 missing/duplicate/error。现有 MultiLine candidate-bank PID `1195368` 继续运行，不暂停、不删除输出。

## 立即执行的工作包

### P1/P2/P4

- `P1.1-OFFICIAL-CAL`：固定 official commit，审计 model/prompt/dataset mapping/canvas/steps/forwards/seed/evaluator/wall/GPU/memory；12-case technical smoke 后自动 full。DreamOn official full 是后续 baseline；rho-EOS 仅 faithful infilling compatibility 后 smoke/full；LR-DLLM 只保留 blocker audit。local CAL/CAL-lite 不得叫 official CAL。
- `P2.1-GROUPED-STATS`：`6707` spans / `20121` results 背后是 `148` base-task groups。primary 是 equal-weight base-task macro accuracy；span-micro 只 descriptive。保持 cluster bootstrap、paired wins/losses、group-aware label-swap、分层 CI、cost frontier 和 8-cell intersection；禁止行级独立显著性。
- `P4.1-EXECREPOBENCH`：首个外部 benchmark 已固定为 ExecRepoBench。固定 dataset/Qwen evaluator commit，审计下载/环境/许可/字段，按 repository group，并完成跨多个 repository 与六种 fill_type 的 evaluator smoke。最终 external result 等方法配置冻结后再开。

### M1.1 Abductive Program-State Bridge

先完成两轮语义/完整性审计与修复：AST/def-use dependency cone 不能只是 identifier 同名；candidate 缺失 suffix obligation 时执行 fixed64 的 64-forward null refinement；generic/M1 使用等 canvas、等 64 forwards、匹配 remask cardinality；stage-one、generic、M1 raw output 分离。随后运行真实 12-case MultiLine smoke；技术通过自动继续全部 `5079` non-frozen MultiLine spans，不设性能 gate。模型为 LLaDA-8B-Base，canvas `16/32/64/128`、seeds `0/1`、64 steps。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：同一 64-forward 预算下 gradual 与 abrupt constraints；约束只来自 prefix/suffix/current candidate/confidence。技术检查后直接 148-case RandomSpanLight full。
- `M3 Birth-Death Canvas Diffusion V0`：同时维护 canvas `16/32/64/128` particles，在固定 total-forward budget 下只用 inference-visible confidence、syntax、prefix/suffix compatibility birth/death；先 12-case smoke 再 148-case RandomSpanLight full。
- `M4 Semantic Particle Assembly V0`：从 8 candidates 提取 AST statement/basic-block/def-use fragments，以 inference-visible obligations 选择；比较 best-single、assembly-without-repair、assembly-with-repair；先 148-case bank offline assembly，再实际 fixed-budget repair。

四个 V0 的数据路线固定为：12--24 technical smoke → 148 RandomSpanLight first full → 927 non-frozen SingleLine development comparison → 5079 MultiLine later validation → method freeze 后 ExecRepoBench。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

候选库保持运行；先启动一个新的 GPU experiment process，10 分钟后审计 memory/utilization/power/OOM/ECC 与各任务吞吐。稳定且保留至少 25 GiB 显存余量时才允许第三个进程；总吞吐显著下降则减少 GPU process，但不停止方法代码开发。每个方法、每个进程必须有独立 output/log directory。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。
