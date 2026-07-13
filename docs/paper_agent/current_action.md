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

两轮语义/完整性审计与修复已在 `1d9ef3f` 完成：AST/def-use dependency cone、seed-0 fixed64 检测、64-forward null fallback、等 canvas/forwards/cardinality generic、分离 raw outputs、forbidden-input/activation/determinism/resume tests 均已落地。真实 12-case MultiLine smoke 已准备，但当前 host GPU launch/审计请求被外部 approval control plane 的 `422 model not found: codex-auto-review` 拒绝；未创建新 PID、未触碰 PID `1195368` 或其输出。control plane 恢复后立即按 `scripts/manual_launch_phase6_multiline_candidate_bank.sh` 启动；技术通过自动继续全部 `5079` non-frozen spans，不设性能 gate。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：`4a91d73` 已完成独立 runner/analysis/brief/tests/launcher；同一 64-forward 预算下 gradual 与 abrupt constraints，约束只来自 prefix/suffix/current candidate/confidence。GPU control plane 恢复后技术 smoke 自动继续 148-case full。
- `M3 Birth-Death Canvas Diffusion V0`：`a874c54` 已完成独立 runner/analysis/brief/tests/launcher；初始 canvas `16/32/64/128` particles，uniform 与 birth/death 各固定 256 forwards/task，只用 inference-visible confidence、syntax、prefix/suffix compatibility。GPU control plane 恢复后先 12-case 再 148-case full。
- `M4 Semantic Particle Assembly V0`：`ed94471` 已完成独立 runner/analysis/brief/tests/launcher；完整 148-case offline structural assembly audit 已实际完成于 `analysis_outputs/m4_semantic_particle_assembly_20260713_v0/offline_summary.json`（best/assembly 均 148/148、zero missing/duplicate/error、frozen sealed/count 0）。从 8 candidates 提取 AST statement/basic-block/def-use fragments，以 inference-visible obligations 选择；GPU control plane 恢复后作 64-forward repair smoke/full。

四个 V0 的数据路线固定为：12--24 technical smoke → 148 RandomSpanLight first full → 927 non-frozen SingleLine development comparison → 5079 MultiLine later validation → method freeze 后 ExecRepoBench。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

候选库保持运行；先启动一个新的 GPU experiment process，10 分钟后审计 memory/utilization/power/OOM/ECC 与各任务吞吐。稳定且保留至少 25 GiB 显存余量时才允许第三个进程；总吞吐显著下降则减少 GPU process，但不停止方法代码开发。每个方法、每个进程必须有独立 output/log directory。

当前仅 host execution approval control plane 临时不可用；这不改变 `iterate_and_execute`、不构成任一方法的科学结果，也不授权绕过。代码、tests、launch scripts 和独立输出路径已准备，待该外部状态恢复再执行 GPU jobs。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。
