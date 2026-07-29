# 2026-07-29 实验完成度、结果判断与下一阶段建议

## 结论摘要

截至 2026-07-29，Execution Sprint V1 已授权的长任务已经全部结束，当前没有运行中的项目 GPU 进程。M1--M4 都完成了当前预注册的 148-group 技术运行与正式 grouped analysis，但没有一个满足 296/927/selected-only-5079 的晋级条件；这些扩展没有运行是科学 gate 主动止损，不是遗漏。

official CAL 的 12-case smoke 和 4,990-case full generation/evaluation pipeline 已完整结束，`4990/4990`、missing/error=`0/0`，但最终 outcome analyzer 尚未在解盲前冻结，因此目前只能确认技术与数据完整性，不能判断 CAL 的最终准确率是正向还是负向。其他 official reproduction 中，DreamOn 只完成官方源码/协议审计，尚未运行 GPU smoke/full；rho-EOS 当前官方实现是 completion-only，不能忠实冒充 infilling baseline；LR-DLLM 缺少可执行 Stage I/II 细节；ExecRepoBench 的 pinned audit 已准备，但真实 checkout/evaluator smoke 尚未完成。

当前没有可直接作为 paper primary 的新方法。已有证据最适合支持“严格协议下的 failure taxonomy、成本/准确率边界和跨 backbone 诊断”，还不足以支持强 CCF-A 方法论文的核心正向 claim。下一阶段应优先完成 official CAL 最终分析、DreamOn official-source reproduction 和 ExecRepoBench evaluator smoke；若继续方法研究，应新建独立 preregistration，而不是在 M1--M4 的已见 outcome 上调参、融合或恢复 5,079。

## 一、M1--M4 完成状态与正式结果

统一状态源为 [`method_portfolio.current.json`](../method_portfolio.current.json) 和 [`runtime_status.current.json`](../runtime_status.current.json)。四条路线的 frozen controller test 均保持 `sealed`，`test_evaluation_count=0`。

| 方法 | 当前阶段是否完成 | 公平主要比较 | 正向性判断 | 科学决定 |
|---|---|---|---|---|
| M1 Abductive Program-State Bridge | 是；stage-one `1332/1332`，generic/full 各 `148/148` | full − equal-compute generic，均为 576 forwards | `−0.68pp`，CI `[−2.03, 0.00]pp`，help/harm=`0/1`；cone 仅激活 `17/148` | `reviewed_not_promoted_v0`；不进 296/927/5079 |
| M2 Constraint-Homotopy | 是；vanilla/gradual/abrupt 各 `148/148` | gradual − vanilla，均为 64 forwards | `+1.35pp`，CI `[−4.05,+6.76]pp`，help/harm=`9/7`；机制激活但长分桶回退且 grouped evidence 不可靠 | `constraints_activated_no_reliable_grouped_advantage`；不晋级 |
| M3 Birth-Death Canvas Diffusion | 是；uniform/birth-death 各 `148/148` | birth-death − uniform，均为 256 forwards | `−4.05pp`，CI `[−10.14,+2.03]pp`，help/harm=`8/14`；token-forward 降约 24.9%，但 accuracy 也下降 | accuracy/efficiency gate 均失败；不进 296/927/5079 |
| M4 Semantic Particle Assembly | 是；best/assembly/repair 各 `148/148` | assembly-without-repair − best-single，均为 512 forwards | `−12.84pp`，CI `[−18.24,−7.43]pp`，help/harm=`0/19`；hash 改变 `83/148`，说明机制激活但明显有害 | `reviewed_not_promoted_v0_multilinecore_cross_source`；不调参、不融合、不扩展 |

正式结果文件：

- [M1 RandomSpanLight 正式结果](m1_abductive_program_state_bridge_20260717_randomspanlight_result.zh.md)
- [M2 RandomSpanLight 正式结果](m2_constraint_homotopy_20260716_randomspanlight_result.zh.md)
- [M3 RandomSpanLight 正式结果](m3_birth_death_canvas_20260717_randomspanlight_result.zh.md)
- [M4 MultiLine-Core 正式结果](m4_semantic_particle_assembly_20260728_multilinecore_result.zh.md)

### 如何理解四条路线

- M1 是弱负/近似持平，但其 targeted mechanism 覆盖过低，且没有胜过同预算 generic refinement。不能用 fixed64 对 576-forward full 的计算差异制造正向表述。
- M2 是四条路线中唯一公平点估计为正的 V0，但不稳定、CI 跨零、p 值弱且长分桶明显回退。它可以作为“promising but uncertain premise”记录，不能作为当前 primary 或直接扩展。
- M3 确实节省 token-forward，也确实发生 birth/death，但预注册的 efficiency gate 要求 accuracy 点估计不低于 uniform；当前不满足。因此不能宣称正向效率结果。
- M4 不是 no-op，而是强负结果。M4 使用 40,632-row MultiLine bank 的 outcome-blind 148-group manifest，与 M1/M3 RandomSpanLight 属于 cross-source；不能把四个 delta 当作完全 apples-to-apples 排名，但 M4 自身的公平比较已足以停止 V0。

若只按各自公平 primary gate 判断，结论不是“从四个里勉强选一个”，而是“无人合格”。因此没有 paper primary，也没有自动启动 296/927。

## 二、未运行的 M-series 扩展是否算未完成

| 路线 | 状态 | 原因 |
|---|---|---|
| 296 MultiLine-Core | 未运行 | M1--M4 均未过预注册 148 promotion gate |
| 927 non-frozen SingleLine development | 未运行 | 只有 296 仍正向才允许进入；当前不存在候选 |
| selected-method-only 5,079 MultiLine | 未运行 | 只有 296 和 927 均保持正向后才允许；当前没有 selected method |
| 历史 M1 5,079 | 部分运行后安全暂停 | stage-one `27217/45711`，generic/dependency-cone 各 `12/5079`；raw append-only、未读部分性能、可 resume，但按当前科学 gate 不应恢复 |

因此，“M1--M4 系列是否完成”的准确回答是：四个 V0 的决策阶段已全部完成；更大开发扩展按预注册规则被主动取消。恢复历史 M1 5,079、直接补跑 296/927 或融合四条方法，都会构成 outcome-driven scope change。

## 三、official reproduction 与外部 benchmark

| 项目 | 技术/协议状态 | outcome 状态 | 是否完成 | 下一步 |
|---|---|---|---|---|
| official CAL | pinned CAL `741e8418`、evaluator `88062ff`；smoke `12/12`，full `4990/4990`，0 missing/error | final outcome analysis 尚未打开 | 运行完整；科学结论未完成 | 先冻结 analyzer，再读取 passed outcome并形成 formal report |
| DreamOn | official source `8a0a549` 已审计；存在真实 FIM/HumanEval-Infilling dynamic-expansion route | 未运行本地 GPU smoke/full | 未完成 | pin evaluator，明确 official 8-GPU 与单 H200 拓扑差异，先做 12-case technical smoke |
| rho-EOS | official source `69992ca` 已审计；当前是 completion-only | 无 faithful FIM result | 当前不可执行为 official infilling reproduction | 只有出现/定义可审计 faithful FIM protocol 后再考虑 GPU |
| LR-DLLM | 文献/协议 blocker audit 已完成 | 无可执行 Stage I/II adapter/result | 未完成且真实 blocked | 等官方代码或足够精确的算法协议；不得自造“LR-DLLM”结果 |
| ExecRepoBench | dataset `fa61028c`、Qwen evaluator `33bc6aab` 的 pinned audit/scaffold 已准备 | checkout/evaluator smoke 未完成 | 未完成 | 重试 host checkout、license/schema audit 和 six-fill evaluator smoke；没有 primary 前不开 final benchmark outcome |

official CAL 的准确标签只能是 **official CAL reproduction on the 4,990-case non-frozen common subset**，不能写成完整 5,715-case paper-number reproduction。其 final outcome 未分析不等于失败，也不等于正向。

DreamOn/rho-EOS 的协议审计见 [`analysis_outputs/p1_dreamon_rhoeos_protocol_audit_20260728_v1/report.zh.md`](../../../analysis_outputs/p1_dreamon_rhoeos_protocol_audit_20260728_v1/report.zh.md)，LR-DLLM blocker 见 [`lrdllm_protocol_audit.zh.md`](../lrdllm_protocol_audit.zh.md)。

## 四、其他已经完成的主要 full/大规模证据

以下项目都已完成，但必须与 official reproduction 分开表述。

| 证据块 | 完成情况 | 结果与强度 |
|---|---|---|
| Shared MultiLine candidate bank | `40,632/40,632 = 5,079×8`，0 duplicate/error，只读保留 | 基础设施与候选库完成；不是独立方法胜利 |
| Phase 5 RandomSpanLight bank/premise audit | base `1332/1332`，alpha `728/728` | deterministic combined proxy 有排序信号，但 corrected grouped gate 失败；conditional V0 被 kill |
| P2.1 grouped statistics | `6,707` spans / `20,121` results / `148` task groups | 10,000 cluster bootstrap、paired/group-aware statistics 完成；是统计基础设施，不是新方法结果 |
| H200 LLaDA-Base core reruns | 1,033-row Control/Midcons/Route2/V6/Local-CAL 全部完成 | Control `787`、Midcons `794`、Route2 `795`、V6 `796`、Local CAL `769`；存在 hardware outcome drift，不能与 A6000 静默合并；V6 仅小幅本地正向 |
| Dream-Coder full allowed SingleLine diagnostic | non-frozen `927` cases、3 policies=`2781` rows、frozen=0 | primary/simple/oracle=`735/744/858`；simple help/harm=`25/16`，说明小幅 policy gain 和显著 oracle canvas ceiling，但仍是 second-backbone diagnostic |
| 1,033-row local same-backbone matrix | 多个 backbone 的 baseline/candidate pair 均完整 | DreamCoder-Base `+7` tasks；LLaDA-MoE `+24` tasks 为最强本地 transfer；Dream/DiffuCoder/LLaDA-1.5 各约 `+1`；LLaDA-Instruct `−2`、DreamCoder-Instruct `−14`。这些是 local comparisons，不是对应论文 official reproduction或 external SOTA |
| Trace/controller/action-bank full diagnostics | 1,033-row traces、927-task action bank、validation controller audits 完成 | 多数 route/controller gate 失败或只弱正向；支持 failure taxonomy，不支持开启 frozen test 或宣称 deployable controller success |

历史 local full 的详细数值和边界见 [`experiment_results.zh.md`](../experiment_results.zh.md) 与 [`evidence_snapshot.md`](../evidence_snapshot.md)。目前最值得保留的正向证据是 LLaDA-MoE local same-backbone `801/1033` 对 `777/1033`（`+2.32pp`，31 wins/7 losses），但它更慢，且没有 protocol-matched external baseline，不能替代 paper primary。

## 五、当前论文就绪度

### 已有优势

- 实验完整性、resume/dedup、forward/token ledger、group-aware statistics 和 frozen-test discipline 较强。
- 已经形成多条有信息量的负结果，而不是只有工程失败：M1 稀疏激活、M2 不稳定、M3 accuracy/cost trade-off、M4 明显 harm。
- 有跨 backbone local transfer matrix，其中 LLaDA-MoE 是较明确的本地正向结果。
- official CAL 大规模 raw 已完整生成，离正式结论只差 outcome-blind analyzer freeze 和分析。

### 仍缺的关键证据

- 没有一个通过开发 gate 的 paper primary method。
- official CAL 尚无 final performance report；DreamOn 尚无 protocol-matched本地结果。
- 没有完成 external benchmark evaluator smoke/final result。
- frozen test 始终未开；在没有稳定 primary 与完整 baseline pack 前也不应开。

结论：当前可以写一篇较强的诊断/负结果与评测方法论文草稿，但若目标是以“新方法显著提升 infilling”为中心的 CCF-A 投稿，仍需要新的独立方法设计和至少一条稳定、同预算、跨数据/跨 backbone 的正向证据。

## 六、是否继续优化 M1--M4

不建议在当前 148 outcome 上直接调 M1--M4 的阈值、轮次、canvas、seed、selector 或进行融合。这样会破坏已有 outcome-blind 证据，且容易把噪声写成方法进步。

可以继续研究，但应采用“新版本、新机制、新 preregistration、新开发 manifest”的方式：

1. 先用 official CAL final analysis 判断动态长度方法在 4,990 common subset 上的真实优势、失败桶和成本。
2. 用 DreamOn official-source smoke/full 判断 training-based dynamic canvas 的可复现上界与资源代价。
3. 基于现有 failure taxonomy，重新设计一个同时改善 candidate existence 与 risk-controlled selection 的方法；不要只增加 remask heuristic。
4. 在读取新方法 outcome 前固定 fair baseline、统计单位、成本、activation 与 promotion gate；先 12→148，再决定 296/927。

M2 的正点估计和 M3 的 token saving 可以作为新设计的 hypothesis source，但不能作为继续调 V0 的依据。M4 的强负结果建议停止该 assembly V0 family，除非未来提出本质不同、可预注册的新 assembly mechanism。

## 七、推荐的下一阶段顺序

推荐按以下顺序审批执行：

1. **CPU 优先：official CAL final analyzer freeze 与正式结果。** 先审计 4,990 common subset 上可用的同协议 controls，提交 analyzer/tests，再读取 outcome；若没有同 subset control，只报告 absolute accuracy/cost，不能虚构 paired improvement。
2. **CPU/network：ExecRepoBench checkout/evaluator technical smoke。** 只验证 provenance、schema、license、repository grouping 和 six fill types；没有 primary 前不打开 final benchmark result。
3. **GPU：DreamOn 12-case official-source technical smoke。** 先固定 evaluator、manifest、seed、模型与拓扑标签。若单 H200 只改变并行度且算法/数据/evaluator不变，可在 smoke 后按明确标签进入 full；若会改变算法或需要官方 8-GPU 才可运行，则安全停止并记录精确 blocker。
4. **方法研究：新 M5 brief/preregistration。** 只做设计与 CPU feasibility，待 baseline evidence 齐全后再审批 GPU；不恢复 M1 5,079，不跑旧 V0 的 296/927。

可直接审批的执行 prompt 见 [`20260729_next_stage_approval_prompt.zh.md`](../prompts/20260729_next_stage_approval_prompt.zh.md)。

## 八、可复现性与仓库状态

- 权威分支：`codex/ccfa-execution-sprint-v1`。
- 本报告审计起点 HEAD：`a0e460f651a0fc426b121bef5084415cb158068b`。
- raw `outputs_clean/`、logs、模型权重和 trace-heavy artifacts 未提交或改写。
- 用户已有未跟踪目录 `analysis_outputs/m1_randomspanlight_20260715_v1/` 与 `analysis_outputs/m2_constraint_homotopy_20260715_sprint_v1/` 保持未改动。
- frozen controller test=`sealed`，`test_evaluation_count=0`。
