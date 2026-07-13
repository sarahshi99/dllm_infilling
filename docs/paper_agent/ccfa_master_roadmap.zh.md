# CCF-A Master Roadmap（唯一权威路线）

更新时间：2026-07-13 UTC
证据基线：`8b348f979f09cda07811e04b7ad3dee60e56373b`
目标：CCF-A/FSE 级论文，而不是继续堆叠零散实验编号。

## 1. 本文件的权威性

后续 Codex 恢复研究时，按以下顺序读取和执行：

1. 本文件：决定研究主线、命名、优先级和不可跨越的证据边界；
2. `current_action.md`：当前最小可执行动作；
3. `experiment_queue.md`：已经选中的实验；
4. `idea_board.md`：尚未进入执行队列的方法假设；
5. `decision_log.md` 与具体实验报告：保存历史结论。

历史 Phase、EXP、V1--V8 编号只用于追溯，不得覆盖本文件。若其他文档与本文件矛盾，以本文件和最新 verified result report 为准，并把矛盾视为研究完整性问题。

## 2. 当前最强但受限的论文结论

当前证据支持：unknown-length DLLM code infilling 中，`canvas adequacy`（空间是否足够）和 `rescue adequacy`（给足空间后当前生成过程是否能产生正确内容）应分开分析；oracle-sufficient canvas 可以恢复一部分失败，但当前可部署的长度选择和 rescue 策略容易造成 harm。

当前证据不支持：

- SOTA 或正式正方法结论；
- frozen-test improvement；
- “oracle canvas 没救回就证明 backbone 无能力”的因果表述；
- 把 `802/1033`（A6000 historical）或 `796/1033`（H200 evidence base）当作严格 held-out 性能；
- 把 `6707` 行当作 `6707` 个独立任务；
- 把 local CAL/CAL-lite 当成官方 CAL 复现；
- 把 Phase 5 fixed AST/def-use proxy 当成完整 Semantic Bridge 方法。

“oracle canvas 没救回”只表示在当前 model、seed、decoding algorithm、oracle-length definition、generation budget 和 evaluator 下，指定 action 没有救回。backbone limitation、generation limitation 和 evaluator limitation仍是竞争解释。

## 3. Track A：论文可信度与 CCF-A 缺口

| ID | 工作 | 当前状态 | 已有证据 | 关闭条件 |
|---|---|---|---|---|
| `P0-NARRATIVE` | 统一仓库叙事 | `closed_for_current_snapshot` | official second-regime、Dream-Coder full、Phase 5 result 已存在 | 每个 decision-bearing result 后同步 readiness、claim、paper、table、queue 和 ledger |
| `P1-BASELINES` | 协议匹配强 baseline | `open_highest_priority` | local controls 和 literature anchors 存在，但不是正式复现 | 顺序固定为 official CAL -> DreamOn -> rho-EOS compatibility；CAL technical smoke 通过后直接 full；DreamOn 为 training-based stratum；rho-EOS 只有忠实 infilling adaptation 才进入 smoke/full；LR-DLLM 仅保留 blocker audit |
| `P2-GROUP-STATS` | 6707-row group-aware statistics | `executing` | row-level totals 已完成；Phase 5 另有 148-group bootstrap | primary estimand 是每个 base task 内先算 policy accuracy、再对 task 等权平均（equal-weight base-task macro accuracy）；span-micro rate 仅 descriptive；以 `HumanEval/<id>` 为 cluster，补 10,000 fixed-seed bootstrap、paired wins/losses、group-aware label-swap、分层 CI、cost frontier 和 help/harm/oracle 交集 |
| `P3-SELECTIVE` | 真正两阶段/三头 selective controller | `deferred_until_official_CAL` | V1--V3 是旧 validation route closure，不等于该新设计 | official CAL 完成后再决定是否启动；只可使用新数据源和 grouped split；canvas/rescue/harm 分头；禁止 V4 式阈值续调 |
| `P4-EXTERNAL` | 非 HumanEval、真实软件工程外部评价 | `executing_ExecRepoBench_first` | MultiLine/RandomSpan 仍是 HumanEval 变体 | 首选且唯一当前首发 benchmark 为 ExecRepoBench；固定 dataset/Qwen evaluator commit、许可和字段；按 repository cluster；M1 配置冻结前只做多 repo、六类 fill_type 的 evaluator smoke，不打开最终 external result |

### P1 baseline 的统一协议表

每个 baseline 必须记录：official repository 和 commit、是否 training、模型/checkpoint、dataset/split、prompt/FIM format、candidate lengths 或初始 canvas、decode steps、实际 forward 次数、seed、temperature、evaluator、wall-clock、GPU、峰值显存、是否使用 oracle，以及与本项目结果属于 `apples_to_apples`、`partially_matched` 还是 `literature_only`。

截至 2026-07-13 的 source check 与执行顺序：

- official CAL 是第一条执行 baseline：固定 upstream commit，审计 model、prompt、dataset mapping、canvas、steps、forward count、seed、evaluator、wall time、GPU 和 memory；只在 exact current-manifest mapping 或明确交集比较；12-case technical smoke 通过后自动跑完整 allowed population。local CAL/CAL-lite 永远不能改名为 official CAL。
- DreamOn 是第二条执行 baseline：固定 official repo/checkpoint/训练与推理协议，作为 training-based stratum 单列；official CAL full 完成后才启动其 official full。
- rho-EOS 排在 DreamOn 后：先做 infilling compatibility audit；只有 faithful adaptation（不把 completion quick-start 偷换成 FIM）才进入 smoke/full。
- LR-DLLM 当前没有可执行 official/local Stage I/II adapter；状态为 `paper_audited_code_or_adapter_blocked`，本轮只维护 blocker audit，不造 adapter、不伪造复现。

### P2 的独立统计单位

Full allowed second-regime 有 `6707` span rows、`20121` policy-result rows，但只有 `148` 个 allowed HumanEval base-task groups。primary estimand 固定为：每个 base task 内对属于该 task 的 span 先计算 policy accuracy，再对 `148` 个 tasks 等权平均。span-micro pass rate 只作 descriptive。所有不确定性、paired wins/losses 与 permutation 都以 base task cluster 为单位；不得将行级 `n=6707` 用作独立样本量或行级显著性检验。

早期 `1033` rows 经 V1--V8、Route2、threshold 和 selector 多轮观察，统一标记为 `development / mechanism-discovery population`。`802/1033` 是 A6000 historical 研究结果；`796/1033` 是 H200 evidence-base 结果。二者均可保留为各自硬件/协议下的历史证据，但都不能作为最终 held-out method performance，且不能静默混合。

## 4. Track B：四个独立方法与一个辅助诊断

四个方法使用固定名称，不再随 Phase 改名：

| Canonical ID | 正式名称 | 当前状态 | 已完成的只是 | 下一独立实验 |
|---|---|---|---|---|
| `M1` | Abductive Program-State Bridge | `implementation_started` | `M1-D0` fixed AST/def-use proxy；该公式在 Phase 5 gate 下被 kill | 立即实现 suffix backward obligations、prefix/candidate forward facts、contradiction set、dependency-cone targeted remasking 与 safe fallback；12-case MultiLine technical smoke 后直接 full `5079` non-frozen MultiLine spans |
| `M2` | Constraint-Homotopy Infilling | `cheap_diagnostic_not_prerequisite` | 只有文献/novelty 登记 | 可独立保留 cheap diagnostic，但不是 M1 implementation/full 的前置条件 |
| `M3` | Birth--Death Canvas Diffusion | `cheap_diagnostic_not_prerequisite` | 只有文献/novelty 登记 | 可独立保留 cheap diagnostic，但不是 M1 implementation/full 的前置条件 |
| `M4` | Semantic Particle Assembly | `deprioritized` | `M4-D0/F1`：61 个 all-fail tasks 中仅 6 个有互补正确 semantic units | 降级；不在本轮抢占 M1。仍不得使用 tests/reference 选择片段 |
| `A1` | Metamorphic Equivariance Evaluator | `stopped_negative_auxiliary` | 728 pairs/91 tasks，delta AUC `0.0143`，CI 跨 0 | 停止新工作；保留为负诊断证据，不进入 fusion |

重要命名边界：

- `AST/def-use bridge proxy V0` = `M1-D0`，不是 M1 完整方法；
- F1 = `M4-D0` premise diagnostic，不是 Semantic Particle Assembly；
- F2 = `A1` auxiliary diagnostic，不是 correctness oracle；
- F3/F4 = `M1-D0` feature/ranking audit，不是四个方法都已实现；
- `P3-SELECTIVE` 是基于旧 controller 证据的新 selector 路线，单独记账，不叫 Controller V4，也不混入 M1--M4。

## 5. 哪些工作可以并行

可以并行：

- P1 official baseline protocol audits/adapters；
- P2 对已有 6707 rows 的 group-aware CPU analysis；
- P4 外部 benchmark feasibility、dataset/evaluator audit；
- M1 full implementation 与 P1/P2/P4；M2/M3 cheap diagnostics 不构成 M1 前置门；
- 不同 GPU、独立输出目录、固定协议下的 baseline full runs。

并行指实验作业，不授权 subagent。每个方法必须有独立输入、输出、指标和结论，不能因另一个方法的结果而临时改公式。

必须串行或在结果后重新冻结：

- 根据同一 development outcomes 修改方法后再把同一数据称为验证；
- 方法 fusion；
- frozen test；
- 会抢占相同 GPU、覆盖输出或改变 evaluator 的运行。

## 6. 新的 gate 规则

探索阶段只有技术/完整性 gate：runner、schema、evaluator、无 frozen rows、无 forbidden inputs、resume/dedup、硬件安全。技术 smoke 通过后，成本可承受的实验直接 full；不得再要求“先显著优于所有 baseline 才允许实现”。

论文 promotion 阶段才使用性能 gate：fresh validation、group-aware uncertainty、compute-matched baselines、help/harm、cost 和跨 dataset/model transfer。Frozen test 只在方法、配置、claim 和统计计划全部冻结后运行一次。

## 7. 当前执行顺序

三条工作流并行推进，但互不替代：

1. `P1-BASELINES`：official CAL full -> DreamOn official full -> rho-EOS infilling compatibility（忠实适配才 smoke/full）；LR-DLLM 只做 blocker audit。
2. `P2-GROUP-STATS` 与 `P4-EXTERNAL`：立刻并行 CPU grouped statistics 和 ExecRepoBench 版本/许可/字段/evaluator audit；P4 不再做泛泛三候选调研。
3. 方法线：M1 full implementation 立即开始；M2/M3 cheap diagnostics 不是前置条件；M4 降级、A1 停止；P3 等 official CAL 后再决定。M1 smoke 技术通过后直接 full，性能 gate 仅服务于论文 promotion。

P1/P2/P4 未关闭前，可以做探索实验，但任何方法结果都不得宣称 CCF-A-ready 或 SOTA。外部 baseline、严谨统计和真实场景不能被新方法实验替代。
