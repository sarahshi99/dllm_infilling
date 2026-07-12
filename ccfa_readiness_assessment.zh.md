# CCF-A Readiness Assessment

更新时间：2026-07-12 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`

最新方法证据：`52f07457a7974fafa69d59914bbc412a2d83e305`

## 直接结论

当前工作仍不是 CCF-A submission-ready。问题不再是 second-regime 数据缺失，也不再是 Phase 5 没运行；真正的四个最大缺口是：协议匹配强 baseline、对 6707 span rows 的 base-task-group 统计、非 HumanEval 真实软件工程评价，以及一个在 fresh validation 上成立的正方法。

当前 framing 仍是 `diagnostic-driven mixed candidate`。Phase 5 为结构信号提供了正向点估计，但 fixed AST/def-use proxy 未通过预注册 gate；这不能升级为 standalone method claim，也没有否定完整 Abductive Program-State Bridge。

## P0--P4 完成度

| Priority | 状态 | 已完成 | 仍缺 |
|---|---|---|---|
| P0 仓库叙事统一 | `closed_for_current_snapshot` | central claim、readiness、paper skeleton、main table、queue、gap ledger 已按 2026-07-12 evidence 对齐 | 新结果后必须再次同步 |
| P1 强 baseline | `open_highest_priority` | fixed64、oracle、CAL-lite、本地控制；LR-DLLM blocker audit | official CAL、rho-EOS、DreamOn 的协议审计/可运行复现；统一 compute 表 |
| P2 6707 group statistics | `open_highest_priority` | row totals 与 config/bucket taxonomy；Phase 5 另有 148-group bootstrap | 6707 rows 按 148 base-task clusters 的 CI、paired test、cost frontier、intersection figure |
| P3 新 selective controller | `not_started` | 旧 V1--V3 route closure | fresh second-regime/cross-model grouped split 上的 canvas/rescue/harm heads；禁止 V4 阈值续调 |
| P4 外部评价 | `open_highest_priority` | HumanEval MultiLine/RandomSpan 与 Dream-Coder second backbone | 至少一个非 HumanEval、长函数/项目级、可执行的真实代码 infilling benchmark |

## 已核实的证据边界

- Early `1033` rows：经过 V1--V8、Route2 和多次 selector/threshold 观察，只能标为 development/mechanism evidence；`802/1033` 不是 held-out final result。
- Full allowed second-regime：`6707` span rows 来自 `148` 个 allowed HumanEval base-task groups。`2019/6707`、`1464/6707`、`3180/6707` 是描述性 row totals，论文显著性和 CI 必须按 base task 聚类。
- Official second-regime 数据已存在并完成 full allowed run；任何“官方 MultiLine/RandomSpan 数据仍缺失”的表述均已删除或标记为历史。
- Dream-Coder full allowed SingleLine 已完成，但仍属于 HumanEval 和 second-backbone diagnostic，不等于真实软件工程外部验证。
- Frozen controller test 仍 sealed，`test_evaluation_count=0`。

## Phase 5 的准确结论

- Full bank：`1332/1332` over `148` tasks；alpha auxiliary `728/728` over `91` tasks；无 missing/duplicate/error/frozen。
- M1-D0 combined proxy cross-canvas pairwise accuracy `0.6273`，选择相对 fixed64/confidence 的 net 为 `+15/+14`。
- 预注册的 all-baseline grouped-bootstrap lower-bound 条件失败；fixed proxy verdict 为 `killed_corrected_within_task_gate_failed`。
- M4-D0/F1 仅发现 `6/61` all-fail tasks 有互补正确 semantic units；A1/F2 equivariance delta AUC `0.0143` 且 CI 跨 0。
- 因此四个正式方法并未完成：M1 完整方法、M2、M3、M4 都仍未实现。

## 外部 baseline 状态

官方 CAL、rho-EOS 和 DreamOn 均已有公开代码，因此“没有代码可用”不能再作为统一理由。下一步必须分别判断：

- CAL：最接近本项目 setting，应第一优先适配 official HumanEval-Infilling Rest splits；
- rho-EOS：training-free 且代码公开，但公开 setting 不是现成 infilling adapter，先做 protocol compatibility；
- DreamOn：代码公开但需要 training/专用 checkpoint，必须与 training-free 方法分层；
- LR-DLLM：论文已审计，但当前仍缺可执行 official/local Stage I/II adapter，保留 blocker，不做伪复现。

所有 baseline 必须记录 model、prompt、dataset/split、candidate/canvas policy、decode steps、forward count、seed、evaluator、training、oracle、wall-clock、GPU 和 peak memory。

## 统计与因果风险

论文不得把“oracle canvas 没救回”写成唯一原因已经定位。允许的表述是：在当前 action family、seed、decode budget、oracle-length definition 与 evaluator 下未恢复。backbone、decoding、候选多样性、等价长度和 evaluator 严苛性仍是竞争解释。

6707-row 主结果必须补：base-task grouped bootstrap、paired group-aware permutation 或等价检验、help/harm CI、config/length/error strata CI、accuracy--cost frontier，以及 canvas recoverability 与 harm 的交集图。

## 严格投稿判断

当前 verdict：`not_ready_but_recoverable`。

要达到有竞争力的 CCF-A/FSE 投稿，至少需要：

1. official CAL 和至少一个 rho-EOS/DreamOn 的可审计协议结果或明确 incompatibility report；
2. 6707-row group-aware statistical package；
3. 一个非 HumanEval 的真实代码 infilling benchmark；
4. 一个独立方法在 fresh validation 上取得 compute-matched、group-aware、help/harm 可解释的增益，或者把论文彻底重构成足够强的 empirical diagnostic study。
