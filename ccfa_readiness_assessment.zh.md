# CCF-A Readiness Assessment

更新时间：2026-07-11 UTC

## Phase 5 CCF-A Gap Ledger

完整 ledger：`analysis_outputs/phase5_method_portfolio_20260711_v1/ccfa_gap_ledger.csv` 与 `.json`。状态只能是 `closed`、`partially_closed`、`open`、`blocked`、`superseded`。

| Gap | Status | 当前证据 / 解除条件 |
|---|---|---|
| 独立、非 heuristic 的方法 premise | `partially_closed` | 五个想法已分开登记；只有 Semantic Bridge 进入本轮 falsification |
| 148-case RandomSpanLight shared bank | `open` | manifest 已精确审计，GPU rows 尚未生成 |
| candidate/AST/semantic-fragment diversity | `open` | F1 blocked on bank |
| alpha-renaming equivariance 独立预测力 | `open` | F2 blocked on bank；不把 stability 称为 correctness |
| deterministic AST/def-use proxy 的 within-task 优势 | `open` | corrected F3/F4 blocked；要求 cross-canvas primary delta CI `>0`、paired help/harm 与 short safety |
| inference-visible within-task ranking | `open` | F4 blocked on bank |
| AST/def-use bridge proxy V0 Pass@1/help-harm | `blocked` | 仅在 corrected within-task gate 通过后解除 |
| frozen controller held-out result | `blocked` | Phase 5 不打开；`test_evaluation_count=0` |
| resumable/duplicate-safe long run | `partially_closed` | code/tests 已有，需真实 smoke/full audit |
| raw-code publication safety | `partially_closed` | compact schema 无 raw code，需 final tracked scan |
| Controller V4 / heuristic-cal-lite continuation | `superseded` | 由 independent premise falsification 取代 |
| Dream-Coder E/F/G 扩展 | `superseded` | 本轮禁止 |
| CCF-A submission readiness | `open` | 仍缺 credible method result 或强 falsification-centered package |

Phase 5 当前 decision：`blocked`。Blocker 是 approved H200 process launch 前的 approval-service `422 model not found: codex-auto-review`，不是方法结果。Claim Readiness Gate 仍为 `not_ready`；不得把未运行的 F1–F4 或 V0 写成证据。

## 总体判断

当前项目已经从“继续调 controller”转为 `diagnostic-driven mixed paper`。这不是一个 positive controller paper：H200 Controller V1/V2/V3 都显示 deployable risk-controlled intervention 只有弱验证信号，frozen test 仍然 sealed，`test_evaluation_count=0`。

Claim Readiness Gate verdict：`diagnostic_mixed_candidate_with_blocking_gaps`。

最稳妥的论文主张是：

> Unknown-length DLLM infilling has separable canvas-limited and rescue-limited regimes. Missed true-long cases expose substantial oracle-canvas recoverability, while already-triggered long failures remain resistant to longer trajectories and trace-guided remasking. Deployable risk-controlled control shows weak but insufficient validation signal, revealing a gap between diagnostic upper bound and safe inference-time intervention.

## 已有强证据

1. H200 evidence base 已被研究者接受，主表和后续 controller 以 H200 rerun 为准；A6000 只保留为 historical reference。
2. H200 core baselines 已全量重跑：Control `787/1033`，Midcons `794/1033`，Route2 `795/1033`，V6 `796/1033`，Local CAL `769/1033`。
3. H200 action bank 覆盖 frozen train/calibration/validation：`927` tasks × `5` actions = `4635` rows，test rows `0`。
4. Oracle action-bank validation upper bound 为 `106/127`，相对 primary `17/0` wins/losses，说明 action bank 中确实存在可恢复空间。
5. True-long attribution 的核心机制清晰：C oracle-sufficient canvas 恢复 `29/89` hard cases，全部来自 missed failed-long；triggered failed-long 为 `0/33`，支持 canvas inadequacy 与 rescue inadequacy 分离。
6. Controller V1/V2/V3 的负结果是受控的：所有开发都停在 validation，frozen test 未打开。

## Weak Evidence

1. Controller V3 有弱 validation signal：保守 top-k policy 可做到 `90/127`、`1/0` wins/losses、population harm upper95 `2.33%`，但净增只有 `+1`。
2. Family A 的最高 pass-count 点为 `91/127`、`3/1` wins/losses、net `+2`，但出现一个 `<=8` short-bucket loss，不能授权 frozen test。
3. Dream-Coder Base fresh oracle-sufficient diagnostic 已完成：15-case subset 上 primary/control `7/15`，best simple policy `7/15`，oracle-sufficient canvas `14/15`，short regressions `0`。这说明 second-backbone 上 oracle canvas 有强恢复空间。
4. Dream-Coder 的 qualitative agreement 是 mixed：missed-long oracle recoveries `3`，triggered-long oracle recoveries `3`，不干净复刻 LLaDA H200 的 missed-vs-triggered split，因此仍不足以声称 model-agnostic generalization。

## Negative Evidence

1. Controller V1 H200 replay 仍为 zero intervention：validation `89/127`，wins/losses `0/0`。
2. Controller V2 最好的非零点为 `90/127`、`5/4` wins/losses，population harm upper95 `7.06%`，未通过 primary gate。
3. Controller V3 没有满足 frozen-test gate 的 deployable policy；最终 route decision 为 `weak_validation_signal_test_sealed`。
4. LR-DLLM final attempt verdict 为 `blocked_missing_algorithmic_detail`：仓库中没有 protocol-matched Stage I/II adapter，不能称为 official reproduction。
5. Second-regime audit 发现 MultiLine/RandomSpan alias 存在，但本机 `data/` 缺少所需 JSONL；后续已构造并运行明确标记的 `synthetic_second_regime_minimal`。该 synthetic subset 18/18 全部通过，说明 runner/data unblock 成功，但 subset 过易，不能替代 official second-regime benchmark。

## CCF-A Blocking Gaps

1. 缺少 official 或更强 second-regime stress evidence：`HumanEval-MultiLineInfilling` / `HumanEval-RandomSpanInfilling` 本地 JSONL 仍缺，synthetic subset 太易。
2. Dream-Coder fresh oracle diagnostic 是 mixed，而不是 clean cross-backbone confirmation；若要强泛化主张，需要扩大 second-backbone subset 或加入更难 case。
3. deployable controller 没有达到 frozen-test gate；因此没有 sealed test improvement。
4. LR-DLLM 没有同协议 baseline，只能作为 blocked baseline 记录。
5. 当前 paper contribution 需要靠机制诊断、负结果严谨性和泛化 audit 支撑，不能写成 SOTA 方法论文。

## Next Required Experiments

1. 补齐 `HumanEval-MultiLineInfilling` 或 `HumanEval-RandomSpanInfilling` 本地 JSONL，运行 official second-regime minimal diagnostic；当前 synthetic result 只能作为 unblock/sanity evidence。
2. 若要增强 CCF-A 竞争力，优先扩大 Dream-Coder diagnostic 到更多 true-long/missed/triggered cases，判断 mixed split 是否稳定，而不是继续在人类验证集上调 Controller V4。
3. 如果 LR-DLLM 官方代码或足够算法细节释放，再重做 protocol-matched Stage I sanity；在此之前不要把 heuristic local adapter 写成 LR-DLLM reproduction。

## Paper Framing Decision

采用 `diagnostic-driven mixed paper`：

- 主贡献：揭示 unknown-length DLLM infilling 中 canvas-limited 与 rescue-limited 两种机制，以及 action-bank oracle upper bound 与 safe deployable controller 之间的缺口。
- 正结果：oracle/action-ceiling 与部分 validation top-k 证明恢复空间存在。
- 负结果：V1/V2/V3 controller 无法安全转化为 frozen-test policy。
- 不声称：SOTA、controller success、unknown-length solved、frozen test improvement、model-agnostic generalization。

当前可以开始论文骨架和正式写作，但 CCF-A 强投稿仍需要至少一个泛化诊断补强：second-backbone fresh diagnostic 或 second-regime diagnostic。
