# Experiment Queue

## 2026-10-13 最终结论与收尾修复

- 完整10330行、十组各1033，164基础题；16行检查通过，缺失/重复/失败均0；累计GPU作业约0.802小时。没有启动后续实验。
- 训练贡献 D−C：global +9（帮助/伤害15/6），frontier +18（29/11）；路径 C−B：global 0（0/0），frontier −5（6/11）。原 D−B +9/+13 分解为 +9+0 / +18−5，不能归因为路径的正收益。
- 关闭诊断、剔除评测和写盘后，KL头相对匹配K1纯生成速度 global=1.657×、frontier=1.818×（时间减少39.66%/45.01%）；无头K2=1.712×/1.899×。共享H200且配置按行轮换，外部负载仍可能影响计时。
- 真重算 E−D：global +25（28/3）、frontier +41（48/7）；E−C +34/+59。A对/B错人口救回：D=11/55、21/75；E=33/55、59/75，其他人口相对B新增伤害D=6/11、E=1/6。
- 统计限制：global头收益行加权95%区间含零，frontier头收益基础题等权区间跨零；E相对D的global基础题等权区间跨零，frontier两种权重均为正。重复开发人口、单生成种子，不能推广为独立测试结论。
- 下一轮最值得改的是右词修正的近似质量，依据E还能额外救回；E成本接近K1，仍不能当成可加速的正式验证器。建议只写入队列，不启动新方法。
- 后台生成/分析退出0；自动commit前因ledger末尾空行失败。已修复末尾换行和报告对两种权重区间的措辞，保留原生成版本与计时数据，补做focused commit/push。

## 2026-10-13T02:32:58.796268+00:00 K2 诊断后台任务：完成并停止

- 用户后续授权共享 H200；十配置串行、按行轮换，GPU 背景负载另存。共享负载可能影响纯生成耗时，不能称空卡测量。
- 状态 `completed_generation`，退出码 `0`；成功行 `10330/10330`，smoke `passed`，完整性 `passed`，GPU 作业累计 `0.8017h`（上限12h，排队等待不计）。
- 代码 `a55b1fd78795ae5f83845302a76f31780dda8cd2`；报告 `analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/report.zh.md`，全部配对结论见报告。
- 本轮全部完成；没有后续 GPU 作业。方法建议只写入报告，不自动运行。

## 2026-10-13 用户授权共享 H200 后台执行

- 本条取代之前“有其他进程即停止”的资源限制；不抢占、不终止其他进程。等待可用显存至少32GiB后开始，等待阶段不加载模型、不计GPU预算。
- exact command：`tmux new-session -d -s dreamon_markov_k2_diagnosis_20261013 'bash scripts/background_dreamon_markov_k2_diagnosis_20261013.sh > analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/background.log 2>&1'`（diagnosis 工作树内）。
- 47项CPU检查通过。后台顺序：16行开关检查→估时/12h预算→10330行→聚类分析/完整性审计→更新compact记录→focused commit/push→停止。协议差异/预算/异常会保存阻塞，不改参数或范围。
- tmux=`dreamon_markov_k2_diagnosis_20261013`；日志与结果目录=`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。无需持续人工监督。共享背景负载记录保留，报告明确计时限制。
- 旧阻塞产物移入同目录 `before_shared_gpu_authorization/`；原人口、头、种子和方法不变。

## 2026-10-13 DreamOn + Markov K2 诊断：代码已保存，GPU 资源阻塞

- 分支 `codex/dreamon-markov-k2-diagnosis-v1`；实现提交 `66c5c2b`，结果目录 `analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。从已审阅 `b8d6e58` 后最新 `0e97f61` 复用，原工作区和旧结果未改动。
- A/B/C/D/E × 两策略 runner/analyzer 已实现；C 为同类同秩零输出投影、lambda=1；E 只填原左词、保留原右位与未来 mask；真正关闭概率/排名/replay/画布诊断，分离 CUDA 同步生成、评测、写盘计时。
- 46 项 CPU 检查通过（含随机采样 RNG/诊断一致性、事务恢复与去重、合成十组完整统计链）；真实模型 16 行检查 **未运行**，正式批次 **0/10330**，完整性 **incomplete_or_invalid**。CPU 检查不是实验完成。
- 已实际执行 `bash scripts/manual_run_dreamon_markov_k2_diagnosis_20261013.sh`，退出 `3` / `blocked_gpu_busy`：唯一 H200 上仍有 6 个其他进程、约 91761 MiB 占用。没有加载模型、消耗 GPU 预算或抢占进程。
- 当前结论收窄：旧 D−B 的 +9/+13 净行数尚未隔离路径贡献，不能全归训练头；旧 wall 不能回答公平纯生成速度；E 恢复能力未测量。旧报告保留作为历史证据，不宣称过去所有收益都是假的。
- 唯一恢复项：空闲 H200 上运行同一入口，先固定稳定 ID 的 16 行开关等价性/估时，通过且不超过累计 12h 后自动全量十组。若资源仍忙，入口退出；不缩小人口，不留下 watcher。
- 下一方法建议等待 D−C / C−B / E−C,D 证据后决定。本轮不启动训练、上下文头、正式验证器、K4/K8 或其他新方法。

<!-- markov-training-audit-20260908 -->
## 2026-09-08 当前状态：训练完成，协议偏差已确认

- 决策：`iterate`；原训练提交 `32e8004`，修正分支 `codex/dreamon-markov-training-audit-fix-20260908`。
- 完整冻结清单存在跨集重复：代码 1,239 组，题目文本 3,102 组（两者重叠，不可相加）。不能再声称严格隔离。
- 已修复无头报表；保守剔除后的外部测试保留 13,387 条，TV/KL 距离缩小 26.69%/27.79%，是事后敏感性结果，不是新的独立测试。
- 旧权重仍属小批次参考损失均值版本；整批归一化已修复代码但未重训。两条训练候选代码与 HumanEval/13 的标准最大公约数实现匹配，实际是否训练需查服务器 SQLite；其余单断言候选需人工复核。
- 下一步：执行 `analysis/audit_markov_bank_membership.py`；实现固定左前沿双词元与全局 top-2 相邻短链。仅在基准隔离问题澄清后，用原验证选定 KL、λ=1 完成 1,033 条 Pass@1/速度比较。确认污染则保留实现与外部检查，暂停正式 HumanEval，不自行重训。
- 权威报告：`analysis_outputs/dreamon_markov_head_training_20260901_v1/report.zh.md`；原权重、原始诊断与冻结清单保留不动。以下旧日期条目仅作历史记录，不是当前运行指令。


Updated: 2026-07-28 UTC

Only selected, executable work belongs here. Canonical IDs and priorities come from `docs/paper_agent/ccfa_master_roadmap.zh.md`. Frozen test remains sealed unless a future fresh-validation protocol explicitly authorizes it.

<!-- method-portfolio-status: M1=reviewed_not_promoted_v0; M2=reviewed_not_promoted; M3=reviewed_not_promoted_v0; M4=reviewed_not_promoted_v0_multilinecore_cross_source -->

## 2026-07-28 Reconciliation Override

`method_portfolio.current.json` is the sole current M1--M4 register. official CAL's 4,990-case full has completed its integrity protocol (`4990/4990`, zero missing/error; no partial accuracy read while running). M1/M2/M3 are formally reviewed and not eligible for 296/927/5079. The mandatory 40,632-row M4 bank had no visible-context match to RandomSpanLight, so M4 ran the committed one-span-per-group MultiLine-Core repair; it is formally negative and not eligible for 296/927/5079. No paper primary is selected.

## Active FAST-SPRINT-01 Queue

M1--M4 are independent candidate methods; no paper primary method is selected. CPU/IO work may run concurrently. The old shared MultiLine candidate bank is complete (`40632/40632`, final audit passed) and must not be restarted. Every process/method has its own output directory. Frozen test remains sealed with `test_evaluation_count=0`.

### P1.1: official CAL protocol audit, smoke, then full

Status: CAL completed integrity reproduction. Official source is `https://github.com/NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`; the run is labeled **official CAL reproduction on the 4,990-case non-frozen common subset**, not a 5,715-case paper-number reproduction. Smoke `12/12` and full `4990/4990` have zero missing/error; no partial accuracy was read while it ran. DreamOn `8a0a549` is source-audited: an official FIM/evaluator path exists, but it is training-based and GPU smoke is not yet approved by a new brief. rho-EOS `69992ca` is completion-only with no suffix/FIM or HumanEval-Infilling evaluator route, so no faithful smoke/full is queued. LR-DLLM remains blocker audit only.

### P2.1: full-allowed grouped statistics

Status: completed CPU (`c66678a`). Input existing `6707` spans / `20121` results; verified `148` groups. Primary: equal-weight base-task macro accuracy (within-task policy accuracy then task mean). Secondary: span-micro descriptive totals. Completed 10,000 fixed-seed cluster bootstrap, task paired wins/losses, group-aware label-swap permutation, config/length/error strata with group count + cluster CI, accuracy-cost frontier, and control/CAL-lite/oracle 8-cell intersection. No row-independent significance and no frozen test.

### P4.1: ExecRepoBench external evaluator preparation

Status: pinned audit implementation ready; host checkout network currently blocked by approval-control-plane `422`. Dataset is fixed at `fa61028ce495c9ceff58398b8a7c47b5ae9f5276`; Qwen evaluator is fixed at `33bc6aabd7791ad7b32f7e92104f11f2359ba890`. `experiments/p4_execrepobench_audit.py` verifies those revisions, schema/fields, repository grouping and emits a six-fill, multi-repository smoke plan without code or external scores. Complete download/environment/license/field audit and actual evaluator smoke once checkout is available; do not open final external results before the eventual method configuration freeze.

### M1.1: Abductive Program-State Bridge

Status: formally reviewed, not promoted. Historical 5079-case MultiLine M1 is `safely_paused_resumable`, not killed or abandoned. Its original raw directories are immutable/append-only; future recovery is selected-method-only and requires `--auto-full --selected-method-only-5079` with existing-key dedup. Independent RandomSpanLight is complete (`1332/1332`, `148/148`, `148/148`; 0 duplicate/error); frozen full-vs-generic is `-0.68pp` with help/harm `0/1`, so this V0 must not resume, retune, or enter 296/927/5079.

### M2: Constraint-Homotopy V0

Status: completed and formally reviewed. All three arms are `148/148`, fixed 64 forwards/4096 token-forwards, 0 missing/extra/duplicate/error, and frozen sealed/count=0. Its predeclared conclusion is `constraints_activated_no_reliable_grouped_advantage`; no automatic 296/927/5079 promotion. Result: `docs/paper_agent/experiments/m2_constraint_homotopy_20260716_randomspanlight_result.zh.md`.

### M3: Birth-Death Canvas Diffusion V0

Status: formally reviewed, not promoted. Uniform and birth/death are each `148/148`, 0 duplicate/error, exact 256 forwards/task, and frozen sealed/count=0. Frozen birth-death-vs-uniform task-macro is `-4.05pp` with help/harm `8/14`; lower token-forward does not meet the accuracy or efficiency promotion rule. Do not retune or enter 296/927/5079.

### M4: Semantic Particle Assembly V0

Status: completed and formally reviewed. The historical 148-case RandomSpanLight offline structural audit remains historical. The required 40,632-row MultiLine bank cannot be substituted into RandomSpanLight (visible-context match `0/164`), so the committed outcome-blind hash selection used exactly one non-frozen MultiLine source per `148` task groups. Technical integrity was exact (`148×3`, zero missing/duplicate/error, frozen sealed/count=0); fair assembly-without-repair − best-single=`-12.84pp` CI `[-18.24,-7.43]pp`, help/harm=`0/19`. Fragment/assembly activation occurred but is negative. Do not retune, fuse, or enter 296/927/5079.

### A1/P3 status

`A1` remains a completed negative auxiliary, not a candidate method. `P3-SELECTIVE` remains deferred pending official CAL evidence; neither changes M1--M4's independent portfolio status.

## Historical Phase 5 Queue

## EXP-007: Phase 5 full RandomSpanLight shared candidate bank

Linked idea: IDEA-011, IDEA-015
Status: completed
Hypothesis: A full-first shared bank can falsify candidate-diversity, equivariance, semantic-bridge, and ranking premises without method-specific generation distributions.
Inputs: all `148` allowed non-frozen `HumanEval-RandomSpanInfillingLight` rows; LLaDA-8B-Base; canvas `16/32/64/128`; seeds `0/1`; fixed64 `(64,0)`; one diagnostic oracle ceiling.
Smoke/full rows: exactly `108` then `1332`; alpha-renaming mirrors are a separate auxiliary block after the full gate.
Command: exact approved H200 command in `docs/paper_agent/current_action.md`.
Success criterion: canvas-128 same-protocol validation, schema/evaluator/resume/row-count/duplicate audits pass, zero frozen rows, unchanged test lock; smoke automatically continues full.
Failure criterion: any substitution for 128, missing/duplicate/error row, frozen intersection, evaluator/schema failure, raw code in compact outputs, or changed test lock.
Historical blocker: approval service rejected earlier launches before process creation. User-authorized manual launch subsequently completed.
Expected report: `analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1/report.md`
Result: base `1332/1332` over `148` tasks and alpha auxiliary `728/728` over `91` verified tasks; zero missing/duplicate/extra/error/frozen rows; `test_evaluation_count=0`.

## EXP-008: Phase 5 F1–F4 premise falsification

Linked idea: IDEA-011, IDEA-014, IDEA-015
Status: completed
Hypothesis: One or more inference-visible premises add stable grouped ranking signal; failure is a valid kill result.
Inputs: EXP-007 full bank only.
Outputs: `analysis_outputs/phase5_premise_falsification_20260711_v1/`.
Success/failure: F1/F2/F4 always report. F3A supervised probes are diagnostic only. F3B passes only under the corrected deterministic within-task/cross-canvas, paired-selection, and short-safety gate.
Stop: no deployable method implementation if F3 fails.
Result: F1–F4 completed. F2 was negative. F3/F4 combined proxy had cross-canvas accuracy `0.6273`, but the corrected gate failed `positive_primary_delta_vs_all_baselines`.

## EXP-009: AST/def-use bridge proxy V0 conditional reranker

Linked idea: IDEA-011
Status: killed_gate_failed
Hypothesis: If the corrected gate passes, a standalone deterministic AST/def-use bridge proxy improves Pass@1 relative to fixed64 and confidence reranking on the same eight candidates.
Inputs: EXP-007 full bank and EXP-008 grouped OOF combined scores.
Outputs: `analysis_outputs/phase5_semantic_bridge_v0_20260711_v1/`.
Kill: if the corrected gate fails, write `killed_corrected_within_task_gate_failed`; no supervised-score fallback, extra generation, homotopy, birth–death, particle assembly, or fusion.
Result: `killed_corrected_within_task_gate_failed`; V0 was not implemented as a deployable reranker and frozen test stayed sealed at count `0`.

## EXP-001: CPU claim-boundary consolidation for diagnostic mixed paper

Linked idea: IDEA-004, IDEA-010
Status: completed
Hypothesis: Existing compact artifacts already support a bounded diagnostic claim, but only if Dream-Coder, second-regime, controller, and LLaDA attribution evidence are separated by strength and status.
Inputs:
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/combined_results.csv`
- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/summary.json`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/topk_policy_curves.csv`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/validation_action_selection.csv`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/validation_summary.json`
- `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/results.csv`
Commands:
```bash
python - <<'PY'
# CPU-only consolidation executed by Codex on 2026-07-08.
# It reads the inputs above and writes:
# analysis_outputs/research_planning_20260708_cpu_claim_audit/{summary.json,report.md,dreamcoder_case_taxonomy.csv}
PY
```
Outputs:
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/report.md`
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/summary.json`
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/dreamcoder_case_taxonomy.csv`
Success criterion: Report cleanly separates strong LLaDA attribution evidence, mixed Dream-Coder evidence, weak/negative controller evidence, and insufficient synthetic second-regime evidence without opening frozen test.
Failure criterion: Any output mixes historical A6000 with H200 evidence, treats synthetic second-regime as official benchmark, or uses frozen-test rows/results.
Stop condition: Stop after compact report and taxonomy are written; no GPU and no raw generated code.
Expected report path: `analysis_outputs/research_planning_20260708_cpu_claim_audit/report.md`

## EXP-002: Dream-Coder expanded oracle-sufficient diagnostic on bounded non-test manifest

## DreamOn order × parallelism × Markov premise diagnostic（completed）

Status: completed / reframe. Output: `analysis_outputs/dreamon_singleline_order_parallelism_markov_diagnostic_20260823_v2/`.

Decision: Do not train a Markov head or a fixed-K left-to-right controller from this evidence. C1 reproduces `951/1033`; top-K is often locally left-clustered, and K>1 has real parallelism, but L2 does not beat C2 while L4 is materially worse than C4. Online stale→fresh changes increase strongly with offset, but correctness-direction and fresh global-rank promotion were not measured. A future, separately authorized oracle-only diagnostic would need those fields before any training proposal.

Linked idea: IDEA-002, IDEA-005
Status: completed
Hypothesis: Dream-Coder's 15-case mixed result will remain oracle-canvas recoverable on a larger non-test manifest, but its missed-vs-triggered split may differ from LLaDA H200.
Inputs:
- `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/case_manifest.csv`
- `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/summary.json`
- Existing Dream-Coder runner in `experiments/phase4_continuation.py`
- Local cached model `Dream-org/Dream-Coder-v0-Base-7B`
Commands:
```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase4_continuation.py \
  --mode dream-oracle \
  --timestamp 20260708_dreamcoder_expanded37_v1 \
  --manifest analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/case_manifest.csv
```
Outputs:
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/summary.json`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/combined_results.csv`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/stratum_summary.csv`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/stratum_taxonomy.csv`
Success criterion: Expanded manifest yields nontrivial oracle-canvas recovery, no uncontrolled short-case regression, and a case taxonomy that clarifies whether Dream-Coder's triggered/missed behavior is model-dependent.
Failure criterion: Oracle-canvas recovery disappears, report is all-pass/all-fail without diagnostic separation, or output cannot be produced from cached model.
Stop condition: Stop after one bounded 37-case oracle-sufficient run; do not add E/F/G actions because Dream-Coder has no trace-remasking adapter; do not use frozen test.
Expected report path: `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md`
Result: Completed on 2026-07-08. Primary/control `11/37`, best simple `11/37`, oracle-sufficient canvas `26/37`. Stratum oracle pass: missed_failed_long `3/12`, triggered_failed_long `4/4`, medium_near_long_underselection `6/8`, short_primary_pass_harmable `6/6`, positive_control_recoverable `7/7`. Interpretation remains second-backbone diagnostic evidence, not model-agnostic confirmation.

## EXP-003: Second-regime stress gate before any benchmark claim

Linked idea: IDEA-001, IDEA-009
Status: completed_full_allowed_done
Hypothesis: A useful second-regime diagnostic requires source-labeled official data with genuine control/deployable failures; if oracle-sufficient canvas recovers a nonzero subset, full allowed official population plus fixed hard-tail diagnostics can support mixed-stress taxonomy without becoming a positive deployable controller claim.
Inputs:
- `analysis_outputs/second_regime_feasibility_20260708_phase4_v4/compatibility_report.md`
- `data/HumanEval-MultiLineInfilling.jsonl`
- `data/HumanEval-RandomSpanInfilling.jsonl`
- `data/HumanEval-RandomSpanInfillingLight.jsonl`
- `analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- `analysis_outputs/grouped_split_20260702_accel2/test_tasks.json`
Commands:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/official_second_regime_diagnostic.py --mode manifest

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/official_second_regime_diagnostic.py --mode diagnostic
```
Outputs:
- CPU manifest gate: `analysis_outputs/second_regime_official_manifest_20260708_v1/report.md`
- Official bounded GPU diagnostic: `analysis_outputs/second_regime_official_diagnostic_20260708_v1/report.md`
- CPU hard-tail manifest: `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/report.md`
- Bounded 48-case hard-tail diagnostic: `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/report.md`
- Supplemental fixed full104 hard-tail stress diagnostic: `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/report.md`
- Full allowed official second-regime diagnostic: `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/report.md`
Success criterion: The selected manifest has nontrivial failures under control/deployable policy and a measurable oracle-canvas recoverability or rescue-limited fraction.
Failure criterion: Official files remain missing and synthetic stress is all-pass, too small, or too artificial; the paper must keep second-regime as unresolved.
Stop condition: Stop after the full allowed official second-regime diagnostic unless a concrete bug appears. Do not run synthetic stress, add policies, tune cal-lite, open frozen test, or start Controller V4.
Expected report path: `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/report.md`
Result: Completed on 2026-07-10. Manifest gate produced `120` official cases, `40` per config and `10` per bucket, with frozen-controller-test rows `0` and evaluator smoke `12/12`. First-pass GPU results: control fixed64 `34/120`, best deployable cal-lite `36/120`, oracle-sufficient canvas `49/120`, oracle gain vs control `26`, deployable harm vs control `15`, oracle harm vs control `11`. The approved 48-case hard-tail diagnostic used balanced sample groups (`12/12/12/12`), source configs `16/16/16`, buckets `12/12/12/12`, frozen rows `0`, and did not alter first-pass labels. 48-case result: control `12/48`, deployable `16/48`, oracle `28/48`, genuine canvas-recoverable `24`, rescue/non-canvas `12`, deployable help `13`, deployable harm `9`, oracle harm vs control `8`, first-pass label changes `0`. Reviewer-requested full104 fixed hard-tail stress result: control `18/104`, deployable `20/104`, oracle `33/104`, genuine canvas-recoverable `26`, rescue/non-canvas `60`, deployable help `17`, deployable harm `15`, oracle harm vs control `11`, first-pass label changes `0`. Full allowed official population result: `6707` rows, frozen rows `0`, control fixed64 `2019/6707`, best deployable cal-lite `1464/6707`, oracle-sufficient canvas `3180/6707`, oracle gain vs control `1633`, deployable help `620`, deployable harm `1175`, oracle harm vs control `472`, rescue/non-canvas-limited `3055`. Interpretation: official second-regime is mixed stress evidence, not a positive deployable controller claim; the full allowed diagnostic strengthens the mixed diagnostic claim, the 120-case first pass remains the preregistered/unbiased diagnostic estimate, and full104 remains supplemental failure taxonomy/robustness evidence. Second-regime GPU work is stopped.

## EXP-004: Controller V1/V2/V3 route-closure consolidation

Linked idea: IDEA-004, IDEA-010
Status: completed
Hypothesis: The controller route can be closed as validation-only weak/negative evidence because oracle action-bank headroom does not translate into a deployable policy that passes harm/frozen-test gates.
Inputs:
- `analysis_outputs/controller_validation_h200_20260707_v1_replay/report.md`
- `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/report.md`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/report.md`
Commands:
```bash
# CPU-only consolidation generated by Codex on 2026-07-08.
```
Outputs:
- `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`
- `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.csv`
- `analysis_outputs/controller_route_closure_20260708_v1/summary.json`
Success criterion: One table records oracle upper bound, selected policy, interventions, validation wins/losses, validation pass, harm upper95, and frozen-test decision for V1/V2/V3.
Failure criterion: Any row implies frozen-test authorization or positive deployable controller evidence.
Stop condition: Do not add Controller V4 unless a new evidence source appears outside repeated validation tuning.
Expected report path: `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`
Result: Completed. V1 selects zero intervention; V2 best nonzero has `5/4` wins/losses and harm upper95 `7.06%`; V3 exploratory top-k has `1/0` wins/losses but fails frozen-test gate. Frozen test remains sealed with evaluation count `0`.

## EXP-005: Paper evidence consolidation for diagnostic mixed paper

Linked idea: IDEA-010
Status: completed
Hypothesis: The completed evidence base can support a CCF-A diagnostic mixed paper if every table row explicitly separates support, limits, allowed claims, forbidden claims, and paper placement.
Inputs:
- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/summary.json`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/summary.json`
- `analysis_outputs/second_regime_official_diagnostic_20260708_v1/summary.json`
- `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/summary.json`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/summary.json`
- `analysis_outputs/controller_route_closure_20260708_v1/summary.json`
- `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/summary.json`
Commands:
```bash
python experiments/paper_evidence_consolidation.py
```
Outputs:
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/claim_matrix.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/claim_matrix.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/main_results_table.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/main_results_table.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/official_second_regime_writeup.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/failure_taxonomy_table.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/failure_taxonomy_table.csv`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/paper_claim_rewrite.md`
- `analysis_outputs/paper_evidence_consolidation_20260710_v1/summary.json`
Success criterion: The consolidation names the final central claim, three contributions, claim matrix, main results table, official second-regime write-up, failure taxonomy, and forbidden claims without opening frozen test or upgrading negative evidence into positive controller claims.
Failure criterion: Any artifact mixes H200 with historical A6000 evidence, treats hard-tail full104 as an unbiased benchmark, or claims deployable controller success.
Stop condition: Stop after compact Markdown/CSV/JSON outputs and web-review-ready handoff updates; no generation or GPU needed for this experiment.
Expected report path: `analysis_outputs/paper_evidence_consolidation_20260710_v1/paper_claim_rewrite.md`
Result: Completed on 2026-07-10. Produced `7` claim-matrix rows, `6` main-results rows, `8` failure-taxonomy rows, and a paper claim rewrite. Verdict: `paper_evidence_consolidation_completed`.

## EXP-006: Dream-Coder full allowed SingleLine optional diagnostic

Linked idea: IDEA-002, IDEA-005
Status: completed
Hypothesis: A full non-frozen Dream-Coder SingleLine diagnostic can reduce sampling noise from expanded37 while preserving frozen-test discipline, but it should remain second-backbone evidence rather than model-agnostic confirmation.
Inputs:
- Local allowed non-frozen `HumanEval-SingleLineInfilling` rows
- Existing Dream-Coder runner and cached `Dream-org/Dream-Coder-v0-Base-7B`
- Frozen-controller-test split lock for exclusion only
Commands:
```bash
# Optional full-run branch executed on 2026-07-10 with fixed policies:
# primary/control, best simple length policy, oracle-sufficient canvas.
```
Outputs:
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/report.md`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/summary.json`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/manifest.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/results.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/stratum_summary.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/taxonomy_summary.csv`
- `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/comparison_vs_expanded37.csv`
Success criterion: Manifest includes only non-frozen allowed SingleLine rows; all three fixed policies complete; results clarify whether oracle canvas recovery persists at full allowed scale.
Failure criterion: Any frozen-controller-test row appears, policies are tuned after results, E/F/G actions are added for Dream-Coder, or outputs require raw generated-code dumps.
Stop condition: Stop after one full allowed Dream-Coder SingleLine run; do not add Dream-Coder E/F/G actions or tune policy after results.
Expected report path: `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/report.md`
Result: Completed on 2026-07-10. Manifest has `927` cases, frozen rows `0`, and `2781` policy rows. Primary/control `735/927`, best simple length policy `744/927`, oracle-sufficient canvas `858/927`; oracle gain vs primary `137`, simple help/harm `25/16`, oracle harm vs primary `14`, rescue/non-canvas-limited `55`. Interpretation: optional full allowed second-backbone SingleLine diagnostic, not model-agnostic confirmation.

## Execution Sprint V1 queue update — 2026-07-17

Current M1--M4 statuses are centralized in `docs/paper_agent/method_portfolio.current.json`; the queue below is execution order, not a competing status register.

1. **M1/M3 results** — formal fixed analyses are complete. M1 full loses its fair generic comparison and M3 birth-death has lower point accuracy than uniform; neither V0 may enter 296/927/5079 or be retuned on its 148-case outcome. No M1 historical 5079 resume.
2. **official CAL** — smoke `12/12` passed with success-only canonical raw, separate failure journal, documented evaluator enablement provenance, frozen seal and resume checks. The official 4,990-case CAL-Rest common full is running; do not inspect partial accuracy or label it as 5,715/5,079 exact evaluation.
3. **M4** — remains independent: a persistent non-destructive supervisor waits for CAL t+10 growth, ECC=0, `>=25GiB` free memory and no third GPU research process before repair `12→148`. M2 remains completed/reviewed and not promoted.
4. **P2.1** completed; **P4 ExecRepoBench** remains final-method-freeze only. Historical 40632 bank score-only analysis is complete and explicitly non-M1-full.

<!-- dreamon-markov-head-training-20260901-v1 -->
## 2026-09-01 DreamOn external Markov-head training v1

Status: `completed_external_test_opened`. TV pilot/full=`True/True`; KL pilot/full=`True/True`; winner=`kl`. This is external OpenCoder training/validation evidence, not a HumanEval method result. Artifacts: `analysis_outputs/dreamon_markov_head_training_20260901_v1/`.

## 2026-10-15 并行扩删坐标修补 v2（追加）

基线dbdd0bd，独立分支codex/dreamon-structure-coordinate-fix-20261015-v1，旧v1输出不动。CPU完整重放37722行：37596逐轮画布等价，126首分歧（A/global原生37；C/global链无头B28、零头C28、训练头D33），未知0，多行配对并集38。冻结清单筛选不看passed；技术smoke在冻结后按预定规则覆盖旧通过/失败。CPU77项含最小反例、预算/截断、边界、旧K1/链K4/frontier/Base原生回归及分组bootstrap公式通过。下一步一次可恢复队列，GPU0等待32GiB空闲，smoke→126行补验→全人口表/修补与学习分开配对→归档推送退出。恢复analysis_outputs/structure_coordinate_fix_20261015_v2/recovery.zh.md。无思维导图修改、新采样适配器或候选四头实现/训练。

2026-10-15 已实际启动唯一后台队列dreamon_structure_coordinate_fix_20261015_v1，queue PID1171091、worker PID1171098；正式修正560e98a，状态running_affected_full。smoke→126行补验→v2合并分析→提交推送全自动，无需用户触发下一阶段。

## 2026-10-15 并行扩删坐标补验 v2 自动完成记录

状态complete；新生成126，有证明复用37596；修补评分救回/改错5/2。证据analysis_outputs/structure_coordinate_fix_20261015_v2/report.zh.md。旧v1表不改，旧单行K2仅历史参照，修补贡献与学习贡献分开；无思维导图修改或后续四头训练。
