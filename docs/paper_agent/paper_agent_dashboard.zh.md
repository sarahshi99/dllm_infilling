# Paper Agent Dashboard

更新时间：2026-07-07 UTC

## H200 新服务器迁移状态（2026-07-05）

当前状态：`PARTIAL — H200 REPRODUCTION OR CONTROLLER V2 INCOMPLETE`。

Bootstrap artifacts：

- `docs/paper_agent/new_server_h200_bootstrap.zh.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/report.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/environment_manifest.json`
- `analysis_outputs/h200_bootstrap_20260705_103617/copied_artifact_hashes.csv`

Bootstrap verdict：`host_h200_available_sandbox_gpu_hidden`。默认 Codex 沙箱内看不到 `/dev/nvidia*`，所以 sandboxed `nvidia-smi` 失败且 `dllm_env` 中 PyTorch 报告 `cuda_available=false`；approved host/unsandboxed check 中 `nvidia-smi` 正常，H200 空闲，`dllm_env` 报告 `cuda_available=true`、`gpu_count=1`。

Tier 1 H200 core baselines 已完成，compact audit 为 `analysis_outputs/h200_repro_audit_20260707_tier1_v2/`，verdict 为 `h200_material_outcome_drift`。结果：Control `787/1033`，Midcons `794/1033`，Route2 `795/1033`，V6 `796/1033`，Local CAL `769/1033`。Route2/V6 相对旧 A6000 均为 `-6`，CAL 为 `-5`；该 drift 已被研究者接受为新 evidence-base 决策，不再阻塞 Controller V2。

H200 action bank 与 Controller V1 replay 也已完成：

- Action bank：`analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/`，`927` tasks，`4635` action rows，test rows `0`，benefit/harm non-KEEP `193/1237`。
- Controller V1 replay：`analysis_outputs/controller_validation_h200_20260707_v1_replay/`，selected zero-intervention controller，validation `89/127`，wins/losses `0/0`，gate failed，test decision `sealed`。
- Comparison audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`，old-vs-H200 action-bank outcome agreement `94.95%`，server verdict remains `h200_material_outcome_drift`，Controller V2 allowed `false`。
- Material drift triage：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`，verdict `material_drift_confirmed_controller_v2_blocked`，core public-split flip rows `53`，action-bank flip/label-change rows `267`，row-level test details suppressed。
- Evidence-base decision：`docs/paper_agent/h200_evidence_base_decision.zh.md`。后续 controller、validation、action bank 和论文主表以 H200 rerun 结果为准；A6000 保留为 historical reference。

Phase 3 H200 Controller V2 已完成 train/calibration/validation。Feasibility audit `analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility/` verdict 为 `mixed_controller_failure`。Controller V2 validation `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/` verdict 为 `risk_certification_limited_test_sealed`：`ordinal_only + probe_trace_fused` 是唯一非零信号，validation `90/127`，wins/losses `5/4`，interventions `41`，但 population harm upper95 `7.06%` 且净增仅 `1`，未通过 gate。Frozen test 继续 sealed。

GitHub 状态：SSH deploy key 已生效，`ssh -T git@github.com` 认证成功；`git ls-remote origin refs/heads/codex/risk-controlled-dynamic-rescue` 返回 `2b0662bfe9fdab787a5249dc9cbefea12d683af1`，与本地 HEAD 一致。见 `analysis_outputs/h200_bootstrap_20260705_103617/GITHUB_REMOTE_VERIFICATION.md`。

Frozen test：仍为 `sealed`，`test_evaluation_count=0`。

## Codex Phase 2 frozen controller 更新（2026-07-03）

当前阶段：`Frozen Risk-Controlled Canvas Controller`。

最新 central claim：

> Unknown-length DLLM infilling exhibits two coupled but separable regimes: canvas inadequacy and rescue inadequacy. Missed true-long failures are substantially trigger/canvas-limited, whereas already-triggered failures remain rescue-limited under the current longer-trajectory and trace-remasking action family. The main deployable opportunity is therefore risk-controlled, non-oracle prediction of when and how much to expand.

中文：unknown-length DLLM infilling 至少有 canvas inadequacy 与 rescue adequacy 两个耦合但可分离的 regime。missed true-long failures 在 oracle-sufficient canvas 下有明显可恢复空间；already-triggered failures 在当前 E/F/G action family 下仍然 rescue-limited。当前可部署机会是风险受控、非 oracle 地预测何时扩展以及扩展到多长。

Phase 2 compact artifacts：

- C/E/F/G attribution：`analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/report.md`
- frozen protocol/test lock：`docs/paper_agent/frozen_controller_protocol.zh.md`，`analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- deployable action bank：`analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/report.md`
- controller validation：`analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/report.md`
- LR-DLLM audit/sanity：`docs/paper_agent/lrdllm_protocol_audit.zh.md`，`analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/`

核心结果：

- C already recovers `29/89` hard cases; E/F/G add only `2` incremental hard recoveries over C.
- Missed failed-long recoveries under oracle-sufficient canvas：`31/56`; triggered failed-long recoveries：`0/33`。
- Action bank covers train/calibration/validation only：`927` tasks × `5` actions = `4635` rows; test remains sealed.
- Validation oracle action-bank upper bound：`104/127 = 81.89%` with `14` wins and `0` losses.
- Current logistic controller cannot pass calibration risk control: no nonzero-intervention point satisfies 5% harm upper bound.
- Validation-selected controller is zero-intervention: `90/127 = 70.87%`, equal to V6, `0/0` wins/losses.
- Frozen test was not run: `test_status=sealed`, `test_evaluation_count=0`.
- LR-DLLM verdict：`protocol_mismatch_blocked`; no full run.

Controller verdict：`no_validation_signal_test_sealed`。这不是 deployable success；它是一个受控 negative result，说明当前 inference-visible logistic controller 无法安全选择 action-bank 中的恢复空间。

## Codex Phase 1b distinct-candidate ceiling 更新（2026-07-02）

Codex 已在 `codex/risk-controlled-dynamic-rescue` 完成 Phase 0 审计、Phase 1 action-ceiling dry-run scaffold、严格限定的 3-case A/B/C/D GPU pilot、旧 pilot action-equivalence 审计，以及 Phase 1b distinct-candidate generation ceiling pilot。审计记录见 `docs/paper_agent/codex_repository_audit.zh.md`，最新交接见 `docs/paper_agent/codex_handoff.latest.zh.md`。

审计结论：`paper-agent-overnight` 是当前最新研究分支，领先默认 `main` 26 个 commits；V6 `802/1033` 是当前 LLaDA-Base 最高 full result，但只是 selector polish；V7/V8 是全局比例放长路线负结果；下一步最有决策价值的是 small, pre-registered true-long action-ceiling matrix，而不是继续参数 sweep。

Phase 1 dry-run scaffold：`experiments/action_ceiling/action_ceiling_matrix.py`，dry-run 输出 `analysis_outputs/action_ceiling_20260702_dryrun/report.md`。Dry-run 覆盖 `9` 个 cases、`36` 个 planned actions，case pool 为 positive controls / triggered failed-long / missed failed-long；其中 `25+` cases 为 `6` 个。

Strict 3-case GPU pilot：`analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/pilot_report.md`。Verdict 为 `positive_control_only`：`116/L0` replay A=fail/B=pass 且 C/D pass；`85/L0` 与 `113/L3` 在 A/B/C/D 下均 fail。Determinism check 通过，historical replay mismatch 为 `[]`。后续 action-equivalence 审计 `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/action_equivalence.md` 显示旧 D `steps96` 与 C 在三例中输出等价，因此不再把单纯增加 step budget 当作有效新 action。

Phase 1b distinct-candidate pilot：`analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/pilot_report.md`。Verdict 为 `candidate_diversity_without_correctness`。Action-distinctness gate 在 `85/L0` 通过，因为 E 禁用 early commit 后确实执行更多 forward、F remasked `4` 个 token 并执行 refinement；但 `85/L0` 最终 hash 仍与 C 相同并保持 `SyntaxError`。正式三例中，`116/L0` C/E/F 三个 seeds 全 pass 且只有一个 hash；`85/L0` C/E/F 三个 seeds 全 fail 且只有一个 hash；`113/L3` 的 E 产生第二个 hash，但 C/E/F 三个 seeds 全为 `UnitTestFailure`。没有 non-positive-control correct candidate，不建议扩展当前 generation family 到 9 cases 或 full benchmark。

## 当前研究目标

将当前 DLLM 代码 infilling 项目推进为有竞争力的 CCF-A 论文：把已有 LCAL/LCAS 经验进展转化为有原则的 length-control 贡献，并配套可复现实验证据。

## 当前 central claim

DLLM 代码 infilling 的 inference-time length control 不能简化为单一长度预测问题。当前最可信的主张是：保守长度控制与 selective rescue 可以低风险修复一部分 medium/near-long under-selection；但对 true-long cases，length sufficiency is necessary but often not sufficient，剩余失败同时受 trigger recall、rescue generation quality、candidate selection 和 backbone capability 约束。

术语口径：文档中的 previous/local baseline 或 local control 是本项目早前跑出的用户自有方法/控制版本，不是 CAL、LR-DLLM 或 DreamOn 方法的本地复现。论文报告值应与“我们之前的方法”和“当前方法”放在同表比较，但列名必须区分。

## 当前实验方案版本

当前处于 Phase 1b：`Distinct-Candidate Generation Ceiling`。上一轮 `positive_control_only` 已被接受；旧 A/B/C/D matrix 不扩展到 9 cases；`96/128` 等单纯增加 step budget 停止作为新 action。Phase 1b 结果为 `candidate_diversity_without_correctness`：当前 E/F 可以产生可审计轨迹变化或个别新 hash，但未在 hard cases 中产生正确候选。下一步不应是 full benchmark，而应分析错误类型或设计更强、预注册、有限的 candidate generator。

`v8`：用户提出的“直接改 CAL-like 公式、让长长度按比例获得更大 reward”已完成 GPU1 sequential full runs。结果为 negative：V8a `786/1033 = 76.09%`，V8b `781/1033 = 75.61%`，V8c `782/1033 = 75.70%`，均低于 current `midcons` `795/1033 = 76.96%`、Route2 precision `len32` `801/1033 = 77.54%` 和 V6 short override `802/1033 = 77.64%`。当前 best LLaDA-Base follow-up 仍是 V6 short override。比例思想不能作为全局 scoring reward 继续；若保留，只能作为 under-selection detector / risk guard 下的局部机制。

## 本次会话已完成

- 阅读了 `AGENTS.md`、近期计划、结果报告、run registry、文献记录，以及核心 clean runner/analysis 代码。
- 创建了 `paper-agent-overnight` 分支。
- 确认 GPU 0,1,2,3 已有其他 Python 任务占用；未启动重型 GPU 实验。
- 初始化了双语 `docs/paper_agent/` 研究跟踪结构。
- 新增经过测试的 evidence snapshot builder，并从本地 raw outputs 重新生成当前 A6000 evidence snapshot。
- 新增经过测试的 probe-curve signal audit；它没有找到 strict viable single-feature threshold，并确认当前 outputs 没有保存 stopping traces。
- 使用 `8` 个 focused tests、compile checks、audit regeneration、JSON assertions 和 `git diff --check` 验证了 probe-curve milestone。
- 已进入优雅暂停流程；新增 `docs/paper_agent/pause_checkpoint.current.md`，未启动新的研究或 GPU 实验。
- 以低 token 模式恢复，并将已过期 checkpoint 与当前 strict-split probe diagnostic files 对齐。
- 新增并运行 CPU-only strict-split probe-score diagnostic。它未通过 held-out safety gate，因此当前 probe-curve fields 本身不足以支持 GPU smoke run。
- 恢复后核对 dashboard/checkpoint verification 状态，并对 strict-split diagnostic 完成 fresh focused verification：unit test、py_compile、audit regeneration、JSON assertions 和 `git diff --check` 均通过。
- 已完成用户确认的 `GSAI-ML/LLaDA-8B-Instruct + midcons` full cross-model run；第一次直连 HuggingFace 因网络不可达中断，成功 run 使用 `HF_ENDPOINT=https://hf-mirror.com`。
- 核对 CAL、LR-DLLM、DreamOn 的 source PDFs，并将 literature-backbone rerun matrix 写入 `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- 用户批准沙箱外执行后，DreamCoder Base 和 DreamCoder Instruct 2-sample smoke 均通过；随后完成两个 full `1033` sample runs，Base 在 GPU2，Instruct 在 GPU3。
- 通过 `/tmp` 下的本地 Git/LFS checkout 下载/探测 `Dream-org/Dream-v0-Base-7B`，完成 2-sample smoke 后启动并完成 Dream-7B full local same-backbone pair：cal_lite baseline 在 GPU2，LCAL official bounded-repair candidate 在 GPU3。
- 通过已配置 proxy 的 Git/LFS 路径在 `/tmp` 下载/探测 `apple/DiffuCoder-7B-Base`，完成 2-sample smoke 后启动并完成 DiffuCoder-Base full local same-backbone pair：cal_lite baseline 在 GPU2，LCAL official bounded-repair candidate 在 GPU3。
- 通过已配置 proxy 完成 `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe。该 repo 直连 HuggingFace 比 `hf-mirror.com` 更可靠；6 个权重分片已下载到 `/tmp/llada15_probe_20260609` 并完成 byte-size 校验。
- 完成 `GSAI-ML/LLaDA-1.5` 2-sample smoke pair 和 full local same-backbone pair，使用共享 GPU `2/3`。Baseline 在 GPU2 完成，candidate 在 GPU3 完成；两条日志均 exit `0`，两个输出目录均有 `1033` 个 valid rows 和 `summary.json`。
- 完成 `inclusionAI/LLaDA-MoE-7B-A1B-Base` 下载恢复、local API/weight validation、flash-attn 环境根因排查、2-sample smoke gate 和 full local same-backbone pair。Candidate `801/1033 = 77.54%`，local `cal_lite` baseline `777/1033 = 75.22%`，pairwise `31` wins / `7` losses。
- 在不使用 Superpowers skills、subagents、reviewer discovery 或 Goal tools 的约束下完成 trace-long-rescue Task 1/2/3。Action brief/current action 通过 markdown hygiene；已实现 trace feature extraction、route analysis 和 report renderer；focused verification 通过 `Ran 6 tests` / `OK`、py_compile 和 `git diff --check`。
- 完成串行 Task 4 full trace collection：previous local method trace run 在 GPU `2` 上得到 `769/1033 = 74.44%` 和 `35257` trace rows；current `midcons` trace run 在 GPU `3` 上复现 `795/1033 = 76.96%`，并得到 `35768` trace rows。两条日志均以 `COMMAND_EXIT_CODE="0"` 结束，两个 trace files 均覆盖 `1033` 个 task ids。
- 完成 Task 5 offline Route 1/2/3 analysis。Route 1 和 Route 2 在两个 trace sources 上均触发 `0` 行；Route 3 因 single-canvas traces 且 Route 1/2 没有信号，不足以支持 multi-canvas policy cost。
- 按 `superpowers:executing-plans` 串行实现并运行 CPU-only `trace_feature_audit_v2`。最终输出 `analysis_outputs/trace_feature_audit_v2_20260613_204721`；decision 为 `diagnostic_only`：previous source 有 policy-level 候选，midcons source 只有 diagnostic-only，跨源稳定性不足。
- 在用户确认继续后，用现有 tracked runner `clean_scripts/run_route2_trace_rescue.py` 完成两条 LLaDA-Base Route 2 trace-gated long-rescue full follow-up runs。Broad plateau 为 `801/1033 = 77.54%`，pairwise `7/1/794/231`；precision top1/conf 为 `800/1033 = 77.44%`，pairwise `5/0/795/233`。两条日志均 exit `0`，两个输出均有 `1033` valid rows 和 `summary.json`。
- 按用户紧急要求将 precision `len32` follow-up 切换为 GPU3-only clean full run；GPU1 partial run 在约 `405/1033` 被中断且不作为证据。GPU3-only run 成功完成：`801/1033 = 77.54%`，pairwise `6/0/795/232`，trigger `57`，trigger true-long precision `61.40%`，avg sec including probe `5.4622`。
- 按 Superpowers local fallback 完成 Route2 error analysis Discovery V3 的 CPU-only 诊断。实现 `analysis/route2_error_analysis.py` 和 `tests/test_route2_error_analysis.py`，输出 `analysis_outputs/route2_error_analysis_20260617_165806`。诊断复现 `1033` joined rows、pairwise `6/0/795/232`、`33` triggered failed-long、`56` missed failed-long、`31/33` triggered failed-long rescue length >= oracle；decision 为 `mixed_rescue_quality_and_gate_recall`。本动作未启动 GPU。
- 按用户要求继续 true-long signal search，完成 Discovery V4 literature brainstorm 和 executable plan。新增 `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`、`docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`、`docs/paper_agent/experiments/20260617_discovery_v4_literature_brainstorm.md`。本动作未启动 GPU。
- 实现并运行 CPU-only `analysis/discovery_v4_signal_audit.py`，测试 `tests/test_discovery_v4_signal_audit.py` 通过 `Ran 5 tests` / `OK`。输出 `analysis_outputs/discovery_v4_signal_audit_20260618_000000`，记录文档 `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`。真实数据 dry run 暴露并修复 `true_long` 和 `triggered_rescue_failure_*` 两类泄漏 candidate；最终 V4 decision 为 `route2_polish_only`，不启动 GPU。
- 按用户要求完成导师汇报材料和 CCF-A readiness 评估。输出 `docs/paper_agent/presentations/20260618_advisor_project_report.pptx`、同名 Markdown/HTML 讲稿，以及 `ccfa_readiness_assessment.zh.md`。结论为 `weak_candidate`：项目有清晰问题和局部正结果，但还未达到 CCF-A submission-ready。
- 完成比例式长度放宽 V7 的 CPU audit、expanded-grid GPU smoke、runner 边界修复、guard smoke 和 clean full run。Clean full 输出为 `outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`，结果 `792/1033 = 76.67%`，相对 current `midcons` 净损 `-3` tasks；比例 promotion `10` 行但 true-long promotion `0` 行。
- 完成 V8 proportional CAL score 公式级 full runs。三条均在 GPU1 串行完成并写出 `1033` 行和 `summary.json`：V8a `786/1033 = 76.09%`，pairwise vs `midcons` `3/12/783/235`；V8b `781/1033 = 75.61%`，pairwise `7/21/774/231`；V8c `782/1033 = 75.70%`，pairwise `6/19/776/232`。V8c 的 cap 是 reward cap，不是候选长度 hard cap。

## 最新结果摘要

- A6000 control：`787/1033 = 76.19%`。
- A6000 `midcons`：`795/1033 = 76.96%`，相对 same-hardware control 为 `+8` wins、`0` losses。
- Long buckets 仍未改善：`17-24 = 20.73%`，`25+ = 16.13%`。
- Offline long-underestimate sweep 没有从当前 result fields 中找到安全 heuristic rule。
- 最新 snapshot：`docs/paper_agent/evidence_snapshot.md`，由 `analysis/build_paper_agent_evidence_snapshot.py` 生成。
- Probe-curve audit：`4106` 个 thresholds，`0` 个 strict viable；最佳 short-risk 为 `8.70%`，高于 `5%` GPU gate。
- Strict-split probe-score audit：`5` 个 deterministic task-id folds、`24` 个 features、aggregate held-out `63` triggers、`47.62%` true-long precision、`32.97%` failed-long recall、`22.22%` short-risk、`7.94%` current-pass risk；`strict_heldout_pass=False`。
- Verification：早前 probe-curve audit 的 `8` 个 focused tests 通过；strict-split diagnostic 在 2026-06-04 fresh verification 中通过 `Ran 3 tests` / `OK`、py_compile、audit regeneration、JSON assertions 和 `git diff --check`。
- LLaDA-Instruct cross-model run：`815/1033 = 78.90%`，低于同 backbone 历史 LCAS-v3 baseline `817/1033 = 79.09%`。Pairwise：`17` wins、`19` losses、`798` tie-pass、`199` tie-fail。Runtime 从 `6.8661s` 降到 `4.1766s` per sample including probe，但这是 negative transfer evidence，不是 claim upgrade。
- Literature-backbone plan：`docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- DreamCoder Base smoke：有效沙箱外 smoke 已通过，`2/2`，前两题与 same-backbone baseline 为 `2` tie-pass，avg total sec including probe `3.1439`。
- DreamCoder Instruct smoke：有效沙箱外 smoke 已通过，`2/2`，前两题与 same-backbone baseline 为 `2` tie-pass，avg total sec including probe `3.0555`。
- DreamCoder Base full run：`832/1033 = 80.54%`，高于同 backbone 本地 official-canvas cal_lite baseline `825/1033 = 79.86%`；pairwise 为 `27` wins、`20` losses、`805` tie-pass、`181` tie-fail；avg total sec including probe 为 `3.7763`，baseline 为 `3.7847`。这是小幅 local positive result，但还不是强 claim。
- DreamCoder Instruct full run：`834/1033 = 80.74%`，低于同 backbone 本地 official-canvas cal_lite baseline `848/1033 = 82.09%`；pairwise 为 `21` wins、`35` losses、`813` tie-pass、`164` tie-fail；avg total sec including probe 为 `3.8472`，baseline 为 `3.8657`。这是 negative transfer evidence。
- DreamCoder 文献位置：Base 高于 CAL DreamCoder-Base anchors（`70.2` average、`76.2` best shown），低于 LR-DLLM DreamCoder-7B `81.6`；DreamOn DreamCoder `92.1` 是 training-based 结果。这些是 anchors，不是 protocol-matched claims。
- Dream-7B local pair：candidate `803/1033 = 77.73%`，本地同 backbone cal_lite baseline `802/1033 = 77.64%`；pairwise 为 `28` wins、`27` losses、`775` tie-pass、`203` tie-fail；avg total sec including probe 为 `3.7337`，baseline 为 `3.6494`。这是 near-tie / slight local positive，不是强 claim。
- Dream-7B 文献位置：candidate 高于 LR-DLLM Dream-7B single-line anchor `76.7`；DreamOn Dream-7B `88.6` 是 training-based 且明显更高。这些是 anchors，不是 protocol-matched claims。
- DiffuCoder-Base local pair：candidate `839/1033 = 81.22%`，本地同 backbone cal_lite baseline `838/1033 = 81.12%`；pairwise 为 `25` wins、`24` losses、`814` tie-pass、`170` tie-fail；avg total sec including probe 为 `3.7538`，baseline 为 `3.6562`。这是 near-tie / slight local positive，不是强 bounded-repair improvement claim。
- DiffuCoder 文献位置：本地两条结果均高于 CAL DiffuCoder-Base anchors（`68.0` average、`74.8` best shown），但 DreamOn DiffuCoder-7B `92.2` 是 training-based 且明显更高。这些是 anchors，不是 protocol-matched claims。
- LLaDA-1.5 probe：本地路径 `/tmp/llada15_probe_20260609`；architecture `LLaDAModelLM`；`model_type=llada`；config `mask_token_id=126336`；tokenizer 中 `<|mdm_mask|>` 解析为 `126336`，但 `tokenizer.mask_token` 为 `None`；6 个 shards byte-size 全部匹配，总大小 `16,031,197,144` bytes。这只是 API/download result，不是 pass-rate result。
- LLaDA-1.5 local pair：candidate `818/1033 = 79.19%`，本地同 backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`；pairwise 为 `18` wins、`17` losses、`800` tie-pass、`198` tie-fail；avg total sec including probe 为 `6.6453`，baseline 为 `5.4224`。这是 near-tie / slight local positive，不是强 claim。Candidate 在 oracle `<=8` 净损失 `6` 个任务，在 oracle `>=13` 合计净增 `+7`，official-repair true-long precision 只有 `10.91%`。
- LLaDA-MoE local pair：candidate `801/1033 = 77.54%`，本地同 backbone `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`；pairwise 为 `31` wins、`7` losses、`770` tie-pass、`225` tie-fail；avg total sec including probe 为 `10.6107`，baseline 为 `8.7025`。所有 oracle buckets 均为正增益或持平：`<=8 +9`、`9-12 +5`、`13-16 +4`、`17-24 +6`、`25+ 0`。这是当前最强 local transfer result，但仍不是 external SOTA。
- LLaDA-Base full trace diagnostics：previous local method trace `769/1033 = 74.44%`、`35257` trace rows；current `midcons` trace `795/1033 = 76.96%`、`35768` trace rows。Route 1/2/3 在 offline accounting 下均未通过 Gate A 或 Gate B；不应创建 route-specific GPU policy runner。
- Trace feature audit v2：final output `analysis_outputs/trace_feature_audit_v2_20260613_204721`，decision `diagnostic_only`。Previous source：`1033` rows、`113` true-long、`96` failed-long、source decision `policy_candidate`；midcons source：`1033` rows、`113` true-long、`91` failed-long、source decision `diagnostic_only`。最有希望的 midcons 候选是 `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`，held-out 上 `18` triggers、`9` failed-long、`2` short-risk、`0` current-pass risk，但仍不足以支持 full GPU policy run。
- Route 2 trace-gated full follow-up：baseline `midcons` 为 `795/1033 = 76.96%`。Broad plateau 得到 `801/1033 = 77.54%`，净增 `+6`，但有 `1` 个 loss / short loss；触发 `73` 行，trigger true-long precision `53.42%`。Precision top1/conf 得到 `800/1033 = 77.44%`，净增 `+5`，`0` losses；触发 `57` 行，trigger true-long precision `61.40%`。Precision policy 是更干净的候选，但 oracle `17-24` 仅 `+1`，`25+` 不变，不能宣称 true-long 已解决。核心诊断：`91` 个 baseline failed-long rows 中，Broad 触发 `39` 但只救回 `2`，Precision 触发 `35` 但只救回 `1`。
- Route 2 precision len32 GPU3-only follow-up：output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`；`801/1033 = 77.54%`，净增 `+6`，pairwise `6/0/795/232`，trigger `57`，trigger true-long precision `61.40%`，avg sec including probe `5.4622`。Bucket net 为 `<=8 +2`、`9-12 +2`、`13-16 0`、`17-24 +2`、`25+ 0`；triggered `25+` 为 `0/11` pass。解释：低风险小幅正收益，但仍没有解决 `25+` true-long。
- V6 short override：`802/1033 = 77.64%`，是当前 LLaDA-Base follow-up 中最高 full-run pass count；相对 Route2 precision len32 为 `1` win / `0` losses / `801` tie-pass / `231` tie-fail。解释：这是小幅 selector polish，不是 long-length 根本解决。
- V7 proportional widening：output `/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`；`792/1033 = 76.67%`，相对 current `midcons` `795/1033 = 76.96%` 为 `-3` tasks，pairwise `1/4/791/237`。比例放宽 promotion `10` 行，true-long promotion `0`，short promotion `1`，long-bucket win `0`，short-bucket loss `2`；promoted rows 平均长度绝对误差从 `1.1` 恶化到 `6.9`。解释：当前比例放宽参数不成立，不能作为主线。
- V8 proportional CAL score：outputs `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`、`outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`、`outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`；三条 full 结果分别为 `786/1033 = 76.09%`、`781/1033 = 75.61%`、`782/1033 = 75.70%`。Pairwise vs `midcons` 分别为 `3/12/783/235`、`7/21/774/231`、`6/19/776/232`；short losses 分别为 `7/13/11`，long wins 分别为 `2/3/2`。解释：公式内比例奖励确实会增加少量 long wins，但伤害 short/medium 更明显；V8 是全局比例放长路线的负结果。
- Phase 1 action-equivalence audit：output `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/action_equivalence.md`；`116/L0` 与 `85/L0` 的 B/C/D hash-equivalent，`113/L3` 的 A/B hash-equivalent、C/D hash-equivalent。解释：旧 D `steps96` 配置上不同但输出等价；必须区分 `candidate_level_canvas_effects` 与 `pass_level_canvas_effects`，不能把“无 pass improvement”写成“canvas 没改变生成”。
- Phase 1b distinct-candidate ceiling：output `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot`；verdict `candidate_diversity_without_correctness`。Unique hashes：`116/L0 = 1`、`85/L0 = 1`、`113/L3 = 2`。C/E/F seeds `0,1,2` 只在 positive control pass；`85/L0` 保持 `SyntaxError`，`113/L3` 保持 `UnitTestFailure`。解释：当前 runner 有有限 trajectory exploration space，但 hard cases 无正确候选；不建议扩展当前 generation family。
- Route2 error analysis Discovery V3：output `analysis_outputs/route2_error_analysis_20260617_165806`；report `analysis_outputs/route2_error_analysis_20260617_165806/report.md`。它确认 Route2 的 `6` 个 wins 均来自 triggered rescue，但 `33` 个 triggered failed-long 中 `31` 个 rescue length 已经 >= oracle，同时还有 `56` 个 failed-long rows 未触发。结论：不要盲目加长 canvas；下一步应同时查 rescue generation/selection quality 和 probe-trace fusion gate recall。
- Discovery V4 design：spec `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`；plan `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`。V4 不再把问题看成单 feature 枚举，而是 risk-controlled action selection：`MissedLongHead` 找 missed failed-long 的 probe-trace fusion signal，`RescueQualityHead` 解释长度足够仍失败的 triggered rows，最后由 Policy Distillation 生成可审稿的 training-free rule/action。
- Discovery V4 signal audit：output `analysis_outputs/discovery_v4_signal_audit_20260618_000000`；report `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`；decision `route2_polish_only`。Joined rows `1033`，true-long `113`，baseline failed-long `91`。Best non-leaking candidate `broad_len24_triggered >= 1` 触发 `73` 行、missed failed-long `4`、triggered rescue-failure `33`、short risk `10`、current-pass risk `1`、true-long precision `0.534`、stable folds `3/5`，因此 reject。结论：不要从 V4 audit 直接启动 GPU full run。

## 关键方案调整

- 将 LLaDA-Base `midcons` 视为真实的 short/medium checkpoint，而不是完整解决方案。
- 将 LLaDA-Instruct `midcons` 视为 negative transfer evidence。
- 在定义更强 signal 前，停止把 GPU 投入当前 official-CAL true-long trigger family。
- 优先从当前 outputs 做 probe-curve multivariate/learned scoring；trajectory diagnostics 需要 trace-enabled smoke run。
- 将第一个 simple strict-split linear probe score 视为 negative evidence，而不是 candidate GPU policy。
- 将 trajectory features、learned length classification、DreamOn-style dynamic canvas control 或 LR-DLLM-style length regularization 提升为下一阶段 paper-level 方向。
- 将未来 GPU 实验限制在卡 `2,3`，并采用等待而非中断已有任务的方式。
- 将 Route 2 precision policy 记录为“paper-cleaner incremental positive evidence”，而不是新的最终主方法；broad policy 只作为更激进但有 short-loss 风险的对照。Route2 error analysis V3 进一步确认：当前问题是 rescue generation/selection quality 与 gate recall 的混合瓶颈，而不是单纯 rescue length 不够。V4 将下一步从“找一个 feature”升级为“找稳定 slice/action signal 并蒸馏成 rule”。

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` 和 `research_design.current.*.md` 包含 research community/user、need、minimum publishable contribution；日志 `2026-05-31 12:39 CST` 记录初始化 paper-agent documents | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 没有单独 CLI transcript；完成依据是产出的 research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` 的 `CEO-Style Stress Review` 覆盖 novelty、importance、reviewer appeal、scope、central claim、weakest assumption 和 CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 结论：当前还不是 CCF-A claim，但有可推进 foothold |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` 的 `Engineering Review Summary`、datasets、baselines、metrics、ablations/kill criteria/compute budget/reproducibility sections | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 当前版本为 `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | 当前方向由 gstack-style research design 和 experiment plan 承接；下次若改变 spec，应先运行 |
| Superpowers `writing-plans` | completed | 当前上下文显示已读取该 skill；`experiment_plan.current.*.md` 和 history 给出可执行计划 | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 没有另建 Superpowers plan 文件 |
| Superpowers `systematic-debugging` | completed | 2026-06-04 verification 中第一次 JSON assertion 使用了错误 schema path；检查 audit JSON 和脚本后确认真实 path 为 `cross_validation.aggregate_heldout` | `docs/paper_agent/current_action.md` | Root cause 是 assertion 命令写错，不是 audit metrics 改变 |
| Superpowers `verification-before-completion` | completed | 日志 `2026-05-31 21:47 CST` 和 `2026-05-31 22:11 CST` 记录早前 probe-curve audit verification；2026-06-04 fresh strict-split verification 记录 unit test、py_compile、audit regeneration、JSON assertions、`git diff --check` | `docs/paper_agent/current_action.md`, `docs/paper_agent/paper_agent_dashboard.zh.md`, `docs/paper_agent/probe_curve_split_score_audit.md` | 关键验证：probe-curve audit `Ran 8 tests` / `OK`；strict-split diagnostic `Ran 3 tests` / `OK` 且 `strict_heldout_pass=False` |
| Superpowers `requesting-code-review` | blocked | 日志 `2026-05-31 21:47 CST` 记录 no independent Task/subagent reviewer tool visible；使用 local diff review + fresh tests 作为 fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | 独立 reviewer 未完成；风险已记录 |

## 当前最大风险

当前提升幅度较小且 heuristic 色彩较强；除非下一阶段产生有原则的 long-length control 或强 cross-model/protocol-matched validation，否则不足以支撑 CCF-A 贡献。

## 下一步计划

1. 不扩展当前 Route2-based generation family 到 9 cases 或 full benchmark。
2. 对 `113/L3` 的 C/F hash 与 E hash 做 compact diff/error analysis，并定位 `85/L0` 固定 `SyntaxError`；负结果也要记录。
3. 若继续 true-long recovery，必须先设计更强但有限、预注册的 candidate generator；不能继续只加 steps、seeds 或局部 if-rule。
4. 启动 grouped split protocol，为后续 risk-controlled controller 留出 calibration/frozen validation/final held-out evaluation。
5. 任何 GPU 前必须写新 action brief、success/kill criteria，并确认不会干扰他人任务。

## 需要用户决策的问题

目前没有正在运行的 GPU 实验需要接管。Phase 1b 不支持继续当前 generation family 的 9-case expansion。下一步需要研究者决定：是先做 hard-case error analysis，还是批准一个新的、预注册的 candidate generator；同时建议尽快建立 grouped split protocol。
