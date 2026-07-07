# Experiment Plan v3：Probe-Curve-First Length Modeling

更新时间：2026-07-07 UTC

## H200 迁移更新（2026-07-05）

当前计划先完成 H200 bootstrap 修正，bootstrap verdict 为 `host_h200_available_sandbox_gpu_hidden`。H200 GPU 在 `/proc/driver/nvidia` 中可见；默认 Codex 沙箱内 `nvidia-smi` 无法与 driver 通信，`/dev/nvidia*` 设备节点缺失，`dllm_env` 中 PyTorch 报告 `cuda_available=false` 和 `gpu_count=0`。但 approved host/unsandboxed check 显示 H200 正常空闲，`dllm_env` 中 PyTorch 报告 `cuda_available=true` 和 `gpu_count=1`。

Tier 1 full baselines 已通过 approved unsandboxed/escalated GPU commands 完成。Compact audit `analysis_outputs/h200_repro_audit_20260707_tier1_v2/` 给出 `h200_material_outcome_drift`：Control `787/1033`，Midcons `794/1033`，Route2 `795/1033`，V6 `796/1033`，Local CAL `769/1033`。H200 action bank `analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/` 和 Controller V1 replay `analysis_outputs/controller_validation_h200_20260707_v1_replay/` 已完成；comparison audit `analysis_outputs/h200_repro_audit_20260707_action_bank_v1/` 仍保持 `h200_material_outcome_drift`。CPU-only material drift triage `analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/` verdict 为 `material_drift_confirmed_controller_v2_blocked`，并只写 train/calibration/validation row-level flips，不泄露 test row-level details。研究者已接受当前 H200 rerun 结果为新的 evidence base，见 `docs/paper_agent/h200_evidence_base_decision.zh.md`；`h200_material_outcome_drift` 不再作为 Controller V2 停止条件。GitHub SSH auth 和 branch freshness 已在 2026-07-06 UTC 验证。Bootstrap artifacts：

- `docs/paper_agent/new_server_h200_bootstrap.zh.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/report.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/environment_manifest.json`
- `analysis_outputs/h200_bootstrap_20260705_103617/copied_artifact_hashes.csv`

只有在 approved unsandboxed context 中 `nvidia-smi` 正常、`torch.cuda.is_available() = true`，才继续 H200 GPU reruns；默认 sandbox 的 CUDA probe 不作为 host GPU failure 证据。Phase 3 现在以 H200 rerun evidence base 推进 Controller V2；frozen test 在 validation gate 通过前继续 sealed，`test_evaluation_count=0`。

## Phase 2 冻结 controller 更新（2026-07-03）

当前阶段已从 oracle action-ceiling diagnostics 进入 `Frozen Risk-Controlled Canvas Controller`。本轮完成：

- C/E/F/G incremental attribution：`analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/`
- frozen split/test lock：`analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- train/calibration/validation action bank：`analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/`
- controller validation：`analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/`
- LR-DLLM audit/sanity：`docs/paper_agent/lrdllm_protocol_audit.zh.md` 与 `analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/`

Validation verdict：`VALIDATION FAILURE — TEST REMAINS SEALED`。在 calibration split 上，当前 logistic controller 的所有非零干预 operating points 都无法满足 5% harm upper-confidence budget；risk-calibrated selected controller 因此选择零干预。validation 上它与 V6 同为 `90/127 = 70.87%`，但没有 pass gain，也没有触发 frozen test。test split 仍然 sealed，`test_evaluation_count=0`。

下一步若继续 controller 主线，必须只在 train/calibration/validation 上改进 inference-visible feature/model/calibration，不得查看 sealed test labels。优先考虑 ordinal/survival canvas adequacy head 或更强但仍可解释的 calibration，而不是扩展 E/F/G remasking action。

## Codex Phase 0 更新（2026-07-02）

V7/V8 full runs 已将“全局比例放长 reward”路线降级为负结果。Codex 审计后建议把下一阶段优先级从继续 probe/threshold sweep 转为 `True-long action-ceiling matrix`：用小规模、预注册、可 dry-run 的 action matrix 区分 canvas adequacy、rescue generation、candidate selection 和 trigger recall。该实验仍必须先写 action brief 和 dry-run manifest，不应直接启动 full GPU。

Phase 1 scaffold 已新增：`experiments/action_ceiling/action_ceiling_matrix.py` 和 `docs/paper_agent/experiments/20260702_action_ceiling_matrix_action.md`。首个 dry-run 输出为 `analysis_outputs/action_ceiling_20260702_dryrun/`，覆盖 `9` 个 cases 和 `36` 个 planned actions；pilot/GPU 尚未运行。

## Objective

从当前 `midcons` A6000 checkpoint 出发，推进到 CCF-A 级别的 DLLM code infilling length-control method。本计划优先进行可复现 diagnostics，再启动昂贵 GPU runs。

## Baseline And Environment

- Dataset：`HumanEval-SingleLineInfilling`，test split，`1033` tasks。
- Current model：`GSAI-ML/LLaDA-8B-Base`。
- Same-hardware baseline：A6000 union control，`787/1033 = 76.19%`。
- Current checkpoint：A6000 `midcons`，`795/1033 = 76.96%`。
- GPU policy：除非用户显式更改分配，未来实验必须只使用 GPU `2,3`。不要 kill、抢占或中断已有进程；如果卡 `2,3` 被占用，应等待或排队。本计划继续从 CPU-only analysis 推进，直到 offline gates 证明值得启动 smoke run。

## Engineering Review Summary

Architecture：保持 legacy `expvision_dllm/` 和 `scripts/` 冻结。新工作放入 `expvision_dllm_clean/`、`clean_scripts/`、`analysis/`、`analysis_outputs/` 和 `docs/paper_agent/`。

Data flow：

```text
existing results.jsonl
  -> compact evidence builder / diagnostic script
  -> paper-agent evidence snapshot and probe-curve audit
  -> experiment_results and dashboard
  -> GPU runner only if offline criteria pass
```

Failure posture：任何 GPU policy 都不应基于未通过 offline short-risk 与 recall checks 的 heuristic 启动。Raw outputs 保留在本地；tracked artifacts 只包含 compact summaries。

## Datasets

Primary：

- `HumanEval-SingleLineInfilling`，`1033` tasks。

Required next validation：

- 如果 protocol alignment 可行，加入 HumanEval-Infilling multi-line。
- 只有当 single-line story 稳定后，才加入 SantaCoder-FIM 或另一个 FIM-style benchmark。

## Baselines

Internal baselines：

- fixed length。
- oracle length。
- CAL-lite。
- LCAS-v3b。
- LCAL。
- official-CAL bounded repair。
- A6000 union control。
- A6000 `midcons`。

External anchors：

- DreamOn dynamic canvas。
- LR-DLLM length regularization。
- Autoregressive code infilling models 只在 explicitly matched settings 下比较。

## Metrics

Primary：

- pass@1。
- 相对 same-hardware control 的 pairwise wins/losses。
- oracle-length bucket pass rates：`<=8`、`9-12`、`13-16`、`17-24`、`25+`。

Secondary：

- selected-minus-oracle length error。
- under-selection rate。
- 按 source 统计 trigger count 和 trigger precision。
- proposed long detectors 的 short-risk rate。
- current-pass risk rate。
- decode/probe overhead。

## Experiment Phases

### E0：Evidence Snapshot

从已有 raw outputs 构建紧凑 paper-agent evidence snapshot。必须验证 row counts、pass rates、bucket metrics、wins/losses 和 long-failure under-selection counts。

Success：生成 summary 与已有报告一致，并作为 compact documentation 跟踪。

### E1：Long-Signal Diagnostic Expansion

将 CPU-only diagnostics 扩展到现有 scalar result fields 之外，并从 probe-curve shape features 开始，因为当前 A6000 full-run outputs 没有保存 `stopping_trace` 或 `step_traces`。

当前可用：

- length-probe curve shape features；
- base、official-CAL 和 long probe selections 之间的不一致；
- 不需要 inference-time oracle 的 failure signatures。

当前 full run 不可用：

- 从 `stopping_trace` 或 step traces 提取的 denoising trajectory summaries。

最新 CPU audit result：

- `analysis/analyze_probe_curve_long_signals.py` 评估了 `4106` 个 single-feature probe-curve thresholds。
- `strict_viable_thresholds = 0`。
- 最佳 threshold 有 `63.04%` true-long precision 和 `31.87%` failed-long recall，但 `8.70%` short-risk，高于 `5%` safety gate。

Success：找到 low short-risk 且有足够 failed-long recall 的 candidate signal 或 learned score，值得启动 smoke GPU run。

Kill：没有 candidate 同时满足 short-risk `<=5%` 和至少 `10` 个 failed-long triggers，除非有清晰记录的 lower-precision/high-recall tradeoff。

### E2：Learned Length Classifier Or Scorer

如果 E1 single-feature rules 失败，在已有 diagnostic fields 上训练或拟合 lightweight length-risk classifier，并使用严格 split discipline 验证。Inference time 不能使用 oracle。

Success：held-out diagnostic precision/recall 超过 hand rules，并保持 short safety。

Kill：classifier 依赖不能跨 model family 或 environment transfer 的 run-specific artifacts。

### E3：Trace-Enabled GPU Smoke Then Full Run

只有在 E1 或 E2 通过 offline gates 后，才在 GPU `2,3` 可用或可以安全等待时运行 smoke experiment。如果下一项 hypothesis 依赖 trajectory information，则 smoke run 必须启用 `--save-step-traces`，确保实际捕获缺失的 trajectory signal。只有 smoke results 未显示 short regression 时，才运行 full `1033`。

Command pattern：

```bash
CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false <runner command>
```

Success：full candidate 改善 total pass rate 和 long buckets，且无 short-bucket regression。

Kill：任何 controlled short-bucket regression，除非被强且有记录的 long-bucket gain 抵消。

### E4：Cross-Model Protocol Alignment

在可用缓存上，用 stable protocol 重跑 Dream-Coder Base/Instruct 和 LLaDA Instruct。只有在模型缓存或网络权限可用时才加入 Dream/DiffuCoder。

Success：method effect transfer，或产生可解释的 non-transfer result。

Kill：prompt/canvas mismatch 使 apples-to-apples interpretation 不成立。

## Compute Budget

CPU-only diagnostics 应先运行，并且现在可以继续。GPU runs 应等待卡可用或使用 wait/queue scripts。任何覆盖 `1033` samples 的 full run 都必须有 manifest，注明 model、command、GPU set、output directory 和 comparison baseline。

## Reproducibility Requirements

每个报告实验必须包含：

- baseline；
- environment；
- GPU set；
- model；
- command；
- output directory；
- total pass rate；
- bucket metrics；
- wins/losses；
- same-hardware 或 cross-hardware comparison label。

## Priority

1. Evidence snapshot 和 paper-agent docs。
2. Long under-selection 的 probe-curve multivariate 或 learned diagnostic scoring。
3. 只有 diagnostic gates 通过后，才启动 trace-enabled GPU smoke experiment。
4. Full A6000 candidate。
5. Cross-model protocol-matched validation。

## Current Decision

使用 `midcons` 作为当前 checkpoint。不要从已有 official-CAL gate family 或 single-feature probe-curve threshold 启动另一个 full true-long GPU run。
