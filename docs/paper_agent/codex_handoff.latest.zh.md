# Codex Handoff Latest

更新日期：2026-07-07 UTC

## 0. H200 新服务器迁移状态

当前恢复已完成 bootstrap 审计、Tier 1 H200 core baseline full reruns、H200 action-bank rebuild、Controller V1 replay 和 CPU-only material drift triage。研究者已接受当前 H200 结果作为新的 evidence base；`h200_material_outcome_drift` 继续作为 reproducibility/audit 事实记录，但不再阻塞 Controller V2。旧 A6000 结果保留为 historical reference，后续 controller、validation、action bank、baseline 和论文主表以 H200 rerun 结果为准。

- Bootstrap output：`analysis_outputs/h200_bootstrap_20260705_103617/`
- Bootstrap report：`analysis_outputs/h200_bootstrap_20260705_103617/report.md`
- Environment manifest：`analysis_outputs/h200_bootstrap_20260705_103617/environment_manifest.json`
- New server doc：`docs/paper_agent/new_server_h200_bootstrap.zh.md`
- Bootstrap verdict：`host_h200_available_sandbox_gpu_hidden`
- GPU 状态：默认 Codex 沙箱内 H200 不可见，`nvidia-smi` 失败且 `torch.cuda.is_available() = false`；approved host/unsandboxed check 中 `nvidia-smi` 正常，H200 空闲，`dllm_env` 中 `torch.cuda.is_available() = true`、`gpu_count = 1`。后续 GPU 实验必须使用 approved unsandboxed/escalated command。
- GitHub：SSH deploy key 已生效，`ssh -T git@github.com` 认证成功；`git ls-remote origin refs/heads/codex/risk-controlled-dynamic-rescue` 返回 `2b0662bfe9fdab787a5249dc9cbefea12d683af1`，与本地 HEAD 一致。见 `analysis_outputs/h200_bootstrap_20260705_103617/GITHUB_REMOTE_VERIFICATION.md`。
- Tier 1 audit：`analysis_outputs/h200_repro_audit_20260707_tier1_v2/`
- Tier 1 H200 results：Control `787/1033`、Midcons `794/1033`、Route2 `795/1033`、V6 `796/1033`、Local CAL `769/1033`。
- Old-vs-H200 drift：Route2 `-6`、V6 `-6`、CAL `-5`; report records paired H200 wins/losses and candidate-hash agreement.
- H200 action bank：`analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/`，`927` train/calibration/validation tasks，`4635` rows，test rows `0`，benefit/harm non-KEEP `193/1237`。
- H200 Controller V1 replay：`analysis_outputs/controller_validation_h200_20260707_v1_replay/`，selected controller remains zero-intervention `probe_only + benefit_only` at threshold `999.0`; validation `89/127`，wins/losses vs primary `0/0`，gate failed，test decision `sealed`。
- H200 action-bank/V1 comparison audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`；old-vs-H200 action-bank outcome agreement `94.95%`，V1 old `90/127` vs H200 `89/127`，Controller V2 allowed `false`。
- H200 material drift triage：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`；triage verdict `material_drift_confirmed_controller_v2_blocked`，core public split flip rows `53`，action-bank flip/label-change rows `267`，row-level test details suppressed。
- H200 evidence-base decision：`docs/paper_agent/h200_evidence_base_decision.zh.md`；drift accepted, Controller V2 authorized on H200 train/calibration/validation。
- Frozen test：仍为 `sealed`，`test_evaluation_count = 0`。
- 未完成：Controller V2 feasibility/validation、true-long replay、frozen test。

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 分支：`codex/risk-controlled-dynamic-rescue`
- 本轮 H200 action-bank/V1 replay source commit：`c05d2f8191c8ff31cec2e7472970262ed9d01526`。
- 当前阶段：Phase 3 H200 Controller V2；H200 drift accepted as evidence-base decision；历史阶段为 Phase 2 `Frozen Risk-Controlled Canvas Controller`
- working tree：push 前包含本轮代码、compact results 和文档；push 后应为 clean。
- test lock：`analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- test status：`sealed`
- test evaluation count：`0`

当前最可信结论：

> Unknown-length DLLM infilling exhibits two coupled but separable regimes: canvas inadequacy and rescue inadequacy. Missed true-long failures are substantially trigger/canvas-limited, whereas already-triggered failures remain rescue-limited under the current longer-trajectory and trace-remasking action family. The main deployable opportunity is therefore risk-controlled, non-oracle prediction of when and how much to expand.

中文：unknown-length DLLM infilling 至少有 canvas inadequacy 与 rescue adequacy 两个耦合但可分离的 regime。missed true-long failures 在 oracle-sufficient canvas 下有明显可恢复空间；already-triggered failures 在当前 E/F/G longer-trajectory 与 trace-remasking family 下仍然没有恢复。当前可部署机会是风险受控、非 oracle 地预测何时扩展以及扩展到多长。

## 2. 本轮完成内容

H200 Tier 1 reproduction：

- Control H200：`/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_control_20260706_tier1_20260706_031441`，`787/1033`。
- Midcons H200：`/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_midcons_20260706_tier1_20260706_042029`，`794/1033`。
- Route2 H200：`/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_route2_20260707_tier1_rerun_tmux_20260707_032537`，`795/1033`。
- V6 H200：`/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_v6_20260707_tier1_tmux_20260707_044307`，`796/1033`。
- Local CAL H200：`/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_cal_20260707_tier1_tmux_20260707_051939`，`769/1033`。
- Compact audit：`analysis_outputs/h200_repro_audit_20260707_tier1_v2/`，verdict `h200_material_outcome_drift`。

H200 action-bank and Controller V1 replay：

- Action bank：`analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/`，`4635` rows，`927` tasks，split rows train `3225` / calibration `775` / validation `635`，actions all `927`，test rows `0`。
- H200 action labels：pass count `2506`，benefit labels non-KEEP `193`，harm labels non-KEEP `1237`。
- Controller V1 replay：`analysis_outputs/controller_validation_h200_20260707_v1_replay/`；selected controller remains zero-intervention and gate failed; validation `89/127 = 70.08%`，oracle action-bank upper bound `106/127 = 83.46%` with `17` wins and `0` losses。
- Comparison audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`；repro verdict remains `h200_material_outcome_drift`，Controller V2 allowed `false`。

H200 material drift triage：

- Output：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`
- Verdict：`material_drift_confirmed_controller_v2_blocked`
- Core aggregate drift remains Route2 `-6`、V6 `-6`、CAL `-5`; row-level core flip CSV only covers train/calibration/validation (`53` rows) and suppresses test row details。
- Action bank outcome agreement remains `94.95%`; action-bank flip/label-change rows `267`，split limited to train/calibration/validation。
- Frozen test remains `sealed`，`test_evaluation_count=0`。

Controller V2 status：authorized on H200 evidence base. Do not use test split unless validation gate passes; frozen test remains sealed with `test_evaluation_count=0`.

代码：

- 新增 `experiments/action_ceiling/oracle_canvas_attribution.py`，区分 C oracle-sufficient canvas 与 E/F/G refinement 的增量。
- 新增 `experiments/action_ceiling/frozen_canvas_controller.py`，支持 `freeze`、`bank`、`merge-bank`、`controller`、`frozen-test` 模式。
- 新增 `experiments/action_ceiling/lrdllm_protocol_sanity.py`，完成 LR-DLLM 10-case protocol sanity。
- 新增 `tests/test_frozen_canvas_controller.py`，覆盖 action canvas、feature schema、logistic round-trip、risk bound、keep/reject、`action_passed` evaluation。

结果：

- C/E/F/G incremental attribution：`analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/`
- frozen split/test lock：`analysis_outputs/frozen_controller_20260703_phase2_freeze/`
- controller action bank merged：`analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/`
- controller validation：`analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/`
- LR-DLLM sanity：`analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/`
- LR-DLLM audit doc：`docs/paper_agent/lrdllm_protocol_audit.zh.md`

注意：`controller_validation_20260703_phase2_controller_validation/` 和 `_v2/` 是路径解析与 action outcome 统计修复前的 invalid attempts，不作为研究结论。

## 3. 精确运行方式

冻结 split 和 test lock：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/frozen_canvas_controller.py \
  --mode freeze \
  --timestamp 20260703_phase2_freeze
```

Action bank shard：

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/frozen_canvas_controller.py \
  --mode bank --timestamp 20260703_phase2_bank_shard0 --shard-index 0 --shard-count 2

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/frozen_canvas_controller.py \
  --mode bank --timestamp 20260703_phase2_bank_shard1 --shard-index 1 --shard-count 2
```

Merge：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/frozen_canvas_controller.py \
  --mode merge-bank \
  --timestamp 20260703_phase2_bank_merged \
  --merge-dirs analysis_outputs/controller_action_bank_20260703_phase2_bank_shard0,analysis_outputs/controller_action_bank_20260703_phase2_bank_shard1
```

Controller validation：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/frozen_canvas_controller.py \
  --mode controller \
  --timestamp 20260703_phase2_controller_validation_v3 \
  --bank-dir analysis_outputs/controller_action_bank_20260703_phase2_bank_merged
```

LR-DLLM sanity：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/lrdllm_protocol_sanity.py \
  --timestamp 20260703_phase2_lrdllm_sanity
```

## 4. 结果

### C/E/F/G Attribution

- hard cases：`89`
- C hard recoveries：`29`
- C missed failed-long recoveries：`29/56`
- C triggered failed-long recoveries：`0/33`
- E/F/G any hard recoveries：`31`
- sufficient canvas effect：`29`
- refinement incremental effect over C：`2`
- C fail but E/F/G pass：`SingleLineInfilling/HumanEval/108/L2`, `SingleLineInfilling/HumanEval/89/L4`

### Action Bank

- train/calibration/validation tasks：`927`
- action rows：`4635`
- actions：`KEEP_PRIMARY`, `EXPAND_16`, `EXPAND_24`, `EXPAND_32`, `EXPAND_48`
- merge coverage：expected `4635`, observed `4635`, missing `0`, extra `0`
- source duplicates：`10` duplicate rows from a transient duplicate shard1 resume; merged bank is key-deduplicated by `(task_id, action)`.
- benefit labels non-keep：`191`
- harm labels non-keep：`1242`

### Controller Validation

- controller family：logistic two-head scoring, variants over `probe_only`, `trace_only`, `probe_trace_fused` and three policy variants.
- calibration harm budget：5% upper confidence bound.
- selected controller after gate search：`probe_only` + `benefit_only`, threshold `999.0`.
- calibration result：no nonzero-intervention operating point satisfies the 5% upper-bound harm budget.
- validation controller Pass@1：`90/127 = 70.87%`, equal to V6 on validation.
- validation interventions：`0/127`.
- validation decision：`VALIDATION FAILURE — TEST REMAINS SEALED`.
- frozen test：not run; `test_evaluation_count=0`.

Important ablation signals:

- always-expand-16/24/32/48 all regress strongly on validation.
- oracle action-bank upper bound is `104/127 = 81.89%` with `14` wins and `0` losses, showing bank-level recoverable space exists.
- current inference-visible controller cannot safely select that space under the preregistered harm budget.

### LR-DLLM

- 10-case protocol sanity verdict：`protocol_mismatch_blocked`
- generation executed：`false`
- full run：not run
- reason：no executable official/local Stage I/II LR-DLLM implementation or adapter exists in this repository.

## 5. 研究解释

数据直接支持：

- `31/89` hard-case recoveries mostly come from sufficient canvas, not E/F/G refinement: `29` C-pass vs `2` refinement-incremental.
- Triggered failed-long remains `0/33` recovered under C/E/F/G.
- The deployable action bank has validation oracle upper bound, but unconditional expansion has large harm.
- The current small interpretable controller fails the preregistered risk gate and must not be evaluated on sealed test.

合理推断：

- Main deployable opportunity is canvas/trigger prediction for missed true-long, but the first logistic controller is too conservative under a strict harm budget.
- Candidate selection gap remains plausible but not proven by a successful deployable selector.

尚未验证：

- Whether richer but still interpretable calibration/modeling can recover nonzero validation interventions under risk control.
- Whether any deployable method improves the sealed test split.
- LR-DLLM same-protocol performance.

## 6. 阻塞和风险

- Frozen test remains sealed; no final held-out method result exists.
- Calibration split is small, so a 5% upper confidence harm budget is strict: even zero observed harms at very low intervention counts often has a high upper bound.
- Current feature table is intentionally inference-visible but may be too weak; do not add test-derived features.
- LR-DLLM is blocked by missing official/local implementation, not by result quality.

## 7. 下一步建议

1. Diagnose calibration risk curve and labels.
   - 科学问题：是否因为 calibration sample too small / harm labels too dense 导致 all-zero controller？
   - 输出：risk-coverage diagnostic only, no test access。

2. Improve interpretable controller within frozen protocol.
   - 科学问题：ordinal/survival canvas adequacy head 是否比 current logistic benefit head 更能选择 missed-long cases？
   - 输出：new validation-only controller; test still sealed unless gate passes。

3. Locate or implement LR-DLLM Stage I adapter.
   - 科学问题：same-protocol LR-DLLM 是否是 stronger baseline？
   - 输出：10-case sanity with verdict `local_stage1_adaptation` or `protocol_matched_lrdllm` before any full run。

## 8. 文件索引

优先读：

- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/report.md`
- `analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/report.md`
- `analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/report.md`
- `analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/validation_baselines.csv`
- `analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/verdict.md`
- `docs/paper_agent/frozen_controller_protocol.zh.md`
- `docs/paper_agent/lrdllm_protocol_audit.zh.md`
- `docs/paper_agent/review_manifest.latest.json`
