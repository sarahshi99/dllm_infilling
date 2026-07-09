# Codex Handoff Latest

更新日期：2026-07-08 UTC

## 0. H200 新服务器迁移状态

当前恢复已完成 bootstrap 审计、Tier 1 H200 core baseline full reruns、H200 action-bank rebuild、Controller V1 replay、CPU-only material drift triage、Controller V2 validation 和 Controller V3 candidate screen。研究者已接受当前 H200 结果作为新的 evidence base；`h200_material_outcome_drift` 继续作为 reproducibility/audit 事实记录。旧 A6000 结果保留为 historical reference，后续 controller、validation、action bank、baseline 和论文主表以 H200 rerun 结果为准。

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
- H200 action-bank/V1 comparison audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`；old-vs-H200 action-bank outcome agreement `94.95%`，V1 old `90/127` vs H200 `89/127`。This remains reproducibility evidence, not a Controller V2 stop condition。
- H200 material drift triage：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`；pre-acceptance audit found material drift，core public split flip rows `53`，action-bank flip/label-change rows `267`，row-level test details suppressed。This is superseded by the H200 evidence-base decision。
- H200 evidence-base decision：`docs/paper_agent/h200_evidence_base_decision.zh.md`；drift accepted, Controller V2 authorized on H200 train/calibration/validation。
- Frozen test：仍为 `sealed`，`test_evaluation_count = 0`。
- Controller V2 feasibility audit：`analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility/`，verdict `mixed_controller_failure`。
- Controller V2 validation：`analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/`，verdict `risk_certification_limited_test_sealed`；best nonzero signal is `ordinal_only + probe_trace_fused` with validation `90/127`, wins/losses `5/4`, interventions `41`, population harm upper95 `7.06%`，未通过 gate。
- Controller V3 candidate screen：`analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/`，三族均完成。Family A best top-k 为 `probe_trace_fused, k=15`，validation `91/127`，wins/losses `3/1`，net `+2`，population harm upper95 `3.68%`，但有 `<=8` bucket loss；exploratory route 选择更保守的 Family A `probe_only, k=5`，validation `90/127`，wins/losses `1/0`，population harm upper95 `2.33%`。Frozen-test gate 未通过，route decision `weak_validation_signal_test_sealed`。
- 未完成：true-long replay、frozen test。

## 0.5 Phase 4 泛化审计与论文骨架

当前路线已经停止 Controller V4，不继续新 remasking action、新 V8/V9 heuristic、H200 drift triage 或 frozen test。论文 framing 转为 `diagnostic-driven mixed paper`，不是 positive controller paper。

Phase 4 outputs：

- Phase 4 script：`experiments/phase4_generalization_audit.py`
- Second-backbone feasibility：`analysis_outputs/second_backbone_feasibility_20260708_phase4_v4/`
- Second-backbone diagnostic：`analysis_outputs/second_backbone_diagnostic_20260708_phase4_v4/`
- Second-regime feasibility：`analysis_outputs/second_regime_feasibility_20260708_phase4_v4/`
- LR-DLLM final attempt：`analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/`
- Paper skeleton：`paper/diagnostic_mixed_draft/`

Phase 4 findings：

- Second-backbone feasibility verdict：`recommended_backbone_available`。推荐 `Dream-org/Dream-Coder-v0-Base-7B`，因为本机 Hugging Face cache、official-canvas runner、evaluator compatibility 和历史 full SingleLine evidence 均存在。
- Second-backbone diagnostic status：initial extraction `analysis_outputs/second_backbone_diagnostic_20260708_phase4_v4/` 已被 Full Access fresh oracle run `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/` 补齐；结果为 mixed second-backbone evidence，而不是 clean cross-backbone confirmation。
- Second-regime feasibility verdict：先前缺少 local official MultiLine/RandomSpan JSONL；已从公开 `loubnabnl/humaneval_infilling` 恢复/export 三个 official second-regime configs 到本地 `data/`，并完成 source-labeled official 120-case manifest、CPU smoke gate、bounded GPU diagnostic、CPU hard-tail manifest、48-case hard-tail diagnostic 和 reviewer-requested full104 fixed hard-tail stress diagnostic。
- LR-DLLM final verdict：`blocked_missing_algorithmic_detail`。当前仍无 protocol-matched official/local Stage I/II adapter；不要称为 official reproduction。
- Frozen test：仍为 `sealed`，`test_evaluation_count=0`。

Phase 4 continuation（2026-07-08）：

- New script：`experiments/phase4_continuation.py`。
- Dream-Coder fresh oracle diagnostic：已完成，输出 `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/`。15-case subset 上 primary/control `7/15`，best simple length policy `7/15`，oracle-sufficient canvas `14/15`；missed-long oracle recoveries `3`，triggered-long oracle recoveries `3`，short regressions `0`。Qualitative agreement 为 mixed：Dream-Coder subset 不干净复刻 LLaDA H200 的 missed-vs-triggered split，因此只能作为 second-backbone diagnostic evidence，不可写成 model-agnostic confirmation。
- Second-regime unblock：先前构造过明确标记的 synthetic minimal regime：`analysis_outputs/second_regime_unblock_20260708_phase4_continue_v3/`，verdict `synthetic_second_regime_minimal_constructed`，`18` cases，short/medium/long 各 `6`。该 synthetic set 仍仅作为 runner/data unblock；官方 first-pass 证据见下。
- Second-regime minimal diagnostic：已在 synthetic subset 上完成，输出 `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/`。Regime 明确标记为 `synthetic_second_regime_minimal`，不是 official MultiLine/RandomSpan benchmark。Control fixed、best deployable cal-lite、oracle-sufficient canvas 均为 `18/18`，canvas-limited fraction `0.0`，rescue-limited fraction `0.0`，deployable policy gap `0`。该结果证明 runner/data unblock 成功，但 subset 太易，不能作为强 second-regime stress evidence。
- Research planning pass：新增协作文件 `docs/paper_agent/idea_board.md`、`docs/paper_agent/experiment_queue.md`、`docs/paper_agent/decision_log.md`。本轮提出并排序 10 个 idea，selected immediate experiments 为 second-regime stress gate、Dream-Coder expanded oracle diagnostic、CPU claim-boundary consolidation。
- CPU claim-boundary consolidation：已完成低成本 CPU 分析，输出 `analysis_outputs/research_planning_20260708_cpu_claim_audit/`。结论：LLaDA H200 attribution 仍是最强机制证据（C oracle canvas `29/89` hard recoveries，triggered-long `0`，refinement incremental `2`）；Dream-Coder 15-case taxonomy 为 `7` canvas-recoverable、`7` stable-pass、`1` rescue/noncanvas-limited，支持 oracle-canvas recoverability 但 missed-vs-triggered split mixed；Controller V3 frozen-gate passing points `0`；synthetic second-regime 仍只是 unblock/sanity。
- Dream-Coder expanded manifest：已完成 CPU-only manifest，输出 `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/`。Manifest 有 `37` 个 train/calibration/validation-derived cases，frozen test rows `0`；预期 GPU 命令已写入 `docs/paper_agent/experiment_queue.md` 的 `EXP-002`。
- Dream-Coder expanded37 oracle diagnostic：已按 queued command 完成一次 bounded GPU run，输出 `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/`。Primary/control `11/37`，best simple `11/37`，oracle-sufficient canvas `26/37`，short regressions `0`。分层 oracle pass：missed_failed_long `3/12`，triggered_failed_long `4/4`，medium_near_long_underselection `6/8`，short_primary_pass_harmable `6/6`，positive_control_recoverable `7/7`。解释：这是 second-backbone diagnostic evidence；因为 Dream-Coder triggered_failed_long proxy 在 oracle canvas 下 `4/4` recoverable，而 LLaDA H200 triggered-long C/E/F/G 为 `0`，不可写成 model-agnostic confirmation。
- Official second-regime data recovery：已完成 CPU-only recovery/schema/evaluator gate，输出 `analysis_outputs/second_regime_official_data_recovery_20260708_cpu_v1/`。从公开 `loubnabnl/humaneval_infilling` 导出本地 `data/HumanEval-MultiLineInfilling.jsonl` (`5815` rows)、`data/HumanEval-RandomSpanInfilling.jsonl` (`1640` rows)、`data/HumanEval-RandomSpanInfillingLight.jsonl` (`164` rows)；first 3 tasks per config schema/evaluator smoke 全通过。
- Official second-regime 120-case manifest gate：已完成，输出 `analysis_outputs/second_regime_official_manifest_20260708_v1/`。Manifest 为 `120` cases，三 config 各 `40`，每 config 的 short/medium/long/extreme 各 `10`；bucket 由 LLaDA tokenizer reference/middle token length 定义。Frozen-controller-test rows `0`，evaluator smoke `12/12`。
- Official second-regime bounded GPU diagnostic：已完成一次 bounded run，输出 `analysis_outputs/second_regime_official_diagnostic_20260708_v1/`。Control fixed64 `34/120`，best deployable cal-lite `36/120`，oracle-sufficient canvas `49/120`；oracle gain vs control `26`，deployable harm vs control `15`，oracle harm vs control `11`。Verdict：`official_second_regime_nontrivial_failures_oracle_recovers_subset_hard_tail_needed`。这不是 easy/near-ceiling，但也不是 clean positive controller evidence；它支持 official second-regime 作为 main diagnostic claim 的压力测试，同时要求 hard-tail 分析。
- Official second-regime hard-tail manifest：按 stop rule CPU-only 构造，输出 `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/`，`104` cases，frozen rows `0`。
- Official second-regime 48-case hard-tail diagnostic：网页/user 已批准 smaller bounded hard-tail GPU diagnostic；已完成一次 bounded run，输出 `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/`。采样组 `12/12/12/12`，source configs `16/16/16`，length buckets `12/12/12/12`，frozen rows `0`。结果：control fixed64 `12/48`，best deployable cal-lite `16/48`，oracle-sufficient canvas `28/48`；genuine canvas-recoverable `24`，rescue/non-canvas `12`，deployable help `13`，deployable harm `9`，oracle harm vs control `8`，first-pass label changes `0`。Verdict：`official_second_regime_mixed_stress_evidence`。二阶段官方数据支持 canvas-sufficiency diagnostic，但同时是 deployable cal-lite 和 rescue/non-canvas limitation 的 scope boundary；full104 reviewer supplemental run 见下一条。
- Official second-regime full104 hard-tail stress diagnostic：reviewer 要求在 GPU budget 不受限时完成固定 104-case hard-tail taxonomy/stress run；已完成一次 bounded run，输出 `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/`。它是 post-first-pass fixed hard-tail stress set，不是 unbiased benchmark aggregate；official first-pass 120-case 仍是 official diagnostic estimate。结果：control fixed64 `18/104`，best deployable cal-lite `20/104`，oracle-sufficient canvas `33/104`；genuine canvas-recoverable `26`，rescue/non-canvas `60`，deployable help `17`，deployable harm `15`，oracle harm vs control `11`，first-pass label changes `0`。Verdict 仍为 `official_second_regime_mixed_stress_evidence`，不授权 deployable controller claims；除非具体 bug 或 preregistered follow-up，停止 second-regime GPU work。
- Controller route-closure table：真实 artifact 已生成于 `analysis_outputs/controller_route_closure_20260708_v1/`，含 `route_closure_table.md`、`route_closure_table.csv`、`summary.json`。V1/V2/V3 均作为 validation-only weak/negative evidence；frozen test remains sealed，`test_evaluation_count=0`，不授权 Controller V4。

CCF-A readiness：可以开始正式写作 diagnostic/mixed paper skeleton，但仍不是 submission-ready。最关键 blocking gaps 已从“official second-regime manifest/GPU evidence 缺失”转为：official second-regime 是 mixed stress evidence 而非 clean deployable-controller claim、Dream-Coder diagnostic 为 mixed/model-dependent、deployable controller 没有 frozen-test improvement、LR-DLLM protocol-matched baseline blocked。

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 分支：`codex/risk-controlled-dynamic-rescue`
- 上一轮已 push HEAD：`99bfd08d67df48090824bb57ff7f7581ad7a1d2c`
- 本轮用户接受的起点 HEAD：`aeea238f5c9752d4e1df60ca29a101eb52f4c211`
- 本轮 hard-tail 任务接受的起点 HEAD：`d9d741f0061b456ffdc43e5fd5f7233fb478a25a`
- 本轮 H200 action-bank/V1 replay source commit：`c05d2f8191c8ff31cec2e7472970262ed9d01526`。
- 当前阶段：Phase 4 generalization audit and diagnostic mixed paper skeleton；Controller V3 已完成且不继续 V4；H200 drift accepted as evidence-base decision。
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
- Comparison audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`；repro verdict remains `h200_material_outcome_drift` as a reproducibility caveat only。

H200 material drift triage：

- Output：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`
- Pre-acceptance audit conclusion：H200 drift is material enough to record and separate from A6000 historical evidence; this no longer blocks Controller V2 after `docs/paper_agent/h200_evidence_base_decision.zh.md`.
- Core aggregate drift remains Route2 `-6`、V6 `-6`、CAL `-5`; row-level core flip CSV only covers train/calibration/validation (`53` rows) and suppresses test row details。
- Action bank outcome agreement remains `94.95%`; action-bank flip/label-change rows `267`，split limited to train/calibration/validation。
- Frozen test remains `sealed`，`test_evaluation_count=0`。

Controller V2 status：authorized on H200 evidence base. Do not use test split unless validation gate passes; frozen test remains sealed with `test_evaluation_count=0`.

Controller V2 result：

- Feasibility audit verdict：`mixed_controller_failure`。Validation recoverable rows `17/127`，harmable rows `67/127`，action-row benefit/harm `31/155`。Best validation recoverability AUC `0.7342` (`trace_only`)，但 calibration recoverable count and harm imbalance make certification hard。
- Variant A `ordinal_only`：best `probe_trace_fused` validation `90/127`，wins/losses `5/4`，interventions `41`，population harm upper95 `0.0706`，gate failed。
- Variant B `pairwise_only`：all feature variants selected zero intervention，validation `89/127`，wins/losses `0/0`，gate failed。
- Variant C `ordinal_pairwise_harm`：selected zero intervention，validation `89/127`，wins/losses `0/0`，gate failed。
- Frozen test remains `sealed`，`test_evaluation_count=0`。

Controller V3 result：

- Script：`experiments/action_ceiling/controller_v3_candidate_screen.py`
- Output：`analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/`
- Required files present：`report.md`, `topk_policy_curves.csv`, `validation_predictions.csv`, `validation_action_selection.csv`, `validation_summary.json`, `validation_baselines.csv`, `validation_ablation.csv`, `oracle_win_distillation.csv`, `cost_risk_pareto.csv`。
- Family A `targeted_missed_long` best top-k by pass count：`probe_trace_fused, k=15`，validation `91/127`，wins/losses `3/1`，net `+2`，population harm upper95 `0.0368`，oracle-win hits `5`；this fails exploratory gate because it has one short-bucket loss。
- Family B `two_stage_rejector` best top-k：`probe_only, k=5`，validation `90/127`，wins/losses `1/0`，net `+1`，population harm upper95 `0.0233`，oracle-win hits `2`；exploratory gate passed, frozen-test gate failed。
- Family C `oracle_win_distillation` diagnostic：`probe_only, k=5`，validation `90/127`，wins/losses `1/0`，net `+1`，population harm upper95 `0.0233`，oracle-win hits `2`；diagnostic only, cannot authorize frozen test。
- Selected exploratory policy：Family A `targeted_missed_long`, `probe_only`, `k=5`，interventions `5`，wins/losses `1/0`，validation `90/127`，population harm upper95 `0.0233`，conditional harm `0/5` with upper95 `0.4507`。
- Route decision：`weak_validation_signal_test_sealed`。
- Frozen-test gate：failed because no deployable policy satisfies the stricter frozen-test criteria with net `>=2` and no short-bucket regression. Frozen test remains `sealed` with `test_evaluation_count=0`。

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

1. 审查 official second-regime first-pass、48-case hard-tail 与 full104 hard-tail 结论（最高优先级）。
   - 科学问题：official MultiLine/RandomSpan/RandomSpanLight 的 first-pass `34/120 -> 49/120`、48-case hard-tail `12/48 -> 28/48` 与 full104 hard-tail `18/104 -> 33/104` 是否应写成 supporting evidence、scope boundary，还是 mixed stress evidence？
   - 当前建议：写成 `mixed stress evidence`。它支持 official regime 上的 canvas-sufficiency diagnostic，但不支持 positive deployable controller claim；120-case first-pass 仍是 unbiased official estimate，full104 只用于 taxonomy/robustness；停止 second-regime GPU work，除非发现具体 bug 或新的 preregistered follow-up。

2. 审查 Dream-Coder expanded37 结果。
   - 科学问题：Dream-Coder 的 mixed evidence 是否足以支撑 model-dependent canvas/rescue boundary，而不是 model-agnostic confirmation？
   - 输出：基于 `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md` 的论文 claim rewrite。

3. 做 baseline pack 和 paper table/figure consolidation。
   - 科学问题：LR-DLLM blocked 后，哪些 protocol-matched / audited baselines 可以公平支撑论文？V1/V2/V3 route closure 如何作为 negative/diagnostic evidence 写入主文？
   - 输出：paper table/figure evidence consolidation；使用 `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`，不继续 Controller V4，不打开 frozen test。

## 8. 文件索引

优先读：

- `docs/paper_agent/idea_board.md`
- `docs/paper_agent/experiment_queue.md`
- `docs/paper_agent/decision_log.md`
- `analysis_outputs/research_planning_20260708_cpu_claim_audit/report.md`
- `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/report.md`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/report.md`
- `analysis_outputs/second_regime_official_data_recovery_20260708_cpu_v1/report.md`
- `analysis_outputs/second_regime_official_manifest_20260708_v1/report.md`
- `analysis_outputs/second_regime_official_diagnostic_20260708_v1/report.md`
- `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/report.md`
- `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/taxonomy_summary.csv`
- `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/case_failure_notes.csv`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/report.md`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/comparison_vs_48.csv`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/taxonomy_summary.csv`
- `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/case_failure_notes.csv`
- `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/report.md`
- `analysis_outputs/controller_route_closure_20260708_v1/route_closure_table.md`
- `docs/paper_agent/controller_route_closure_table_plan.md`
- `analysis_outputs/second_regime_diagnostic_20260708_phase4_fullaccess_v1/report.md`
- `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/report.md`
- `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/report.md`
- `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/report.md`
- `analysis_outputs/controller_action_bank_20260703_phase2_bank_merged/report.md`
- `analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/report.md`
- `analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3/validation_baselines.csv`
- `analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/verdict.md`
- `docs/paper_agent/frozen_controller_protocol.zh.md`
- `docs/paper_agent/lrdllm_protocol_audit.zh.md`
- `docs/paper_agent/review_manifest.latest.json`
