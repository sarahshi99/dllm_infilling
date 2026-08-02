# Paper-Agent Activity Ledger

## 2026-07-12 UTC

- action：按用户 audit correction 修复 F3/V0 supervision leakage；starting remote/local HEAD `3315ed82d42770e3ed7d8ae20e9d5f1570940ff6` verified `0/0`，clean isolated worktree。
- defect：原 OOF logistic 使用 `passed` labels，且 V0 读取 `score_combined_bridge`；因此它是 supervised correctness controller，原 `outcomes_used_for_selection=false` 不准确。
- correction：split 为 `supervised_probe_diagnostic` 与 fixed deterministic `AST/def-use bridge proxy V0`；V0 schema rejects supervised scores；primary gate 改为 within-task/cross-canvas ranking、deterministic baseline deltas、paired help/harm 与 short safety；global AUROC secondary-only。
- verification interim：`18` focused tests pass；synthetic end-to-end corrected gate smoke correctly kills an insufficient case。Final fresh verification/commit/push pending before H200 retry。
- frozen test：`sealed`，`test_evaluation_count=0`。
- gpu retry：correction commit `5aa86c7a3368e8024449ccf421febfbf9eb78dd2` push 后 exact registered H200 command authorized retry 一次；approval service 再次 pre-launch `422`。candidate rows `0`、active process `0`。写出 manual tmux launcher；operational=`blocked_infrastructure`，scientific=`blocked`。

## 2026-07-11 UTC

- action：开始 Phase 5 independent-method falsification；fetch 后核对 authoritative remote/local HEAD，保留主 checkout 用户已有 untracked paths，并建立 clean isolated worktree。
- evidence：starting HEAD `b8ae031cf3b430833613aaf29083535edc0738f6`；Stage A registry/gap ledger；candidate-bank、F1–F4、conditional V0 scaffold；real-data read-only `164 -> 148` population audit。
- verification：`13` focused tests / `OK`，三个 Phase 5 scripts `py_compile`，JSON/CSV parse，frozen lock sealed-zero audit，compact raw-code scan，`git diff --check`。
- result：Stage A/scaffold pass；Stage B H200 command 在 process launch 前被 approval service 以 `422 model not found: codex-auto-review` 拒绝。GPU process `0`，smoke rows `0`，frozen test count `0`。
- decision：`blocked`。F1–F4 blocked on bank；V0 blocked on F3；不实现 homotopy/birth–death/particle/fusion。
- resume：用户看到 blocker 后明确批准重试 approved H200 command；命令见 `docs/paper_agent/current_action.md`。

## 2026-07-02 00:00 CST

- action：继续上一轮 `superpowers:brainstorming` 后的下一阶段，使用 `superpowers:writing-plans` local fallback 写出 post-V8 CPU-only 执行计划。没有启动 GPU。
- evidence：新增计划 `docs/superpowers/plans/2026-07-02-post-v8-rescue-quality-and-local-guard-plan.md`；更新 `docs/paper_agent/current_action.md`。
- result：计划将下一步拆成两个 CPU audit：一是 rescue failure anatomy，解释 Route2-triggered rows 为什么在长度够时仍失败；二是 local proportional guard audit，检查 V8 少量 long wins 是否能通过 inference-visible guard 与 short/medium losses 分离。
- success criteria：产出 `new_rescue_action_candidate`、`local_prop_guard_candidate`、`diagnostic_only` 或 `stop_true_long_route_for_now` 之一。没有 CPU-positive evidence 前不启动 GPU。
- next：进入 `superpowers:executing-plans` local fallback：先写 CPU audit action brief，再 tests-first 实现 `analysis/post_v8_rescue_quality_audit.py` 和 `tests/test_post_v8_rescue_quality_audit.py`。

## 2026-07-01 16:20 CST

- action：按用户要求使用 `superpowers:brainstorming` 思考下一步。当前环境没有可读取的 `superpowers:brainstorming` skill 文件，因此按项目 protocol 使用 Superpowers local fallback：系统比较多条研究路线，筛掉低价值路线，并形成下一步 `writing-plans` 候选。没有启动 GPU。
- evidence：新增 brainstorm 文档 `docs/paper_agent/experiments/20260701_next_step_brainstorm_after_v8.md`。
- result：推荐主路线是 rescue-quality anatomy + one targeted rescue action，因为 Route2/V3 诊断显示 `31/33` triggered failed-long rows 已经 rescue length >= oracle，继续盲目加长不合理。比例思想保留为伴随 CPU audit：只分析 V8 changed rows，寻找能保留 V8 少量 long wins 且过滤 short/medium losses 的 local guard。
- rejected：不继续扫 V8 beta；不盲目增加 rescue length；不把 selector-only tweak 作为主线；不默认做全局 multi-canvas reranking。
- next：进入 `superpowers:writing-plans` 阶段时，建议写一个 CPU-only plan：Candidate 1 为 rescue failure anatomy audit，Candidate 2 为 local proportional guard audit；没有 CPU-positive 证据前不启动 GPU。

## 2026-07-01 15:40 CST

- action：按用户要求实现并串行运行 V8 proportional CAL score full runs；只使用 GPU1。V8 是对“直接修改 CAL-like 长度评分公式，让长长度获得比例式额外 reward”的忠实实验，而不是 V7 那种 post-hoc near-best 长候选覆盖。
- evidence：action brief `docs/paper_agent/experiments/20260701_v8_proportional_cal_score_full_gpu1_action.md`；V8a 输出 `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`，日志 `logs/paper_agent/20260701_v8a_propcal_beta002_gpu1.log`；V8b 输出 `outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`，日志 `logs/paper_agent/20260701_v8b_propcal_beta004_gpu1.log`；V8c 输出 `outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`，日志 `logs/paper_agent/20260701_v8c_propcal_beta004_cap32_gpu1.log`。
- verification：V8 implementation focused tests 通过 `Ran 8 tests` / `OK`；`expvision_dllm_clean/length_probe.py`、`expvision_dllm_clean/config.py`、两个 runner `py_compile` 通过；runner `--help` 通过；相关 code/test `git diff --check` 通过。三条 full runs 均写出 `1033` 行和 `summary.json`，runner 进程正常退出。
- result：V8a beta `0.02` no cap 得到 `786/1033 = 76.09%`，pairwise vs `midcons` 为 `3/12/783/235`；V8b beta `0.04` no cap 得到 `781/1033 = 75.61%`，pairwise `7/21/774/231`；V8c beta `0.04` reward cap `32` 得到 `782/1033 = 75.70%`，pairwise `6/19/776/232`。
- diagnostics：V8a/b/c 的 short-bucket losses 分别为 `7/13/11`，long-bucket wins 分别为 `2/3/2`。更强 beta 会增加少量 long wins，但 short/medium regressions 更大。V8c 的 `cap=32` 是 proportional reward cap，不是 hard candidate-length cap，因此结果里仍可能出现 `40/48` 候选。
- interpretation：这不是实现失败，而是全局比例式长度 reward 的 policy tradeoff 失败。V8 是用户原始想法的忠实 full-run 检验；结论是当前全局公式不应作为主线。
- next：保留 V6 short override `802/1033 = 77.64%` 作为当前 LLaDA-Base best follow-up。若继续比例思想，必须改成局部 guarded policy：先判断 under-selection，再在 risk guard 下加长，而不是全局给长长度加分。

## 2026-07-01 02:20 CST

- action：完成用户提出的比例式长度放宽 V7 方案的 GPU smoke、bug fix、guard smoke 和 clean full run；使用 GPU2，不启动新的后续 GPU 实验。
- evidence：action brief `docs/paper_agent/experiments/20260630_proportional_length_widening_v7_gpu_action.md`；clean full 输出 `outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`；clean full 日志 `logs/paper_agent/20260701_v7_prop_widen_full_gridfix_gpu2.log`。
- verification：targeted smoke 正常退出；第一条 full run 在 `761/1033` 处暴露 correction-grid 边界 bug；修复后 `tests/test_lcal_official_bounded_repair_proportional.py` 和 `tests/test_proportional_length_widening_audit.py` 通过 `Ran 10 tests` / `OK`，runner 和 audit `py_compile` 通过，`--help` 和 `git diff --check` 通过；guard smoke 正常退出；clean full 有 `1033` 行和 `summary.json`。
- result：V7 proportional widening 得到 `792/1033 = 76.67%`，低于 current `midcons` `795/1033 = 76.96%`，也低于 V6 short override `802/1033 = 77.64%`。Pairwise vs `midcons` 为 `1` win / `4` losses / `791` tie-pass / `237` tie-fail。
- diagnostics：比例放宽只 promotion `10/1033` 行，其中 true-long `0` 行、short `1` 行、current-pass `8` 行；long-bucket win 为 `0`，short-bucket loss 为 `2`。Promoted rows 的平均长度绝对误差从 `1.1` 变成 `6.9`，说明当前参数会把本来接近正确的 short/medium 估计推宽。
- interpretation：这不是 GPU 或实现失败，而是方法信号失败。简单按比例把 near-best 长长度候选放宽，在当前参数下没有解决 true-long，通过率反而下降。
- next：不要继续这个 V7 参数做主线。若保留比例思想，应先做 CPU-first V7b/V8：严格 cap 到 `24/32`、加入 raw-confirm / trace-risk guard、只在明确 under-selection slice 中启用；在新 action brief 前不要启动新 GPU full run。

## 2026-06-18 11:25 CST

- action：按用户要求梳理当前项目进展，生成导师汇报 PPT、讲稿版 Markdown/HTML，以及 CCF-A readiness assessment。没有启动 GPU。
- evidence：输出 `docs/paper_agent/presentations/20260618_advisor_project_report.pptx`、`docs/paper_agent/presentations/20260618_advisor_project_report.md`、`docs/paper_agent/presentations/20260618_advisor_project_report.html`、`ccfa_readiness_assessment.zh.md`；生成脚本为 `docs/paper_agent/presentations/make_20260618_advisor_project_report.py`；action brief 为 `docs/paper_agent/experiments/20260618_advisor_report_ppt_action.md`。
- verification：Discovery V4 最新结果在整理前 fresh check 通过：`tests/test_discovery_v4_signal_audit.py` 为 `Ran 5 tests` / `OK`，`analysis/discovery_v4_signal_audit.py` 的 `py_compile` 通过。PPTX 通过 `python3 -m zipfile -t`，包含 `22` 个 slide XML；生成脚本 `py_compile` 通过；相关 Markdown/HTML/memo 通过 `git diff --check`。
- result：汇报材料的核心 verdict 为 `weak_candidate`。当前项目有清晰研究问题、局部正结果和系统负证据，但 true-long `25+` 未解决、跨 backbone 效果混合、protocol-matched external baselines 和完整 ablations 不足，因此不是 CCF-A submission-ready。
- next：先向导师汇报并确认 paper framing；若继续推进，应从 rescue generation/selection quality 或 principled length modeling 开始，而不是从 Discovery V4 audit 直接启动 GPU full run。

## 2026-06-18 00:00 CST

- action：实现并运行 CPU-only Discovery V4 signal audit。没有启动 GPU。
- evidence：新增 `analysis/discovery_v4_signal_audit.py`、`tests/test_discovery_v4_signal_audit.py`；输出 `analysis_outputs/discovery_v4_signal_audit_20260618_000000`；记录文档 `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`；更新 `docs/paper_agent/current_action.md`。
- verification：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_discovery_v4_signal_audit.py` 通过，`Ran 5 tests` / `OK`；`py_compile` 通过。
- hygiene：真实数据 dry run 先后暴露 `true_long` 和 `triggered_rescue_failure_*` 两类泄漏 candidate；均已加入 forbidden feature 过滤，最终结果不使用 oracle/pass/outcome-derived policy features。
- result：final decision 为 `route2_polish_only`。joined rows `1033`，true-long `113`，baseline failed-long `91`。precision len32 为 `6/0/795/232`、trigger `57`；precision len24 为 `5/0/795/233`、trigger `57`；broad len24 为 `7/1/794/231`、trigger `73`。
- diagnostics：最佳非泄漏 candidate 是 `broad_len24_triggered >= 1`，触发 `73` 行，missed failed-long `4`，triggered rescue-failure `33`，short risk `10`，current-pass risk `1`，true-long precision `0.534`，stable folds `3/5`，因此 reject。
- next：不要从 V4 audit 直接启动 GPU full run。保留 Route2 precision len32 作为 conservative polish；若继续 true-long recovery，应设计 rescue generation/selection quality 机制，并在 GPU 前写新的 action brief 和 success/kill criteria。

## 2026-06-17 18:20 CST

- action：按用户要求继续 true-long 信号搜索，使用 Superpowers local fallback 完成 Discovery V4 的文献启发 brainstorm 和 executable plan。没有启动 GPU。
- evidence：新增 `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`、`docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`、`docs/paper_agent/experiments/20260617_discovery_v4_literature_brainstorm.md`，并更新 `docs/paper_agent/current_action.md`。
- result：V4 将三层 layer 重新定义为：Error/Action Anatomy、Discovery Model Layer、Policy Distillation Layer。核心设计是两个 discovery heads：`MissedLongHead` 找 `56` 个 missed failed-long 的 probe-trace fusion signal，`RescueQualityHead` 解释 `33` 个 triggered failed-long 中为什么 `31/33` 长度够仍失败。
- literature：吸收 risk-controlled selection、slice/subgroup discovery、RuleFit/rule lists、time-series shapelets/ROCKET/catch22、calibration/OOD residuals、weak supervision、counterfactual/uplift diagnostics。
- next：实现 CPU-only `analysis/discovery_v4_signal_audit.py` 和 `tests/test_discovery_v4_signal_audit.py`。没有新的 CPU candidate 通过 held-out risk gate 前，不启动 GPU；GPU `2/3` 有他人任务时不得启动实验。

## 2026-06-17 17:10 CST

- action：完成 CPU-only Route2 error analysis Discovery V3，用来解释 Route2 precision `len32` 小幅正收益背后的失败形态，并为下一轮 Discovery layer 定方向。没有启动 GPU 实验。
- evidence：实现 `analysis/route2_error_analysis.py`；测试 `tests/test_route2_error_analysis.py`；输出目录 `analysis_outputs/route2_error_analysis_20260617_165806`；报告 `analysis_outputs/route2_error_analysis_20260617_165806/report.md`。输入 baseline 为 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`；输入 Route2 为 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`。
- result：诊断复现 `1033` joined rows、pairwise `6/0/795/232`、Route2 triggers `57`、triggered failed-long `33`、missed failed-long `56`、triggered failed-long 中 rescue length >= oracle 为 `31/33`。Dominant bottleneck 为 `mixed_rescue_quality_and_gate_recall`，recommended next path 为 `rescue_generation_quality+gate_recall`。
- interpretation：Route2 的 `6` 个 wins 都来自 triggered rescue，说明 gate 有真实信号；但大多数 triggered failed-long 已经长度足够却仍失败，说明盲目加长不是默认解。与此同时，`56` 个 failed-long 没触发，说明 gate recall 仍不足。
- next：先做 CPU-first 的 rescue generation/selection audit 和 probe-trace fusion gate 设计。除非有新 action brief、success/kill criteria，并确认 GPU `2/3` 没有他人任务，否则不启动新的 GPU full run。

## 2026-06-14 02:40 CST

- action：按用户紧急要求将 Route2 precision `len32` full run 从 GPU1 partial 改为 GPU3-only clean full run，并持续监督到完成。
- evidence：GPU1 partial run 在约 `405/1033` 被中断，不作为最终证据；clean run 使用 tmux session `route2_precision_len32_gpu3_20260614`，日志 `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`。命令使用 `CUDA_VISIBLE_DEVICES=3`；日志以 `COMMAND_EXIT_CODE=0` 结束；`results.jsonl` 有 `1033` valid rows；`summary.json` 存在；`step_traces.jsonl` 非空。
- result：Route2 precision `len32` 得到 `801/1033 = 77.54%`，相对 current `midcons` baseline `795/1033 = 76.96%` 净增 `+6` tasks；pairwise `6` wins / `0` losses / `795` tie-pass / `232` tie-fail。触发 `57` 行，trigger true-long precision `61.40%`，avg sec including probe `5.4622`。
- diagnostics：bucket net 为 `<=8 +2`、`9-12 +2`、`13-16 0`、`17-24 +2`、`25+ 0`。Triggered oracle `17-24` 为 `2/24` pass；triggered oracle `25+` 为 `0/11` pass。与 precision `len24` 相比，总 pass 多 `+1`，但 `25+` 仍无提升。
- interpretation：这是低风险小幅正收益，且比 precision `len24` 稍好；但它没有解决 true-long，尤其没有解决 `25+`。当前瓶颈不是单纯 rescue length 不够，而是 gate recall 和 rescue generation/selection 同时存在问题。
- next：先做 Route2 error analysis，分开列 triggered-but-still-failed true-long rows、missed failed-long rows、short/medium wins，再决定是否设计 adaptive rescue length、better rescue decoding 或 trace/probe fusion gate。

## 2026-06-13 23:14 CST

- action：收口用户批准后的两条 LLaDA-Base Route 2 trace-gated long-rescue full follow-up runs，并将结果写回恢复入口。
- evidence：broad plateau 日志 `logs/paper_agent/20260613_full_route2_broad_gpu2.log`、输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`；precision top1/conf 日志 `logs/paper_agent/20260613_full_route2_precision_gpu3.log`、输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`。两条日志均以 `COMMAND_EXIT_CODE=0` 结束，两个输出均有 `1033` valid rows、`0` malformed rows 和 `summary.json`。
- result：对照 current `midcons` baseline `795/1033 = 76.96%`，broad plateau 得到 `801/1033 = 77.54%`，pairwise `7` wins / `1` loss / `794` tie-pass / `231` tie-fail；precision top1/conf 得到 `800/1033 = 77.44%`，pairwise `5` wins / `0` losses / `795` tie-pass / `233` tie-fail。
- diagnostics：broad 触发 `73` 行，trigger true-long precision `53.42%`，bucket net 为 `<=8 0`、`9-12 +2`、`13-16 +2`、`17-24 +2`、`25+ 0`。Precision 触发 `57` 行，trigger true-long precision `61.40%`，bucket net 为 `<=8 +1`、`9-12 +1`、`13-16 +2`、`17-24 +1`、`25+ 0`。`91` 个 baseline failed-long rows 中，Broad 触发 `39` 但只救回 `2`，Precision 触发 `35` 但只救回 `1`。
- interpretation：Route 2 trace gate 有真实但温和的 full-run 正信号。Precision policy 更干净：无 losses、无 short loss；broad 净增更大但有 `1` 个 short loss。两者都没有解决 true-long，尤其 `25+` 不变；当前瓶颈更像 fixed `len=24` rescue 质量/长度选择，而不只是 gate 召回。
- next：把 precision policy 当作 paper-cleaner incremental evidence；不要继续无设计地调阈值跑 full。下一步先做 triggered-but-still-failed / missed failed-long error analysis；若追求更强 CCF-A claim，需要设计 adaptive rescue length、更强 generation-side rescue 或更强 length signal。

## 2026-06-13 20:55 CST

- action：实现并运行 CPU-only `trace_feature_audit_v2`，用于检查 v1 Route 1/2 公式零触发后是否仍存在更复杂的 trace signal。
- evidence：实现文件 `analysis/trace_feature_audit_v2.py`，测试 `tests/test_trace_feature_audit_v2.py`，最终输出目录 `analysis_outputs/trace_feature_audit_v2_20260613_204721`，报告 `analysis_outputs/trace_feature_audit_v2_20260613_204721/report.md`。
- result：overall decision 为 `diagnostic_only`。Previous source 有 `1033` rows、`113` true-long、`96` failed-long，source decision 为 `policy_candidate`；current `midcons` source 有 `1033` rows、`113` true-long、`91` failed-long，source decision 为 `diagnostic_only`。
- diagnostics：最有希望的 midcons 候选 `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16` 在 held-out 上触发 `18` 行，包含 `9` 个 failed-long、`2` 个 short-risk、`0` 个 current-pass risk，true-long precision 为 `0.500`。这说明 trace features 有信号，但跨源稳定性和固定规则风险仍不足。
- next：不要直接启动 full GPU policy runner。若继续 Route 2，应先写 small GPU smoke action brief，围绕 low top1 / late plateau / low confidence family 做更严格验证。

## 2026-06-12 19:31 CST

- action：完成 trace-long-rescue 的 Task 4/5 full trace collection 和 offline route analysis。
- evidence：previous trace 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`；midcons trace 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`；route analysis 目录 `analysis_outputs/trace_long_rescue_llada_base_prev_20260612_192611` 和 `analysis_outputs/trace_long_rescue_llada_base_midcons_20260612_192611`；report `analysis_outputs/trace_long_rescue_report_20260612`。
- result：previous trace run 验证为 `769/1033 = 74.44%`，有 `35257` trace rows；current `midcons` trace run 验证为 `795/1033 = 76.96%`，有 `35768` trace rows。Route 1/2 在两个 trace sources 上均触发 `0` 行；Route 3 因无 trace signal 而停止。所有 routes 的 Gate A/B 均失败。
- interpretation：这是当前 trace-only / risk-controlled route family 的 negative diagnostic evidence。不应从这批 traces 启动 route-specific GPU policy full run。
- next：policy-runner 工作在这里停止；除非新的 action brief 定义更强 trace feature family 或不同 route。

## 2026-06-12 18:10 CST

- action：验证 previous local method full trace run 后，启动 current `midcons` full trace run，并保持它是当前唯一 trace-long-rescue GPU job。
- evidence：previous 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`；current tmux session `trace_llada_base_midcons_20260612`；current 日志 `logs/paper_agent/20260612_full_trace_llada_base_midcons_gpu3.log`；current 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`。
- result：previous trace verification 通过：日志 exit `0`、`1033` 个 valid result rows、`35257` 个 trace rows 且覆盖 `1033` 个 task ids、`summary.json` 存在，pass count 为 `769/1033 = 74.44%`。Current `midcons` early health 通过，在 GPU3 上推进到至少 `26/1033` rows 且 traces 非空。
- next：持续监控 `trace_llada_base_midcons_20260612` 到完成；验证 exit `0`、`summary.json`、`1033` valid rows 和非空 linked traces 后，才运行 offline Route 1/2/3 analysis。

## 2026-06-12 17:17 CST

- action：启动 Task 4 previous local method full trace collection，并保持它是 trace-long-rescue 当前唯一 GPU job。
- evidence：tmux session `trace_llada_base_prev_20260612`；日志 `logs/paper_agent/20260612_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`；输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`。
- result：early health 通过。该 run 已加载 `1033` 个任务，创建 `config.json`、`results.jsonl` 和非空 `step_traces.jsonl`，并推进到至少 `204/1033` rows、`6989` trace rows，GPU2 正常工作。
- next：持续监控该 tmux session 到完成；验证 exit `0`、`summary.json`、`1033` valid rows 和非空 traces 后，才允许启动 current `midcons` trace run。

## 2026-06-12 16:59 CST

- action：在启动任何 GPU full trace run 之前，按串行模式完成 trace-long-rescue Task 1/2/3。
- evidence：`docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md` 和 `docs/paper_agent/current_action.md` 通过 markdown diff hygiene；新增 `analysis/trace_long_rescue_features.py`、`analysis/analyze_trace_long_rescue_routes.py`、`analysis/print_trace_long_rescue_report.py` 和 `tests/test_trace_long_rescue_features.py`。
- result：focused verification 通过：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py` 输出 `Ran 6 tests` / `OK`；三个 trace-long-rescue analysis scripts 的 py_compile 通过；Task 1/2/3 相关文件的 `git diff --check` 通过。
- interpretation：offline trace analysis tooling 已就绪，但尚未创建 route-specific policy runner，也尚未启动 GPU trace run。
- next：先打印 exact command/log/output/success/kill criteria，然后只启动 GPU `2` 上的 previous local method full trace run；完成 fresh verification 后才启动 GPU `3` 上的 current `midcons` trace run。

## 2026-06-11 14:42 CST

- action：监控并收口完整 `inclusionAI/LLaDA-MoE-7B-A1B-Base` local same-backbone pair，生成 pairwise/bucket analysis，并同步到 GitHub 可读实验文档。
- evidence：baseline 日志 `logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log` 和 candidate 日志 `logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log` 都以 `COMMAND_EXIT_CODE="0"` 结束。Baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719` 和 candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740` 都有 `1033` 个 valid rows、`0` 个 malformed rows 和 `summary.json`。Pairwise analysis 为 `analysis_outputs/lladamoe_full_pair_20260611_1438`。
- result：candidate `801/1033 = 77.54%`，local `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`；pairwise 为 `31` wins、`7` losses、`770` tie-pass、`225` tie-fail；avg sec including probe 为 `10.6107`，baseline 为 `8.7025`。
- diagnostics：oracle bucket delta 为 `<=8 +9`、`9-12 +5`、`13-16 +4`、`17-24 +6`、`25+ 0`。True-long trigger precision 仍偏弱：`official_repair_true_long_precision = 11.54%`，`official_long_suspicion_true_long_precision = 40.00%`，`official_mid_rescue_true_long_precision = 13.04%`；`under_select_rate_17plus = 91.15%`。
- interpretation：这是当前最强 local transfer result，可支持 LLaDA-MoE 上的 local same-backbone improvement claim，但不是 external SOTA claim。文献 anchors 必须继续与 protocol-matched local controls 分开。
- next：没有新 action brief 前不要启动另一个 GPU full run。下一步研究应先设计更强 length signal 或 ablation plan，或明确选择尚未覆盖的 checkpoint variant。

## 2026-06-11 11:30 CST

- action：完成 `inclusionAI/LLaDA-MoE-7B-A1B-Base` smoke gate，并启动 full local same-backbone baseline/candidate pair。
- evidence：下载恢复日志 `logs/paper_agent/20260611_1100_lladamoe_aria2_resume.log` exit `0`；smoke baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427`，candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452`，两者均为 `2/2` 且日志 exit `0`。Full baseline tmux session `lladamoe_full_base_20260611_1126`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`；full candidate tmux session `lladamoe_full_candidate_20260611_1126`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`。
- result：full run 正在运行，尚无最终 pass rate。早期健康检查显示 baseline 约 `16/1033`、candidate 约 `11/1033`，GPU2/GPU3 各约 `15-16GB` 显存，无 OOM/import failure。
- diagnostics：LLaDA-MoE 需要 `llmxy` Transformers `4.52.3` 的 `modeling_rope_utils`，但 `dllm_env` 的 `flash_attn_2_cuda` 因 `GLIBC_2.32` 导入失败；当前命令使用 `/tmp/no_flash_attn` shim 隐藏 flash-attn，同时保留 `dllm_env` verifier 依赖。
- next：持续监控两个 tmux sessions 到结束；完成后验证 exit code、`1033` 行、summary，再生成 pairwise/bucket/runtime analysis 和本地/历史/文献对比表。

## 2026-06-10 21:36 CST

- action：继续推进 `inclusionAI/LLaDA-MoE-7B-A1B-Base` download/API gate，并在启动任何 GPU 实验前写好 smoke-pair brief。
- evidence：当前 tmux session 为 `lladamoe_aria2_20260610_1950`；当前日志为 `logs/paper_agent/20260610_1950_lladamoe_aria2_download.log`；download/probe brief 为 `docs/paper_agent/experiments/20260610_1941_lladamoe_download_api_probe.md`；smoke plan 为 `docs/paper_agent/experiments/20260610_2136_lladamoe_smoke_pair.md`。
- result：下载仍在进行中。当前本地 shards 看似已有 `14G`，但这是 `aria2c` 预分配；`.aria2` sidecar 文件仍存在，因此 checkpoint 尚未完整。
- next：继续监控到 tmux session 退出且 `.aria2` 文件消失，然后校验 shard bytes，运行本地 API/load probe，之后才启动两个 2-sample smokes。

## 2026-06-10 19:25 CST

- action：持续监控 `GSAI-ML/LLaDA-1.5` full local same-backbone pair 到完成，验证 row counts 和 exit codes，从 raw rows 重新计算 pairwise/bucket/runtime metrics，并更新 GitHub 可读结果文档。
- evidence：baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`；candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`；pairwise analysis `analysis_outputs/llada15_full_pair_20260610_1923`；最终 brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`。
- result：candidate 为 `818/1033 = 79.19%`，本地同 backbone `cal_lite` LCAS-v3b baseline 为 `817/1033 = 79.09%`；pairwise 为 `18` wins、`17` losses，net `+1` task。Avg sec including probe 为 `6.6453`，baseline 为 `5.4224`。
- diagnostics：candidate 在 oracle `<=8` 净损失 `6` 个任务，在 `9-12` 持平，在 oracle `>=13` 合计净增 `+7`。Official repair 触发 `110/1033 = 10.65%` 行，但 true-long precision 只有 `10.91%`；`110` 个 triggers 中 `82` 个是 oracle `<=8`。
- interpretation：这是 near-tie / slight local positive，不是强 claim upgrade。当前 official-CAL repair family 仍然不是精确的 true-long detector。
- next：继续 literature-backbone matrix，下一个为 `inclusionAI/LLaDA-MoE-7B-A1B-Base`；先通过已配置 proxy/mirror 下载并 probe checkpoint，然后跑 tiny baseline/candidate smokes，通过后才启动 full pair。

## 2026-06-10 17:35 CST

- action：在两个 2-sample smokes 通过后，启动 full `GSAI-ML/LLaDA-1.5` local same-backbone pair。
- evidence：full-run brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`；baseline log `logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log`；candidate log `logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log`。
- result：早期健康检查通过。两边都加载了本地 LLaDA-1.5 shards，读取 `1033` 个 HumanEval-SingleLineInfilling tasks，并开始 decode。虽然卡上已有共享任务，加载后 GPU2/GPU3 总显存仍约 `25.8GB` / `25.9GB`，没有 OOM。
- next：持续监控两个 tmux sessions 到结束，然后做 row-count/JSON sanity、raw-row pairwise/bucket/runtime analysis；在此之前不解释性能 claim。

## 2026-06-09 21:55 CST

- action：完成 `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe，并把结果写入文档。
- evidence：本地模型路径 `/tmp/llada15_probe_20260609`；probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`；当前行动说明 `docs/paper_agent/current_action.md`。
- result：通过 proxy 直连 HuggingFace 可用；`hf-mirror.com` 对该 repo 会跳回 HuggingFace，且 `huggingface_hub` mirror 模式失败。6 个 safetensors shards 已全部下载，并按 HF API/index 做 byte-size 校验，总大小 `16,031,197,144` bytes。Local `AutoConfig`、`AutoTokenizer`、`AutoModel.from_config` 和 CPU/local checkpoint loading 均通过。LLaDA-1.5 使用 `LLaDAModelLM`、`model_type=llada`，config `mask_token_id=126336`，tokenizer 中 `<|mdm_mask|>` 也解析为 `126336`；但 `tokenizer.mask_token` 本身是 `None`。
- caveat：这不是 GPU smoke，也不是性能结果。最新 `nvidia-smi` 显示 GPU `2` 和 `3` 被占用，且当前 sandboxed Python 报告 `torch.cuda.is_available() == False`。
- next：等 GPU `2` 或 `3` 释放且允许沙箱外 CUDA/verifier 执行后，先跑 tiny candidate 和同 backbone `cal_lite` baseline smokes，再考虑 full LLaDA-1.5 pair。

## 2026-06-09 21:05 CST

- action：将刚完成的 DiffuCoder-Base full local pair 同步到恢复入口，并为下一个 literature backbone `GSAI-ML/LLaDA-1.5` 写入行动说明。
- evidence：DiffuCoder full-pair brief `docs/paper_agent/experiments/20260609_1925_full_diffucoder_base_parallel_baseline_candidate.md`；下一步行动说明 `docs/paper_agent/current_action.md`；LLaDA-1.5 probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`。
- result：DiffuCoder-Base candidate 为 `839/1033 = 81.22%`，本地同 backbone `cal_lite` baseline 为 `838/1033 = 81.12%`，pairwise 为 `25` wins、`24` losses。这仍是 near-tie / slight local positive，不是强 bounded-repair claim。
- next：运行使用 proxy 的 LLaDA-1.5 metadata/download probe，检查 config/tokenizer/mask token；只有 API contract 兼容后才启动 tiny GPU smoke。

## 2026-06-09 18:27 CST

- action：持续监控 `Dream-org/Dream-v0-Base-7B` full baseline/candidate pair 直到完成，验证 row count/schema/backend/canvas，从 raw `results.jsonl` 重新计算 same-backbone pairwise/bucket/runtime metrics，并更新 GitHub 可读实验文档。
- evidence：baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`；candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`；两者均有 `1033` 行、`summary.json`、`0` 个 malformed JSON rows，并使用预期 backend/canvas。
- result：candidate 为 `803/1033 = 77.73%`，同 backbone 本地 cal_lite baseline 为 `802/1033 = 77.64%`，pairwise `28` wins、`27` losses，net `+1` task。Avg sec including probe 为 `3.7337`，baseline 为 `3.6494`。这是 near-tie / slight local positive，不是强 claim。
- diagnostics：oracle `>=17` long buckets 合计多 `+3` tasks，但 `9-12` 和 `13-16` 合计少 `-5` tasks。Repair 在 `101/1033 = 9.78%` 行触发，但 triggered rows 中只有 `13/101 = 12.87%` 是 true-long；大多数 triggers 是 short。
- literature positioning：candidate 高于 LR-DLLM Dream-7B single-line `76.7`，DreamOn Dream-7B `88.6` 是 training-based 且明显更高。以上只是 anchors，不是 protocol-matched claims。
- next：继续 backbone matrix，下一个为 `apple/DiffuCoder-7B-Base`。先使用 HuggingFace 镜像或 Git/LFS 做 download/API probe，再做 tiny smoke；runner/canvas/verifier contract 通过前不启动 full run。

## 2026-06-09 14:35 CST

- action：检查已完成的 DreamCoder Base/Instruct full runs，从 raw `results.jsonl` 重新计算 same-backbone pairwise/bucket/runtime metrics，并更新 GitHub 可读实验文档。
- evidence：Base 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`；Instruct 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`；两者均有 `1033` 行和 `summary.json`。完成后的 GPU check 显示没有运行中的 GPU 进程。
- result：Base candidate 为 `832/1033 = 80.54%`，同 backbone 本地 baseline 为 `825/1033 = 79.86%`，pairwise `27` wins、`20` losses，net `+7` tasks。Instruct candidate 为 `834/1033 = 80.74%`，同 backbone 本地 baseline 为 `848/1033 = 82.09%`，pairwise `21` wins、`35` losses，net `-14` tasks。Base 是小幅 local positive；Instruct 是 negative transfer evidence。
- literature positioning：Base 高于 CAL DreamCoder-Base anchors（`70.2` average、`76.2` best shown），低于 LR-DLLM DreamCoder-7B `81.6`；DreamOn DreamCoder `92.1` 是 training-based。以上只是 anchors，不是 protocol-matched claims。
- next：继续 backbone matrix，下一个为 `apple/DiffuCoder-7B-Base`。使用 HuggingFace 镜像，先在 GPU 2/3 上做 download/API probe 和 tiny smoke；runner/canvas/verifier contract 通过前不启动 full run。

## 2026-06-09 12:12 CST

- action：为 `Dream-org/Dream-Coder-v0-Base-7B` 启动 official-canvas LCAL/S3 + official-CAL bounded-repair smoke，并排查运行环境。
- evidence：第一次 smoke 日志 `logs/paper_agent/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2.log` 显示模型加载成功但 `datasets` 在只读 `/home/shx/.cache/huggingface/datasets` 写 lock 失败；复制 HF module/dataset cache 到 `/tmp` 后，dataset-load probe 成功读出 `2` 条任务。第二次 smoke 加载模型和任务后，在 HumanEval verifier 的 `multiprocessing.Manager()` 处因 sandbox 禁止 socket/listener 失败；sandbox 内 CUDA probe 显示 `torch.cuda.is_available() == False`。
- result：这是 environment/sandbox blocker，不是 DreamCoder 方法结果。当前没有有效 pass rate 或 runtime。
- next：需要用户显式批准沙箱外执行，先跑 GPU2 上的 2-sample Base smoke；通过后再启动 Instruct smoke/full run，不能把这两个 failed sandbox output 用于论文比较。

## 2026-06-09 12:31 CST

- action：用户批准后，在沙箱外完成 DreamCoder Base 和 DreamCoder Instruct 2-sample smoke，并准备 full parallel runs。
- evidence：Base 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`，Instruct 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`；两者均有 `2` 行 JSON、`summary.json`、tier1/2/3 verifier、`lcal_v3`/`official_cal` metadata。Base/Instruct smoke 均为 `2/2` pass，且对各自 baseline 前两题均为 `2` tie-pass。
- result：official-canvas adapter 的 runner/schema/verifier/GPU-runtime sanity gate 通过；这不是 full performance result。已写 full run action brief，计划 Base 用 GPU2、Instruct 用 GPU3。
- next：启动两个 full `1033` sample runs，完成后分别对同 backbone baseline 和 literature anchors 做结果整理。

## 2026-06-09 12:34 CST

- action：启动 DreamCoder Base/Instruct full parallel runs。
- evidence：Base tmux session `dreamcoder_base_full_20260609_1231`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`，日志 `logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`；Instruct tmux session `dreamcoder_instruct_full_20260609_1231`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`，日志 `logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log`。
- result：早期健康检查通过；Base 已观察到 `15` 行，Instruct 已观察到 `7` 行；GPU2/GPU3 均在 `95%+` util、约 `15-16GB` 显存，无早期 OOM。
- next：等待完成后做 row-count/JSON sanity、读取 `summary.json`、对各自 same-backbone baseline 做 pairwise/bucket/runtime analysis，并整理文献锚点对照。

## 2026-06-09 11:04 CST

- action：检查已完成的 `GSAI-ML/LLaDA-8B-Instruct + midcons` run，并设计 literature-backbone rerun matrix。
- evidence：candidate summary 位于 `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834/summary.json`；历史 baseline summary 位于 `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/summary.json`；同时核对了 CAL、LR-DLLM、DreamOn 的 PDF 表格。
- result：candidate 为 `815/1033 = 78.90%`，低于同 backbone 历史 baseline `817/1033 = 79.09%`；pairwise 为 `17` wins、`19` losses。已写入 `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- next：不升级 claim；下一项实现目标是 DreamCoder official-canvas bounded-repair adapter，先定义 smoke-run criteria，再考虑 full GPU run。

## 2026-06-04 20:31 CST

- action：按用户确认启动 `GSAI-ML/LLaDA-8B-Instruct` + 当前 `midcons` bounded-repair full run，使用 GPU `2,3`。
- evidence：第一轮直连 `huggingface.co` 启动因网络不可达在模型加载前被 Ctrl-C 中断；随后用 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` 重新启动。日志 `logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log` 显示模型 checkpoint shards 已加载，`Loaded 1033 tasks`，输出目录为 `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`。
- result：在该时间点，run 尚未结束；最近检查时 `results.jsonl` 已有 `62` 行，GPU 2/3 正在使用。该 interim state 已被上方 2026-06-09 final result 覆盖。
- next：等待 full run 完成后做 row-count/JSON sanity check、读取 `summary.json`、和 LLaDA-Instruct 历史 baseline `817/1033 = 79.09%` 做 pairwise/bucket analysis，并更新 results/dashboard/checkpoint。

## 2026-06-04 15:40 CST

- action：对 strict-split probe-score diagnostic 做 final focused verification。
- evidence：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py`、`py_compile`、`analysis/analyze_probe_curve_split_score.py`、JSON assertions、`git diff --check`。
- result：fresh verification 通过：`Ran 3 tests` / `OK`，compile exit `0`，audit regeneration 重现 `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`，JSON assertions 确认 `cross_validation.aggregate_heldout` 关键 metrics，diff hygiene 通过。第一次 JSON assertion 因 schema path 写错失败，root cause 是断言命令错误，不是 audit metrics 改变。
- next：当前结果继续作为 negative evidence；下一步默认做 CPU-only conservative high-precision score kill-test，GPU work 继续 blocked。

## 2026-06-04 15:31 CST

- action：恢复长期 paper-agent session 后，核对 dashboard、checkpoint、experiment results 和 overnight log 的 verification 状态。
- evidence：`AGENTS.md`、`docs/paper_agent/research_agent_protocol.md`、`git status --short --branch`、`pause_checkpoint.current.md`、`paper_agent_dashboard.zh.md`、`experiment_results.zh.md`、`open_questions.zh.md`、`activity_ledger.zh.md` 尾部，以及 verification 相关 `rg` 命中。
- result：确认 dashboard 中 `verification-before-completion=completed` 对应较早 probe-curve audit 的 `Ran 8 tests` / `OK`；strict-split diagnostic 只有 focused `Ran 3 tests` / `OK`，仍需 final milestone-level verification、diff review、commit、push。已修正 checkpoint workflow 表述；没有启动 GPU 或新实验。
- next：先运行 strict-split diagnostic 的 final focused verification，再决定是否做 CPU-only conservative high-precision score kill-test；GPU work 继续 blocked。

## 2026-06-01 01:50 CST

- action：以低 token 模式恢复，并将已过期的 pause checkpoint 与当前 dirty files 对齐。
- evidence：`git status --short --branch`、`docs/paper_agent/pause_checkpoint.current.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/evidence_snapshot.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/open_questions.en.md`，以及 `overnight_log.*.md` 尾部。
- result：确认未提交的 strict-split probe diagnostic 是当前计划中的 CPU-only 下一步；没有启动 gstack workflow 或 GPU experiment。
- next：完成 strict-split diagnostic evidence，并记录它是否通过 offline GPU gate。

## 2026-06-01 01:52 CST

- action：运行 strict-split probe-score 单元测试，并从已有 A6000 `midcons` result 生成 CPU-only audit。
- evidence：`tests/test_analyze_probe_curve_split_score.py`、`analysis/analyze_probe_curve_split_score.py`、`docs/paper_agent/probe_curve_split_score_audit.json`、`docs/paper_agent/probe_curve_split_score_audit.md`。
- result：测试通过；aggregate held-out gate 未通过，结果为 `63` triggers、`47.62%` true-long precision、`32.97%` failed-long recall、`22.22%` short-risk、`7.94%` current-pass risk。
- next：双语记录该 negative result，并保持 GPU work blocked。

## 2026-07-05 10:45 UTC

- action：执行 H200 新服务器 bootstrap / Git / artifact integrity 审计。
- evidence：实际 Git root 为 `/home/shx/projects/dllm_infilling/git_workspace`；HEAD `2b0662bfe9fdab787a5249dc9cbefea12d683af1`；H200 在 `/proc/driver/nvidia/gpus/0000:22:00.0/information` 可见，但 `nvidia-smi` 失败，`/dev/nvidia*` 设备节点缺失，`dllm_env` 中 `torch.cuda.is_available() = false`。
- result：生成 `docs/paper_agent/new_server_h200_bootstrap.zh.md` 与 `analysis_outputs/h200_bootstrap_20260705_103617/`；initial sandbox-only verdict 为 `h200_environment_invalid`，后续 2026-07-06 host check 已修正为 `host_h200_available_sandbox_gpu_hidden`；review manifest 增加 `server_migration` 字段；frozen test 仍 `sealed` 且 `test_evaluation_count=0`。
- github：当时 remote URL 存在，但 `git ls-remote` 和 `ssh -T git@github.com` 因 sandbox escalation approval rejection 未执行；该状态已在 2026-07-06 被新的 SSH/branch verification 取代，见 `analysis_outputs/h200_bootstrap_20260705_103617/GITHUB_REMOTE_VERIFICATION.md`。
- next：当时计划修复 GPU device nodes / driver 可用性并验证 GitHub；该路径后续修正为使用 approved unsandboxed/escalated GPU commands。

## 2026-07-05 10:55 UTC

- action：尝试将 H200 bootstrap audit 做 focused staging/commit。
- evidence：计划 stage `docs/paper_agent/new_server_h200_bootstrap.zh.md`、`analysis_outputs/h200_bootstrap_20260705_103617/` 以及相关 handoff/dashboard/manifest docs。
- result：`git add` 需要写 Git index，已按 sandbox policy 请求 escalation，但审批层在执行前拒绝，错误为 `codex-auto-review` model not found / 422。未创建 commit，未 push。
- next：需要在可正常批准 Git index/network 操作的会话或用户终端中执行 focused `git add`、`git commit`、`git push`；不要把未跟踪的 `scripts/bootstrap_remote_10_98_36_183.sh` 或 `scripts/no_flash_attn/` 混入提交。

## 2026-07-06 UTC

- action：恢复 GitHub SSH deploy-key access，并重新验证 H200 bootstrap 的远端状态。
- evidence：`ssh -T git@github.com` 返回 `Hi sarahshi99/dllm_infilling! You've successfully authenticated, but GitHub does not provide shell access.`；`git ls-remote origin refs/heads/codex/risk-controlled-dynamic-rescue` 返回 `2b0662bfe9fdab787a5249dc9cbefea12d683af1`，与本地 HEAD 一致。
- result：GitHub auth / branch freshness blocker 解除；H200 GPU blocker 仍存在：`nvidia-smi` 失败、`/dev/nvidia*` 缺失、`dllm_env` 中 `torch.cuda.is_available() = false`。
- next：做 focused bootstrap commit/push；该 GPU blocker 后续修正为 sandbox visibility issue。

## 2026-07-06 UTC Host GPU Visibility Correction

- action：区分默认 Codex sandbox 与 host/unsandboxed H200 可见性。
- evidence：默认 sandbox 中 `/dev/nvidia*` 不可见，`nvidia-smi` 失败；approved host/unsandboxed `nvidia-smi` 成功，显示 `NVIDIA H200 NVL`、driver `580.159.03`、memory `143771 MiB`、无运行进程；approved host/unsandboxed `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 可见；approved host/unsandboxed `dllm_env` PyTorch 报告 `cuda_available=true`、`gpu_count=1`。
- result：H200 host environment 可用；此前 `h200_environment_invalid` 应修正为 `host_h200_available_sandbox_gpu_hidden`。后续 GPU 实验必须通过 approved unsandboxed/escalated commands 运行。
- next：commit/push bootstrap correction，然后启动 H200 Tier 1 reruns。

## 2026-08-01 UTC DreamOn Progressive V2-Hard

- action：在独立分支 `codex/dreamon-progressive-v2-slots` 冻结 642-row protocol，完成 slot-aware generator/runner 与 41 个 focused tests，并依次执行 V2-Hard smoke、30-case pilot 和 full 642。
- evidence：`repro_results/dreamon_progressive_v2_protocol/`、`repro_results/dreamon_progressive_v2_hard_all642/`、`docs/paper_agent/experiments/20260801_dreamon_v2_hard_full.md`。
- result：V2-Hard `7/642 = 1.09%`，compile `46/642`，exact `3/642`；421 条显式 256-forward cap failure，0 disallowed protocol violation，0 runtime error，0 post-hoc truncation。该 oracle exact-three-line structural diagnostic 是强负结果，削弱 hard-slot-only stable progressive decoding 假设。
- next：按冻结顺序执行 V2-OpenTail，不因 V2-Hard Pass@1 下降停止。

## 2026-08-01 UTC DreamOn V2 decoder protocol iterate

- decision：旧 V2 解码协议被确认存在 newline blanket ban、单-mask EOS delete 和缺失 exact-transition cycle detection 三个混杂因素；研究决策改为 `iterate`。
- archive：V2-Hard-v1 642-row 原始结果保持只读并标注为 decoder/protocol failure diagnostic；V2-OpenTail-v1 partial357 移入 `repro_results/abandoned/dreamon_v2_opentail_protocol_v1_partial357/`，`resumable:false`；Joint-v1 未启动。
- action：本轮只实现 `v2_hard_v2_boundary` protocol version 2，经过严格 smoke/pilot 门禁后才允许 full，并在该方法结束后停止。
- protocol：`repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json`。
- execution：78 tests/static gates 通过；Smoke 5 为 5/5 completed、0 cycle/error；Pilot 30 为 25/30 completed、5 exact cycles、Pass@1 11/30、compile 19/30、exact 5/30。
- gate：pilot 仅通过 Pass@1 floor；completion/zero-cycle/zero-unresolved/zero-protocol/compile floors 失败。按预注册规则 `stop_before_full`，未创建 full 结果，未启动 OpenTail-v2/Joint-v2。

## 2026-08-02 UTC DreamOn V3 iterate_and_execute

- action：完成 A official cumulative budget 与 B oracle nonempty guard 的测试、Cycle5/Affected6、Pilot 和获授权 Full。
- result：A Pilot 14/30，未跑 Full；B Pilot 18/30；B Full 280/642 Pass、471/642 compile、642/642 completed、0 protocol/runtime failure。
- interpretation：budget 解释旧 cycle 误终止，nonempty 解释部分 blank failure；剩余错误未被两者解释。B 低于 one-shot/V1，decision=`reframe`。
