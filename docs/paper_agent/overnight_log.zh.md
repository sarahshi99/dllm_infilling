# Overnight Log

## Entry 2026-05-31 12:36 CST

- timestamp：2026-05-31 12:36 CST
- current phase：初始化与上下文恢复
- what was done：阅读项目 agent rules，启动长期 autonomous research-agent workflow，并创建 `paper-agent-overnight` 分支。
- evidence or files inspected：`AGENTS.md`、`git status --short --branch`、`git branch --list --all`。
- decision made：在 `paper-agent-overnight` 上工作；将已有 `AGENTS.md` 和 `AGENTS.zh.md` 变更视为用户工作，不 stage 或 revert。
- uncertainty/risk：创建分支时 worktree 已有无关变更；后续 commit 必须只 stage paper-agent files。
- next action：在撰写 paper-agent documents 前，回顾已有 research plans 和 result reports。

## Entry 2026-05-31 12:37 CST

- timestamp：2026-05-31 12:37 CST
- current phase：证据回顾
- what was done：回顾 prior CCF-A roadmap、run registry、A6000 reports、long-underestimate diagnostics、literature notes 和 model-generalization registry。
- evidence or files inspected：`docs/superpowers/plans/2026-05-29-ccfa-roadmap-and-next-experiments.md`、`docs/results/run_registry.md`、`docs/results/a6000_midcons_longrescue_report.md`、`docs/results/long_underestimate_detector_report.md`、`docs/results/literature_sota_notes.md`、`docs/results/model_generalization_registry.md`、`analysis_outputs/experiment_scoreboard.md`。
- decision made：使用 `midcons` 作为当前 A6000 checkpoint；除非新 diagnostics 推翻当前证据，否则不再运行另一个 full official-CAL true-long heuristic run。
- uncertainty/risk：在 prompt/canvas/evaluation settings matched 前，literature comparison 仍只是 suggestive。
- next action：初始化双语 paper-agent design、experiment plan、result、dashboard 和 open-question documents。

## Entry 2026-05-31 12:38 CST

- timestamp：2026-05-31 12:38 CST
- current phase：硬件安全检查
- what was done：在启动任何实验前检查 GPU occupancy。
- evidence or files inspected：`nvidia-smi` 输出显示 GPU 0,1,2,3 上均有 Python processes，每张卡约占用 9.7-10.1 GiB memory，且 utilization 较高。
- decision made：当前不启动 heavy GPU jobs。继续 CPU-only analysis 和文档准备。
- uncertainty/risk：GPU 之后可能可用；任何 queued run 都必须避免中断现有任务。
- next action：创建必需的 `docs/paper_agent/` files。

## Entry 2026-05-31 12:39 CST

- timestamp：2026-05-31 12:39 CST
- current phase：paper-agent document initialization
- what was done：创建双语 dashboard、research design、experiment plan、experiment plan history、experiment results、open questions 和本 log。
- evidence or files inspected：基于 prior result reports 和已检查代码创建 `docs/paper_agent/*.md`。
- decision made：将 research design 和 experiment plan 设为 `v1`；创建时保持 `initial` 与 `current` 同步，并在 history 中记录 initial plan revision。
- uncertainty/risk：第一版文档基于已有 reports；下一步应从 raw outputs 独立验证核心 metrics。
- next action：从已有 `results.jsonl` files 构建 compact evidence snapshot。

## Entry 2026-05-31 12:41 CST

- timestamp：2026-05-31 12:41 CST
- current phase：可复现 evidence snapshot
- what was done：新增一个经过测试的 CPU-only evidence snapshot builder，并从已有 raw outputs 生成 `docs/paper_agent/evidence_snapshot.json` 与 `docs/paper_agent/evidence_snapshot.md`。
- evidence or files inspected：`analysis/build_paper_agent_evidence_snapshot.py`、`tests/test_build_paper_agent_evidence_snapshot.py`、A6000 control 和 candidate `results.jsonl` files，以及生成的 snapshot files。
- decision made：将重新计算得到的 snapshot 作为当前 milestone 的 paper-agent evidence anchor。
- uncertainty/risk：Snapshot 依赖本地 raw outputs；如果这些 outputs 被移动，未来运行必须使用明确的 `--run NAME=PATH` 参数。
- next action：验证文档集合、运行测试，并准备一个 focused commit，排除无关 `AGENTS.*` changes。

## Entry 2026-05-31 12:58 CST

- timestamp：2026-05-31 12:58 CST
- current phase：probe-curve signal audit
- what was done：实现并运行 CPU-only probe-curve signal audit，分析 A6000 `midcons` result。该 audit 在已有 probe-curve fields 上评估 single-feature thresholds。
- evidence or files inspected：`analysis/analyze_probe_curve_long_signals.py`、`tests/test_analyze_probe_curve_long_signals.py`、`docs/paper_agent/probe_curve_signal_audit.md`、`docs/paper_agent/probe_curve_signal_audit.json`，以及 raw `results.jsonl` key inspection。
- decision made：将 current experiment plan 从 `v1` 更新为 `v2`：优先从现有 outputs 做 probe-curve multivariate/learned scoring，并将 trajectory diagnostics 推迟到存在 trace-enabled smoke run 后。
- uncertainty/risk：最佳 single-feature threshold 仍有 `8.70%` short-risk，高于 `5%` gate；multivariate scoring 仍可能失败。
- next action：验证、commit 并 push probe-curve audit milestone；然后设计 learned-score diagnostic。

## Entry 2026-05-31 15:24 CST

- timestamp：2026-05-31 15:24 CST
- current phase：plan alignment 与 GPU allocation 更新
- what was done：纳入用户明确补充的 hardware constraint：未来实验应使用 GPU 卡 `2,3`，而不是 `0,1,2,3`。
- evidence or files inspected：`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/research_design.current.en.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/experiment_plan.history.en.md`、`docs/paper_agent/open_questions.en.md`，以及本会话中的用户直接指令。
- decision made：将 experiment plan 从 `v2` 更新为 `v3`；除非用户更改 allocation，未来 GPU commands 必须使用 `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false`，并且必须等待而不是中断已有 jobs。
- uncertainty/risk：两张卡运行可能比此前四张卡草案更慢；所有未来 comparison 都必须显式标注 GPU set。
- next action：重新运行 verification，commit 并 push probe-curve audit 与 GPU-allocation plan update，然后继续 CPU-only multivariate probe-curve scoring。

## Entry 2026-05-31 15:56 CST

- timestamp：2026-05-31 15:56 CST
- current phase：pre-commit verification alignment
- what was done：重新读取 required current plan context，审查 probe-curve audit diff，并将生成的英文 probe-audit interpretation 与中文报告同步。
- evidence or files inspected：`AGENTS.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/research_design.current.en.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/experiment_plan.history.en.md`、`docs/paper_agent/open_questions.en.md`、`analysis/analyze_probe_curve_long_signals.py` 和 `docs/paper_agent/probe_curve_signal_audit.*`。
- decision made：保持本 milestone scope 聚焦于 CPU-only probe-curve diagnostics、双语 paper-agent documentation，以及 GPU `2,3` operational constraint。
- uncertainty/risk：当前环境中 independent subagent review 可能不可用；如果不可用，将记录并使用本地 diff review 与 fresh tests 替代。
- next action：重新生成 probe-curve audit，运行 tests 和 compile checks，请求或模拟 code review，然后只暂存 paper-agent 文件和 analysis/test 改动用于 commit。

## Entry 2026-05-31 21:47 CST

- timestamp：2026-05-31 21:47 CST
- current phase：恢复后的 pre-commit verification 与本地 code review
- what was done：在 context handoff 后，重新阅读项目规则、Superpowers verification 与 code-review 指南、必需的当前 paper-agent context、当前日志和 worktree state。
- evidence or files inspected：`AGENTS.md`、`git status --short --branch`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/research_design.current.en.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/experiment_plan.history.en.md`、`docs/paper_agent/open_questions.en.md`、`docs/paper_agent/overnight_log.*.md`，以及 Superpowers `verification-before-completion` 和 `requesting-code-review` skills。
- decision made：继续在 `paper-agent-overnight` 上工作；不 stage 与本 milestone 无关的用户变更 `AGENTS.md` 或 `AGENTS.zh.md`；由于当前环境中看不到 independent Task/subagent reviewer tool，使用本地 diff review 加 fresh tests 与 regeneration 作为 code-review fallback。
- uncertainty/risk：该 fallback review 弱于独立 reviewer。通过重新阅读计划、检查 diffs、重新生成 derived artifacts，并在 commit 前运行 focused tests 来缓解风险。
- next action：运行 fresh verification suite，重新生成 probe-curve audit，断言关键 JSON facts，检查 diff hygiene；如果所有 gates 通过，则只 commit 并 push 预期 milestone files。

## Entry 2026-05-31 22:11 CST

- timestamp：2026-05-31 22:11 CST
- current phase：commit 前的 probe-curve audit verification
- what was done：运行 focused tests、compile checks，从本地 raw results 重新生成 probe-curve audit，断言关键 JSON facts，执行 diff hygiene checks，并本地审查新增 analysis script、tests 和双语报告。
- evidence or files inspected：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_long_signals.py tests/test_build_paper_agent_evidence_snapshot.py tests/test_diagnose_long_underestimate_policy.py` 报告 `Ran 8 tests` 和 `OK`；`/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_long_signals.py analysis/build_paper_agent_evidence_snapshot.py` 成功退出；`/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_long_signals.py` 重新生成 `docs/paper_agent/probe_curve_signal_audit.json` 与 `.md`，输出 `thresholds=4106 strict_viable=0`；JSON assertion 确认 `strict_viable_thresholds == 0`、`rows_with_stopping_trace == 0`、`rows_with_probe_curve_features == 1033`；`git diff --check -- analysis/analyze_probe_curve_long_signals.py tests/test_analyze_probe_curve_long_signals.py docs/paper_agent` 未返回问题。
- decision made：probe-curve audit milestone 已足以作为 negative diagnostic result commit：single-feature probe-curve thresholds 有信息量，但不应提升为 GPU policy。下一步 research step 保持为 strict-split multivariate 或 learned probe scoring。
- uncertainty/risk：该证据只排除了当前 gates 下的直接 single-feature thresholds；它没有排除 learned probe scoring、trajectory features、dynamic canvas control 或 length regularization。
- next action：只 stage 预期的 analysis、test 和 `docs/paper_agent/` files，排除无关 `AGENTS.md` 与 `AGENTS.zh.md`，然后 commit 并 push 本 milestone。

## Entry 2026-05-31 22:26 CST

- timestamp：2026-05-31 22:26 CST
- current phase：优雅暂停 checkpoint
- what was done：停止新增探索，捕获 `git status --short --branch`，用 pause-state summaries 更新 dashboard 和 evidence snapshot，并写入 `docs/paper_agent/pause_checkpoint.current.md`。
- evidence or files inspected：`git status --short --branch`、`docs/paper_agent/paper_agent_dashboard.en.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/evidence_snapshot.md`，以及已验证 probe-curve audit milestone 的当前上下文。
- decision made：暂停时不改变 experiment plan version；当前计划仍为 `v3`。只 commit 预期的 analysis/test/paper-agent files，并继续排除无关的用户变更 `AGENTS.md` 与 `AGENTS.zh.md`。
- uncertainty/risk：由于当前看不到 Task/subagent reviewer tool，independent code-review workflow 仍处于 blocked。Checkpoint 的 Workflow / Skill Status 表格已显式记录这一点。
- next action：stage 预期文件，使用 `docs: save paper agent pause checkpoint` commit；如果 remote 可用则 push `paper-agent-overnight`，然后停止。

## Entry 2026-05-31 22:33 CST

- timestamp：2026-05-31 22:33 CST
- current phase：恢复后的 CPU-only learned probe diagnostic planning
- what was done：重新阅读 `AGENTS.md`、pause checkpoint、必需的 current paper-agent context、当前 git status，以及相关 Superpowers skills。只为规划一个窄范围后续 diagnostic，检查了已有 probe-curve audit script 和 tests。
- evidence or files inspected：`AGENTS.md`、`docs/paper_agent/pause_checkpoint.current.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/research_design.current.en.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/experiment_plan.history.en.md`、`docs/paper_agent/open_questions.en.md`、`analysis/analyze_probe_curve_long_signals.py`、`tests/test_analyze_probe_curve_long_signals.py` 和 `git status --short --branch`。
- decision made：继续在 `paper-agent-overnight` 上工作，并保留与本任务无关的用户 dirty changes：`AGENTS.md` 与 `AGENTS.zh.md`。将 Superpowers brainstorming 适配为 autonomous-research setting：把 pause checkpoint 和 current experiment plan 视为已批准 spec，然后先写 implementation plan，再改代码。下一项 diagnostic 必须是 CPU-only、deterministic、dependency-free 且 strict-split。
- uncertainty/risk：即使使用 deterministic held-out folds，在同一个 1033-task benchmark 上训练和评估 learned score 仍可能 overfit；该 diagnostic 只能说明是否值得考虑 GPU policy，而不是 deployment policy 本身。
- next action：写 focused implementation plan，先添加 failing tests，再实现 strict-split probe diagnostic，并重新生成 compact evidence。

## Entry 2026-06-01 01:52 CST

- timestamp：2026-06-01 01:52 CST
- current phase：CPU-only strict-split probe-score diagnostic
- what was done：以低 token 模式恢复，将已过期的 pause checkpoint 与当前 dirty files 对齐，运行新的 strict-split probe-score unit test，并从已有 A6000 `midcons` result 生成 CPU-only held-out audit。
- evidence or files inspected：`AGENTS.md`、`git status --short --branch`、`docs/paper_agent/pause_checkpoint.current.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/evidence_snapshot.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/open_questions.en.md`、`overnight_log.*.md` 尾部、`analysis/analyze_probe_curve_split_score.py`、`tests/test_analyze_probe_curve_split_score.py`、`docs/paper_agent/probe_curve_split_score_audit.json` 和 `docs/paper_agent/probe_curve_split_score_audit.md`。
- decision made：将 simple dependency-free multivariate probe-curve score 视为 negative evidence，而不是 candidate GPU policy。
- uncertainty/risk：该 diagnostic 只测试了当前 probe-curve fields 上的一类简单 linear scoring family；它没有排除 constrained high-precision rules、trace-enabled trajectory features、dynamic canvas control 或 length regularization。
- next action：运行 focused verification，更新 dashboard 和 compact docs，然后只 commit 并 push 预期的 analysis/test/documentation files，同时保留无关 `AGENTS.*` changes。

Result summary：

- split discipline：`5` 个 deterministic SHA256 task-id folds，且 thresholds 只在 train folds 上选择。
- rows：`1033`；feature_count：`24`。
- aggregate held-out trigger_count：`63`。
- true_long_precision：`47.62%`。
- failed_long_recall：`32.97%`。
- short_risk_rate：`22.22%`。
- current_pass_risk_rate：`7.94%`。
- strict_heldout_pass：`False`。

## Entry 2026-06-13 23:14 CST

- timestamp：2026-06-13 23:14 CST
- current phase：Route2 full follow-up 收口与下一步 error analysis 准备
- what was done：监控两条 LLaDA-Base Route2 trace-gated long-rescue full runs 到结束，验证日志退出码、row count、summary 和 step traces，并将主表、bucket 表、pairwise、trigger diagnostics 和 failed-long coverage 写入 `experiment_results.*.md`、dashboard、checkpoint 和 current action。
- evidence or files inspected：Broad 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`，Precision 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`，baseline `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`，日志 `logs/paper_agent/20260613_full_route2_broad_gpu2.log` 和 `logs/paper_agent/20260613_full_route2_precision_gpu3.log`。
- decision made：将 Route2 full follow-up 记录为小幅正收益和诊断证据，而不是 true-long problem 已解决。Precision policy 更干净，Broad policy 净增更高但有 short-loss 风险。
- uncertainty/risk：这轮 full run 仍然很 heuristic；`25+` bucket 不提升，且 failed-long triggered rows 大多救不回来，说明 fixed `len=24` rescue 不是充分方案。
- next action：先做 CPU-only Route2 error analysis，比较 triggered-but-still-failed 与 missed failed-long rows，再决定是否设计 adaptive rescue length、更强 generation-side rescue 或新 length signal。
