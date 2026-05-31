# Experiment Plan History

## Plan Revision v0 -> v1

- Timestamp：2026-05-31 12:36 CST
- Trigger：启动长期 autonomous paper-agent 任务，并按要求初始化 versioned paper-agent documents。
- Previous plan：此前不存在 `docs/paper_agent/` plan。最近的 prior plan 是 `docs/superpowers/plans/2026-05-29-ccfa-roadmap-and-next-experiments.md`。
- New plan：Diagnostic-first length modeling plan，使用 A6000 `midcons` 作为当前 checkpoint；在找到更强 offline signal 前，停止继续运行 official-CAL true-long GPU 实验。
- Evidence：`docs/results/a6000_midcons_longrescue_report.md`、`docs/results/long_underestimate_detector_report.md`、`analysis_outputs/experiment_scoreboard.md`、`docs/results/literature_sota_notes.md`、`docs/results/model_generalization_registry.md`，以及已检查的 clean runner/analysis code。
- Reason for change：A6000 `midcons` 对 medium lengths 有正向结果，但 true-long gates 和 offline heuristic sweeps 都是负证据。CCF-A 路径需要更强的 length-modeling mechanism。
- Expected benefit：避免把 GPU 浪费在已知较弱的 trigger family；在保留当前正向证据的同时推进到 paper-level method。
- Risk introduced：计划可能需要更大的工程工作、learned components 或 protocol alignment，下一次正向结果可能不会很快出现。
- What remains unchanged：Raw outputs 保持本地；tracked artifacts 只提交 compact summaries；legacy code 仅作 reference；新实验使用 `expvision_dllm_clean/`、`clean_scripts/` 和 `analysis/`。
- Kill criteria affected：任何新 long detector 在 GPU evaluation 前必须通过 offline short-risk 与 recall gates。
- Related commits or files：`docs/paper_agent/*`，以及 `docs/results/` 和 `analysis_outputs/` 下已有报告。

## Plan Revision v1 -> v2

- Timestamp：2026-05-31 12:58 CST
- Trigger：重新检查 A6000 `midcons` raw output 后发现 full-run results 包含 probe-curve fields，但没有保存 `stopping_trace` 或 `step_traces`；随后运行了 CPU-only probe-curve audit。
- Previous plan：E1 将 trajectory summaries 和 probe-curve shape features 列为候选 diagnostics，且 trajectory diagnostics 排在前面。
- New plan：优先从现有 outputs 做 probe-curve diagnostics 和 learned scoring，再做 trajectory diagnostics。Trajectory diagnostics 现在需要未来的 trace-enabled smoke run，因为当前 full run 的 `rows_with_stopping_trace = 0`。
- Evidence：`analysis/analyze_probe_curve_long_signals.py`、`docs/paper_agent/probe_curve_signal_audit.md`、`docs/paper_agent/probe_curve_signal_audit.json`，以及对 A6000 `midcons` run 的 raw output key inspection。
- Reason for change：现有 `results.jsonl` files 可以立即支持 probe-curve analysis，但不包含 trajectory traces。Probe-curve audit 评估了 `4106` 个 thresholds，发现 `0` 个 strict viable single-feature thresholds。
- Expected benefit：避免规划一个无法从当前 artifacts 完成的 trajectory analysis；在 GPU 被占用时继续保持 CPU-only、evidence-driven 进展。
- Risk introduced：Single-feature probe-curve diagnostics 可能太弱；下一步 method 可能需要 multivariate learned scoring 或 trace-enabled smoke rerun。
- What remains unchanged：offline gates 通过前不启动 GPU run；`midcons` 仍是当前 A6000 checkpoint。
- Kill criteria affected：除非 short-risk 至多 `5%` 且 failed-long trigger count 至少 `10`，否则 single-feature probe-curve threshold 不具备 GPU 资格。
- Related commits or files：`analysis/analyze_probe_curve_long_signals.py`、`tests/test_analyze_probe_curve_long_signals.py`、`docs/paper_agent/probe_curve_signal_audit.*`。

## Plan Revision v2 -> v3

- Timestamp：2026-05-31 15:24 CST
- Trigger：用户明确补充，未来实验应使用 GPU 卡 `2,3`，而不是 `0,1,2,3`，同时不要中断其他用户正在运行的任务。
- Previous plan：Trace-enabled smoke 以及后续 GPU run 会在可用时使用 GPU `0,1,2,3`。
- New plan：除非用户显式更改分配，未来 GPU 实验必须使用 `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false`。Agent 应等待或排队，不得 kill、抢占或中断已有进程。
- Evidence：本会话中的用户直接指令；此前 `nvidia-smi` 检查显示四张卡上均有 Python jobs，因此仍需安全调度。
- Reason for change：当前 compute allocation constraint 比初始计划更窄。尊重 GPU ownership 是 reproducibility 和 collaboration safety 的一部分。
- Expected benefit：避免干扰卡 `0,1` 上的工作，并使后续 experiment manifests 与用户期望的硬件分配一致。
- Risk introduced：两张卡的 full run 可能比原先四张卡计划更慢；任何 same-hardware comparison 都必须明确标注 `2,3` GPU set。
- What remains unchanged：offline gates 通过前不启动 GPU run；`midcons` 仍是当前 A6000 checkpoint；trace-enabled smoke run 仍然需要更强 offline signal。
- Kill criteria affected：科学 gate 不变。Operational gate 现在额外要求 GPU `2,3` 可用，或使用 non-disruptive wait/queue mechanism。
- Related commits or files：`docs/paper_agent/experiment_plan.current.*.md`、`docs/paper_agent/paper_agent_dashboard.*.md`、`docs/paper_agent/open_questions.*.md`、`docs/paper_agent/overnight_log.*.md`。
