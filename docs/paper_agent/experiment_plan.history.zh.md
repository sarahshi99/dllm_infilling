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
