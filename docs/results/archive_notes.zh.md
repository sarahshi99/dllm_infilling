# 结果归档说明

原始实验输出保留在本地：

- `/home/shx/projects/dllm_infilling/outputs_clean`

普通 git 只保存紧凑记录：

- `docs/results/run_registry.md`
- `docs/results/run_registry.json`
- `docs/results/historical_result_analysis.md`
- `docs/results/model_generalization_registry.md`
- `docs/results/model_generalization_registry.json`
- `analysis_outputs/experiment_scoreboard.md`
- `analysis_outputs/` 下的紧凑 CSV/JSON 诊断。

Canonical historical runs：

- `old_union_gpus23`：旧全局 checkpoint，`787/1033 = 76.19%`。
- `union_gpus01_control`：新环境中的同策略 control，`785/1033 = 75.99%`。
- `eval12_nomiddle_gpus01_control`：gate-control run，与 `union_gpus01_control` 有相同 pass/fail 集合。
- `midcons`：新环境中最佳 candidate，`791/1033 = 76.57%`。
- `midaggr`：record-only candidate，wins 更多但短/中长度损失过多。

Cross-model 本地记录：

- `model_generalization_runs/20260512_114917_lcas_v3_full`：LLaDA 与 Dream-Coder 的 LCAS v3 runs。
- `model_generalization_runs/20260513_dreamcoder_official_full`：Dream-Coder official-canvas full runs。

解释：

- `docs/results/historical_result_analysis.md` 说明哪些 runs 是 canonical，哪些方向已被 superseded，以及为什么当前 A6000 `midcons` 是最佳同硬件 checkpoint。

Artifact 策略：

- 不要把完整原始 `outputs_clean/` 提交到普通 git。
- 只有明确需要逐字保留的选定 raw 文件才使用 Git LFS 或外部存储。
- 没有用户明确许可，不要在归档工作中删除本地 raw outputs。
- Registry 默认排除 smoke runs 和缺少 summary 的 incomplete runs。只有做审计时才使用 `analysis/build_run_registry.py --include-smoke` 或 `--include-incomplete`。
