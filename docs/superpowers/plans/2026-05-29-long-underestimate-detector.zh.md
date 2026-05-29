# Long Underestimate Detector 实施计划

> **给 agentic worker：** 英文版是完整可执行计划；本中文版本翻译目标、架构、任务和验证。

**目标：** 构建只读离线 sweep，在消耗 GPU 运行新 long-tail policy 前，识别候选 long-underestimation triggers。

**架构：** 新增一个 analysis script，读取 `results.jsonl`，sweep gate combinations，计算 oracle-labeled diagnostic rates，并写出 compact JSON/CSV/Markdown artifacts。脚本独立于模型加载和现有 runners。

**技术栈：** Python standard library、现有 JSONL result format、`analysis_outputs/long_underestimate_detector/` 输出、直接 `unittest` tests。

---

## Task 1：Diagnostic Sweep Script

新增 `analysis/diagnose_long_underestimate_policy.py`：

- 读取 A6000 `midcons` 的 `results.jsonl`；
- 从已有 fields 中抽取 selected length、oracle length、final source、best long score、ratio、support 等诊断信息；
- 枚举 candidate gates；
- 输出每条规则的 triggers、true-long precision、failed-long recall、short risk、current-pass risk；
- 写出 `sweep.json`、`sweep.csv`、`sweep.md`。

测试：

- 构造小 fixture，确保规则统计和排序正确；
- 运行 `tests/test_diagnose_long_underestimate_policy.py`。

## 执行结果

已对 A6000 `midcons` 结果运行：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/diagnose_long_underestimate_policy.py \
  --results outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/results.jsonl \
  --output-dir analysis_outputs/long_underestimate_detector/a6000_midcons
```

结果：

- 评估 `16776` 条 candidate rules；
- 最佳规则 true-long precision 仅 `35.48%`；
- failed-long recall 为 `36.26%`；
- short-risk 为 `40.86%`；
- 严格可行规则数为 `0`。

结论：当前 result fields 不足以安全触发 true-long rescue。下一步需要 trajectory features、learned length classifier、DreamOn-style dynamic canvas 或 LR-DLLM-style length regularization。
