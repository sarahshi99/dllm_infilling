# K2 诊断复核索引（2026-10-13）

当前：[完整CPU复核报告](report.zh.md) · [日期勘误](errata.20261013.zh.md) · [复算审计](reproduction_audit.json)

- [可携带证据说明](../evidence/README.zh.md)、[压缩逐行原始字段](../evidence/per_case.jsonl.gz)、[证据版本/缺失字段](../evidence/evidence_manifest.json)、[固定人口ID](../evidence/population.json)。
- [原精度全单元格比对](all_cell_comparison.csv)、[旧报告数值表比对](published_report_table_comparison.csv)、[独立summary](independent_summary.csv)、[独立配对与CI](independent_paired.csv)、[净损失追回分母](net_loss_recovery.csv)。
- [逐组评测审计](evaluation_groups.csv)、[原错误文本计数](evaluation_reasons.csv)、[13次超时](timeout_cases.csv)、[仅超时不确定的代数边界](timeout_sensitivity.csv)。
- [匿名GPU时间序列](../evidence/gpu_background.csv)、[GPU摘要](gpu_background_summary.json)、[以后独占计时条件/命令，本轮不执行](exclusive_timing_future.zh.md)。
- [原报告，保留](../report.zh.md)、[原运行清单，GPU进程隐私删节](../run_manifest.json)、[原完整性](../integrity_audit.json)。

从仓库根目录：

```bash
python analysis/review_dreamon_markov_k2_diagnosis.py \
  --result-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 \
  --output-dir /tmp/k2_review
```

起点HEAD `340170ce6ae44477027dd7e94ec7a9c0f0910384`；最后归档HEAD由包含本索引的Git提交及最终回复标明。本轮CPU工作完成后停止。
