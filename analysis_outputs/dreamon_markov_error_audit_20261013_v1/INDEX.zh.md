# 2026-10-13 全量逐样本错误审计索引

- [离线单文件案例集](casebook.html)：默认326已审失败行，全部2066策略行/10330输出可筛选。
- [主要结论与限制](report.zh.md)、[326条逐条中文人工记录](semantic_reviews.jsonl)、[有限队列](review_queue.jsonl)、[覆盖与全部ID](coverage_audit.json)。
- [全部原字段](per_case_compact.jsonl)、[原题/测试/参考补全](task_contexts.json)、[1630份准确拼接候选](candidates.jsonl)、[1220失败两层标签](failure_labels.jsonl)、[静态/安全CPU证据](official_test_replay.jsonl)。
- [32格及ABD八类](joint_outcomes.csv)、[对称配对及帮助/伤害ID](paired_recheck.csv)、[完整性](integrity_recheck.json)、[来源](source_manifest.json)、[未保存证据与最小补证](missing_evidence.json)、[机制证据等级](mechanism_evidence_summary.json)。
- [隔离探测](isolation_probe.json)、[数据验证](verification.json)、[HTML交互验证及无浏览器限制](html_verification.json)、[既有匿名GPU背景引用](gpu_background_summary.json)。

历史诊断、1,588格审核与CI保留在相邻旧目录。唯一新分支`codex/dreamon-markov-error-audit-v1`；本地completion和runner_exit在提交/退出后记录实际远端核验。语义审阅已全量完成，首次生成分歧unknown不等于未审阅。没有后续研究自动运行。
