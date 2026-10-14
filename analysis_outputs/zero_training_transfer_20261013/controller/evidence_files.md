# 2026-10-14 最终证据清单

- 已提交：DreamCoder Base无损逐行压缩 `dreamcoder_base/per_case.jsonl.gz`（8264行），完整性/交付核验、模型/头/生成代码版本manifest、固定16条技术检查、四组两顺序summary、全部六种配对及164题cluster区间、exit/completion状态。
- 保全复用：`a_dreamcoder/dreamon_reused_per_case.jsonl.gz`（10330行），旧DreamOn自身校准及补充配对，未重复GPU生成。
- 来源与阻塞：A/B/C的source_registry、tokenizer审计、strict head加载、原生参考CPU检查；B/C不完整权重与恢复命令，未测量GPU质量。
- 原始SQLite（17MiB）、runner/analysis日志、GPU背景快照JSONL仅本地保留于当前主控工作树 `analysis_outputs/zero_training_transfer_20261013/dreamcoder_base/`，不提交普通Git。全部SQLite payload与压缩逐行证据一致，见delivery_verification.json。
- 预算/资源/运行队列：`controller/gpu_budget.json`、initial/final_resource_snapshot.json、job_queue.json、delivery_summary.json。共享GPU耗时只算成本，未产生正式加速比。
- 原progress.json保留最后一次成功事务的8264行快照（writer当时status仍running）；最终状态以exit.json、run_manifest.json、integrity_audit.json、completion_status.json为准。
- 报告：`docs/paper_agent/transfer_20261013/report.zh.md`；后续训练只准备材料future_training_preparation.md，没有执行。
