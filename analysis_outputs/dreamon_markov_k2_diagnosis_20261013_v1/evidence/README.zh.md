# 可携带的既有逐行证据

这些是已生成、已评测的原始成功记录，不是从汇总表展开的数据。`per_case.jsonl.gz`保留32个原字段和原浮点精度，包含完整输出文本及词元，不需要模型、分词器、SQLite或源数据即可复算。

- 唯一键：`(policy, arm, task_id)`；`base_task_id`是聚类单位，`line_id`仅标识同题行。
- `seed=42`，`row_seed`沿用原SHA-256逐行派生；头/重算的局部右采样种子公式保存在evidence_manifest的run_context中。这是已有种子算法，不是新增数据hash约束。
- `passed`和`failure_reason`是原评测结果；`timed out`也属于已评测成功保存行中的失败标签。不能把未恢复的运行错误=0解释为评测超时=0。
- `output_tokens`等于保存词元数组长度，`normal_token_count`是累计普通提交数，两者不是同一指标；本次所有`final_remaining_masks=0`。
- `forward_count=base_forward_count+recompute_forward_count`，`outer_round_count=base_forward_count`。E额外前向必须计入总数。
- `adjacent_opportunity_count`、`markov_enabled_count`、`recompute_enabled_count`分别是相邻机会、实际头调用、实际重算；`nonadjacent_fallback_count`是global top-2非相邻回退。不能混用覆盖分母。
- `corrected_structural_stop_count`仅是修正/重算后右词变结构词元而只提交左词的停止；`structural_stop_count`还包含顺序屏障。global K1原并行EOS广播路径不经过顺序屏障，不能从其屏障计数0推断没有结构动作。扩展/删除另有`expand_count`、`delete_count`。
- `generation_sec`是原decode前后CUDA同步计时，`evaluator_sec`是原评测调用墙钟时间；`file_write_sec`仅计第一次SQLite INSERT/commit，遗漏随后UPDATE/commit、progress和journal写入，不是全量写盘成本。
- 评测超时/异常来源、子进程退出码、CPU时间/负载、case和evaluator起止UTC、动作全序列没有保存，详见`missing_original_observations`，不倒推补写。
- `population.json`来自已经发布的旧1033行开发人口ID表；不包含新封存测试。`smoke_checks.json`保留原160个检查摘要，不能替代未保存的逐步动作trace。
- `failure_events.jsonl`为空：导出时原failure journal不存在，观察到0条运行失败事件；不声称能证明历史文件从未被删除。
- `gpu_background.csv`是原1052个历史快照，只保留采样时刻和数值；通过本作业既有PID记录区分自己/其他任务，但不输出PID、用户、命令、路径或设备UUID。包括前次阻塞、预加载、smoke和正式运行时段；不是连续监控，也不能精确对齐逐例评测。

仓库根目录执行（Python>=3.10，仅需NumPy；验证版本见requirements.txt）：

```bash
python analysis/analyze_dreamon_markov_k2_diagnosis.py \
  --result-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 \
  --output-dir /tmp/k2_recomputed
```

也可将本文件夹复制到别处，用`--evidence-dir /path/to/evidence --output-dir /tmp/k2_recomputed`；分析器与`analysis/dreamon_k2_evidence.py`两个CPU源码文件须同在`analysis/`目录。原作者绝对路径不是输入依赖。

统计仅条件于单生成种子42、重复开发人口。观测1.657×/1.818×来自共享H200，轮换不能代替独占计时，不称最终公平加速。
