# DreamOn V2-OpenTail protocol-v1 partial357：已废弃

该目录是旧 `v2_opentail` protocol version 1 的部分 full generation，已于 2026-08-01 UTC 从活跃 resume 路径移入 `repro_results/abandoned/`。

- predictions：357 行，357 个唯一 task_id；
- 顺序：冻结 population 的严格前缀；
- 正常 completed：188；
- `forward_cap_with_unresolved_masks`：169；
- runtime/invariant error：0；
- 最后完成任务：`MultiLineInfilling/HumanEval/108/L2_L4`；
- 下一条未完成任务：`MultiLineInfilling/HumanEval/108/L3_L5`；
- full scoring：未执行；
- `resumable: false`。

废弃原因是旧协议已确认存在解码缺陷：hard slot blanket newline ban、EOS 只删除单个 selected mask，以及缺少精确 deterministic-transition cycle detection。该目录不得恢复，不得与 protocol-v2 rows 混合，也不得用于描述完成的 V2-OpenTail 结果。完整机器记录见 `abandonment.json`。
