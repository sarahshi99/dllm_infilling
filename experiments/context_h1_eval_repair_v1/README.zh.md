# H1评测补充（不训练）

入口：`bash experiments/context_h1_eval_repair_v1/launch.sh`。
使用原H1正式best epoch5，固定SingleLine/global/K4/1033/seed42/T=.2/top_p=.9/batch1。
只修复已核验的BF16输入与FP32头适配，不修改模型、权重或解码/评分语义。
保留原始6条成功、1027条技术失败的归档；成功行复用，dtype技术失败允许本次补跑。
新运行独立写入`/home/shx/.cache/dllm_infilling/context_h1_eval_repair_20261111_v1`。
共有pipeline.log、progress.json、exit.json、shell_exit.json、queue.sqlite、generated_rows.jsonl.gz和per_case.jsonl.gz。
同入口恢复，不重训，不重复已完成生成/评分，不重抽普通失败取优。

共享cuda:0，仅一个GPU工作进程、一份DreamOn。显存留4GiB，OOM记录blocked退出，不停止他人任务。
A的三个基线直接复用；追加H2配对。完成后自动报告、归档commit/push和退出。
完成时在原H1 run追加evaluation_supplement_20261111_v1.json索引，旧结果不覆盖。
