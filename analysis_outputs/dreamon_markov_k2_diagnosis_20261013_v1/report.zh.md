# DreamOn + Markov K2 诊断：未完成

状态：`blocked_gpu_busy`。阻塞：H200 is occupied by other compute processes; no GPU job launched/continued。

CPU 检查 `46` 项通过，包含临时目录中的合成 10330 行分析链检查；合成数据没有写入本轮结果。GPU 消耗 `0` 小时，模型未加载。

已完成 `0/10330` 行；16 行真实模型检查 `not_run`。未产生完整实验结论；空表表示未测量，不能解读为零收益。

训练出的头究竟有没有帮忙？尚不能归因；旧 D−B 包含路径变化，必须测 D−C。

公平计时后到底快多少？尚未测量；旧 wall 含评测和不对称诊断，不能复用。

让模型重新算，能不能明显救回来？尚未测量。E 实现和 CPU 检查不能替代真实模型结果。

下一轮最值得改哪一件事，为什么？先在空闲 H200 上完成本轮固定协议，获得归因和机制证据后再选算法改动。不得凭缺失结果选新方法。

人口为反复使用的 1033 行/164 基础题开发/机制诊断人口，单生成种子 42。没有访问新的封存测试集、训练或调参。预算 12 GPU 小时，完整批次估算必须等待 16 行真实模型检查。

恢复命令：

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python experiments/run_dreamon_markov_k2_diagnosis.py --original-config /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-k2-eval-v1/analysis_outputs/dreamon_markov_k2_full_20261011_v1/run_config.json --output-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 --budget-gpu-hours 12
```

代码：`codex/dreamon-markov-k2-diagnosis-v1@66c5c2bd66e8b7cfe6afdb4706d7b341f2247f82`。旧结果与旧判断保留。本轮不自动扩展到新方法。

资源证据：唯一 H200 NVL 上有 6 个其他进程，约 91761 MiB 显存占用；详细快照位于 run_manifest.json。0% 瞬时利用率不代表空闲。没有中断任何已有进程，也没有留下 GPU watcher。
