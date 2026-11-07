# H2 正式训练与恢复

工作目录为本分支git worktree。唯一入口：

```bash
bash experiments/context_h2_training_v1/launch.sh
```

tmux：`dreamon_h2_20261107_v1`。外层launch锁和内层worker锁只保护本任务。
不使用公共GPU独占锁，不修改H1锁、代码或输出。
公共库只读自`/home/shx/.cache/dllm_infilling/context_h1_training_20261107_v1`，
等待最长72小时，每60秒CPU检查，持久化首次截止时间，恢复不会自动延长。
H1 full coverage、源审计、完整初始验证及正式plan共同证明已结束采集；允许H1仍训练时立即接续。
发现公共库缺片、配额/源隔离/配置差异则记录blocked，不补采、不消费部分库。

训练直接调用共享`context_h1_training_v1.train.train`，选择`H2-GRU`、独立run和只读原Bank。
H1默认调用与日程不变，H2没有独立损失或采样器。H2 heads.py原样，full-memory，
单GRUCell宽256，每目标根据相对坐标重读至多三条历史；初始状态由forward每次创建零张量。
FP32 head、旧表示及loss；旧表示已经对齐，适配器只转dtype，不再shift。
共享日程seed20261107，rank256，W_out=0，AdamW lr3e-4、betas(.9,.999)、eps1e-8、wd0、clip1。
有效批128，最后64样本批使用实际样本均值；顺序default_rng(seed+epoch)，5轮，严格完整验证KL择best。

训练阶段无DreamOn；分片访问，不复制280GB缓存。4 GiB保留及空闲显存决定自身上限，原训练器选择
micro32/16/4，OOM保存原日志、按原durable checkpoint恢复并依次减半，micro1失败退出。
last包含优化器、epoch/cursor/step、Python/NumPy/torch/CUDA RNG；每32步保存，故异常时至多重放31步。
这只重放未持久化训练步骤，不重新初始化已完成训练。best若epoch0明确标为初始权重。

所有本地输出在`/home/shx/.cache/dllm_infilling/context_h2_training_20261107_v1`：
`pipeline.log`、`progress.json`、`exit.json`、`shell_exit.json`、`checkpoints/{best,last}.pt`、
`queue.sqlite`、`generated_rows.jsonl.gz`、`per_case.jsonl.gz`、`validation_epoch_*.json`。
本分支`analysis_outputs/dreamon_context_h2_training_20261107_v1`存紧凑结果、证据索引和检查记录。

阶段自动接续：公共依赖 → 技术检查（临时权重丢弃）→ 最多5轮训练 → 复用共同对照 →
固定best/1033行生成 → 官方评分 → 配对统计 → 归档/commit/push → 退出。
评测显存不足记为训练完成、评测受阻；恢复同入口只接续生成，不重训。
普通错误/超时不会重抽。中断的未落盘生成记技术失败，不通过重复生成选择更好结果。
OOM技术中断可在恢复时接续该行，所有尝试均保留在SQLite。

H1当前生成只有6/1033，另1027条dtype失败；H2用单独适配修复，不修改H1。
H1完整1033及配置匹配时才追加配对，否则标待补，不阻碍H2归档退出。
A的三个共同对照使用H1同一复用审计函数，验证模型、人口、原始解码及评分语义后复用3099行。
若语义变化，保留差异证据，先完成H2，不与H1同时自动重跑相同基线。
资源计数区分完整主模型forward、实际提交、head调用、GRUCell执行和有效历史更新；
不声称跨目标缓存加速或公平共享GPU加速。缺少同数据Hctx重训对照，不作多词记忆因果归因。
