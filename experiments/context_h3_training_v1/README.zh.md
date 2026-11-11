# H3 正式训练与恢复

从本worktree启动（所有阶段自动衔接，完成后退出）：

```bash
bash experiments/context_h3_training_v1/launch.sh
```

tmux：`dreamon_h3_20261111_v1`。独立输出 `/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1`；
日志 `pipeline.log`，状态 `progress.json` / `exit.json` / `shell_exit.json`，权重 `checkpoints/{best,last}.pt`。
启动锁和worker锁仅保护本任务。恢复使用同一入口，不重复已经完成的训练或生成。

公共数据只读复用H1 20261107正式库；CPU依赖检查最多72h/每60s，截止时间持久化。
不加载模型等待数据。采集失败/数据损坏/实质协议差异时blocked并退出。公共库不复制。
正式训练直接调用H1的共享train函数；本分支仅在初始化描述中增加H3身份，无独立损失或采样语义。
H2已发布的只读Bank与BF16转FP32适配一并复用，固定代码基底9302a1e。

B heads.py的H3-short-attention保持原样：宽256、4头、一层、learned-attention、全部当前链实际普通历史，
K2≤1条、K4≤3条，无持久案例/链/轮状态。不训练消融或Jacobi。
seed20261107；仅W_out置零，内部随机。FP32、完整词表KL、AdamW、固定lr、有效批128、5轮、
每轮顺序default_rng(seed+epoch)、末批64按实际样本均值，与H1/H2相同。
last每32步持久化优化器、epoch/cursor/step及RNG；OOM减半micro后从最近durable边界恢复，最多重放31步。
micro1仍失败退出，不等待GPU独占。训练阶段不加载DreamOn；评测阶段单份模型，OOM保留训练完成状态直接恢复评测。

技术验证：CPU套件涵盖因果隔离、padding、reset、注意力归一化和单记录权重1、batch/step、梯度、
梯度累积和优化器恢复；GPU技术检查使用真实公共状态并丢弃临时权重/优化器。
初始及每轮完整验证平均KL选best，同分保留更早权重；完整分桶/源等权和部署采样指标一并归档。

训练后best固定执行1033行SingleLine/global/K4，seed42/T=.2/top_p=.9/batch1；
公共Z4、冻结旧头、K1共3099行按A协议审计直接复用；H2完成结果追加配对；H1目前仅6行，标待补。
配对区间按164基础题分组bootstrap，普通失败不重生成。缺少同数据Hctx和消融，不作多词信息或选择性注意力因果主张。

紧凑报告及证据：`analysis_outputs/dreamon_context_h3_training_20261111_v1`。
原始生成/评分/SQLite及权重只存服务器，后台自动归档commit/push；推送失败保留本地提交，不改凭据。
