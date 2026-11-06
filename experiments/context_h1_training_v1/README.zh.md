# H1 第一条正式训练

本入口只训练H1。A/B基点为43a52fd/fca5775，合并提交记录在Git历史。
数据/初始化/优化器/评测规则在本次brief.zh.md中预先固定，不修改原实验方案及思维导图。

## 运行与恢复

仓库根目录：`/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h1-training-20261107-v1`。
环境：`/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python`。

```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h1-training-20261107-v1
# 固定前四轨迹实测；完成数据直接复用，smoke头与优化器丢弃
bash experiments/context_h1_training_v1/launch.sh --probe
# 正式后台启动；同命令恢复，只有本任务的文件锁，不等空卡
 tmux new-session -d -s dreamon_h1_20261107_v1 -c "$PWD" 'bash experiments/context_h1_training_v1/launch.sh'
```

运行目录：`/home/shx/.cache/dllm_infilling/context_h1_training_20261107_v1/`。
入口日志pipeline.log，进度progress.json，完成/阻塞exit.json，shell退出shell_exit.json，
GPU观测gpu_observations.jsonl，精确数据桶data_summary.json/coverage.json，
首批实测pilot_measurement.json，容量capacity.json，checkpoint在checkpoints/。
运行完成复调直接退出，不加载GPU、不重生成。意外中断从已完成轨迹/已存优化器步/验证逐状态/生成逐行恢复。
训练每32步保存（最坏重放31步），每轮边界保留完整状态；OOM微批减半并从durable checkpoint重载，最多5次至1。
本任务不终止/暂停其他GPU进程；batch1主模型仍OOM就保存并报告，不自动量化或换模型。

## 公共库复用

`queue.sqlite` 的state行含split、原record/group、K、history_length、order、span、arm、stage、
抽样key及不可变data/trajectory-*.npz分片/offset。数据完全足额才进入训练。
NPZ每行保留全部4槽（K2零padding）的FP32旧hidden、位置、链ID、selected mask、
目标query索引、实际普通词、token状态和query历史mask、FP32全词表old/teacher logits。
audit_x0与audit_time_ids仅供teacher复核，Bank.batch严格只返回九个公共输入字段及teacher。
teacher来自相同DreamOn，用X0的本链实际前缀单独重算，未来词/答案/其他链新词不写入。
语义重复全库不重收，不跨split；每源组/桶最多4；没有用K4一词冒充K2。

后续H2–H4读取：

```python
from experiments.context_foundation.common import Store
from experiments.context_h1_training_v1.bank import RUN, Bank, config
bank = Bank(Store(RUN/'queue.sqlite'), config())
batch, teacher_logits = bank.batch(bank.indices('train')[:16], 'cuda:0')
```

保持同一source split/32桶/数据、seed20261107、零W_out、各自内部正常随机，
同train.py的accumulated_update/evaluate、AdamW固定设置、effective128和5轮上限。
本轮train函数仅初始化H1；后续路线须另建run/checkpoint目录，不覆盖本次优化器。
H1沿用B的gamma=1+tanh(g)，g最后层为0使gamma严格为1，其余内部层保留随机。
头和损失FP32，full-vocab T1平均KL；初始与逐轮完整验证，best严格按KL并列选早者。
验证另报源程序等权与32桶、T1 TV、T=.2/top_p=.9 TV、相同common-u采样改变/teacher不一致。
实际采样指标不参与best选择。

## 自动后续及归档

释放DreamOn后训练；加载best再串行完整1033 SingleLine/global/K4，seed42/T=.2/top_p=.9/batch1。
A三个对照逐项检查代码/人口/模型权重/采样/评分语义，一致则复用，否则解释差异并跑匹配对照。
不重跑F4。生成技术异常落盘，普通失败和timeout不重生成。官方评分参数沿用A。
配对按164基础题cluster bootstrap，报告H1相对零头/旧静态/K1及纯学习恢复比例。
分母非正不报告比例。best epoch0明确是未训练权重；第5轮仍改善只记录、不追加。

完成/阻塞均自动写独立中文report.zh.md和证据索引，追加项目日志，提交推送。
大缓存、checkpoint、逐行输出只在服务器；Git提交代码、CPU证据、曲线、逐桶汇总和索引。
共享GPU耗时是实际观测，不作为公平推理加速或同数据重训其他头的结论。
本批退出后不自动启动其他训练、补训、seed或模型。
