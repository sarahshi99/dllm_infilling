# 2026-10-13 三起点来源与现有头零训练迁移

主线需要三个起点分别经过DreamOn长度微调后的权重。本轮冻结v2 KL best rank256/λ1，仅检验现有头直迁；没有训练。

|确切DreamOn微调权重|模型及revision|状态|
|---|---|---|
|DreamCoder起点|Dream-org/DreamOn-v0-7B @ 8ccc74750e43177327f29dab9e91882ba759e194|复用完整自身校准；不是跨模型成功|
|Dream起点|缺失|未找到可验证公开权重|
|DiffuCoder起点|缺失|未找到可验证公开权重|

来源边界：固定[官方代码8a0a549](https://github.com/DreamLM/DreamOn/tree/8a0a54918412eda9402a327646f7f067f7160ec8)及[正式ICLR论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/7c020dd308d7413b5078686723b2f857-Paper-Conference.pdf)；[issue6](https://github.com/DreamLM/DreamOn/issues/6)的发布承诺不能当下载地址。v0历史提交、量化及第三方Java衍生不补三底座。官方run脚本明确DreamCoder Base身份，但未披露发布权重实际初始化Base revision。

|原始Base固定64辅助|revision|当前状态|完整行数|
|---|---|---|---|
|Dream-org/Dream-Coder-v0-Base-7B|2346ccd3be517d0d314152b988a3b9bafa7d6d63|running_full|72/8264|
|Dream-org/Dream-v0-Base-7B|6572adb5535263e4d1a337b56942ba48b6dee2a9|blocked_weights_not_ready|0/8264|
|apple/DiffuCoder-7B-Base|a2c33054cbe99b9ab7af074f09b1d13943157ff1|blocked_weights_not_ready|0/8264|

完整普通token→ID、分词规则/文本、特殊词/输出列及一次logits shift审计已存各智能体目录。来源头按原定义strict加载全部参数；零头deepcopy且仅清零output.weight，λ仍1。未命名Base列不赋扩删语义，实际非法token会停目标。

Base原生日程不是严格每轮K；局部参考只补min(K,remaining)与frontier选位，固定64且不读答案长度，EOS留原位，所有组相同截取/skip special规则。现有DreamOn max_gen_len64与论文128分别登记，不能称同一复现。

人口固定1033行/164基础题、单生成种子42、原derive_row_seed；16条字典序等距ID不按正确率挑选。A=K1无头，B=K2无头，C=K2同类同秩零输出头λ1，D=K2来源头λ1。

|DreamOn自身校准顺序|A|B|C|D|D−C帮助/伤害|C−B帮助/伤害|D−B净|D−A净|
|---|---|---|---|---|---|---|---|---|
|global_confidence|951|901|901|910|15/6|0/0|+9|−41|
|left_to_right_frontier|942|877|872|890|29/11|6/11|+13|−52|

以上分母均1033；来源校准D−C净+9/+18。global行加权区间含0，frontier基础题等权区间跨0；单种子重复开发人口，不能推广独立测试或跨模型成功。完整D−C/C−B/D−B和相对K1的帮助/伤害、164题cluster 95%CI见a_dreamcoder/dreamon_reuse_paired.json；保全10330行含历史E机制组，无重跑、无改写旧报告。

Base若完整，逐目标report.zh.md/paired_comparisons.csv给出六个配对和两种权重cluster区间，summary.csv给触发、结构/非相邻回退分母、失败/超时。兼容但无增益只表示现有头本次迁移未成功；只优于无头但不优于零头不能归因为学习；固定长度失败可能受长度误差主导，不自动否定逐模型重训。

本轮累计GPU作业成本 0.134004h / 12.0h；另存历史诊断成本 0.801673h。共享H200计时只作成本，没有正式加速比。互斥锁和所有session在controller/gpu_budget.json，GPU背景/命令/版本/PID/恢复状态在目标manifest及controller清单。下载不计GPU。

后续仅准备材料：future_training_preparation.md列索取信息、论文/代码差异、选择10epochs重建、有效batch128梯度累积、显存理论范围及以后授权pilot估时方法。没有启动任何主模型微调、头训练、K4或隐藏状态头。

命令：scripts/run_zero_training_transfer_dreamcoder_20261013.sh；通用runner支持已登记另两目标，必须先完成权重和技术检查；最终合并报告命令 python scripts/finalize_zero_training_transfer_20261013.py。原工作区未提交修改未动。

验证时HEAD：`d9d19cca9e7bc946fd4719a8fc21590b69eb6db0`。最终交付HEAD见Git提交及最终回复（本文件无需自指未来commit）。
