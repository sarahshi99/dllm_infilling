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
|Dream-org/Dream-Coder-v0-Base-7B|2346ccd3be517d0d314152b988a3b9bafa7d6d63|completed_generation|8264/8264|
|Dream-org/Dream-v0-Base-7B|6572adb5535263e4d1a337b56942ba48b6dee2a9|blocked_download_incomplete_final|0/8264|
|apple/DiffuCoder-7B-Base|a2c33054cbe99b9ab7af074f09b1d13943157ff1|blocked_incomplete_weights|0/8264|

## 2026-10-14 最终结果追加：DreamCoder Base 辅助迁移

2026-10-13 14:06 UTC退出0；完整8264行，缺失/重复/历史技术失败/未恢复错误均0；SQLite与逐行gzip全部payload逐项一致。Dream和DiffuCoder两路仍为本轮下载阻塞，未做GPU评测。

|顺序|K1无头 A|K2无头 B|K2零头 C|K2来源训练头 D|
|---|---|---|---|---|
|global_confidence|596/1033 (57.70%)|469/1033 (45.40%)|465/1033 (45.01%)|486/1033 (47.05%)|
|left_to_right_frontier|592/1033 (57.31%)|362/1033 (35.04%)|387/1033 (37.46%)|482/1033 (46.66%)|

|顺序|配对|帮助/伤害|净行数|行加权Δ95%CI(pp)|基础题等权Δ95%CI(pp)|
|---|---|---|---|---|---|
|global_confidence|D-C|57/36|+21|+2.03 [+0.31,+3.69]|+2.30 [+0.11,+4.59]|
|global_confidence|C-B|0/4|-4|-0.39 [-0.79,-0.09]|-0.26 [-0.59,-0.03]|
|global_confidence|D-B|54/37|+17|+1.65 [-0.09,+3.39]|+2.04 [-0.18,+4.34]|
|global_confidence|D-A|86/196|-110|-10.65 [-13.33,-8.05]|-12.08 [-17.14,-7.30]|
|left_to_right_frontier|D-C|169/74|+95|+9.20 [+6.54,+11.86]|+10.01 [+5.22,+14.85]|
|left_to_right_frontier|C-B|110/85|+25|+2.42 [-0.10,+4.91]|+5.62 [+0.86,+10.60]|
|left_to_right_frontier|D-B|192/72|+120|+11.62 [+8.89,+14.42]|+15.63 [+10.81,+20.58]|
|left_to_right_frontier|D-A|103/213|-110|-10.65 [-14.09,-7.31]|-11.36 [-17.19,-5.68]|

判断：本轮DreamOn来源头在原始DreamCoder Base辅助目标上，相对同路径零头显示可迁移的学习修正贡献，global净+21（+2.03pp）、frontier净+95（+9.20pp），两种权重的164题cluster区间均在0以上。global头−无头净+17的区间仍含0；frontier头−无头净+120包含路径净+25，不能全部归因学习。
但D相对K1两顺序均少110行（−10.65pp）；因此是K2质量损失的部分恢复，未达到K1质量，也不是三份确切DreamOn微调权重的跨模型验证成功。固定长度误差可能影响结果，本轮未做长度归因，不能把落后全部归于长度。单种子42、重复使用开发人口；区间仅反映本种子条件下基础题重采样，不是独立测试泛化证据。

来源训练头触发：global 26444/26462相邻机会=99.93%，frontier 33030/33056=99.92%；global非相邻回退6594/(26462+6594)=19.95%，frontier0/33056。Base无结构动作，结构机会分母0、率不适用；所有画布始终64且无残mask/非法词元。
质量失败A/B/C/D分别global 437/564/568/547、frontier441/671/646/551；其中评测超时global6/4/4/5、frontier10/5/6/8，按原3秒评测规则计失败且不重测挑选。技术失败与评测不通过分开报告。
决策：本轮评测及阻塞交付完成，停止；不自动训练、重试下载、换参或开新实验。原始Base辅助增益支持保留现有头的迁移假设，主线仍需另两份可靠DreamOn权重。

完整普通token→ID、分词规则/文本、特殊词/输出列及一次logits shift审计已存各智能体目录。来源头按原定义strict加载全部参数；零头deepcopy且仅清零output.weight，λ仍1。未命名Base列不赋扩删语义，实际非法token会停目标。

Base原生日程不是严格每轮K；局部参考只补min(K,remaining)与frontier选位，固定64且不读答案长度，EOS留原位，所有组相同截取/skip special规则。现有DreamOn max_gen_len64与论文128分别登记，不能称同一复现。

人口固定1033行/164基础题、单生成种子42、原derive_row_seed；16条字典序等距ID不按正确率挑选。A=K1无头，B=K2无头，C=K2同类同秩零输出头λ1，D=K2来源头λ1。

|DreamOn自身校准顺序|A|B|C|D|D−C帮助/伤害|C−B帮助/伤害|D−B净|D−A净|
|---|---|---|---|---|---|---|---|---|
|global_confidence|951|901|901|910|15/6|0/0|+9|−41|
|left_to_right_frontier|942|877|872|890|29/11|6/11|+13|−52|

以上分母均1033；来源校准D−C净+9/+18。global行加权区间含0，frontier基础题等权区间跨0；单种子重复开发人口，不能推广独立测试或跨模型成功。完整D−C/C−B/D−B和相对K1的帮助/伤害、164题cluster 95%CI见a_dreamcoder/dreamon_reuse_paired.json；触发率、结构停止/总轮次、非相邻回退/双位机会和失败/超时见controller/dreamon_calibration_summary.csv；保全10330行含历史E机制组，无重跑、无改写旧报告。

Base若完整，逐目标report.zh.md/paired_comparisons.csv给出六个配对和两种权重cluster区间，summary.csv给触发、结构/非相邻回退分母、失败/超时。兼容但无增益只表示现有头本次迁移未成功；只优于无头但不优于零头不能归因为学习；固定长度失败可能受长度误差主导，不自动否定逐模型重训。

本轮累计GPU作业成本 3.541866h / 12.0h；另存历史诊断成本 0.801673h。共享H200计时只作成本，没有正式加速比。互斥锁和所有session在controller/gpu_budget.json，GPU背景/命令/版本/PID/恢复状态在目标manifest及controller清单。下载不计GPU。

后续仅准备材料：future_training_preparation.md列索取信息、论文/代码差异、选择10epochs重建、有效batch128梯度累积、显存理论范围及以后授权pilot估时方法。没有启动任何主模型微调、头训练、K4或隐藏状态头。

命令：scripts/run_zero_training_transfer_dreamcoder_20261013.sh；通用runner支持已登记另两目标，必须先完成权重和技术检查；最终合并报告命令 python scripts/finalize_zero_training_transfer_20261013.py。原工作区未提交修改未动。

验证时HEAD：`585b6f342dc974cfc4499092d9a3c6039599ac85`。最终交付HEAD见Git提交及最终回复（本文件无需自指未来commit）。


### 2026-10-14T02:14:14.839450+00:00 最终发布状态追加

最终结果与完整紧凑证据已提交 `8d06cb65583fa21549580713e01ffdd6d33eeb9f`。GitHub visibility: `local_only_not_pushed`；SSH22/443均连接超时，HTTPS非交互尝试无可用用户名/凭据，三次有界尝试均失败。最后确认已推送HEAD为585b6f3（历史确认，不代表当前远端状态）。当前本地报告/全部8264行证据和统计完整，评测不受影响。没有启动新GPU、训练、下载或持续监督；停止发布重试，待网络/凭据恢复后可将当前分支普通push，不force、不重做评测。
