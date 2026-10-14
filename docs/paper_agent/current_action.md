# 2026-10-14 单会话K4/多行批次

用户已授权A DreamOn单行K4(8264)、B DreamCoder Base固定64单行K4(6198)、C DreamOn多行global K2(23260)；seed42，冻结bf16主模型/float32 rank256 v2 KL best lambda1。当前阶段CPU修补→K4实现→合成测试→固定16行技术smoke→自动全量队列。reviewer_gate_disabled，当前会话本地diff审阅与fresh verification。

推荐在现有decoder上新增K4空间连续链路径，保留K1/K2；不采用直接仅改K或多模型并发。理由：第三/第四词必须依赖前驱最终词，结构动作使本轮其它坐标失效。Base保持原生固定画布EOS/完整画布topk及64步时钟。对比B-A提交规则、C-B局部重采样、D-C已学偏置。GPU0共享H200，一次一个worker，加载前至少32GiB空闲；不抢占。官方timeout3s，普通失败是有效结果。

输出analysis_outputs/k4_multiline_20261014_v1；逐行SQLite事务和gzip证据、唯一上下文、manifest、配对表、逐轮离散轨迹、匿名资源记录。技术错误暂停受影响阶段，其它独立阶段继续；无任意全量时间上限。成功：集合精确1033/5815与164基础题、头调用/实际提交/NFE/官方评测可复算，提交推送并核验远端。风险：同164题开发人口、共享GPU观测耗时、Base跨K64步时间表混杂、旧无轨迹不可倒推机制。本轮不训练或下载其它模型，不读取旧封存标签。
