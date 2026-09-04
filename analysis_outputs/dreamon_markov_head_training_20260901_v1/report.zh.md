# DreamOn 外部一阶 Markov 头训练 v1

状态：`completed_external_test_opened`。

1. TV-head 是一个只看刚提交左邻 token 的 rank-256 加性头，用全词表 L1/TV 匹配 fresh DreamOn。
2. KL-head 结构完全相同，唯一差异是使用正向 KL 匹配 fresh DreamOn。
3. 数据仅来自 OpenCoder `educational_instruct`；HumanEval 只用于严格去重，没有用于训练、验证、超参数或结果选择。
4. 两个头从同一个零输出初始化和同一个冻结 transition bank 开始；共同初始化 SHA-256 见 `common_initialization.json`。
5. TV pilot/full=`True/True`；KL pilot/full=`True/True`。
6. 学习目标是 stale→fresh 的 DreamOn 原始分布变化，不是 HumanEval 正确标签。
7. 困难 mismatch transition 的 recovery 与总体 TV 改善见 `validation_comparison.csv`。
8. 简单 stable transition 的过度修正见同表 `stable_corruption`。
9. winner=`kl`；按验证 raw TV、stable corruption、aligned reference NLL 的预注册顺序决定。
10. 本轮不自动授权 HumanEval K=2；只有外部训练比较完成后由下一次研究决策决定。

外部测试打开次数：`1`。本报告不声明 HumanEval Pass@1 改善。
