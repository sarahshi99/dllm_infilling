# 2026-10-26 CPU 实现核验记录

状态：CPU 实现已完成，真实 A 输入尚未发布；没有真实256状态结果，不声称本批数据诊断完成。
固定文献与边界见 reference.zh.md；执行范围、公式、预声明统计与恢复命令见 plan.zh.md。

## 本地审查

`reviewer_gate_disabled`；按用户与AGENTS.md要求，当前会话串行审查，没有第二个智能体。

- 逐行核查 R_raw/T_raw 的构造先于全块 log normalization，没有对条件矩阵逐行重归一化。
- Q 的行列目标来自候选内旧边缘；Q_T 来自同一 T 的块内边缘。KL 方向分别为投影 KL(Q||R) 与评估 KL(T||·)。
- 条件 KL 按 T.row 加权，并核查 KL 链式分解残差；不能把首位边缘截断修正混称依赖学习收益。
- 稀疏支持使用 transport max-flow 检查 Hall 约束，不仅检查空行/空列；容量可行性采用FP64、1e-12总质量余差，IPF仍需独立满足1e-6绝对边缘误差。
- 原始概率为0但log有限时，仅允许原始dtype可以解释的下溢；不对0取log、不加epsilon。缺少log只允许producer明确声明稳定FP64正支持的等价输入。
- 所有状态、来源组、split、order/span、候选/尾/特殊质量及数值失败保留。存在不可用状态时不把complete-case子集均值冒充主均值。
- bootstrap 重采来源题组、保留其所有派生状态；高覆盖门槛固定，没有筛选passed或调低样本门槛。
- ready 与manifest、每一分片在解析前后核验hash；partial缺桶/预算声明不全时阻塞，不修改A。
- 已完成C包核验后退出，C专用flock避免重复consumer；默认中断恢复保留首次等待截止时间。
- 后台推送设置有限超时，不强推；远端HEAD必须等于本地commit才标pushed_verified。失败保留本地commit和精确retry命令。
- 仅显式stage C拥有的路径，普通git commit --only不包含其他任务内容。消费者只写C锁/C_done，不删除ready/其他锁。

## Fresh verification

精确命令、环境、源文件SHA、完整unittest输出与结果存于新增目录：
`analysis_outputs/dreamon_copula_diagnostic_20261015_v1/test_results.json` / `test_results.txt`。
测试覆盖22项：已满足边缘不变、投影边缘、独立乘积、30组正矩阵teacher-oracle KL/Pythagoras、错误边缘伤害、全块/逐行截断反例、可行/不可行零支持（含Hall反例）、零目标与极端log、条件KL链式分解与最大轮数、hash损坏、partial元数据、FP32下溢与缺log、256状态FP32完整流程、组bootstrap、非有限样本保留、中断/半行恢复、输入变更拒绝、有限等待、失败退出、真实本地Git推送/失败保留提交。

测试中的256状态为随机模拟数据，不是A的DreamOn/OpenCoder结果；测试临时包不混入科学结果目录。
当前仅能证明CPU算法与交付流程的技术检查通过；不能判断旧边缘是否有帮助。

剩余风险：没有真实A数据的数值/覆盖证据；有限16×16候选可能严重截断；开发来源与同母状态相关；没有独立reviewer。
收到合规输入后consumer自动给出全体、分层与高覆盖结果及中文结论，归档完成后退出，不开始第二批。
