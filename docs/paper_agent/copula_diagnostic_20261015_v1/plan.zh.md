# 2026-10-26 第一批 C：冻结头 K2 Copula CPU 诊断

起点 `codex/dreamon-structure-coordinate-fix-20261015-v1` / `d39a51d494efc2a1d1661eb28622b18b3b741fa6`。
本工作独立分支 `codex/dreamon-copula-diagnostic-20261015-v1`，contract_id=`dreamon-context-v1-20261015`、schema_version=1。
仅新增 C 的分析、测试与日期记录，不改 A/B、历史报告、采样器或思维导图；不使用 GPU、模型推理、训练、多个智能体，也不运行生成程序。
“不生成代码”按任务的明确 CPU 分析代码交付要求解释为不做模型代码生成推理。

## 行动说明与固定选择

审稿人关心的是：现有冻结 Markov 联合分布 R 是否因强保旧边缘而更接近固定顺序 fresh 参考 T，而非是否胜过独立 I。
候选路线为硬边缘投影、软约束、重新训练；本批只执行已指定的硬投影，后两项留待下一批。
假设：旧边缘可能有帮助，也可能有害；候选截断会改变首位边缘，必须分解收益。
模型与数据仅引用 A 的清洗 OpenCoder train/validation、K4 链首两位置子问题及冻结模型/头 SHA。
T 不是正确代码分布；顺序 old_p1×head_cond2 本身已定义 joint，不能称“没有联合建模”。

实现文件为 `analysis/dreamon_copula_cpu.py`、`scripts/consume_dreamon_copula_cpu.py`，测试为 `tests/test_dreamon_copula_cpu.py`。
新增输出目录 `analysis_outputs/dreamon_copula_diagnostic_20261015_v1/`。
环境仅 Python 标准库和 NumPy；设置 `CUDA_VISIBLE_DEVICES=''`、BLAS/OMP 线程数 1，不导入 torch/模型。
精确验证命令（本 worktree 根目录）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_dreamon_copula_cpu.py -v
```

精确后台命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/consume_dreamon_copula_cpu.py --wait-seconds 86400 --poll-seconds 60
```

共同 Git 目录通过 `git rev-parse --path-format=absolute --git-common-dir` 解析，中转后缀固定 `research_runs/dreamon_context_20261015_v1`。
先查同项 C_done/本地完成记录，避免重跑；只拥有 C_consumer.lock，不删除任何已有锁或 ready。
ready、manifest 与每个分片均验证 SHA256，防止跨 worktree 中 producer 尚未写完或内容改变；这是用户要求的跨进程发布验证，不新增研究 gate。
发布者 complete 包必须实际 256 且四桶各 64。partial 包须在 manifest 明列实际 N、逐桶缺口和预算停止状态/原因，不能从文件存在推断可用。
不兼容、producer 明确失败或等待到期即记录 blocked 与准确恢复命令并退出。等待上限从第一次启动计时；中断恢复不重置期限。
consumer 在完整验证后自动分析、归档、提交、普通 push、验证远端 HEAD 后退出；推送失败保留本地 commit 并如实记录。不启动第二批。

## 统计与成功标准

float64/log-space IPF，最大绝对边缘误差 1e-6，最多 1000 轮，不加 epsilon。优先 log 字段；概率零与有限 log 的明确下溢单独审计。
先构造未归一化 `old_p1*cond2`，记录 R/T 块质量，再分别全块归一化。绝不先逐行归一化 cond2。
I=r⊗c，R，Q=IPF(R;r,c)，Q_T=IPF(R;T.row,T.col)。Q_T 只是不可部署的 teacher-oracle 算法检查。
全部实际状态保留；逐状态 KL/TV、teacher 行边缘加权条件 KL、边缘漂移、覆盖/尾/特殊质量、支持、迭代数、CPU 时间均发布。
统计预设 seed=20261015，2000 次来源题组配对 cluster bootstrap；重采题组保留其全部派生样本，主估计为状态加权均值。无独立测试主张。
主表、order/span/split 分层、四格交叉表与预声明高覆盖表均发布。
高覆盖条件：old_p1≥.95、old_p2≥.95、R_raw≥.90、T_raw≥.90；少于32状态或8来源题组标“高覆盖证据不足”，不调阈值。
使用 KL 链式分解 `Δjoint=Δrow+Δconditional` 判断改善是否仅为首位截断偏移；不能因 Q 胜 I 就声称胜 R。
零支持不可行、未收敛、非有限 KL 不删除，单列状态并阻止肯定校准建议。

成功标准是玩具实质测试通过、真实包完整性与统计可审计、Q_T 符合适用条件下的 KL 预期；是否推荐 pilot 依次看 Q/R、覆盖与分层。
停止条件是预算/等待上限、输入不兼容或数值阻塞；不能靠加 epsilon、降覆盖门槛、自动搜索惩罚系数或训练绕过。
结果回答有限候选概率几何，不回答代码通过率、端到端速度或完整 V² 可部署性。total pass rate / wins/losses / GPU hardware comparison 均不适用。

`reviewer_gate_disabled`：按 AGENTS.md 和用户要求，本地串行 diff review + fresh CPU verification；没有独立 reviewer 的残余风险明确保留。

## 初次检查

2026-10-26：本地/远端无同名 C 分支及已完成 C 结果；中转有 protocol_ready/B_done/A_queue.lock，无 dcd_ready。A 仍在运行。
A 的当前 partial exporter 只给 count/coverage_gap，尚未在 manifest 明列逐桶缺口与预算停止原因；若届时仍如此，C 必须 blocked，不能自动修改 A。
