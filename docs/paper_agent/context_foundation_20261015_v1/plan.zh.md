# 第一批A执行协议与恢复入口

用户已授权本批唯一GPU任务：独立分支 `codex/dreamon-context-foundation-20261015-v1`，起点 `d39a51d494efc2a1d1661eb28622b18b3b741fa6`，已含坐标修补560e98a。原工作区未跟踪文件、旧报告/采样器/结果和思维导图保持原样。不调用Goal，不派生agent，不训练新头，不合并B，不启动B/C任务或第二批。

采用 karpathy-guidelines 进行局部实现和本地diff review。`reviewer_gate_disabled`：依照AGENTS.md，使用当前会话串行审查和fresh CPU verification。用户已给定详细方案，短brief直接执行；无需再确认研究方向。

选择独立采样/状态模块复用已修补的结构操作、冻结头和官方评分器。另一方案是改写旧采样器，会影响历史复现，因此本批实现放在 `experiments/context_foundation/`。共同随机输入只作配对，不能保证分叉后相同坐标仍有相同含义。FP32分布运算及稳定tie规则作为明确的新协议差异，不与旧结果逐例配对。

精确执行命令（从本worktree根目录）：

```bash
bash scripts/launch_context_foundation_queue.sh
```

tmux为 `dreamon_context_foundation_20261015_v1`。控制器为 `scripts/run_context_foundation_queue.py`，唯一GPU worker为 `python -u -m experiments.context_foundation.worker`；GPU0是当前唯一H200，覆盖旧AGENTS默认2,3设置。环境使用仓库既有dreamon-repro，TOKENIZERS_PARALLELISM=false，OMP=2。GPU退出后单独CPU进程CUDA_VISIBLE_DEVICES为空、OMP=1并绑定两个允许的CPU核心，再调用既有bwrap隔离评分器；官方超时仍3秒。

输出目录 `analysis_outputs/dreamon_context_foundation_20261015_v1/`，日志为queue.log/worker.log/analysis.log/publication.log。队列PID、worker PID、本机路径和预算仅放本机运行记录。恢复执行同一命令，SQLite主键按kind/key避免重复生成；官方程序失败是最终有效评分，只有基础设施评分异常有限重试。可恢复原始张量：`bash scripts/launch_context_foundation_queue.sh --restore-missing`，按固定源轨迹/key重放并核验原hash；仍使用原累计预算，绝不自动加时。

完成顺序与上限：

1. 最小CPU技术测试及真实DreamOn技术smoke；失败停下保留诊断，不能看passed调算法。
2. A1/A2合计最多4 GPU作业小时；其中盲候选轨迹采集最多2小时，为teacher/反事实预留余量。源程序、跨度和盲轨迹已先冻结；在已遇候选中按稳定hash保留，每格train/val=192/64，来源零头/旧头各半，每组每来源最多4，总最多8。相同状态不因order/arm标签重复计数；先遇到的确定来源拥有该状态。采集截止由固定清单/时间上限决定，不看teacher差异或评分。只有合法链首普通前缀状态入库。桶不足明确coverage gap。
3. A3最多2小时，提交无pickle的DCD小包并原子发布dcd_ready，C可立即CPU分析实际完成样本。
4. 剩余总预算内运行global K1/Z4/H4-old/F4-chain各1033行，轮换arm顺序，只用seed42；固定hash32案例留详细画布。不跑额外frontier/多行/原生K4/Base/K8/多种子。
5. GPU进程退出并释放模型/锁，CPU评分、分层/分组bootstrap、中文报告、归档、追加项目日志、提交、普通push并核验远端HEAD；网络失败保留本地提交和准确push命令后退出。

所有GPU模型驻留（包含加载、教师、写盘与生成）总计最多12小时。心跳每秒记录累计值；safe checkpoint检查在每次模型前向和生成轮边界，控制器有原子预算监督及有界停止。进程异常退出由控制器补记最后心跳至退出；无法确定的崩溃间隔保守计入，不重置预算。预算达到后只完成当前安全检查点，任何剩余缺口写入completion_matrix和retry matrix。等待GPU锁/至少32GiB可用显存最多累计24小时，等待不计GPU作业；原有外部进程不被终止。专用A_queue.lock与现有 `/tmp/dllm_infilling_h200_diagnosis.lock` 仅约束遵循该锁的进程，不能约束其他作业。

启动前粗估为约4–10 GPU作业小时，可能在12小时内完成；这是基于历史同模型共享H200吞吐的宽范围估计，新增排序、hidden和长链采集可能改变它，不保证满桶。真实smoke后自动保存timing_estimate.json，不因质量中途缩减/扩展计划。

研究问题与成功标准：共同随机输入的零头/无头逐token逐画布相同；h[t-1]/logits对齐且边界可查；教师不写未来/其他链；本轮实际长链覆盖有完整分母；更早词干预有源组bootstrap和32状态数值噪声参考；F4相对旧头的帮助/伤害和净值均报告。分布更近不等于代码通过，F4不是完整K1或正确答案/上界，不宣称其加速。

本批全部manifest在Git中使用仓库相对路径。共享中转通过git-common-dir解析，只存协议/DCD/完成的原子ready文件及A运行锁，不删除其他会话锁。tensor_manifest对大张量使用缓存相对路径、hash、恢复命令；缓存精确FP32 logits及原BF16 hidden的无损FP32表示，不涉及teacher半精度误差。DCD小包包含原完整词表log_softmax抽出的稳定log字段、候选和特殊质量；候选外质量由1−候选质量计算。首位因果锚点是旧p1，16×16是截断诊断而非完整joint。

最终结果以新目录report.zh.md、completion_matrix.json、generation_retry_matrix.jsonl.gz、A_manifest.json和本机publication_status.json为准；“已启动”等待不代表真实smoke或全量完成。残余风险：真实GPU检查尚待资源；单seed重复开发人口、有限长链覆盖、共享负载，以及hash重放可能受未来硬件数值变化影响。
