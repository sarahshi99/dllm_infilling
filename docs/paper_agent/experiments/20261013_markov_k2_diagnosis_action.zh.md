# DreamOn + Markov K2 归因与机制诊断

- 范围：用户指定 A/B/C/D/E × global/frontier，完整 1033 行开发/机制诊断人口；单生成种子 42，不打开新的封存测试、不调参、不训练、不运行 K4/K8。
- 起点：已审阅 `b8d6e58`，复用最新 `0e97f61`；已有 diagnosis 工作树干净，无本轮结果。旧报告 wall 包含 evaluator，K1 诊断不对称，不能用来回答纯生成速度。
- 对照：D−C 隔离训练修正；C−B 测重采样/控制路径；E−C/D 测固定右位置真正重算；全部与匹配 A 配对。
- 推荐实现：扩展现有 decoder，诊断关闭时跳过分布、排名、replay、CPU 画布保存；C 使用同类同秩头并将输出投影清零，lambda=1；E 在同触发位置复制掩码画布，只写左词，再前向一次并单次对齐。另建 runner/analyzer，保留旧运行。
- 未采用：lambda=0（绕路）、重选右位置（改变干预）、正式验证器（超范围）。风险：单种子且重复开发人口，效果小/区间宽应标不确定；E 不是上界或 K1 等价。
- 检查：先 CPU 单元测试，后固定稳定 ID 的 16 行，各配置诊断开关输出/动作/前向一致，再自动完整 10330 行，旋转十配置顺序。失败记录并可恢复，成功主键去重。
- 计时：bf16、batch=1、同一空闲 H200；模型加载/预热、CUDA 同步 decode、评测、写盘分别计时。先据 16 行估算，GPU 总上限 12h（不是预测）。前向计数分基础、额外重算、总计和外层轮数。
- 资源现状：2026-10-13 初查唯一 H200 有 6 个其他进程、约 91761 MiB 占用，故不得启动 GPU。CPU 实现和检查继续；资源仍忙则保存 blocker，不将部分结果当完整结果。
- 结果目录：`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。
- reviewer_gate_disabled：依仓库规则，本地 diff review + fresh verification，不使用 subagent。残余风险：真实模型检查在空闲 GPU 才能完成。
- 成功：10330 成功行、1033 ID/164 基础题、无重复/缺失/未恢复错误、smoke 协议检查通过；或明确记录资源/协议/预算阻塞和恢复命令，不伪造统计结论。
- 结束：本轮结束后停止；下一轮建议仅写入队列，不自动执行。

## 实现与 CPU 验证

- decoder 增加 `diagnostics=False` 真正绕过概率/排名/replay/画布保存，默认保留历史诊断语义；`capture_actions` 仅供检查，正式计时关闭。
- C 复制 v2 KL 同类同秩头并清零输出投影；D 使用原 checkpoint；均走 lambda=1 的计算、加法和局部采样。
- E 在同分支中复制设备画布只写左词，不改其他 mask、原 pair、注意力或结构动作；新 logits 只平移一次，保留 expand mask。额外前向与右采样一起放在原局部种子的 fork_rng 中。
- 新 runner 使用 SQLite `(policy,arm,task_id)` 唯一键事务保存成功行；失败日志保留；按行轮换十配置，CUDA 同步纯生成计时，评测/写盘另列。源码 Git blob 版本复用允许后续仅文档提交后恢复，不重扫大文件。
- 测试先红（新增参数尚未实现），后绿；当前 45 项 CPU 检查通过。覆盖 C/D 同类同秩/同采样路径、E 画布/对齐/扩展禁用/结构屏障/非相邻回退/RNG 隔离、两策略随机采样的诊断开关一致、事务恢复/重复拒绝、cluster 权重和稳定 smoke 选样。
- 本地 `git diff -w` review 确认原诊断计算只增加条件保护，生成选择和原结构变换保留。reviewer_gate_disabled；CPU 合成测试不能替代 16 行真实模型验证。
- 固定入口：`bash scripts/manual_run_dreamon_markov_k2_diagnosis_20261013.sh`。日志重定向到本轮结果目录；不启动资源 watcher，不启动新方法。

## 实际执行状态

- 最终 CPU verification：46 tests OK（新增合成 10330 行分析链检验，临时 fixture 不进入真实结果目录）。py_compile、bash -n、git diff --check 通过。
- `bash scripts/manual_run_dreamon_markov_k2_diagnosis_20261013.sh` 实际执行，退出码 3；唯一 H200 被 6 个其他进程占用，模型加载前阻塞。结果=0/10330、smoke=not_run、GPU=0h、预计全批耗时=null。
- 已保存 compact report/manifest/audit/CSV 表头和本地空 per_case。未来恢复先资源检查，再16行估时，不能把旧 wall 当预测或速度结果。

## 2026-10-13 用户授权共享 H200 后台执行

- 本条取代之前“有其他进程即停止”的资源限制；不抢占、不终止其他进程。等待可用显存至少32GiB后开始，等待阶段不加载模型、不计GPU预算。
- exact command：`tmux new-session -d -s dreamon_markov_k2_diagnosis_20261013 'bash scripts/background_dreamon_markov_k2_diagnosis_20261013.sh > analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/background.log 2>&1'`（diagnosis 工作树内）。
- 47项CPU检查通过。后台顺序：16行开关检查→估时/12h预算→10330行→聚类分析/完整性审计→更新compact记录→focused commit/push→停止。协议差异/预算/异常会保存阻塞，不改参数或范围。
- tmux=`dreamon_markov_k2_diagnosis_20261013`；日志与结果目录=`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。无需持续人工监督。共享背景负载记录保留，报告明确计时限制。
- 旧阻塞产物移入同目录 `before_shared_gpu_authorization/`；原人口、头、种子和方法不变。
