# 已完成 K2 诊断的 CPU 复核与可携带证据归档

- 实际起点：`340170ce6ae44477027dd7e94ec7a9c0f0910384`，工作树干净，无后续提交；不回滚，不修改生成器或旧数据库。
- 范围：只读取既有10330行/164基础题开发人口结果，导出压缩原始字段与匿名GPU历史时序；不执行生成、评测、训练、GPU查询/作业或停止进程。
- 实现：`analysis/dreamon_k2_evidence.py` 导出只读证据；现有 analyzer 增加相对路径离线输入；独立CPU审计脚本复算计数/配对/10000次cluster bootstrap并比对原精度。最终归档脚本纳入证据，review manifest快照和旧条目保留。
- 数据：SQLite只读连接，先核对其payload与已有JSONL逐字段一致，原字段原值导出，不补造缺失信息；完整输出文本/词元随gzip保留，无新增hash约束。
- 评测审计：原有13次timed out，逐组/逐题记录；仅凭错误文本不能反推异常类型或超时原因。缺少case/evaluator起止时间、CPU负载和子进程退出信号，必须明确证据界限，不重新评测。
- 计时：1.657×/1.818×只叫共享H200观测；轮换顺序不替代独占计时。历史GPU日志只输出数值与采样时刻，其他用户命令/姓名/PID/路径不上传；旧含进程路径的manifest作隐私删节，原件仅留本地。
- 验证：临时目录只放压缩证据、人口ID和smoke版本记录，不复制SQLite/原数据/模型；从任意工作目录执行分析，再用独立计算对照原summary/paired全部单元格与CI符号。保留原表，任何差异另存并解释。
- 成功：可携带证据齐全、全部统计复算一致或差异已审计；提交紧凑复核报告、勘误、review索引和代码，push后停止。
- reviewer_gate_disabled：按项目规则只做本地diff review与fresh CPU verification；遵循已读取的karpathy-guidelines，不使用subagent或新broad workflow。

## 2026-10-13 K2诊断CPU复核完成（追加，旧判断保留）

- 起点HEAD `340170ce6ae44477027dd7e94ec7a9c0f0910384`；SQLite只读与JSONL的10330个payload逐字段一致，原库未修改；全部32字段/完整输出导出约940KiB gzip，可随仓库复算。
- 隔离临时目录不含SQLite/作者数据/模型，且禁止SQLite连接和torch/transformers导入；相对路径分析通过。独立聚合与10000次基础题整组bootstrap（seed20261013）复算，与两份原表/原报告数值表共1588单元格一致；无数值差异。
- 新发现并澄清：原未恢复运行错误0不等于评测无超时。13次timed out全部集中HumanEval/32；共享资源对墙钟超时的影响无法排除，缺CPU负载、评测起止UTC、子进程CPU/退出码/超时来源。仅超时标签未知的敏感性边界不改变原标签，D−C净增范围[9,10]/[17,20]。
- 1052个GPU历史快照全部有6个其他任务，已仅导出匿名数值。1.657×/1.818×只称共享H200观测；轮换不能替代独占计时。旧report保留，新日期勘误另存。
- D−B净+9/+13追回B相对A并行净损失的18%/20%，不混用A对/B错子组救回比例。E仍是固定原右位只填左后重算的机制诊断，不是K1等价、数学上界或成功验证算法。
- 当前复核入口：`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/review_20261013/INDEX.zh.md`；review_manifest.latest.json已引用，旧manifest快照和旧条目保留。最终归档脚本后续自动纳入紧凑证据并去除GPU进程身份信息。
- 本轮CPU工作到提交推送即停止；没有生成/评测重跑、训练、GPU实验或停止其他进程。后续最小CPU超时检查/独占计时均仅写条件和命令，未排队，不自动推进新方法。

验证结束：8项CPU测试通过；隔离复算及独立原精度比对1588单元格、差异0；py_compile、git diff --check通过。原数据库只读，旧report/summary/paired/smoke/background_exit未改动，生成器和runner未改动。新证据总量约1.2MiB，复核材料约300KiB；不提交raw SQLite、原大JSONL或私有GPU日志。
