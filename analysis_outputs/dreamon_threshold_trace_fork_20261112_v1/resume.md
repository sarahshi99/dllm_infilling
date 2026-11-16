# 完成状态与恢复入口

本有限实验已完成，status.json为delivery_completed，delivery_verification.json为passed。5168自然路径、2920分叉、64主教师、54从头种子、5168未修改原入口详细对照和同64状态数值校验均完成；本任务两个tmux和模型进程均已结束。当前不需要启动任何模型。

分支：codex/dreamon-threshold-trace-fork-20261112-v1。
实际工作区：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-threshold-trace-fork-20261112-v1。
报告：report.zh.md；病例、全部十个质量翻转逐轮附录：case_studies.zh.md。

## 证据与版本

来源报告285f8f1462481d7c73e1e6659822f45fe7b59a5e；历史执行660f2960ff65fb216f762a1eca0183835de2afbb。相关运行依赖相同，实际run_spec已核对。
回放实际源码418e73e88b9cdce5f89f2675ea6a9ac99aa4d146；分叉编排4e4894089767e67065aa168256b30d913f9ddf68；原入口/教师复核b743a41fcbe459b823d2ebc2e87a086667347c23。分叉前动作定位修正不改解码器、不按结果换面板，详见brief与phase_versions.json。

普通Git保留源码、紧凑逐例证据和索引；raw轨迹、106个快照、SQLite与日志留本目录local/（忽略提交）。旧模型/头位置和校验见source_integrity.json与历史run_spec，不复制权重入Git。

- local/results.sqlite以(kind,key)唯一事务保存，评分以(case,output)复用。
- 自然/从头原始轨迹：local/<config>__<case中斜杠换下划线>__<seed>.json.gz。
- 分叉原始轨迹映射：raw_trace_index.jsonl.gz；快照local/snapshot__*.pt。
- 主阶段local/run.log及attempts.log；后续local/verify_all.log、validation_attempts.log、enrich.log；CPU检查local/tests_final.log、delivery_audit.log。
- 初始CPU概率元数据保留，实际GPU评分更正索引score_numeric_audit.json；阈值变化0。最终教师用teacher_logspace_per_state.jsonl.gz，初始teacher_per_state.jsonl.gz不覆盖。

## 无模型的离线核对

在上述工作区使用原解释器：

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.threshold_trace_fork.enrich
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.threshold_trace_fork.delivery
```

enrich重建机器汇总，delivery核对完整清单、生成预算、计数、事实seed42、评分缓存、逻辑/实际映射、原入口结果与概率数值。人工报告、额外物理头成本说明和CPU日志摘要另行保留；审计重跑不会自动改写人工解释或扩展实验。analyze是生成阶段初稿工具，会覆盖report/case_studies，完成后不要直接用它重建人工报告。

历史运行入口为bash experiments/threshold_trace_fork/launch.sh，再顺序bash experiments/threshold_trace_fork/finish.sh。恢复未完成阶段前必须检查两个tmux及status，禁止重复加载模型；已有完整事务不重复累计，不能混用源代码版本。主阶段OOM最多一次进程恢复、基础设施最多三次；本次均0。普通失败/3秒超时不重抽，科学有效性异常先诊断。本轮所有阶段已完成，固定面板/专项不再扩大。

执行未完成项无；预定N/A为六条顺序/豁免触发不存在、139 .90无扩展；候选效应/交互未识别为研究边界。没有训练、阈值搜索、全程新策略矩阵或主分支合并。
