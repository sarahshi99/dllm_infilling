# 坐标修补 v2 有限队列恢复入口

实现分支codex/dreamon-structure-coordinate-fix-20261015-v1，基线dbdd0bd5cbbf21a049a989213136aa2650844aea。唯一会话dreamon_structure_coordinate_fix_20261015_v1。

```bash
bash scripts/launch_structure_coordinate_queue.sh
```

同会话运行时命令只报告现有入口，不建第二队列。队列PID见queue_status.json，worker PID及正式源码/模型/头/数据/采样/评分版本见run_manifest.json。queue.log/worker.log保存日志，progress.json与per_case.sqlite逐行保存生成和评分两个恢复点。只恢复相同正式源码，成功键跳过，已生成未评分只恢复评分，不重新抽样。

完整CPU重放37722逻辑行、未知0。冻结influence_manifest.json：需重跑126行，单行global原生A4=37，多行global链无头B=28、零头C=28、训练头D=33。多行配对并集38个任务ID，未受影响的其它臂保留完整画布等价证明。没有因passed、扩删计数或EOS残留标签筛影响。原通过/失败只在影响清单冻结后用于技术smoke覆盖，规则见technical_smoke_manifest.json。

CPU77项通过后启动；H200 GPU0加载前等待至少32GiB空闲，持有旧项目全局GPU锁，一次一个worker，不中断别人。固定smoke技术通过后自动补验剩余受影响行；正式自检行并入最终表。Base只用CPU原生参考回归和全6198行重放证明，不重新装GPU模型。

自动阶段：补验→原生官方3秒隔离评分→v2全人口合并核验→修补/学习分别配对、固定跨度分层和基础题bootstrap→压缩归档→提交推送/远端核验→退出。GPU拥挤时后台等待，无评测时长上限；网络操作有界并保留准确重试命令。不会自动训练、实现common-u或四个候选新头、更新思维导图或继续无限排查。

旧analysis_outputs/k4_multiline_20261014_v1全部文件不变。新完整人口表A/B/C/merged_per_case_v2.jsonl.gz及报告report.zh.md，仅新126行见new_rows.jsonl.gz。原数据上下文与模型/头版本见旧目录固定相对引用和本轮run_manifest.json。旧单行K2只历史参照，旧global K2锚点仍限1012无扩展子集，不作为修复版全量配对。

失败/重试原始日志本地保存，归档failure_retry_log.portable.jsonl去除机器路径；无事件时明确空文件与failure_retry_coverage.json时间覆盖。publication_status.json保存最终本地提交、remote HEAD和准确推送重试命令。若失败：

```bash
git -c 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1' push origin codex/dreamon-structure-coordinate-fix-20261015-v1
```
