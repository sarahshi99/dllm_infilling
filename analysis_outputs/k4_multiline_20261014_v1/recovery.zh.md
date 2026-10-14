# 2026-10-14 有限队列恢复

在本分支worktree执行：

```bash
bash scripts/launch_k4_multiline_queue.sh
```

会话dreamon_k4_multiline_20261014_v1；queue_status.json保存队列PID，queue.log保存日志；各批progress.json、run_manifest.json及per_case.sqlite保存事务恢复点。A→B→C一次一个GPUworker，共享GPU0，加载前空闲至少32GiB，不终止他人进程。无任意全量时长上限。

每阶段固定16行技术smoke通过后自动全量，不按质量调参。已生成但未评分的行只恢复官方评分；已评分的普通错误/超时跳过。技术错误暂停该阶段并记录failure_journal；继续其它独立阶段，末尾正常退出，不自动训练。修复解码/协议代码后请使用新版本目录，不混入旧manifest；只恢复相同代码/参数可原目录续跑。

完整性、分析、压缩归档、focused commit/push由queue自动执行。网络推送有45秒边界，仅约束网络操作；不影响本地评测。publication_status.json保存本地提交、远端核验与准确重试命令。SQLite/log/临时错误栈不上传，逐行gzip和唯一上下文可独立聚合。

旧迁移已完整：bf62fdf，codex/dreamon-zero-training-transfer-20261013；错误审计CPU修补：f6ce43f，codex/dreamon-markov-error-audit-v1（含fa29d5c、74d6610）。本次SSH22/443尝试超时，本地提交保留。重试：

```bash
git -c 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12' push origin codex/dreamon-zero-training-transfer-20261013
git -c 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12' push origin codex/dreamon-markov-error-audit-v1
git -c 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12' push origin codex/dreamon-k4-multiline-eval-20261014-v1
```

旧错误审计缺326匹配生成轨迹、13超时卡点未补造。Dream、DiffuCoder下载仍待办，本轮不启动。
