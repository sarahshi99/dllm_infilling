# 第一批 C 独立恢复入口

时间：2026-10-26T10:57:21Z
状态：**waiting**
原因：Awaiting validated A dcd_ready; no empirical conclusion yet.

分支：`codex/dreamon-copula-diagnostic-20261015-v1`
输出：`analysis_outputs/dreamon_copula_diagnostic_20261015_v1/`。
本批仅CPU，不训练、不调用模型或生成、不启动第二批。等待不代表诊断完成。

恢复命令（仅手动新等待窗口允许重设24小时期限；中断恢复默认保留原期限）：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-copula-diagnostic-20261015-v1 && /home/shx/miniconda3/bin/python -u /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-copula-diagnostic-20261015-v1/scripts/consume_dreamon_copula_cpu.py --wait-seconds 86400 --poll-seconds 60
```

若partial manifest缺字段：由A发布者补充实际count、四桶missing_buckets（global/single、global/crossline、frontier/single、frontier/crossline的缺数），
以及budget_stop={stopped:boolean,reason:非空字符串}，核验所有文件hash后原子重发dcd_ready；C不能自行补写A的声明。
字段缺失、hash不符或数值支持不可恢复时，不得仅重新运行C绕过校验；先恢复producer包。

运行记录：
```json
{
  "deadline_epoch": 1793098641.2926953,
  "poll_seconds": 60.0
}
```
