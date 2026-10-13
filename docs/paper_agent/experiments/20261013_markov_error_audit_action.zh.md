# 2026-10-13 DreamOn + Markov 已有输出全量错误审计

用户已授权单会话连续完成既有1033行/164基础题开发人口的CPU错误审阅及新分支commit/push；禁止GPU、重新生成、训练、方法改动和多智能体。

推荐复用c843847逐行gzip与1588格审核，补题目/候选、官方失败证据、326逐行语义审阅和离线casebook。未采用重新生成与重复完整统计，两者不解决具体代码错误问题。风险：无过程trace无法识别生成因果；无安全隔离则不得执行候选；单种子复用人口只做描述性结论。

启动命令（工作树内）：

```bash
tmux new-session -d -s dreamon_markov_error_audit_20261013 -c /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-error-audit-v1 'python scripts/run_dreamon_markov_error_audit_session.py'
```

现有Codex配置/认证原样继承；CUDA_VISIBLE_DEVICES为空。tmux中的单一Codex负责语义审阅，不只是Python评测脚本。所有日志与checkpoint位于 `analysis_outputs/dreamon_markov_error_audit_20261013_v1/`。

成功标准：326条队列全部经过实际逐行检查；每个失败输出有实际事实/解释或明确证据缺口；10330条可重算引用保留；分组与标签验证；HTML可离线筛选；focused commit/push已核验；completion明确剩余未知。真实来源/权限/运行环境阻塞才停止，记录缺口，不无限重试。

reviewer_gate_disabled；同一Codex本地diff review和fresh verification。此行动不改变研究方向或central claim，历史报告保留。

## 完成产物（2026-10-13，追加）

326行已由本单个Codex逐行阅读后写入semantic_reviews并同步唯一队列；1220失败全部定位，10330原字段逐值一致。官方CPU取证单并发、bwrap命名空间/只读运行时/空环境/seccomp/RLIMIT隔离探测通过；248次不同程序官方执行，重复程序证据明确复用，13超时各一次仍超时。插桩不改原断言、不额外调用，760条插桩记录与原测试结果一致；早期211个唯一插桩清理故障补取后仍保留初次失败。

exact commands：`python analysis/audit_dreamon_markov_errors.py prepare`、`python analysis/audit_dreamon_markov_errors.py replay`；repair仅补插桩，13超时不重复执行。逐条解释落盘后生成`python analysis/build_dreamon_error_casebook.py`；fresh checks为`python analysis/verify_dreamon_error_audit.py`与`node analysis/verify_dreamon_error_casebook.cjs`。完整性交叉检查和HTML DOM/JS交互通过；没有原生浏览器，视觉布局未验证。索引`analysis_outputs/dreamon_markov_error_audit_20261013_v1/INDEX.zh.md`，报告/覆盖/源码/单HTML完整保存。

代码事实支持D-C所救44行中37个原语法/缩进失败；机制无trace，326首次生成分歧unknown。D/E恢复空间76行，但E也新增10行失败；不据此扩大头或启动方法。所有研究方向建议仅记录，结束后停止；最终实际提交/推送核验见本地completion/runner_exit。
