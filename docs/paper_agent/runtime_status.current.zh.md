# Runtime Status — 2026-07-16 UTC

审计时间：`2026-07-16T15:37:00Z`；审计时 HEAD 为 `8440af1`。Frozen controller test 仍为 `sealed`，`test_evaluation_count=0`。

## M2：完整并已正式复核

- tmux/PID 已自然退出；vanilla、gradual、abrupt 均 `148/148`、unique=`148`、duplicate/error=`0`，row-key 集合一致。
- 每个 ok row 都是 canvas=`64`、seed=`0`、actual forwards=`64`、token-forwards=`4096`；末行可解析，full technical gate passed，run manifest=`completed`。
- 正式 10,000 task-cluster grouped 统计与无代码 activation audit 已完成：`analysis_outputs/m2_constraint_homotopy_20260716_grouped_v1/`；结论是 `constraints_activated_no_reliable_grouped_advantage`。详见 `docs/paper_agent/experiments/m2_constraint_homotopy_20260716_randomspanlight_result.zh.md`。

## M3：已启动并自然完成 12→148

- pre-launch 修正已在 `8440af1`：移除不能改变 token 排序的 scalar penalty；within-particle remasking 只按 ordinary confidence，syntax/obligation/contradiction 只作 particle ranking、birth/death 与最终选择。
- M3 tmux/PID 已自然退出；smoke 和 full technical gate 均通过，run manifest=`completed`。
- uniform、birth-death 均 `148/148`、unique=`148`、duplicate/error=`0`、row-key 集合一致、每 row=`256` forwards。uniform token-forward 固定=`15360`；birth/death 实测 min/mean/max=`8192/11532.11/24832`，共发生 `444` 个 birth/death events（148/148 rows 非零）。这支持 equal-forward、**不**支持 equal-token 的表述。
- peak CUDA allocation=`16,383,515,136` bytes；full wall=`8036.64s`。作业在可取得独立 t+10 snapshot 前已到达两臂 `148`，故以 completion audit 取代 t+10 行数审计；无 OOM/Traceback/磁盘错误。

## M1：未受本轮干预

- 历史 5079 MultiLine M1 继续 `safely_paused_resumable`，完全未动、未读部分性能。
- 新 RandomSpanLight M1 没有 live PID/tmux；raw 只读审计显示 stage1=`1332/1332`、generic=`148/148`、dependency-cone=`148/148`，均 unique、0 duplicate/error。smoke technical gate 已通过；launcher 未留下 final run manifest。本轮不 resume、不重启、不改写 M1，也不读其 outcome performance。

## GPU 与下一步

- M3 启动时曾两次出现短暂 `nvidia-smi` driver communication failure；随后恢复。最终 H200 audit：utilization=`95%`、free=`120054 MiB`、power=`177.99W`、temperature=`41°C`、ECC=`0`。
- 本轮没有启动 M4、official CAL 或 DreamOn。SciPy 仍缺失，official CAL 保持 corrected-protocol 的 smoke 前依赖状态。
