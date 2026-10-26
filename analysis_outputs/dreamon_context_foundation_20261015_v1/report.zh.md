# 第一批A：协议已封存，后台队列待启动

状态为启动准备，尚未产生真实GPU结果。新分支起点d39a51d，冻结DreamOn与旧v2 KL/full/best rank256，common-u新协议单独报告。

源缓存/成员/原train-validation划分已核查，106491条记录身份唯一、关联排除组交集0；固定8060条盲轨迹。16项CPU技术检查通过，真实DreamOn smoke将在GPU资源允许后首先运行。CPU隔离官方评分器另有合成通过/断言失败检查。

预计约4–10 GPU作业小时，可能在12小时内完成，长链桶不保证满额。A1/A2合计最多4小时，DCD最多2小时；达到上限记录缺口，原预算不重置。GPU等待最多累计24小时，不计GPU作业成本。

执行/恢复：`bash scripts/launch_context_foundation_queue.sh`。详细协议和操作说明见docs/paper_agent/context_foundation_20261015_v1/plan.zh.md；本目录protocol_schema.json、frozen_protocol.json、source_audit.json与cpu_tests.txt供审阅。后台结束会自动用实际完整/部分结果重写本新版本报告、归档、提交推送并退出；旧记录不改。
