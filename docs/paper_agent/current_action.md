# 当前有限动作：threshold trace/fork

2026-11-16。用户已授权的机制诊断，范围以本实验brief为准，无训练或阈值搜索。
114/114真实技术检查通过，原入口/日志/快照恢复逐轮一致；CPU22项通过。当前执行五配置5165忠实回放及规定的有限分叉、教师64和焦点从头54。
分支codex/dreamon-threshold-trace-fork-20261112-v1，从285f8f1建立；原执行660f296。
命令：`bash experiments/threshold_trace_fork/launch.sh`；单模型cuda:0，当前映射H200，共享负载，不干预他人。
输出/状态：analysis_outputs/dreamon_threshold_trace_fork_20261112_v1/{status.json,resume.md}；日志local/run.log。
科学有效性异常先停止依赖分叉；普通失败和3秒超时保留。reviewer_gate_disabled，本地diff review + fresh verification。

中间完整配对已核验：K2/K4无头.95各1033行精确复现；通过952/953。105例K4前向增加（73例最终词元相同），其中81例扩展更多。总ΔR815=ΔO(-85)-ΔS109+ΔZ1009；完整配对表见k2_k4_replay_verified.json。其余配置与因果分叉尚未完成，这些是配对描述，不据此归因。
