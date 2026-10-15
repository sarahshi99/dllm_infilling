# 2026-10-15 并行扩删坐标修补与受影响样本补验

用户已授权此有限维护批次，基线dbdd0bd，独立分支codex/dreamon-structure-coordinate-fix-20261015-v1。只把扩展前EOS删除资格与插入/截断同步，不改变EOS广播、预算、选位、RNG、结构屏障、头或生成参数。替代方案（删除全部EOS、禁广播、单结构限制）均改变方法，不采用。无subagent，karpathy-guidelines；reviewer_gate_disabled，本地diff与fresh CPU verification。

先重放封存离散轨迹和复用来源，确认旧画布可复算后定位首分歧；首分歧后不把旧动作当修正模型输出。影响筛选不读passed，未知纳入补验。A/global原生K4最多1033、C/global B/C/D最多17445，上限18478；K1、链K4、frontier、Base须有非干扰/重放依据才复用。独立目录analysis_outputs/structure_coordinate_fix_20261015_v2，旧analysis_outputs/k4_multiline_20261014_v1任何文件不改。

主模型bf16与v2 KL/full/best rank256 float32冻结，lambda1、seed42、旧row_seed和边seed、不接入common-u。正常官方错误/3s超时保留，只有基础设施中断原seed恢复。GPU0单H200，一个worker，沿用32GiB加载余量、全局mutex；当前GPU拥挤则后台等待，不抢占、不设评测时长上限。CPU技术检查→固定规则smoke→影响清单全量→全人口合并/配对/基础题bootstrap→归档/提交推送→退出。不训练、不改思维导图、不扩大旧单行K2锚点。

报告按全5815、单行跨度1033、跨行4782及跨行2–3/4–7/至少8行；总体恢复和纯学习恢复分母分别使用其无头/零头，分母非正不适用。bootstrap按旧10000次、seed20261013、基础题整组、行加权与基础题等权。共享GPU子集时间不混成全人口公平加速。最后明确评分变化、学习净增、结论保留/更正并关闭此项。

只读核对：2026-10-15 ls-remote全部heads无坐标修补/后继分支，原分支远端仍dbdd0bd；本机全部worktree/同用户进程及tmux未发现同项队列；原ABC完成矩阵完整。本机主worktree的既有未跟踪文件不碰。下载Dream/DiffuCoder及四个新头方向均保持下阶段待办。
