# DreamOn SingleLine Markov 前提补充诊断 v3：结果

状态：`completed`。分支 `codex/dreamon-markov-premise-rerun-v3`，base `5d5d5f2eb9c550e77327e35e827a9ed2ca27b34d`。

本轮只运行全局置信度逐词刷新诊断和固定左前沿逐词刷新诊断，不重跑 C2/C4/L2/L4，不训练 Markov head。官方 SingleLine development/full-allowed population 完成 `1033×2=2066`，共同 task `1033`、基础 task group `164`、duplicate/missing/error=`0/0/0`；Pass@1 精确复现 `951/1033` 与 `942/1033`。

reference-prefix aligned 的 offset-1 raw reference ΔlogP 分别为 `+0.0676 [0.0402,0.0966]` 和 `+0.0778 [0.0516,0.1057]`；wrong→correct / correct→wrong 分别 `72/30` 与 `80/28`。actual-decode finite ΔlogP 点估计为正，但 CI 分别 `[-0.0011,0.0387]` 与 `[-0.0015,0.0387]`，均跨 0。offset=2/3 的 reference 方向更强，但 stale→fresh TV 增长、top-1 agreement 下降，因此它们主要是长块风险证据，不能与 offset=1 混合解释。

全局置信度普通 step 中，i+1 连续右邻存在率 `95.52%`；存在时 source top-2/top-4 占比 `93.24%/93.26%`；这些 source 候选在 fresh forward 后的 top-2/top-4 retention 为 `97.17%/97.71%`。这支持未来另行预注册的 global top-K 连续短链设计验证，但不证明 Markov head 可学习。

2026-08-23 的六臂质量/效率结果仍有效；其 Markov stop 判断因两个预注册字段未采集而被本轮补充诊断取代，字段缺失不解释为负面结果。权威中文报告、CSV、完整性审计和压缩逐 case/transition 数据见 `analysis_outputs/dreamon_markov_premise_rerun_20260830_v3/`。
