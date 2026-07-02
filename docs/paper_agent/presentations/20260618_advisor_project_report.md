# DLLM 代码填空长度控制项目汇报

生成时间：2026-06-18 11:25 CST

说明：本 Markdown 是 PPT 的讲稿版。每页先给 slide bullet，再给“讲法”。

---

## 1. DLLM 代码填空长度控制项目汇报

**面向导师汇报的项目梳理与 CCF-A readiness 评估**

- DLLM：Diffusion Language Model，扩散式语言模型，用逐步去噪生成文本/代码。
- 代码填空：code infilling，给定前缀和后缀，模型补中间缺失代码。
- 核心问题：模型不只要生成对，还要先选对“中间要留多长”。
- 当前结论：已有可信进展，但还没到 CCF-A 投稿就绪水平。

讲法：开场先说清楚：本项目不是普通 prompt 调参，而是在研究 DLLM 做代码填空时的长度选择问题。导师最关心的是问题是否重要、方法是否有贡献、实验是否够硬、离 CCF-A 还差什么。

---

## 2. 一句话结论

- 项目已经形成了清晰研究问题、可复现实验链和若干局部正结果。
- 最可靠正结果：medium rescue，即中等长度低估可以被较安全地修复。
- 最大短板：true-long，即真实答案很长的样本，尤其 oracle 25+ 仍没有解决。
- CCF-A：CCF 推荐 A 类会议，通常要求清晰新问题、强方法、严格对照和充分消融；当前 verdict 是 weak_candidate。

讲法：这一页先给导师一个不绕弯的判断：有潜力，但不能现在包装成顶会主张。weak_candidate 的意思是有论文雏形和局部证据，但还缺少决定性的主方法与外部可比证据。

---

## 3. 为什么长度控制重要

- 在 DLLM 代码填空中，模型要先放置若干 mask，也就是可生成的空位。
- canvas：生成画布，指留给中间代码的 mask 长度。
- 长度过短：答案被截断或语义不完整；长度过长：短答案任务可能被干扰。
- 所以 length control：长度控制，是代码填空性能的核心变量，不是小工程细节。

讲法：可以用一个直觉例子讲：如果正确答案需要 20 个 token，但模型只给 8 个空位，再强的解码也很难写完整；但如果所有题都给 32 个空位，又可能破坏短答案。

---

## 4. 研究目标

- 目标：在不知道真实答案长度的情况下，为 DLLM 自动选择合适填空长度。
- 同时满足两个条件：提高中长答案通过率，并避免短答案回退。
- pass@1：一次生成就通过测试的比例，本项目用 pass rate 表达。
- 核心问题：能否用 inference-time 信号，在推理阶段不训练模型也改善长度选择？

讲法：强调 inference-time 的含义：模型权重不变，主要靠推理时的长度探测、置信度曲线和轨迹信号做控制。这是当前方法的优势，也是局限。

---

## 5. 项目里几个名字的关系

- baseline：对照方法，用来判断当前方法是否真的更好。
- same-backbone：同一底座模型比较，避免模型能力差异污染结论。
- CAL-lite / LCAS / LCAL：项目内部的长度选择与修复策略族。
- official-CAL bounded repair：用官方 CAL 风格长度作为受限修复候选，不是无条件改长。
- literature anchor：文献报告值，只能定位相对水平，不能当成本地同协议 baseline。

讲法：这一页是防止导师听乱的术语表。尤其要讲清楚：表里“论文报告值”和“我们之前的方法”不是同一个东西，不能偷换成 apples-to-apples 结论。

---

## 6. 方法主线：从短安全到中等长度修复

- 第一步：在候选长度集合里打分，先选一个短安全长度。
- 第二步：如果长候选和短候选置信度接近，触发 LCAL 修正。
- 第三步：如果 official-CAL 明显认为应该更长，再做 bounded repair。
- 设计原则：宁可少触发，也不能让短答案大面积损失。

讲法：可以把它讲成一个保守医生：先开温和药，如果检查信号强才加治疗，避免把本来没病的短题也改坏。

---

## 7. 长度探测公式

- 对每个候选长度 l，模型先看 prefix + [MASK]^l + suffix。
- raw_score(l)：中间每个 mask 位置最大 token 概率的平均值。
- length_power：score(l) = raw_score(l) x l^alpha，用很小长度奖励减少短答案偏置。
- 选择规则：选 score(l) 最大的长度；并用短优先 tie-break 控制风险。

讲法：这里不用深讲数学，关键直觉是：如果只看置信度，短答案天然更容易高分；乘以 l^alpha 是温和地补偿长答案。

---

## 8. bounded repair 的触发逻辑

- 只有当短安全长度 s3_selected 足够短时，才考虑修复。
- official_selected：official-CAL 建议的长度。
- 触发条件示例：official_selected 在允许区间内，且 delta = official_selected - s3_selected 在边界内。
- mid_rescue 还要求 long_ratio 足够高，并且来源符合 base/source 约束。

讲法：这一页可配合代码讲：不是 official-CAL 说多长就多长，而是受区间、差值、ratio、source 多个门控限制。

---

## 9. Route2：用解码轨迹再救一次长样本

- trace：解码轨迹，指逐步去噪过程中的置信度、top1 等动态信号。
- Route2 先跑 primary 结果，再看轨迹是否像“卡住的长答案”。
- precision policy：top1_median <= 0.464844 且 confidence_max <= 0.84375 时触发。
- 触发后用固定 rescue length 重新生成；不触发就保留 primary。

讲法：Route2 是后续诊断推动出来的，不是最初主方法。它的价值在于无 loss 的小幅提升和对 failure mode 的解释。

---

## 10. Discovery V4：最新本地诊断

- Discovery V4：CPU-only 信号审计，不启动 GPU，只分析已有结果和轨迹。
- 目标：找 missed failed-long 和 triggered rescue-failure 的稳定低风险信号。
- 最新决策：route2_polish_only，没有发现比 Route2 polish 更强的低风险 V4 信号。
- 含义：下一步不能盲目再跑 full GPU policy，需重做更有原则的 length model。

讲法：这页说明最新进展不是又跑出强结果，而是把一条可能路线排除了。负结果对论文也有价值，但要诚实定位。

---

## 11. 实验设置

- 数据集：HumanEval-SingleLineInfilling，HumanEval 派生的单行代码填空评测，共 1033 条本地任务。
- 指标：pass rate、pairwise wins/losses、oracle length buckets、runtime。
- oracle length bucket：按真实答案长度分桶，<=8、9-12、13-16、17-24、25+。
- 比较口径：本地 same-backbone 是主要证据；文献 reported numbers 只是 anchors。

讲法：这里要强调评测口径。如果导师问“为什么你高于某些论文数值还说不够”，答案是协议不同，不能直接宣称 SOTA。

---

## 12. A6000 主线结果

- control baseline：787/1033 = 76.19%。
- midcons：795/1033 = 76.96%，相对 control +8 wins / 0 losses。
- Route2 precision len32：801/1033 = 77.54%，相对 midcons +6 wins / 0 losses。
- 但 25+ bucket 仍然 0 增益；true-long 没有真正解决。

讲法：这是最适合汇报的主线表述：数字是正的，风险也小，但幅度不够大，而且长尾没解决。

---

## 13. 跨 backbone 结果总览

- DreamCoder Base：80.54%，比本地 baseline +0.68pp，小幅正结果。
- DreamCoder Instruct：80.74%，比本地 baseline -1.36pp，negative transfer。
- Dream-7B / DiffuCoder / LLaDA-1.5：基本 near-tie，只多 +0.10pp 左右。
- LLaDA-MoE：77.54%，比本地 baseline +2.32pp，是当前最强 local transfer。

讲法：讲的时候要把 MoE 作为亮点，但同时说它仍不是外部 SOTA，因为文献协议和模型训练范式不同。

---

## 14. 最强局部亮点：LLaDA-MoE

- baseline：777/1033 = 75.22%。
- candidate：801/1033 = 77.54%，+24 tasks / +2.32pp。
- pairwise：31 wins / 7 losses / 770 tie-pass / 225 tie-fail。
- 各 bucket 非负：<=8 +9，9-12 +5，13-16 +4，17-24 +6，25+ 0。

讲法：这一页是最能打的实验结果。但注意最后一个 25+ 仍然 0，这正好说明核心难点还在。

---

## 15. 失败分析：true-long 为什么难

- true-long：真实答案长度 >=17 的长样本，是当前最难区域。
- midcons 的 failed long 中，90/91 是 under-selected，即模型留的长度不够。
- Route2 precision len32 触发 35 个 failed-long，但只救回 2 个。
- 33 个触发后仍失败的长样本里，31 个 rescue length 已经 >= oracle，说明不只是加长问题。

讲法：这是非常关键的分析：问题不是简单“再加长一点”，而是 gate recall 和 rescue generation quality 同时有瓶颈。

---

## 16. 目前证据支持什么

- 支持：中等长度低估可以被较安全地修复。
- 支持：轨迹信号有弱信息，能带来无 observed loss 的小幅 polish。
- 支持：LLaDA-MoE 上有清楚的同 backbone 本地提升。
- 支持：当前 official-CAL gate family 不是精确 true-long detector。

讲法：这一页把可以说的 claim 列出来，避免讲成“什么都不行”。项目已经有不少扎实证据。

---

## 17. 目前证据不支持什么

- 不支持：已经解决 true-long code infilling。
- 不支持：当前方法是 SOTA。SOTA 即 state of the art，当前最先进水平。
- 不支持：跨 backbone 稳定提升，因为有多条 negative/near-tie 结果。
- 不支持：现在直接投 CCF-A，因为外部对照、消融和主方法力度不够。

讲法：这一页要讲得坦诚。导师一般更喜欢清楚知道边界，而不是把小结果硬说大。

---

## 18. CCF-A readiness gate

- central claim：已有精确窄 claim，但还不是强主张。
- baselines：本地 same-backbone 充分，protocol-matched external baselines 不足。
- ablation：消融实验，即拆开方法组件验证贡献来源；目前还不完整。
- error analysis：错误分析较强，但还需要转化为新方法或严谨 negative-result 论文。

讲法：这页对应项目 protocol 的 Claim Readiness Gate。verdict 不是 paper_candidate，而是 weak_candidate。

---

## 19. 离 CCF-A 还差什么

- 一个更有原则的 long-length method，而不只是 heuristic gate。
- protocol-matched：同数据、同 prompt、同 evaluation setting 下复现强外部 baseline。
- 完整 ablation：LCAL、official repair、Route2 trace、rescue length、runtime 分别贡献多少。
- 稳定性：更多 seed/split 或 deterministic split discipline 下结果稳定。

讲法：这一页是导师最需要的计划判断。建议不要把下一步说成“继续调规则”，而是提升为方法层面的长度建模。

---

## 20. 推荐论文故事

- 当前最好故事：diagnostic-plus-method，而不是单纯性能 SOTA。
- 主线：DLLM 代码填空的长度选择存在 medium 与 true-long 分裂。
- 贡献 1：安全中等长度修复方法和跨模型证据。
- 贡献 2：系统证明 confidence/trace heuristic 难以解决 true-long，并提出下一代 length modeling。

讲法：如果后续没有强 long 方法，可以转成负结果加启发式方法的论文；如果后续有新方法，则这套诊断就是论文动机和消融基础。

---

## 21. 下一步建议

- 短期：把 Discovery V4 写入 dashboard/results，并提交未跟踪脚本与报告。
- 方法：设计 adaptive rescue generation 或 learned/dynamic length controller。
- 实验：先 CPU gate，再 GPU smoke，再 full 1033；继续保护 GPU 2/3 不抢占。
- 论文：补 protocol-matched external baseline、消融、失败案例和复现包。

讲法：最后给导师一个可执行路线：不是马上大跑 GPU，而是先把方法方向定清楚，避免继续小修小补。

---

## 22. 汇报时可以这样收束

- 我目前已经把问题从“调长度”收敛到“medium 可以安全修复，但 true-long 需要新信号”。
- 现有结果能支撑弱论文候选和扎实研究计划，但还不能支撑 CCF-A 投稿。
- 我建议下一阶段把目标从 heuristic repair 升级为 principled length modeling。
- 导师需要拍板：走 method paper，还是 diagnostic negative result + new controller paper。

讲法：这一页可以作为口头总结。重点是请导师决策论文路线，而不是只汇报一堆跑分。
