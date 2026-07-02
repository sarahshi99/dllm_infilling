#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[2]
DECK_MD = OUT_DIR / "20260618_advisor_project_report.md"
DECK_HTML = OUT_DIR / "20260618_advisor_project_report.html"
DECK_PPTX = OUT_DIR / "20260618_advisor_project_report.pptx"
ASSESSMENT = ROOT / "ccfa_readiness_assessment.zh.md"


SLIDES = [
    {
        "title": "DLLM 代码填空长度控制项目汇报",
        "subtitle": "面向导师汇报的项目梳理与 CCF-A readiness 评估",
        "bullets": [
            "DLLM：Diffusion Language Model，扩散式语言模型，用逐步去噪生成文本/代码。",
            "代码填空：code infilling，给定前缀和后缀，模型补中间缺失代码。",
            "核心问题：模型不只要生成对，还要先选对“中间要留多长”。",
            "当前结论：已有可信进展，但还没到 CCF-A 投稿就绪水平。",
        ],
        "notes": "开场先说清楚：本项目不是普通 prompt 调参，而是在研究 DLLM 做代码填空时的长度选择问题。导师最关心的是问题是否重要、方法是否有贡献、实验是否够硬、离 CCF-A 还差什么。",
    },
    {
        "title": "一句话结论",
        "bullets": [
            "项目已经形成了清晰研究问题、可复现实验链和若干局部正结果。",
            "最可靠正结果：medium rescue，即中等长度低估可以被较安全地修复。",
            "最大短板：true-long，即真实答案很长的样本，尤其 oracle 25+ 仍没有解决。",
            "CCF-A：CCF 推荐 A 类会议，通常要求清晰新问题、强方法、严格对照和充分消融；当前 verdict 是 weak_candidate。",
        ],
        "notes": "这一页先给导师一个不绕弯的判断：有潜力，但不能现在包装成顶会主张。weak_candidate 的意思是有论文雏形和局部证据，但还缺少决定性的主方法与外部可比证据。",
    },
    {
        "title": "为什么长度控制重要",
        "bullets": [
            "在 DLLM 代码填空中，模型要先放置若干 mask，也就是可生成的空位。",
            "canvas：生成画布，指留给中间代码的 mask 长度。",
            "长度过短：答案被截断或语义不完整；长度过长：短答案任务可能被干扰。",
            "所以 length control：长度控制，是代码填空性能的核心变量，不是小工程细节。",
        ],
        "notes": "可以用一个直觉例子讲：如果正确答案需要 20 个 token，但模型只给 8 个空位，再强的解码也很难写完整；但如果所有题都给 32 个空位，又可能破坏短答案。",
    },
    {
        "title": "研究目标",
        "bullets": [
            "目标：在不知道真实答案长度的情况下，为 DLLM 自动选择合适填空长度。",
            "同时满足两个条件：提高中长答案通过率，并避免短答案回退。",
            "pass@1：一次生成就通过测试的比例，本项目用 pass rate 表达。",
            "核心问题：能否用 inference-time 信号，在推理阶段不训练模型也改善长度选择？",
        ],
        "notes": "强调 inference-time 的含义：模型权重不变，主要靠推理时的长度探测、置信度曲线和轨迹信号做控制。这是当前方法的优势，也是局限。",
    },
    {
        "title": "项目里几个名字的关系",
        "bullets": [
            "baseline：对照方法，用来判断当前方法是否真的更好。",
            "same-backbone：同一底座模型比较，避免模型能力差异污染结论。",
            "CAL-lite / LCAS / LCAL：项目内部的长度选择与修复策略族。",
            "official-CAL bounded repair：用官方 CAL 风格长度作为受限修复候选，不是无条件改长。",
            "literature anchor：文献报告值，只能定位相对水平，不能当成本地同协议 baseline。",
        ],
        "notes": "这一页是防止导师听乱的术语表。尤其要讲清楚：表里“论文报告值”和“我们之前的方法”不是同一个东西，不能偷换成 apples-to-apples 结论。",
    },
    {
        "title": "方法主线：从短安全到中等长度修复",
        "bullets": [
            "第一步：在候选长度集合里打分，先选一个短安全长度。",
            "第二步：如果长候选和短候选置信度接近，触发 LCAL 修正。",
            "第三步：如果 official-CAL 明显认为应该更长，再做 bounded repair。",
            "设计原则：宁可少触发，也不能让短答案大面积损失。",
        ],
        "notes": "可以把它讲成一个保守医生：先开温和药，如果检查信号强才加治疗，避免把本来没病的短题也改坏。",
    },
    {
        "title": "长度探测公式",
        "bullets": [
            "对每个候选长度 l，模型先看 prefix + [MASK]^l + suffix。",
            "raw_score(l)：中间每个 mask 位置最大 token 概率的平均值。",
            "length_power：score(l) = raw_score(l) x l^alpha，用很小长度奖励减少短答案偏置。",
            "选择规则：选 score(l) 最大的长度；并用短优先 tie-break 控制风险。",
        ],
        "notes": "这里不用深讲数学，关键直觉是：如果只看置信度，短答案天然更容易高分；乘以 l^alpha 是温和地补偿长答案。",
    },
    {
        "title": "bounded repair 的触发逻辑",
        "bullets": [
            "只有当短安全长度 s3_selected 足够短时，才考虑修复。",
            "official_selected：official-CAL 建议的长度。",
            "触发条件示例：official_selected 在允许区间内，且 delta = official_selected - s3_selected 在边界内。",
            "mid_rescue 还要求 long_ratio 足够高，并且来源符合 base/source 约束。",
        ],
        "notes": "这一页可配合代码讲：不是 official-CAL 说多长就多长，而是受区间、差值、ratio、source 多个门控限制。",
    },
    {
        "title": "Route2：用解码轨迹再救一次长样本",
        "bullets": [
            "trace：解码轨迹，指逐步去噪过程中的置信度、top1 等动态信号。",
            "Route2 先跑 primary 结果，再看轨迹是否像“卡住的长答案”。",
            "precision policy：top1_median <= 0.464844 且 confidence_max <= 0.84375 时触发。",
            "触发后用固定 rescue length 重新生成；不触发就保留 primary。",
        ],
        "notes": "Route2 是后续诊断推动出来的，不是最初主方法。它的价值在于无 loss 的小幅提升和对 failure mode 的解释。",
    },
    {
        "title": "Discovery V4：最新本地诊断",
        "bullets": [
            "Discovery V4：CPU-only 信号审计，不启动 GPU，只分析已有结果和轨迹。",
            "目标：找 missed failed-long 和 triggered rescue-failure 的稳定低风险信号。",
            "最新决策：route2_polish_only，没有发现比 Route2 polish 更强的低风险 V4 信号。",
            "含义：下一步不能盲目再跑 full GPU policy，需重做更有原则的 length model。",
        ],
        "notes": "这页说明最新进展不是又跑出强结果，而是把一条可能路线排除了。负结果对论文也有价值，但要诚实定位。",
    },
    {
        "title": "实验设置",
        "bullets": [
            "数据集：HumanEval-SingleLineInfilling，HumanEval 派生的单行代码填空评测，共 1033 条本地任务。",
            "指标：pass rate、pairwise wins/losses、oracle length buckets、runtime。",
            "oracle length bucket：按真实答案长度分桶，<=8、9-12、13-16、17-24、25+。",
            "比较口径：本地 same-backbone 是主要证据；文献 reported numbers 只是 anchors。",
        ],
        "notes": "这里要强调评测口径。如果导师问“为什么你高于某些论文数值还说不够”，答案是协议不同，不能直接宣称 SOTA。",
    },
    {
        "title": "A6000 主线结果",
        "bullets": [
            "control baseline：787/1033 = 76.19%。",
            "midcons：795/1033 = 76.96%，相对 control +8 wins / 0 losses。",
            "Route2 precision len32：801/1033 = 77.54%，相对 midcons +6 wins / 0 losses。",
            "但 25+ bucket 仍然 0 增益；true-long 没有真正解决。",
        ],
        "notes": "这是最适合汇报的主线表述：数字是正的，风险也小，但幅度不够大，而且长尾没解决。",
    },
    {
        "title": "跨 backbone 结果总览",
        "bullets": [
            "DreamCoder Base：80.54%，比本地 baseline +0.68pp，小幅正结果。",
            "DreamCoder Instruct：80.74%，比本地 baseline -1.36pp，negative transfer。",
            "Dream-7B / DiffuCoder / LLaDA-1.5：基本 near-tie，只多 +0.10pp 左右。",
            "LLaDA-MoE：77.54%，比本地 baseline +2.32pp，是当前最强 local transfer。",
        ],
        "notes": "讲的时候要把 MoE 作为亮点，但同时说它仍不是外部 SOTA，因为文献协议和模型训练范式不同。",
    },
    {
        "title": "最强局部亮点：LLaDA-MoE",
        "bullets": [
            "baseline：777/1033 = 75.22%。",
            "candidate：801/1033 = 77.54%，+24 tasks / +2.32pp。",
            "pairwise：31 wins / 7 losses / 770 tie-pass / 225 tie-fail。",
            "各 bucket 非负：<=8 +9，9-12 +5，13-16 +4，17-24 +6，25+ 0。",
        ],
        "notes": "这一页是最能打的实验结果。但注意最后一个 25+ 仍然 0，这正好说明核心难点还在。",
    },
    {
        "title": "失败分析：true-long 为什么难",
        "bullets": [
            "true-long：真实答案长度 >=17 的长样本，是当前最难区域。",
            "midcons 的 failed long 中，90/91 是 under-selected，即模型留的长度不够。",
            "Route2 precision len32 触发 35 个 failed-long，但只救回 2 个。",
            "33 个触发后仍失败的长样本里，31 个 rescue length 已经 >= oracle，说明不只是加长问题。",
        ],
        "notes": "这是非常关键的分析：问题不是简单“再加长一点”，而是 gate recall 和 rescue generation quality 同时有瓶颈。",
    },
    {
        "title": "目前证据支持什么",
        "bullets": [
            "支持：中等长度低估可以被较安全地修复。",
            "支持：轨迹信号有弱信息，能带来无 observed loss 的小幅 polish。",
            "支持：LLaDA-MoE 上有清楚的同 backbone 本地提升。",
            "支持：当前 official-CAL gate family 不是精确 true-long detector。",
        ],
        "notes": "这一页把可以说的 claim 列出来，避免讲成“什么都不行”。项目已经有不少扎实证据。",
    },
    {
        "title": "目前证据不支持什么",
        "bullets": [
            "不支持：已经解决 true-long code infilling。",
            "不支持：当前方法是 SOTA。SOTA 即 state of the art，当前最先进水平。",
            "不支持：跨 backbone 稳定提升，因为有多条 negative/near-tie 结果。",
            "不支持：现在直接投 CCF-A，因为外部对照、消融和主方法力度不够。",
        ],
        "notes": "这一页要讲得坦诚。导师一般更喜欢清楚知道边界，而不是把小结果硬说大。",
    },
    {
        "title": "CCF-A readiness gate",
        "bullets": [
            "central claim：已有精确窄 claim，但还不是强主张。",
            "baselines：本地 same-backbone 充分，protocol-matched external baselines 不足。",
            "ablation：消融实验，即拆开方法组件验证贡献来源；目前还不完整。",
            "error analysis：错误分析较强，但还需要转化为新方法或严谨 negative-result 论文。",
        ],
        "notes": "这页对应项目 protocol 的 Claim Readiness Gate。verdict 不是 paper_candidate，而是 weak_candidate。",
    },
    {
        "title": "离 CCF-A 还差什么",
        "bullets": [
            "一个更有原则的 long-length method，而不只是 heuristic gate。",
            "protocol-matched：同数据、同 prompt、同 evaluation setting 下复现强外部 baseline。",
            "完整 ablation：LCAL、official repair、Route2 trace、rescue length、runtime 分别贡献多少。",
            "稳定性：更多 seed/split 或 deterministic split discipline 下结果稳定。",
        ],
        "notes": "这一页是导师最需要的计划判断。建议不要把下一步说成“继续调规则”，而是提升为方法层面的长度建模。",
    },
    {
        "title": "推荐论文故事",
        "bullets": [
            "当前最好故事：diagnostic-plus-method，而不是单纯性能 SOTA。",
            "主线：DLLM 代码填空的长度选择存在 medium 与 true-long 分裂。",
            "贡献 1：安全中等长度修复方法和跨模型证据。",
            "贡献 2：系统证明 confidence/trace heuristic 难以解决 true-long，并提出下一代 length modeling。",
        ],
        "notes": "如果后续没有强 long 方法，可以转成负结果加启发式方法的论文；如果后续有新方法，则这套诊断就是论文动机和消融基础。",
    },
    {
        "title": "下一步建议",
        "bullets": [
            "短期：把 Discovery V4 写入 dashboard/results，并提交未跟踪脚本与报告。",
            "方法：设计 adaptive rescue generation 或 learned/dynamic length controller。",
            "实验：先 CPU gate，再 GPU smoke，再 full 1033；继续保护 GPU 2/3 不抢占。",
            "论文：补 protocol-matched external baseline、消融、失败案例和复现包。",
        ],
        "notes": "最后给导师一个可执行路线：不是马上大跑 GPU，而是先把方法方向定清楚，避免继续小修小补。",
    },
    {
        "title": "汇报时可以这样收束",
        "bullets": [
            "我目前已经把问题从“调长度”收敛到“medium 可以安全修复，但 true-long 需要新信号”。",
            "现有结果能支撑弱论文候选和扎实研究计划，但还不能支撑 CCF-A 投稿。",
            "我建议下一阶段把目标从 heuristic repair 升级为 principled length modeling。",
            "导师需要拍板：走 method paper，还是 diagnostic negative result + new controller paper。",
        ],
        "notes": "这一页可以作为口头总结。重点是请导师决策论文路线，而不是只汇报一堆跑分。",
    },
]


ASSESSMENT_TEXT = """# CCF-A Readiness Assessment

更新时间：2026-06-18 11:25 CST

## 总体判断

当前项目进展是：已经有明确研究问题、可复现实验链、多个同 backbone 本地对照、跨模型结果、trace 诊断和一批负结果；但还没有达到 CCF-A 会议论文的投稿就绪水平。

Claim Readiness Gate verdict：`weak_candidate`。

含义：项目已经不是零散实验，具备论文雏形；但还不能称为 `paper_candidate` 或 `submission_candidate`，因为核心贡献的“方法新意 + 强结果 + 外部可比性”还不够硬。尤其是 true-long code infilling 仍未解决，当前提升主要是 medium rescue 和 Route2 polish。

## 1. 研究背景和动机

DLLM（Diffusion Language Model，扩散式语言模型）通过逐步去噪生成文本或代码。和自回归模型不同，DLLM 做 code infilling（代码填空，即给定前缀和后缀，生成中间缺失代码）时，通常要先确定中间 canvas（生成画布，也就是留多少个 mask/token 位置）。

这个长度选择不是小工程细节。若 canvas 太短，真实答案写不完整；若 canvas 太长，短答案任务可能被干扰，甚至把本来能过的题改坏。本项目把问题聚焦为：

> 在不知道 oracle length（真实答案长度，只能在离线评测时知道）的情况下，DLLM 如何在推理阶段自动选择或修复 infilling length，并尽量提升中长答案而不伤害短答案？

这个问题有 CCF-A 潜力，因为它触到 DLLM code infilling 的实践限制：fixed canvas（固定长度画布）不灵活，oracle length 不现实，单纯 confidence-based length selection 又容易低估长答案。

## 2. 当前核心贡献和方法

当前最诚实的 central claim 是：

> Inference-time length control（推理阶段长度控制，不改模型权重，只在生成时选择/修复长度）可以较安全地恢复 medium-length under-selection（中等长度低估），但 true-long infilling（真实答案很长的填空）仍主要受 length underestimation（长度低估）和 rescue quality（修复生成质量）限制。

已形成的贡献雏形：

1. `LCAL / midcons`：用长度探测曲线识别中等长度低估，在较低短答案风险下修复一部分 medium bucket。
2. `official-CAL bounded repair`：只在保守边界内采用 official-CAL 风格的更长长度建议，避免无条件加长。
3. `Route2 trace-gated rescue`：用 trace（解码轨迹，即去噪过程中 top1/confidence/plateau 等动态信号）触发二次 rescue，得到小幅无 loss polish。
4. 系统负结果：single-feature probe、strict-split probe score、trace route analysis、Discovery V4 都显示当前信号不足以安全解决 true-long。

目前 strongest local positive：

- A6000 LLaDA-Base：Route2 precision len32 为 `801/1033 = 77.54%`，相对 midcons `+6` wins / `0` losses。
- LLaDA-MoE：candidate `801/1033 = 77.54%`，local baseline `777/1033 = 75.22%`，`+24` tasks / `+2.32pp`，pairwise `31/7/770/225`。

但是这些还不足以成为 CCF-A 级别的强主张，因为 true-long `25+` 基本没有改善，跨 backbone 结果混合，且外部 literature anchors 还不是 protocol-matched baselines。

## 3. 方法具体实现

### 3.1 长度探测

对每个候选长度 `l`，构造：

```text
prefix + [MASK]^l + suffix
```

然后让模型前向一次，计算中间 mask 位置的平均最大 token 概率：

```text
raw_score(l) = mean_t max_v P(v | prefix, [MASK]^l, suffix)
```

为了避免模型总偏向短答案，`length_power` 模式使用：

```text
score(l) = raw_score(l) * l^alpha
```

其中 `alpha` 很小，例如 `0.06` 或 `0.10`。实现位置：

```python
def adjust_length_probe_score(raw_score, mask_length, cfg):
    if cfg.decode.cal_lite_score_mode == "length_power":
        return raw_score * (mask_length ** cfg.decode.cal_lite_length_alpha)
```

对应文件：`expvision_dllm_clean/length_probe.py`。

### 3.2 LCAL / midcons

LCAL 先在 compact length grid 上选择 base length，再判断是否需要更长的 correction。直觉是：如果长候选的分数接近或支持度足够，说明短答案选择可能低估了真实长度。

简化伪代码：

```python
base = select_length(base_grid, alpha=0.06)
if long_score_is_competitive(base, long_grid):
    long = select_length(long_grid, alpha=0.10)
    selected = max(base, long)
else:
    selected = base
```

项目里的实现还加入了 `weak_window`、`strong_min_len`、`support_count`、`long_score_floor`、`ratio_trigger_threshold` 等门控，目的是减少 short-risk（短答案风险，即在真实短答案上误触发长修复）。

### 3.3 official-CAL bounded repair

bounded repair 的核心不是“official-CAL 说多长就用多长”，而是只有满足边界才修复：

```text
delta = official_selected - s3_selected
trigger = length_condition and delta_condition and optional_mid_rescue_conditions
```

简化伪代码：

```python
if s3_selected <= repair_max_s3_len:
    official_selected = official_cal_select()
    delta = official_selected - s3_selected
    if (
        repair_min_official_len <= official_selected <= repair_max_official_len
        and repair_min_delta <= delta <= repair_max_delta
    ):
        selected = official_selected
```

对应文件：`clean_scripts/run_lcal_official_bounded_repair.py`。关键设计是保守触发，避免短答案回退。

### 3.4 Route2 trace-gated rescue

Route2 先运行 primary 方法并保存 step traces（逐步去噪轨迹）。如果轨迹像“长答案被卡住”，才跑 fixed-length rescue。

两个已测试策略：

```python
POLICIES = {
    "broad_plateau": [
        ["top1_last", "<=", 0.667969],
        ["max_remaining_plateau_steps", ">=", 16.0],
    ],
    "precision_top1_conf": [
        ["top1_median", "<=", 0.464844],
        ["confidence_max", "<=", 0.84375],
    ],
}
```

对应文件：`clean_scripts/run_route2_trace_rescue.py`。

`precision_top1_conf + rescue_len32` 的结果最干净：`801/1033 = 77.54%`，pairwise `6/0/795/232`，无 observed loss。

### 3.5 Discovery V4

Discovery V4 是 CPU-only signal audit（只用 CPU 的信号审计，不启动 GPU），用于判断是否存在新的低风险 V4 policy。它构造 row-action table，合并 baseline、Route2 len24/len32、broad len24、probe/trace fields，再搜索 slice/rule/trace-shape/calibration/uplift 信号。

它的候选评分为：

```text
score =
  8 * missed_failed_long
+ 2 * triggered_rescue_failure
+ 5 * route2_win
+ 4 * true_long_precision
- 5 * short_risk
- 3 * current_pass_risk
```

其中 current-pass risk 指当前 baseline 已经通过但候选规则可能干预的风险。

最新结果：

- output：`analysis_outputs/discovery_v4_signal_audit_20260618_000000`
- decision：`route2_polish_only`
- reason：没有找到比 Route2 polish 更稳定的低风险 V4 信号
- verification：`tests/test_discovery_v4_signal_audit.py` 5 个单测通过，`py_compile` 通过

## 4. 实验 setting 和结果

### 4.1 统一设置

- Dataset：`HumanEval-SingleLineInfilling`，HumanEval 派生的单行代码填空任务，本地共 `1033` rows。
- Main metric：pass rate / pass@1，即一次生成通过单元测试的比例。
- Pairwise W/L/TP/TF：同一题上 candidate 相对 baseline 的 win、loss、tie-pass、tie-fail。
- Oracle buckets：按真实答案长度分桶：`<=8`、`9-12`、`13-16`、`17-24`、`25+`。
- Comparison type：local same-backbone comparison 是主证据；CAL、LR-DLLM、DreamOn 等文献数值只是 literature anchors，除非本地复现同协议。

### 4.2 A6000 LLaDA-Base 主线

| Run | Pass | Rate | Delta |
|---|---:|---:|---:|
| control | `787/1033` | `76.19%` | baseline |
| midcons | `795/1033` | `76.96%` | `+8`, `0` losses |
| Route2 precision len32 | `801/1033` | `77.54%` | `+6` vs midcons, `0` losses |

Bucket 结论：medium bucket 有收益，但 oracle `25+` 没有提升。

### 4.3 跨 backbone 总结

| Backbone | Local baseline | Current/candidate | Delta |
|---|---:|---:|---:|
| LLaDA-8B-Base Route2 | `795/1033` midcons | `801/1033` | `+6` vs midcons |
| LLaDA-8B-Instruct | `817/1033` | `815/1033` | `-2` |
| DreamCoder Base | `825/1033` | `832/1033` | `+7` |
| DreamCoder Instruct | `848/1033` | `834/1033` | `-14` |
| Dream-7B | `802/1033` | `803/1033` | `+1` |
| DiffuCoder-Base | `838/1033` | `839/1033` | `+1` |
| LLaDA-1.5 | `817/1033` | `818/1033` | `+1` |
| LLaDA-MoE | `777/1033` | `801/1033` | `+24` |

结论：LLaDA-MoE 是当前最强 local transfer；其他多数是 near-tie 或 negative transfer，不能宣称跨 backbone 稳定强提升。

### 4.4 True-long failure evidence

关键发现：

- `midcons` 的 failed long 中 `90/91 = 98.90%` 是 under-selected。
- Route2 precision len32 的 `57` triggers 中 true-long precision 为 `61.40%`。
- `35` 个 failed-long 被 precision len32 触发，但只 rescue 成功 `2` 个。
- `33` 个 triggered failed-long 仍失败，其中 `31/33` 的 rescue length 已经 >= oracle。

这说明 true-long 的主要瓶颈不是简单“长度再加大”，而是 gate recall 与 rescue generation/selection quality 同时不足。

## 5. 是否达到 CCF-A 水平

### 5.1 支持点

- 问题重要：unknown-length DLLM code infilling 是真实限制。
- 证据链较完整：有 same-hardware、same-backbone、cross-backbone、trace diagnostics、negative evidence。
- 项目纪律较好：报告中记录 baseline、环境、GPU、输出目录、pairwise、bucket metrics。
- 已有局部亮点：midcons 无 loss 正收益、Route2 polish、LLaDA-MoE +2.32pp。

### 5.2 不足点

- 方法新意不足：当前规则仍偏 heuristic，缺少一个有原则的 long-length controller。
- 主结果不够强：多数 backbone 是小幅正、near-tie 或 negative transfer。
- true-long 没解决：oracle `25+` 基本没有改善。
- 外部比较不足：文献结果不是 protocol-matched baseline，不能写 SOTA。
- 消融不足：还需要系统拆解 LCAL、official repair、Route2 trace、rescue length、runtime cost。
- 统计稳定性不足：需要更多 split/seed 或严格 deterministic split discipline。

### 5.3 Verdict

当前不是 CCF-A submission-ready。

最高合理表述：

> The project is a weak but credible paper candidate: it has a real problem, careful diagnostics, and localized positive evidence, but it still needs a principled long-length method or a stronger diagnostic-negative-result framing with protocol-matched baselines.

中文汇报表述：

> 目前项目已经从“调长度策略”推进到“发现 medium 与 true-long 两类错误机制不同”的阶段。它可以支撑一个有潜力的研究方向，但还不能支撑 CCF-A 投稿。下一阶段必须把 heuristic repair 升级为 principled length modeling，或者明确转成 rigorous diagnostic/negative-result paper。

## 6. 建议下一步

1. 先把 Discovery V4 的脚本、测试、report 写入 dashboard/results，并做 focused commit。
2. 重新定义主方法：adaptive rescue generation、learned length classifier、dynamic canvas controller 或 length regularization。
3. 建立 protocol-matched external baselines：同数据、同 prompt、同 canvas/evaluation setting 下对 CAL、LR-DLLM、DreamOn 或其可复现部分做对照。
4. 做系统消融：移除/替换 LCAL、official repair、Route2 trace、rescue length、score mode，报告 total、bucket、wins/losses、runtime。
5. 做错误案例分析：尤其是 `25+`、triggered-but-still-failed、missed failed-long。
6. 论文 framing 二选一：method paper，或 diagnostic negative result + new controller paper。
"""


def _escape(text: str) -> str:
    return escape(text, {"'": "&apos;", '"': "&quot;"})


def write_markdown() -> None:
    lines = [
        "# DLLM 代码填空长度控制项目汇报",
        "",
        "生成时间：2026-06-18 11:25 CST",
        "",
        "说明：本 Markdown 是 PPT 的讲稿版。每页先给 slide bullet，再给“讲法”。",
        "",
    ]
    for idx, slide in enumerate(SLIDES, 1):
        lines.append(f"---\n\n## {idx}. {slide['title']}")
        if slide.get("subtitle"):
            lines.append(f"\n**{slide['subtitle']}**")
        lines.append("")
        for bullet in slide["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append(f"讲法：{slide['notes']}")
        lines.append("")
    DECK_MD.write_text("\n".join(lines), encoding="utf-8")


def write_html() -> None:
    sections = []
    for idx, slide in enumerate(SLIDES, 1):
        subtitle = f"<p class='subtitle'>{html.escape(slide.get('subtitle', ''))}</p>" if slide.get("subtitle") else ""
        bullets = "\n".join(f"<li>{html.escape(item)}</li>" for item in slide["bullets"])
        sections.append(
            f"<section class='slide'><div class='kicker'>Slide {idx:02d}</div><h1>{html.escape(slide['title'])}</h1>{subtitle}<ul>{bullets}</ul><p class='notes'><b>讲法：</b>{html.escape(slide['notes'])}</p></section>"
        )
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DLLM 代码填空长度控制项目汇报</title>
<style>
body {{ margin: 0; font-family: "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; background: #f4f6f8; color: #17202a; }}
.slide {{ width: 1120px; min-height: 630px; margin: 28px auto; padding: 54px 68px; box-sizing: border-box; background: white; border: 1px solid #d9e0e7; box-shadow: 0 10px 28px rgba(21, 36, 52, .10); }}
.kicker {{ color: #0f766e; font-size: 18px; font-weight: 700; margin-bottom: 18px; }}
h1 {{ font-size: 38px; line-height: 1.15; margin: 0 0 20px; color: #102033; }}
.subtitle {{ font-size: 24px; color: #4b5563; margin: -6px 0 22px; }}
ul {{ font-size: 24px; line-height: 1.42; margin: 0 0 26px 0; padding-left: 30px; }}
li {{ margin: 10px 0; }}
.notes {{ margin-top: 28px; padding-top: 18px; border-top: 1px solid #e5e7eb; font-size: 18px; line-height: 1.5; color: #374151; }}
@media print {{ .slide {{ page-break-after: always; margin: 0; box-shadow: none; border: none; }} }}
</style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""
    DECK_HTML.write_text(doc, encoding="utf-8")


def para(text: str, size: int, color: str, bullet: bool = False, bold: bool = False) -> str:
    ppr = ""
    if bullet:
        ppr = '<a:pPr lvl="0" marL="342900" indent="-171450"><a:buChar char="•"/></a:pPr>'
    return (
        f"<a:p>{ppr}<a:r><a:rPr lang=\"zh-CN\" sz=\"{size}\" b=\"{'1' if bold else '0'}\">"
        f"<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill></a:rPr><a:t>{_escape(text)}</a:t></a:r></a:p>"
    )


def textbox(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: list[str]) -> str:
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{_escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
</p:sp>"""


def slide_xml(slide: dict, idx: int) -> str:
    title = slide["title"]
    bullets = slide["bullets"]
    title_box = textbox(2, "Title", 530000, 360000, 11100000, 780000, [para(title, 3300, "102033", bold=True)])
    subtitle_box = ""
    if slide.get("subtitle"):
        subtitle_box = textbox(3, "Subtitle", 560000, 1070000, 10500000, 520000, [para(slide["subtitle"], 1900, "4B5563")])
        body_y = 1620000
        shape_id = 4
    else:
        body_y = 1290000
        shape_id = 3
    bullet_size = 2050 if len(bullets) <= 4 else 1850
    bullet_paras = [para(item, bullet_size, "17202A", bullet=True) for item in bullets]
    body_box = textbox(shape_id, "Body", 760000, body_y, 10800000, 3950000, bullet_paras)
    note = slide["notes"]
    if len(note) > 126:
        note = note[:123] + "..."
    notes_box = textbox(shape_id + 1, "SpeakerHint", 760000, 5850000, 10300000, 560000, [para("讲法：" + note, 1250, "475569")])
    footer = textbox(shape_id + 2, "Footer", 10000000, 6500000, 1600000, 260000, [para(f"{idx}/{len(SLIDES)}", 1000, "64748B")])
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title_box}
      {subtitle_box}
      {body_box}
      {notes_box}
      {footer}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def write_pptx() -> None:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(SLIDES) + 1)
    )
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  {slide_overrides}
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    slide_ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, len(SLIDES) + 1))
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""
    pres_rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, len(SLIDES) + 1):
        pres_rels.append(
            f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    presentation_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(pres_rels)
        + "</Relationships>"
    )
    slide_master = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""
    slide_master_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""
    slide_layout = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""
    slide_layout_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""
    theme = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="AdvisorReport">
  <a:themeElements>
    <a:clrScheme name="Advisor"><a:dk1><a:srgbClr val="102033"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="17202A"/></a:dk2><a:lt2><a:srgbClr val="F4F6F8"/></a:lt2><a:accent1><a:srgbClr val="0F766E"/></a:accent1><a:accent2><a:srgbClr val="2563EB"/></a:accent2><a:accent3><a:srgbClr val="B45309"/></a:accent3><a:accent4><a:srgbClr val="BE123C"/></a:accent4><a:accent5><a:srgbClr val="475569"/></a:accent5><a:accent6><a:srgbClr val="64748B"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="AdvisorFonts"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="AdvisorFmt"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>DLLM 代码填空长度控制项目汇报</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex PPTX generator</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(SLIDES)}</Slides><Company></Company>
</Properties>"""
    pres_props = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>'
    view_props = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>'

    with zipfile.ZipFile(DECK_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        zf.writestr("ppt/presentation.xml", presentation)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/presProps.xml", pres_props)
        zf.writestr("ppt/viewProps.xml", view_props)
        zf.writestr("ppt/theme/theme1.xml", theme)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        for idx, slide in enumerate(SLIDES, 1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide, idx))
            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>',
            )


def main() -> None:
    write_markdown()
    write_html()
    write_pptx()
    ASSESSMENT.write_text(ASSESSMENT_TEXT, encoding="utf-8")
    manifest = {
        "markdown": str(DECK_MD),
        "html": str(DECK_HTML),
        "pptx": str(DECK_PPTX),
        "assessment": str(ASSESSMENT),
        "slide_count": len(SLIDES),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
