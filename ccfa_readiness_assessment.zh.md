# CCF-A Readiness Assessment

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
