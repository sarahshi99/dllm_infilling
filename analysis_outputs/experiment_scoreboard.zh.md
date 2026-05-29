# LCAL Infilling 实验 Scoreboard

更新时间：2026-05-29 Asia/Shanghai

## 当前决策

`midcons` 现在也是 A6000 rerun 中的最佳策略。它相对 A6000 control 增加 `+8` wins、`0` losses，达到 `795/1033 = 76.96%`。

当前 long-tail 分支是负面证据：`true_long` 和 `combined` 与 A6000 control 完全持平，因为 official-CAL true-long trigger 在安全 gates 后触发次数为 0。下一轮长样本实验应使用独立的 long-underestimation detector，而不是继续放宽同一个 official-CAL trigger。

历史 RTX 5090 决策：

`midcons` 曾是 RTX 5090 / torch 2.11 cu128 环境中的最佳策略，但由于环境迁移本身引入了小幅 short-bucket regression，它当时还不是相对旧服务器 union checkpoint 的干净 global best。

术语使用：

- **Old global checkpoint:** `old_union_gpus23`，`787/1033 = 76.19%`。
- **New-environment control:** `union_gpus01_control`，`785/1033 = 75.99%`。
- **Best new-environment strategy:** `midcons`，`791/1033 = 76.57%`。

## 主表

| Run | 环境 | 方法 | Pass | 相对 old union | 相对 gpus01 union | <=8 | 9-12 | 13-16 | 17-24 | 25+ | 最佳状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `old_union_gpus23` | old server / gpus23 | union: S3 short-safe + bounded repair off6..9 + long suspicion off>=16 | `787/1033` `76.19%` | baseline | n/a | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | old global best |
| `union_gpus01_control` | new 5090 / torch cu128 | 精确复跑 old union config | `785/1033` `75.99%` | `-2` | baseline | `89.30%` | `77.59%` | `54.44%` | `20.73%` | `16.13%` | environment control |
| `eval12_nomiddle_gpus01_control` | new 5090 / torch cu128 | old union，但 official probing 扩展到 S3<=12；无 mid rescue | `785/1033` `75.99%` | `-2` | `0` | `89.30%` | `77.59%` | `54.44%` | `20.73%` | `16.13%` | 仅 gate-control |
| `midcons` | new 5090 / torch cu128 | union + conservative mid rescue off11..13, delta3..7, ratio>=0.8 | `791/1033` `76.57%` | `+4` | `+6` | `89.30%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | gpus01 环境策略最佳 |
| `midaggr` | new 5090 / torch cu128 | union + aggressive mid rescue off11..15, delta3..8, ratio>=0.8 | `789/1033` `76.38%` | `+2` | `+4` | `88.96%` | `77.59%` | `61.11%` | `20.73%` | `16.13%` | 仅记录；伤害 short |

## A6000 受控运行

| Run | 方法 | Pass | 相对 A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `a6000_control` | union control: S3 short-safe + bounded repair + long suspicion | `787/1033` `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | A6000 baseline |
| `midcons` | union + conservative mid rescue off11..13, delta3..7, ratio>=0.8 | `795/1033` `76.96%` | `+8` (8W/0L) | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | A6000 best |
| `mid_precision` | mid rescue + support/best-len/short-jump guards | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | 负面证据 |
| `true_long` | true-long rescue off>=17, delta>=8, ratio>=0.85, support>=2 | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | 负面证据 |
| `combined` | mid precision + true-long rescue | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | 负面证据 |

## 受控对比

| 对比 | Wins | Losses | Net | 解释 |
|---|---:|---:|---:|---|
| `union_gpus01_control` vs `old_union_gpus23` | 4 | 6 | -2 | 跨服务器/torch 变化真实但较小，主要影响 <=8 和 9-12。 |
| `eval12_nomiddle_gpus01_control` vs `union_gpus01_control` | 0 | 0 | 0 | 只扩展 official evaluation 到 S3<=12，在不开启 mid rescue 时不改变输出。 |
| `midcons` vs `union_gpus01_control` | 7 | 1 | +6 | Conservative mid rescue 在同环境中确实有效。 |
| `midaggr` vs `union_gpus01_control` | 9 | 5 | +4 | Aggressive mid rescue 有更多 wins，但引入过多 short/mid false positives。 |

## 第一份详细 breakdown 的含义

Breakdown 按以下维度分解新结果：

- 相对 control run 的 pairwise wins/losses。
- 每个 win/loss 的 oracle-length bucket。
- 最终来源：`base`、`official_bounded_repair`、`official_long_suspicion`、`official_mid_rescue`、`strong_correction`、`weak_correction`。
- 长度转移，例如 `S3 6 -> official 13`。
- loss 是否直接由 `official_mid_rescue` 导致，还是由环境/run-stack drift 导致。

这很重要，因为早期 quick summary 把 `midcons` 直接与 `old_union_gpus23` 比较，混合了策略变化和服务器/PyTorch 变化。

## 为什么离线预期与 full run 不完全一致

Conservative mid rescue 的离线预期大约是相对 union `+8 / 0 loss`。完整受控结果相对 `union_gpus01_control` 是 `+7 / 1 loss`。

原因：

1. **环境 drift 存在。** 在 5090 stack 上复跑 exact old union 改变了 10 个任务：4 wins、6 losses，net -2。因此直接和旧 `gpus23` 结果比较会夸大或误归因部分损失。
2. **Gate-control 是无辜的。** 没有 mid rescue 时，`official_eval_max=12` 与 union control 的 pass/fail 集完全一致。所以更宽的 official probing window 本身不是结果变化来源。
3. **Mid rescue 对 short oracle cases 仍偏宽。** Conservative mid rescue 触发 24 次：18 pass、6 fail；造成 7 个 controlled wins 和 1 个 controlled loss。Aggressive mid rescue 触发 43 次：30 pass、13 fail，带来 5 个 controlled losses。False positives 多为 true-short 或 9-12 oracle tasks，被 official-CAL 过度扩展到 13-15。
4. **它不解决 true-long under-selection。** `17-24` 与 `25+` bucket pass rate 仍为 `20.73%` 和 `16.13%`。当前 mid rescue 主要改善 13-16 和部分 9-12，不是达到 80% 所需的 long tail。

## 设计诊断

当前 mid rescue 规则：

```text
S3 <= 12
official_len in 11..13 or 11..15
delta in 3..7 or 3..8
long_ratio >= 0.8
source == base
```

有效部分：

- 恢复了 S3 选 6-9、official-CAL 选 11-13 的 under-selected medium cases。
- Conservative 版本在同环境中把 13-16 bucket 从 `54.44%` 提高到 `58.89%`。
- 相对 gpus01 union control，它没有降低 aggregate <=8 bucket pass rate，尽管仍有一个 direct short loss 和一个 short win。

失败部分：

- `long_ratio >= 0.8` 选择性不够。Short false positives 的 ratios 也常在 `0.83..0.97`。
- Official selected length 约 13-15 不是可靠 true-long signal；它也可能是 oracle <=8 或 9-12 的 over-extension。
- Aggressive off11..15/delta3..8 虽改善 13-16，但损害 <=8 和 9-12，因此不能在“不牺牲 short”的准则下作为最佳 checkpoint。

## 下一步实验

优先级：

1. **Mid rescue precision pass。**
   保留 conservative range，但加 guards 去除 direct loss，同时尽量不丢失 wins：
   - 要求 `official_len <= 13`。
   - 要求 `delta <= 7`。
   - 增加 raw/curve agreement：只有 base long evidence 不平坦/不嘈杂时才 official rescue，例如 `best_long_len in {13,14,15,16}` 或 adjacent support count >= 2。
   - 当 base/S3 <= 5 且 official 跳到 13 时，加 short-protection veto，除非 long curve support 很强。

2. **True-long rescue branch，与 mid rescue 分离。**
   当前 long buckets 没变，达到 80% 需要不同分支：
   - 不允许该分支影响 <=8/9-12。
   - 只在 high official length、强 long curve support、明显 failure/under-selection signature 时触发。
   - 如果会牺牲 short，先作为 record-only 实验评估。

3. **同环境 checkpoint 策略。**
   新服务器策略决策以 `union_gpus01_control` 为 baseline。
   在受控环境对比中出现 total 改善且不回退 short buckets 之前，`old_union_gpus23` 保留为旧 global checkpoint。

## 文件

- Old union: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826`
- gpus01 union control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_174051`
- gpus01 eval12/no-mid control: `outputs_clean/full_lcal_official_bounded_repair_union_eval12_nomiddle_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_182208`
- midcons: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_gpus01_20260520_201658`
- midaggr: `outputs_clean/full_lcal_official_bounded_repair_union_midaggr_off11_15_d3_8_r08_gpus01_20260520_210248`
