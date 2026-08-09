# Frontier-Gated DreamOn V0 结果报告

本报告只覆盖 Pilot-30 development/mechanism population 与 fixed-full-1000 development/validation population；不是官方 5815 full，也不是 frozen test。旧 DreamOn 逐行 V1/V2/V3 路线保持冻结，其负向/混合证据不删除。

## 实现与等价门

- 初始动态中间区是单一连续的 64 个 `[MASK]`；`max_new_tokens=64`。
- 唯一方法改动是 frontier 窗口内未解决 mask 的位置 eligibility；完整 prefix/middle/suffix 始终参加官方 DreamOn forward。
- 换行是普通 token；没有 line slot、separator、截断、重试、nonempty、compile gate、blacklist、BoundaryShift 或 repair。
- `w=∞` 与未经修改的官方 DreamOn 在 55 条真实样本上 final tokens/text、逐步位置/动作与停止原因完全一致，并覆盖 normal、换行后继续生成、expand、delete。

## Pilot-30

| w | Pass@1 | 编译 | 完成 | 平均 forwards | 平均生成耗时 | 晋级 |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 28/30 (93.33%) | 30/30 (100.00%) | 30/30 (100.00%) | 20.77 | 0.399s | 是 |
| 4 | 27/30 (90.00%) | 30/30 (100.00%) | 30/30 (100.00%) | 20.23 | 0.379s | 是 |
| 8 | 27/30 (90.00%) | 30/30 (100.00%) | 30/30 (100.00%) | 20.23 | 0.378s | 是 |
| 16 | 27/30 (90.00%) | 30/30 (100.00%) | 30/30 (100.00%) | 20.23 | 0.379s | 是 |
| inf | 27/30 (90.00%) | 30/30 (100.00%) | 30/30 (100.00%) | 20.23 | 0.379s | 是 |

自动晋级：w=1, w=4, w=8, w=16, w=inf。门槛严格固定为至少 16/30。

## Fixed-full-1000

| w | Pass@1 | 编译率 | 完成率 | 平均 forwards | 平均 wall | GPU time 合计 | token-forwards |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 555/1000 (55.50%) | 97.30% | 100.00% | 27.88 | 0.589s | 588.9s | 8389414 |
| 4 | 555/1000 (55.50%) | 97.50% | 100.00% | 28.07 | 0.595s | 594.5s | 8476428 |
| 8 | 554/1000 (55.40%) | 97.20% | 100.00% | 27.89 | 0.591s | 590.8s | 8409108 |
| 16 | 553/1000 (55.30%) | 97.20% | 100.00% | 27.86 | 0.590s | 590.5s | 8404894 |
| inf | 553/1000 (55.30%) | 97.20% | 100.00% | 28.04 | 0.595s | 595.0s | 8465708 |

### 配对比较（有限窗口 vs w=∞）

| w | help | harm | tie | Pass@1 差值 | base-task cluster bootstrap 95% CI | compile help/harm | broadcast 更多/更少/相同 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 22 | 954 | +0.20pp | [-1.12, +1.50]pp | 12/11 | 14/22/964 |
| 4 | 4 | 2 | 994 | +0.20pp | [-0.28, +0.73]pp | 5/2 | 5/6/989 |
| 8 | 1 | 0 | 999 | +0.10pp | [+0.00, +0.34]pp | 2/2 | 4/3/993 |
| 16 | 1 | 1 | 998 | +0.00pp | [-0.30, +0.29]pp | 2/2 | 4/2/994 |

### 动作、未完成与主要失败类别

- `w=1`：expand 0，delete action 959，single delete 8，broadcast delete 951（951 rows），未完成 0，cycle/oscillation rows 0；失败类别 {"passed": 555, "functional_failure_after_compile": 418, "compile_failure": 27}。
- `w=4`：expand 134，delete action 980，single delete 17，broadcast delete 963（950 rows），未完成 0，cycle/oscillation rows 2；失败类别 {"passed": 555, "functional_failure_after_compile": 420, "compile_failure": 25}。
- `w=8`：expand 134，delete action 986，single delete 18，broadcast delete 968（950 rows），未完成 0，cycle/oscillation rows 2；失败类别 {"passed": 554, "functional_failure_after_compile": 418, "compile_failure": 28}。
- `w=16`：expand 71，delete action 982，single delete 17，broadcast delete 965（951 rows），未完成 0，cycle/oscillation rows 1；失败类别 {"passed": 553, "functional_failure_after_compile": 419, "compile_failure": 28}。
- `w=inf`：expand 199，delete action 1002，single delete 18，broadcast delete 984（947 rows），未完成 0，cycle/oscillation rows 3；失败类别 {"passed": 553, "functional_failure_after_compile": 419, "compile_failure": 28}。

### 广播删除型提前终止检查

- `w=1` 相对 `w=∞`：broadcast delete 次数更多的样本 14，更少 22，相同 964；其中“更多 broadcast 且由 baseline pass 变为 fail” 1。
- `w=4` 相对 `w=∞`：broadcast delete 次数更多的样本 5，更少 6，相同 989；其中“更多 broadcast 且由 baseline pass 变为 fail” 0。
- `w=8` 相对 `w=∞`：broadcast delete 次数更多的样本 4，更少 3，相同 993；其中“更多 broadcast 且由 baseline pass 变为 fail” 0。
- `w=16` 相对 `w=∞`：broadcast delete 次数更多的样本 4，更少 2，相同 994；其中“更多 broadcast 且由 baseline pass 变为 fail” 0。
