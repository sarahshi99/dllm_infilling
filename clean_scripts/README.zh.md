# clean_scripts 索引

该目录包含用于 length-selection 和 stopping 实验的小型入口脚本。
部分文件名具有历史沿革，因此以本文件作为当前地图。

## 当前主线

| 脚本 | 用途 | 状态 |
| --- | --- | --- |
| `run_cal_lite_lcas_v3.py` | 基础 CAL-lite 长度选择，加 LCAS-v3 stopping；没有 long-aware correction。 | 稳定 baseline。 |
| `run_cal_lite_lcal_v3.py` | LCAL-v3：先做基础 CAL-lite 选择，仅当 `base_selected_length >= 13` 时再做 long-aware correction。 | 稳定 LCAL-v3 baseline。 |
| `run_cal_lite_lcal_v3_t2_ratio.py` | LCAL-v3 实验性 T2：加入 `best_long_score / best_score` 的 ratio trigger 分支。 | 实验性；适合分析，不建议作为新默认。 |

## 较旧或辅助脚本

| 脚本 | 用途 | 状态 |
| --- | --- | --- |
| `run_cal_lite.py` | 朴素 CAL-lite 入口。 | Legacy/base utility。 |
| `run_cal_lite_lcas.py` | 早期 LCAS 入口。 | Legacy。 |
| `run_cal_lite_lcas_b.py` | 早期 LCAS-B 入口。 | Legacy。 |
| `run_cal_lite_stop.py` | 带 stopping 控制的 CAL-lite。 | 辅助。 |
| `run_fixed_stop.py` | 带 stopping 控制的 fixed-length decode。 | 辅助。 |
| `run_vanilla_fixed.py` | Vanilla fixed-length decode。 | Baseline 工具。 |
| `run_vanilla_oracle.py` | Vanilla oracle-length decode。 | 诊断 baseline。 |
| `run_lcas_v3_model_sweep.py` | LCAS-v3 模型 sweep 入口。 | 辅助。 |
| `analyze_length_failure.py` | 失败分析工具。 | 分析工具。 |

## 术语

- CAL-lite：从 probe grid 中选择一个 mask length。
- LCAS：length-conditional adaptive stopping；在长度确定后控制何时停止 decoding。
- LCAL：建立在 CAL-lite 上的 long-aware correction；只对看起来偏长的样本尝试更强 long grid。
- T2 ratio branch：额外的 LCAL 触发条件。除 `base_selected_length >= 13` 外，如果最佳 long candidate 分数足够接近全局最佳分数，也会触发。

## 当前建议

为保证可复现比较，保留旧文件名，用 experiment name 编码变体。
不要把 `run_cal_lite_lcal_v3_t2_ratio.py` 设为默认：它改善了一些中长样本，但会引入短样本损失。
