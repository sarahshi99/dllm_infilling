# 代码审计与重构计划

## 上传项目中发现的问题

### 1. Baseline 与历史分支相互缠绕

- `decode_structured.py` 有 1688 行，混合了多代 repair、verifier 和 remask 逻辑。
- `config.py` 在同一个共享 dataclass 中承载 P0、P1、G、R/M、S 系列开关。
- runner 层在多个脚本中重复初始化逻辑，增加后续回滚和实验卫生管理难度。

### 2. 当前 baseline 链条隔离不足

- `decode_vanilla.py` 仍然混合核心 decode、细粒度重建诊断和完整 trace logging。
- step-level logging 捕获大量文本 payload 和重复诊断，成本高且容易被改坏。

### 3. Snapshot/selector 分析代码已经长成独立子系统

- `snapshot_pipeline.py` 与 `snapshot_selector.py` 合计超过 1400 行。
- 这些代码对离线分析有用，但不应继续留在下一阶段 critical path 上。

### 4. 项目需要冻结 legacy 区和干净实验区

- 旧代码应保留，用于复现实验历史。
- 新的 phase-0/phase-1 实验应放到新目录，并只共享最小核心。

## 已创建内容

新增了 `expvision_dllm_clean/` 干净 package，以及 `clean_scripts/` 下的专用 clean scripts。

### 新 clean package

- `expvision_dllm_clean/config.py`
- `expvision_dllm_clean/dataset.py`
- `expvision_dllm_clean/modeling.py`
- `expvision_dllm_clean/verifier.py`
- `expvision_dllm_clean/decode.py`
- `expvision_dllm_clean/evaluation.py`
- `expvision_dllm_clean/logging.py`
- `expvision_dllm_clean/runner.py`

### 新 clean scripts

- `clean_scripts/run_vanilla_fixed.py`
- `clean_scripts/run_vanilla_oracle.py`

## 设计决策

### A. Legacy 代码保持不动

原始 `expvision_dllm/` 和 `scripts/` 目录保留在原位，现在作为冻结 legacy 区，而不是新工作的推荐基础。

### B. Clean package 只支持当前下一步

新 package 有意只覆盖：

- 数据集加载；
- fixed-length vanilla decode；
- oracle-length vanilla 诊断；
- 共享 verifier、logging 和 evaluation 工具。

这样做是刻意的：它阻止旧 G/R/M/S 逻辑泄漏到下一轮 2x2 实验基础中。

### C. Mask length selection 成为显式接口

Clean package 只有一个显式开关：`mask_length_source = fixed | oracle`。
这是 phase 0 的正确抽象边界。后续 CAL-lite 可以作为第三个值加入，而不会污染 vanilla 链条。

## 推荐下一步编码工作

1. 运行新的 clean fixed-length baseline，并在小 smoke subset 上确认能复现已有 P0 结果。
2. 运行新的 oracle-length 诊断，并与 clean fixed baseline 做样本级 win/loss/tie 对比。
3. 只有在上述步骤完成后，再新增小型 `length_probe.py` 模块实现 CAL-lite。
4. 只有 CAL-lite 稳定后，再单独增加 stopping 模块。

## 现在不要做的事

- 不要把 CAL-lite 直接加进 legacy `decode_vanilla.py`。
- 暂时不要把 G/R/M/S 逻辑迁移进新的 clean package。
- 在 clean 2x2 baseline 建立前，不要迁移到 DreamOn、DAEDAL 或 token-level stopping。
