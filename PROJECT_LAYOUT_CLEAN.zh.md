# 干净项目布局建议

## 保留两个区域

```text
project/
  expvision_dllm/           # legacy experimental code, frozen for reference
  scripts/                  # legacy runners, frozen for reference

  expvision_dllm_clean/     # clean package for the new phase
  clean_scripts/            # dedicated clean runners

  outputs/                  # legacy outputs
  outputs_clean/            # new clean outputs
```

## 为什么采用这个布局

下一阶段只需要一个最小 baseline core 和显式 length-control 实验。
把 clean package 与 legacy package 分离后，后续 CAL-lite 和 stopping 实验可以独立加入，不会把历史分支重新拖回 critical path。

## 立即可用的实验入口

- `python clean_scripts/run_vanilla_fixed.py ...`
- `python clean_scripts/run_vanilla_oracle.py ...`

## 计划新增文件

Phase 0 完成后，下一批新增文件应为：

- `expvision_dllm_clean/length_probe.py`
- `clean_scripts/run_cal_lite.py`
- `expvision_dllm_clean/stopping.py`
- `clean_scripts/run_cal_lite_stop.py`

核心规则是：每个新实验都应新增模块和 runner，而不是重新打开 legacy monolith。
