# Long Underestimate Detector Sweep

该表记录对 A6000 `midcons` 结果进行的离线 long-underestimation 规则 sweep。所有规则都只用于诊断，不改变生成结果。

| 规则 | 触发数 | True-long precision | Failed-long recall | Short risk | Current-pass risk |
|---|---:|---:|---:|---:|---:|
| `sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.4|supp>=0|src=base` | 93 | 35.48% | 36.26% | 40.86% | 27.96% |
| `sel<=3|best>=13|gap>=4|ratio>=0.55|raw>=0.5|supp>=0|src=base` | 32 | 37.50% | 13.19% | 37.50% | 18.75% |
| `sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.5|supp>=0|src=base` | 33 | 36.36% | 13.19% | 36.36% | 18.18% |
| `sel<=3|best>=13|gap>=4|ratio>=0.55|raw>=0.4|supp>=0|src=base` | 33 | 36.36% | 13.19% | 39.39% | 21.21% |

完整规则表见英文版 `sweep.md`。核心结论是：没有规则同时达到足够高的 true-long precision、足够低的 short risk 和有意义的 failed-long recall。因此，不应基于这些现有字段继续发起同类 true-long GPU 实验。
