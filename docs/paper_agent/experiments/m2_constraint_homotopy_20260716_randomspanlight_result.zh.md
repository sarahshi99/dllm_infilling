# M2 Constraint-Homotopy V0 — RandomSpanLight 148-case result

状态：`completed_reviewed`。这是一项独立候选方法结果，不选择论文主方法，也不授权自动进入 296/927/5079。

## 完整性

- vanilla、gradual、abrupt 均为 `148/148`；三者各有 `148` unique candidate keys，row-key 集合完全一致。
- 每臂 missing/extra/duplicate/error 均为 `0`；所有 ok rows 为 canvas `64`、seed `0`、`64` forwards、`4096` token-forwards，末行可解析。
- full technical gate passed，run manifest 为 `completed`；frozen controller test 保持 `sealed`，`test_evaluation_count=0`。
- 运行中 peak CUDA allocation 为 `16,496,130,048` bytes（约 `15.36 GiB`）；full run wall time 为 `12,248.53s`。该 wall-clock 发生在共享 GPU 环境，仅是 contention measurement。

## 预注册 grouped 统计

Primary estimand 是 148 个 base tasks 的 equal-weight macro accuracy；span-micro 只作 descriptive（本数据每 task 一个 span，故数值相同）。所有 CI 为 fixed-seed `10,000` task-cluster bootstrap；p 为 group-aware paired label-swap。

| Method | passed/148 | task macro, 95% CI | descriptive micro | mean wall/case |
|---|---:|---:|---:|---:|
| vanilla fixed64 | 38 | 25.68% [18.92%, 33.11%] | 25.68% | 28.57s |
| gradual constraints | 40 | 27.03% [20.27%, 34.46%] | 27.03% | 25.75s |
| abrupt constraints | 35 | 23.65% [16.89%, 30.41%] | 23.65% | 28.38s |

| Comparison | macro delta, 95% CI | wins/losses/ties | help/harm | label-swap p |
|---|---:|---:|---:|---:|
| gradual − vanilla | +1.35pp [−4.05pp, +6.76pp] | 9/7/132 | 9/7 | 0.8113 |
| abrupt − vanilla | −2.03pp [−6.08pp, +2.03pp] | 3/6/139 | 3/6 | 0.5108 |
| gradual − abrupt | +3.38pp [0.00pp, +7.43pp] | 7/2/139 | 7/2 | 0.1783 |

长度分桶仅为 descriptive：gradual 在 extreme `34.92%` 对 vanilla `30.16%`，但在 long `13.04%` 对 vanilla `26.09%`；没有进行行级独立显著性或基于分桶的调参。

## Activation audit

仅写入 candidate hash、计数和动态聚合，不含 raw generated code。

- gradual vs vanilla：`105/148` full-code/middle hashes 不同。
- abrupt vs vanilla：`68/148` 不同。
- gradual vs abrupt：`78/148` 不同。
- 三臂 `effective_update_steps` 与 `total_token_changes` 均为每例 `64`，所以 schedule 确实改变候选输出，但这些两个 aggregate dynamics 本身不提供额外的差异方向证据。

## 科学判断

结论固定为：`constraints_activated_no_reliable_grouped_advantage`。Gradual 的点估计高于 vanilla，但 task-macro CI 跨零且 group-aware p=`0.8113`；abrupt 相对 vanilla 为负方向。故本 148-case 结果不支持“逐步引入约束”的可靠准确率优势，也不支持 abrupt schedule；它是机制被激活但方向尚未得到 grouped evidence 支持的结果。此判断不会阻止 M3，且不触发 M2 的自动规模扩展。

Formal artifacts: `analysis_outputs/m2_constraint_homotopy_20260716_grouped_v1/`.
