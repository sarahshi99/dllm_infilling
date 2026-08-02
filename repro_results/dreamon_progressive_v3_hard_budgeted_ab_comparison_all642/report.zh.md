# DreamOn V3 A/B Full642 对比

生成日期：2026-08-02（UTC）

## 实验身份

- Population：642 rows / 115 base problems，exact-three-line development/mechanism population，不是 held-out test。
- A：`v3_hard_budgeted`。A Pilot30 为 14/30，未通过原始 18/30 Full 门槛；本次 Full 是用户后续明确授权的 post-hoc 机制诊断，不能改写成预注册性能证据。
- B：`v3_hard_budgeted_nonempty_oracle`。只在 A 上增加 hard-slot nonempty guard；属于 oracle structural diagnostic，不可部署。
- 两个 Full 都是 642/642 completed，0 runtime/protocol error，task 顺序与 manifest 一致。

## 主结果

| 方法 | Pass@1 | Compile | Exact | Task-macro Pass@1 |
|---|---:|---:|---:|---:|
| One-shot | 301/642 (46.88%) | 638/642 (99.38%) | 83/642 (12.93%) | 43.95% |
| 历史 V1 | 316/642 (49.22%) | 585/642 (91.12%) | 122/642 (19.00%) | 46.46% |
| A: V3-Budgeted | 281/642 (43.77%) | 538/642 (83.80%) | 97/642 (15.11%) | 41.17% |
| B: Budgeted+Nonempty | 280/642 (43.61%) | 471/642 (73.36%) | 106/642 (16.51%) | 42.46% |

## B 对 A 的配对结果

- Pass：B wins 15 / losses 16 / both pass 265 / both fail 346，净变化 -1 条（-0.16%）。
- Exact McNemar p=`1`；row bootstrap 95% CI=[-1.87%, 1.56%]；115-base-problem cluster bootstrap 95% CI=[-2.60%, 2.09%]。
- Task-macro：A 41.17%，B 42.46%，差值 1.29%。
- Compile：B help 6 / harm 73；总体从 538/642 降到 471/642。
- Exact match：B help 9 / harm 0；总体从 97/642 升到 106/642。

## 机制隔离

- A 有 137 个 blank-slot rows；B guard 恰好在这 137 条上触发，B blank-slot rows 为 0。
- 其余 505 条未触发 guard 的样本，在 completion、score、动作计数、budget、forwards 和 token-forwards 上全部与 A 一致；隔离审计差异行数为 0。
- Guard 共拒绝 9009 个候选，但没有产生 `nonempty_guard_no_valid_action` terminal failure。
- 非空约束消除了空槽位并增加 exact match，但没有提高整体 Pass@1；同时造成明显 compile harm。这说明“reference 为三条非空行”这一 oracle 结构先验并不是稳定的功能正确性改进。

## 计算对比

| 指标 | A | B | B-A |
|---|---:|---:|---:|
| Forwards | 35,738 | 44,067 | +8,329 |
| Token-forwards | 9,126,663 | 11,412,369 | +2,285,706 |
| Generation wall time | 2702.23s | 4061.30s | +1359.07s |
| Peak CUDA memory | 15,819,467,264 | 15,819,467,264 | +0 |
| Budget exhausted rows | 138 | 217 | +79 |

B 相对 A 使用 23.31% 更多 forwards、25.04% 更多 token-forwards。墙钟时间还受到同期其他 GPU 作业影响，因此机制计算比较优先采用 forwards/token-forwards。

## 结论

本次全量结果不支持 B 的 nonempty guard 能提高功能正确率：B 比 A 少 1 个 Pass，配对统计和 cluster bootstrap 均不支持稳定收益；其代价是 67 个 compile 净损失和约 25% 更多 token-forwards。B 的 9 个 exact-match 净收益说明它更贴近 oracle 三行表面结构，但这没有转化成功能收益。当前证据更支持保留 A 作为协议忠实性控制，并把 nonempty guard 视为负向/混合机制结果，而不是可推进的方法。
