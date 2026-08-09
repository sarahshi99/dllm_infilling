# Codex Handoff：Frontier-Gated DreamOn V0

日期：2026-08-09 UTC

## Git

- 起始 HEAD：`3eac683fcfdb2d8936e3d88416848b553ed4f69a`
- 分支：`codex/frontier-gated-dreamon-64`
- 阶段 commits：`06dc6fa`（实现/manifest/equivalence）、`c083e01`（Pilot-30）；最终结果/文档 commit 见该分支最新 HEAD。
- 远端：`origin/codex/frontier-gated-dreamon-64`

## 方法与门禁

- 一个连续 64-mask dynamic middle；`max_new_tokens=64`，一轮提交 1 token。
- 唯一方法变化是 frontier 位置 eligibility；完整 prefix/middle/suffix 始终参加官方 forward。
- newline 是普通 token；无 slot、separator、截断、重试、nonempty、compile gate、blacklist、BoundaryShift 或 repair。
- `w=infinity` 对 55 条真实样本与官方 native 完全等价，覆盖 normal/newline-after-continue/expand/delete。

## Population

- Pilot-30 manifest SHA256：`99ab1af2cd7cd68ecd848864cf76731340f47402c4afb8d4e753ccc6ad3163b9`。
- Fixed-full-1000 manifest SHA256：`5ca4fc25ac5ca5dbb9623e25f93a440ac883ff76c15ed019effeb17d597cec2d`。
- 两者均是 development population；fixed 1000 不是官方 5815 full，也不是 frozen test。

## 结果

- Pilot Pass：`w=1 28/30`；`w=4,8,16,infinity 27/30`。全部达到 16/30 并自动晋级。
- Fixed Pass：`555,555,554,553,553 / 1000`。
- Fixed compile：`973,975,972,972,972 / 1000`。
- 所有窗口 1000/1000 completed；0 exception；0 frontier violation。
- 配对 help/harm vs infinity：`w=1 24/22`，`w=4 4/2`，`w=8 1/0`，`w=16 1/1`。
- 结论：near-tie/mixed；有限 frontier 没有稳健优越性，不能隐藏 harm 或只报平均值。

## 关键文件

- `experiments/frontier_gated_dreamon/report.zh.md`
- `experiments/frontier_gated_dreamon/fixed_full_1000_summary.json`
- `experiments/frontier_gated_dreamon/review_manifest.latest.json`
- `experiments/frontier_gated_dreamon/results/fixed_full_1000_results.index.json`
- `experiments/frontier_gated_dreamon/run_manifest.json`

## 恢复/复核命令

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m pytest -q tests/test_frontier_gated_dreamon.py
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.analyze
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.audit
```

不要自动运行 5815、增加窗口、调参或组合旧机制。任何新验证需重新预注册。
