# LR-DLLM Preregistration and Pure-Core Action Brief

日期：2026-07-31 UTC

状态：`preregistration_ready_for_commit_before_implementation`

标签：**paper-guided, author-unverified reimplementation of LR-DLLM**。

## Scope

本 action 只冻结作者代码搜索结论、歧义选择，并实现不加载模型/数据的纯算法核心和 synthetic unit tests。它不运行 GPU、不生成 benchmark outcome、不打开 frozen controller test，也不创建尚不可执行的 full manifest。

## Exact commands

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests.test_lrdllm_algorithm
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile experiments/lrdllm_algorithm.py tests/test_lrdllm_algorithm.py
git diff --check
```

GPU：none。Env：`dllm_env`。Input：synthetic fixtures only。Output：test stdout；不写 raw。Log：当前 Codex session。预算：<1 CPU minute。

Success gate：全部 preregistered pure functions和 ledger/resume guards测试通过；没有 torch/model/dataset/evaluator dependency；没有 frozen/reference/oracle/test字段入口。

Kill criteria：需要猜测尚未登记的算法语义；测试依赖正式 outcome；实现读取 benchmark 或 frozen 文件；diff 扩张到 DreamOn、M1--M5、PPT 或 ExecRepoBench final。

本 action 的 docs commit/push 必须先于 `experiments/lrdllm_algorithm.py` 的创建。完成 pure core 后，再单独冻结 DreamCoder 12/64 smoke manifests、GPU command、log/output、预算和 kill criteria。
