# LR-DLLM Stage I-only 本地适配实现与 technical smoke 报告

日期：2026-07-31 UTC

状态：`reviewer_hardened_cpu_preflight_verified_no_gpu_execution`

准确标签：**local LR-DLLM Stage-I-only adaptation from arXiv:2602.07546v1 Algorithm 1**

运行标签：**Stage-I-only selector + fixed-canvas decode**

## 1. 执行摘要

本轮已实际完成核心实现、独立 adapter、launcher、reviewer-hardening tests、CLI/compile/diff 检查、smoke/full immutable manifest preflight、真实 pinned decoder source 受控审计和完整 model artifact provenance。没有实现 Stage II，没有修改 CAL-lite、official CAL `ARMS`、M1--M4、pinned upstream source 或 official CAL raw outputs。

GPU technical smoke 未执行。2026-07-31 07:25 UTC 只读审计发现物理 GPU 0 已有三个外部计算进程：

| PID | 命令摘要 | GPU memory |
|---:|---|---:|
| 755980 | `python -u src/detection/orthrus_gnn_fl.py CADETS_E3 --1_hop` | 17,136 MiB |
| 810890 | `python -u src/detection/orthrus_gnn_fl.py CADETS_E3 --1_hop` | 17,136 MiB |
| 818373 | `python temp_FedMend.py --dataset lanl --encoder GCN --rnn GRU --seed 42` | 70,720 MiB |

合计约 104,992 MiB。ECC volatile/aggregate correctable 与 uncorrectable 均为 0。按单 H200 安全规则，本轮未抢占、未结束或干扰这些进程，smoke 精确标记为 `blocked_by_existing_gpu_process`。

## 2. Git 基线与隔离

- base branch：`origin/codex/ccfa-execution-sprint-v1`
- original base HEAD：`48661dcc52ee7b2c840f5a2dde9eb9b35f8af078`
- rebased authoritative base HEAD：`619515789d9297903242fe1412d6123eeb1e1ec8`
- branch：`codex/lrdllm-stage1-adapter-v1`
- clean worktree：`/home/shx/projects/dllm_infilling/git_workspace/.worktrees/lrdllm-stage1-adapter-v1`
- 开始时 `git status --short`：空
- `current_action.md` 顶部仍为未完成的 `CAL-PHASE1B-SINGLELINE-838`，本轮没有覆盖。

## 3. 实现文件与边界

新增：

- `expvision_dllm_clean/lrdllm_stage1.py`
- `experiments/p1_lrdllm_stage1_adapter.py`
- `experiments/p1_lrdllm_stage1_audit.py`
- `tests/test_lrdllm_stage1.py`
- `tests/test_p1_lrdllm_stage1_adapter.py`
- `scripts/manual_launch_lrdllm_stage1_singleline_20260731.sh`
- `docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md`
- 本报告

更新：

- `docs/paper_agent/lrdllm_protocol_audit.zh.md`
- `docs/paper_agent/baseline_population_and_protocol_matrix.current.zh.md`
- `docs/paper_agent/baseline_population_and_protocol_matrix.current.json`

独立 arm 为 `local_lrdllm_stage1_fixed_decode`。它没有加入 `experiments/p1_official_cal_adapter.py::ARMS`，不会混淆 official CAL 身份。

## 4. Stage I 数学与数值实现

Probe 长度严格为 `2^m <= MAX_LENGTH`。primary `MAX_LENGTH=128`：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

每个长度构造 `prefix + L masks + suffix`，恰好一次 forward。只对 mask span 的 logits 计算 float32 负熵：

```text
log_probs = log_softmax(mask_logits.float())
probs = exp(log_probs)
conf_i = sum_v probs_i(v) * log_probs_i(v)
A(L) = mean_i conf_i
```

拟合使用全部八个指数点、自然对数和带截距 OLS：

```text
x_j = log(L_j)
y_j = A(L_j)
k_hat = sum((x_j-x_mean)(y_j-y_mean)) / sum((x_j-x_mean)^2)
intercept = y_mean - k_hat*x_mean
CL(L) = A(L) - k_hat*log(L)
```

选择 `CL` 最大长度；exact tie 选更短长度。`k_hat` 不 clamp、不取绝对值，没有 clipping、ridge、robust regression、upper-half 重拟合或全局 demo bias。所有 raw entropy、log length、OLS 参数、R² 和 `CL` 必须 finite。

selector 只接收由 `prompt/suffix` 构成的安全 task view；不读取 reference、canonical solution、oracle length、passed、task ID 模式、tests 或 evaluator。模型为 eval，运行于 `torch.no_grad()`。

每行计划保存论文身份、probe grid、forward/token counts、全部 entropy/log/CL maps、`k_hat`、intercept、R²、raw entropy argmax、selected length、tie-break、fit scope 和 finite gate。

## 5. Fixed decode 协议与证据

pinned decoder：CAL commit `741e8418a88a732b4c92812424d4f03cab1f7b1f` 的 `llada_cal.generate`。

本地固定调用模板：

```text
steps=None
gen_length=selected_length
block_length=None
temperature=0.0
cfg_scale=0.0
span=1
max_gen_length=128
dstep=-1
use_bias=False
```

证据：

1. `dstep=-1` 使 pinned upstream 完全跳过 CAL length search，必须返回 `search_steps=0`。
2. upstream 对 `steps=None`、`block_length=None` 分别解析为 `curr_steps=L`、`curr_block_length=L`。
3. mock argument-capture 测试证明 `L=32` 时上述显式参数与 existing `official_fixed32` 逐项相等。
4. 对其他 `L`，调用模板唯一改变的是 `gen_length`；steps/block length 仍由同一 pinned upstream 规则解析为 `L`。
5. formal output middle token count必须等于 Stage I selected length，不允许 post-selection expansion/contraction。

真实源码审计进一步固定 `llada_cal.py` SHA256=`b1d684040334ea1ffeacd92279cfa98e007ec5c9ce1a39840054e98420989225`。CPU controlled execution 在 `L=4` 下观察到 resolved steps/block length=`4/4`、formal forwards=`4`、search forwards=`0`、output canvas=`4`。任一 hash/语义变化均产生 `fixed_decode_protocol_ambiguity` 并禁止 GPU。

fixed canvas 内仍使用 upstream 标准 LLaDA iterative denoising；没有 LR-DLLM Stage II 的逐 token commit 或动态长度搜索。本轮 temperature/cfg 跟 existing official fixed decoder 保持一致，不采用论文完整 Stage II 的 `temperature=0.2/top_p=0.9`。

## 6. Manifest、frozen 与数据访问

CPU preflight 通过：

- smoke12：12 rows，SHA256 `56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`
- full：838 rows / 143 clusters，SHA256 `52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`
- SingleLine dataset SHA256：`1277073e7ddce163ef68598c2bca777f688a53df99422f097e37362090f88e1f`
- evaluator commit：`88062ff9859c875d04db115b698ed4b0f0395170`
- frozen controller test：`sealed`，`test_evaluation_count=0`
- certificate：`sealed_files_opened=false`

full manifest 仅 preflight，未执行。dataset preflight 只做 opaque SHA256；实际 adapter loader 只对 manifest 指定的 source row 执行 JSON parse，其余 gzip lines 保持 opaque，不把完整 1,033-row dataset 解析进内存。

Manifest schema 已改为 exact allowlist；任何未登记字段都会 fail-stop。

## 7. Model/evaluator provenance hardening

- resolved model revision：`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- model artifact-set SHA256：`923e056fe742858e8b817406ae942feccb53ace80116187070de81b29901719a`
- config SHA256：`5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a`
- weight-index SHA256：`28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408`
- tokenizer config SHA256：`6e9f41633217287fcf9a58890efb26e91e905bd6ae2234b534b65e0c36f4dd3c`
- tokenizer SHA256：`ee1ef8e5f6d9493ac25480b7b7337ff5d2c1b946190afff18d12c86ca738ae00`
- evaluator source/enabled/driver SHA256：`b96a5b82...` / `06c25200...` / `5c2ec640...`
- environment：Python `3.10.20`、torch `2.5.1+cu121`、transformers `4.38.2`、torch CUDA `12.1`、mask token id `126336`

六个 weight shards 和全部 remote-code/tokenizer artifacts 的逐文件 SHA256 写入 CPU preflight，并计划写入 run manifest和每条 raw。任一 mismatch 在 model forward 前 fail-stop。

## 8. 测试与静态检查

执行：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest \
  tests/test_lrdllm_stage1.py \
  tests/test_p1_lrdllm_stage1_adapter.py \
  tests/test_lrdllm_algorithm.py \
  tests/test_lrdllm_dreamcoder_adapter.py \
  tests/test_build_lrdllm_common_manifests.py \
  tests/test_p1_official_cal_adapter.py \
  tests/test_p1_official_cal_source_audit.py \
  tests/test_build_cal_singleline_rest_manifest.py \
  tests/test_baseline_grouped_analyzer.py
```

Reviewer hardening 新增覆盖：full-gate env bypass、真实 source semantic audit、source/model hash mismatch、manifest extra field、wrong probe grid、selected/generated mismatch、wrong source/model/evaluator provenance、existing failure journal、probe-only no completion/no evaluator，以及 unified analyzer 的 length-selection 命名。

结果：`Ran 72 tests ... OK`。相对 rebase 后原测试集合新增 9 个 test methods；其中 corrupted-row test 另含 20 个逐项 corruption subcases。

以下也通过：

- `python -m py_compile expvision_dllm_clean/lrdllm_stage1.py experiments/p1_lrdllm_stage1_audit.py experiments/p1_lrdllm_stage1_adapter.py`
- adapter `--help`
- launcher `--help`
- launcher `bash -n`
- CPU `preflight`
- `git diff --check`

## 9. 成本账本状态

代码与测试强制：

```text
total_forwards = stage1_probe_forwards + formal_decode_forwards
total_token_forwards = stage1_probe_token_forwards + formal_decode_token_forwards
```

`MAX_LENGTH=128` 每行必须 `stage1_probe_forwards=length_selection_forwards=8`，并要求 selector 自报值与透明 proxy 实际值一致。`search_forwards` 仅为 analyzer 兼容别名，不表示 CAL search。formal decoder 的 upstream search forward 必须为 0。

由于 GPU smoke 未执行，目前没有真实 row ledger、aggregate forward/token totals、wall time 或 peak memory；不得伪造。单元测试中的 ledger 守恒全部通过。

## 10. Probe-only / smoke / full 完整性状态

| Gate | 状态 |
|---|---|
| protocol freeze | pass |
| CPU tests / compile / CLI / diff | pass |
| fixed32 protocol ambiguity | none；mock equivalence pass |
| smoke manifest hash/rows | pass，12 rows |
| full manifest preflight | pass，838/143 |
| ECC | pass，0 errors |
| output path conflict | pass，目标 raw/analysis/log 路径均不存在 |
| existing GPU process | **blocked**，3 processes |
| technical smoke execution | **not executed** |
| probe-only execution | **not executed，0/1** |
| evaluator count from this arm | 0 |
| selected-length distribution | unavailable；无 raw rows |
| per-row probe forwards | unavailable empirically；code gate固定为 8 |
| failure journal | not created；run未启动 |
| resume no-op | not executed；run未启动 |

没有报告 smoke accuracy，因为没有生成或 evaluator outcome。

## 11. Full gate 与公平比较

- 直接设置 `ALLOW_LRDLLM_STAGE1_FULL=1` 不能绕过 adapter 内部 strict gate。
- full 必须读取同 commit 的 probe-only audit、12-row smoke final audit、resume-noop progress和完整 provenance。
- 任一已有 failure journal 都要求新 output version；旧 journal不删除、不覆盖。
- 本 arm 不是论文完整 LR-DLLM，不得与论文主表数字直接比较。
- 838 scientific run必须先具备同 838 keys、同 decoder 的 `official_fixed32` control。
- official CAL 是独立 arm；Fixed64 不得称 equal-compute。

本轮未运行 probe-only、smoke、838 full、official_fixed32、official CAL 或 Fixed64。

## 12. 论文未完全明确之处

1. “long-span regime” 没有阈值；primary 按 Algorithm 1 使用所有指数点。
2. 论文没有 argmax tie-break；本地 exact tie 选更短。
3. 论文未定义小于两个 probes 的退化 OLS；本地 fail-stop。
4. 没有 audited author code，无法确认作者实现的具体点集、精度、tie 或 decoder glue。
5. 论文完整 LR-DLLM 使用 Stage II；本轮 fixed-canvas decode 是项目共同协议下的本地适配，不等价于论文最终方法。

## 13. 状态区别与后续命令

- official full LR-DLLM：`still blocked / no audited author code`
- local Stage-I-only Algorithm-1 adaptation：`implemented and CPU-verified; smoke blocked_by_existing_gpu_process`
- Stage II in this local arm：`not implemented`
- 838 full：`not started`

下一步只能先运行 probe-only sanity：

```bash
bash scripts/manual_launch_lrdllm_stage1_singleline_20260731.sh probe-only
```
