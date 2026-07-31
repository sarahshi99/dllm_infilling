# External Baseline Population and Protocol Matrix

更新时间：2026-07-31 UTC

权威机器可读版本：`docs/paper_agent/baseline_population_and_protocol_matrix.current.json`。

## 本轮边界

- 只补全和规范化 external baselines；不设计 M5，不恢复/修改 M1--M4，不做 PPT，不运行 ExecRepoBench final benchmark。
- frozen controller test 保持 `sealed`，`test_evaluation_count=0`；Phase 0 未打开两份 sealed 文件。
- 当前服务器只有一张 `NVIDIA H200 NVL`，真实 device index=`0`。后续所有 GPU 命令使用 `CUDA_VISIBLE_DEVICES=0`，最多一个项目 GPU 进程，覆盖旧 GPU 2/3 约定。

## 三层报告口径

1. `literature-reported numbers`：只引用论文，不能与本地结果直接算 delta。
2. `official-source/paper-protocol reproduction on project non-frozen subset`：使用官方代码或论文协议，但 population 是项目非冻结子集；不能称完整 paper-number reproduction。
3. `common-protocol local comparison`：只有 backbone、immutable manifest、evaluator、sampling、seed、steps/decoder 和 normalized candidate keys 全部相同才允许 paired delta/CI。

Training-free 与 training-based DreamOn 分栏。Fixed64 是标准 fixed-length baseline，但不是 CAL、DreamOn、LR-DLLM 的 equal-compute control；只有 forward/token budget 实际匹配时才可写 compute-matched。

## Population ledger

| Dataset | Official full rows | Project non-frozen rows | Frozen rows | Non-frozen clusters |
|---|---:|---:|---:|---:|
| SingleLine | 1033 | 927 | 106 | 148 |
| MultiLine | 5815 | 5079 | 736 | 148 |
| RandomSpan | 1640 | 1480 | 160 | 148 |
| RandomSpanLight | 164 | 148 | 16 | 148 |

CAL seed-42：SingleLine=`1033/100 Demo/933 Rest`，CAL-Rest∩non-frozen=`838 rows/143 clusters`；MultiLine=`5815/100 demo-linked exclusion/5715 Rest`，CAL-Rest∩non-frozen=`4990 rows/143 clusters`。禁止生成 6707 或 7634 合并准确率。

## 术语归一化（覆盖旧文档标签，不改历史 raw/path）

| 历史标签 | 当前准确标签 |
|---|---|
| CAL-lite | local uncalibrated short-range selector |
| Dream-Coder primary | DreamCoder + local uncalibrated short-range selector |
| Dream-Coder simple | DreamCoder + local bounded-repair selector |
| offline best/oracle | reference-length diagnostic |

上述本地方法都不是 Fixed64、official CAL、DreamOn 或 LR-DLLM。历史 raw 文件名、JSON keys 和 output directory 保持不变，仅在 current docs/report labels 中规范化。

## 当前执行矩阵摘要

| Baseline / label | Backbone | Dataset/population | Protocol class | Status | Paired comparison |
|---|---|---|---|---|---|
| official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset | LLaDA-8B-Base | MultiLine 4990/143 | official/direct + documented evaluator overlay | analyzed：row `32.9259%`；task-macro `27.8471%`，95% CI `[24.3313,31.2715]%` | 当前无 identical-key official_fixed32，只报绝对表现/成本 |
| official_fixed32 | LLaDA-8B-Base | MultiLine 4990/143 | official/direct | not run | 完成后可与 CAL 配对 |
| official-source CAL initial32 | LLaDA-8B-Base | SingleLine CAL-Rest 838/143 | official-source SingleLine loader adaptation | immutable manifest frozen；smoke pending | 中心控制是 official_fixed32 |
| Fixed4/8/16/64 sensitivity | LLaDA-8B-Base | SingleLine CAL-Rest 838/143 | official-source adaptation | adapter extension not frozen | Fixed64 不自动等于 equal-compute |
| DreamCoder CAL | DreamCoder Base | SingleLine/MultiLine | official-source audit pending | blocker：同一 100-demo bias/CAL algorithm 未证明 | 不可配对 |
| DreamOn official-source single-H200 reproduction | DreamOn-v0-7B@8ccc7475 | SingleLine 927/148 | official/direct source route | 五个 released-source arms 完成并分析；row=`88.46..91.69%`，macro=`78.11..85.13%` | min4/8/16/32/64 deltas=`+58.464/+33.243/+13.943/+19.060/+21.211pp`；完整系统差异 |
| DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder | DreamCoder Base@2346ccd3 | SingleLine 927/148 | common-protocol local controls；min=max，EOS contraction保留 | 五个smoke/resume/full全部完成并配对分析 | same-manifest；不是equal-compute |
| DreamOn official-source MultiLine reproduction via benchmark-only adapter | DreamOn-v0-7B@8ccc7475 | MultiLine 5079/148 | 同一pinned generator；仅dataset/manifest/evaluator-row namespace变化 | 五个smoke/resume gates通过；min4 full运行，min8/16/32/64资源队列 | DreamCoder matched controls未完成前只报绝对指标 |
| LR-DLLM | DreamCoder Base | SingleLine 927；RandomSpan 1480；MultiLine 5079 | paper-guided, author-unverified reimplementation | SingleLine paired完成；RandomSpan primary−Fixed64=`-15.135pp` CI完全低于0；MultiLine Fixed64运行中 | 仅同dataset/same-key配对；不合并Mean |
| LR-DLLM | LLaDA-8B-Base | 同上 | DreamCoder technical credibility 后移植 | blocked by predecessor | 不可配对 |
| CAL authors’ DAEDAL FIM adaptation | LLaDA-8B-Base | SingleLine 838；MultiLine 4990 | pinned CAL `daedal_cal` source path | source present，adapter audit pending | decoder/fixed control 冻结后才可配对 |
| DreamCoder + local uncalibrated short-range selector | DreamCoder Base | SingleLine 927 | historical local diagnostic | complete | 仅限历史三臂内部，不对 official baseline 直接算 delta |
| DreamCoder + local bounded-repair selector | DreamCoder Base | SingleLine 927 | historical local diagnostic | complete | 同上 |
| DreamCoder reference-length diagnostic | DreamCoder Base | SingleLine 927 | offline diagnostic | complete | 不可部署、不可作 baseline delta |
| rho-EOS | LLaDA family | FIM datasets | unsupported | completion-only，无 faithful FIM/evaluator route | 不运行 |
| FlexMDM | 未固定 | FIM datasets | unsupported | 无 audited official FIM route | 不运行 |

## Immutable manifests

- MultiLine CAL-Rest common：`analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl`，SHA256=`e805489111e788e315bcf14838b077b09d0c23c44900ad8220d9e9afcbb95c48`。
- SingleLine CAL-Rest non-frozen：`analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl`，`838` rows/`143` clusters，SHA256=`52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`。
- SingleLine CAL smoke12：`analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl`，SHA256=`56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`。

Unsupported/blocked 组合不生成伪 manifest。DreamOn、DreamCoder CAL、LR-DLLM、DAEDAL 的新 manifest 只有在 source/checkpoint/decoder/evaluator 可执行性和协议边界完成冻结后创建。

## CAL MultiLine 4,990 解盲结果

- row Pass@1=`1643/4990=32.9259%`
- equal-weight 143-cluster task-macro=`27.8471%`
- 10,000 cluster-bootstrap 95% CI=`[24.3313%,31.2715%]`
- search/decode/total forwards=`76,807/156,841/233,648`
- token-forwards=`63,545,173`；mean wall=`3.8762 s/row`；peak memory=`15.42 GiB`
- no official_fixed32 same-key result，故 paired delta/help-harm=`not reported`

正式报告：`docs/paper_agent/experiments/20260731_official_cal_multiline_4990_result.zh.md`。

## Unified analyzer contract

`analysis/baseline_grouped_analyzer.py` 统一报告：

- row-level Pass@1；
- equal-weight base-function task-macro Pass@1；
- 10,000 次 base-function cluster bootstrap 95% CI；
- paired row-level 和 cluster-level help/harm；
- search/decode/total forward calls、token-forwards、wall time、peak GPU memory；
- dynamic selected-length、expansion、contraction、termination distributions。

配对使用 manifest 中跨方法共享的 normalized `candidate_key`。key 集合不完全相同、task-group 不一致、missing/extra/duplicate、cluster 数错误或 forward 分账不守恒时 fail-stop。CAL-Rest 明确使用实际 `143` clusters，不硬写 148。
