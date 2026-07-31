# Official-source CAL MultiLine 4,990 Grouped Analysis

准确标签：**official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset**。

这不是完整 5,715-case paper-number reproduction，也不代表 CAL 论文四种初始长度的整张表。报告层次是 `official-source/paper-protocol reproduction on project non-frozen subset`。

## Provenance and integrity

- CAL：`741e8418a88a732b4c92812424d4f03cab1f7b1f`
- evaluator：`88062ff9859c875d04db115b698ed4b0f0395170`
- model：`GSAI-ML/LLaDA-8B-Base`，cache revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- config：initial length=`32`、span=`1`、dstep=`4`、max length=`128`、bias=`true`、temperature/CFG=`0`、oracle=`false`、seed=`42`
- raw：`4990` rows、`4990` unique candidate keys、`143` clusters；missing/extra/duplicate/error/failure=`0/0/0/0/0`
- raw SHA256：`1dfaf0b79bbb081ce696e6589d2083e7f40a328ad68159e230414a506ea3b04a`
- forward/token accounting errors=`0/0`
- frozen test=`sealed`，`test_evaluation_count=0`

## Accuracy

| Metric | Result |
|---|---:|
| Row-level Pass@1 | `1643/4990 = 32.9259%` |
| Equal-weight base-function task-macro Pass@1 | `27.8471%` |
| 10,000 cluster-bootstrap 95% CI | `[24.3313%, 31.2715%]` |
| Paired delta / paired CI | `not reported`：无 identical-key completed `official_fixed32` |
| Paired row/cluster help-harm | `not available`：无合法配对控制 |

## Offline length buckets

Bucket 只按 outcome-blind manifest 中 canonical-solution UTF-8 bytes 的描述性四分位定义（cuts=`51/103/184`），不参与方法选择。

| Bucket | Rows / clusters | Row Pass@1 | Task macro | Mean selected length | Mean forwards | Mean token-forwards |
|---|---:|---:|---:|---:|---:|---:|
| q1 short | `1251 / 131` | `57.3141%` | `50.1578%` | `29.41` | `45.17` | `12,229.96` |
| q2 medium | `1246 / 124` | `50.7223%` | `38.0136%` | `30.11` | `45.35` | `12,049.42` |
| q3 long | `1251 / 97` | `21.1831%` | `14.1046%` | `32.74` | `47.66` | `12,670.94` |
| q4 extreme | `1242 / 59` | `2.3349%` | `1.7709%` | `33.47` | `49.12` | `13,994.02` |

## Cost

| Quantity | Total | Mean per row |
|---|---:|---:|
| Search forward calls | `76,807` | `15.3922` |
| Formal decode forward calls | `156,841` | `31.4311` |
| Total forward calls | `233,648` | `46.8232` |
| Token-forwards | `63,545,173` | `12,734.50` |
| Per-case wall-time sum | `19,342.38 s` (`5.373 h`) | `3.8762 s` |
| Peak GPU memory | `16,560,579,584 bytes` (`15.42 GiB`) | n/a |

Hardware：single `NVIDIA H200 NVL`；这是同一现有 run 的绝对成本，不与其他硬件结果做静默比较。

## Dynamic length behavior

- selected length：min=`2`、p50=`32`、p90=`40`、max=`77`、mean=`31.4311`；70 个 unique lengths。
- 相对 initial32 的最终净状态：`2418` rows `<32`、`453` rows `=32`、`2119` rows `>32`。
- selected-length buckets：`<=8:34`、`9-16:148`、`17-24:561`、`25-32:2128`、`33-48:1974`、`49-64:135`、`65+:10`。
- search forwards：min=`9`、p50=`14`、p90=`23`、max=`54`。
- 历史 raw 记录了最终 selected length 和 search forwards，但没有 event-level expansion/contraction count 或 termination reason。这里明确报告 unavailable，不从 Pass@1 反推或伪造。

## Scientific boundary

这是 CAL 的绝对 subset reproduction result，不足以证明相对固定长度的提升。下一项 CAL 中心比较是同 manifest/decoder 的 `official_fixed32`；在它完成前不得写 paired improvement。Fixed64 仍可作标准固定长度 baseline，但不是自动 compute-matched control。
