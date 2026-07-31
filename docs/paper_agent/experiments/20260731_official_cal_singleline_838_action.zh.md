# Official-source CAL SingleLine 838 Smoke→Full Action Brief

日期：2026-07-31 UTC

状态：`adapter_verified_ready_for_commit_then_smoke`

准确标签：**official-source CAL, initial length 32, on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset**。

## Protocol

- source：`NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`
- evaluator：`openai/human-eval-infilling@88062ff9859c875d04db115b698ed4b0f0395170`
- model：`GSAI-ML/LLaDA-8B-Base@0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- config：initial=`32`、span=`1`、dstep=`4`、max=`128`、bias=`true`、temperature/CFG=`0`、seed=`42`
- bias：复用 pinned source 的固定 `(a,b,c,d,e)=(1.00,1.77,0.56,0.06,0.24)`；不在 CAL-Rest outcome 上拟合。
- full manifest：`838 rows/143 clusters`，SHA256=`52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`
- smoke manifest：`12 rows/12 clusters`，SHA256=`56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`

Adapter change 只参数化 `single-line`/`multi-line` dataset config、full count和 candidate-key namespace；旧 MultiLine行为与 raw contract必须保持兼容。

## Execution

- GPU：physical index `0`；最多一个项目 GPU process。
- output：`outputs_clean/official_cal_singleline_20260731_v1/`
- log：`logs/paper_agent/20260731_official_cal_singleline.log`
- full tmux：`cal_singleline_838_20260731`
- environment：`dllm_env`，`TOKENIZERS_PARALLELISM=false`

Exact smoke/full commands 与 `docs/paper_agent/current_action.md` 一致。

Launcher：`scripts/manual_launch_official_cal_singleline_20260731.sh`。Smoke 直接调用 `smoke` mode；full 由独立 tmux `cal_singleline_838_20260731` 调用 `full` mode。每次启动向同一 log append exact escaped command、start/end UTC 和 exit code；canonical raw 继续 append-only/dedup。

同 manifest 的中心控制使用 launcher 第二参数：smoke=`bash scripts/manual_launch_official_cal_singleline_20260731.sh smoke official_fixed32`；full=`bash scripts/manual_launch_official_cal_singleline_20260731.sh full official_fixed32`。默认省略第二参数时仍是 `official_cal_primary`。Fixed64仅作长度敏感性/其他 common-protocol baseline，不称 CAL equal-compute。

## Gates

Smoke：12 exact unique rows；0 missing/duplicate/error/failure；forward/token守恒；成本字段完整；evaluator正常；resume no-op写入0行。通过后立即 tmux full。

Full：838 exact unique rows、143 clusters；0 missing/duplicate/error/failure；append-only raw；progress manifest/ETA；frozen=`sealed/0`。

Kill：revision/config/population不一致、failure journal非零、OOM/ECC、accounting错误、无关 GPU process。运行中不读取 partial accuracy。

Verification：11 个 CAL/builder tests `OK`；`py_compile`、launcher `bash -n`、CLI help、SingleLine population certificate/hash preflight、diff hygiene全部通过。`reviewer_gate_disabled`，local review未发现 blocker。
