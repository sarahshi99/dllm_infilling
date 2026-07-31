# CAL Authors’ DAEDAL FIM Adaptation Action Brief

日期：2026-07-31 UTC

状态：`adapter_preflight_verified_smoke_pending_gpu_blocked_after_CAL_priority`

准确标签固定为：**CAL authors’ DAEDAL FIM adaptation**。不得称为原始 DAEDAL 官方 FIM 方法。

## Source and license boundary

- source：`NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`。
- algorithm file：`daedal_cal/daedal_cal.py`，SHA256=`9c89bc1fc31b35af3da107cce63e8caa945fe1652291b53fb274d8ec8d34372a`。
- evaluation file：`daedal_cal/daedal_code_infilling.py`，SHA256=`338d0409432f00fc16d497bb78099695e07644058c27edb2722da0d878bd79e9`。
- pinned checkout 没有 LICENSE 文件：只允许内部审计/运行；不把其源码复制进本项目发布 artifact。
- evaluation script 当前以 `prefix=`/`suffix=` 调用 `generate`，但真实签名是 `prefix_ids=`/`suffix_ids=`，直接执行会 TypeError。适配只直接调用 pinned `generate` 并修正外层参数名，不修改生成算法。

## Frozen arms

Dynamic：initial=`8`、Stage1=`off`、Stage2=`on`、max=`128`、block=`32`、temperature/CFG=`0`、high/low confidence=`0.90/0.30`、expansion factor=`2`、EOS confidence=`0.5`、expand-EOS confidence=`0.9`、EOS scan tokens=`8`、seed=`42`。

Control：`daedal_fixed8_control` 使用相同 pinned generate、model、manifest、evaluator、seed和 decoder，但 Stage1/Stage2都关闭并固定长度8。它是同协议固定长度控制，不宣称 equal-compute。Fixed64 仍是 LR-DLLM/common-protocol中心 baseline，不替代本项 Fixed8。

Forward accounting：Stage1关闭，因此 source 内所有实际 model calls 都登记为 decode forwards；search forwards=`0`。Dynamic 扩张与 denoising在同一 Stage2 forward中交织，不能伪拆成额外 search calls。

## Populations and commands

- SingleLine CAL-Rest common：838 rows/143 clusters；full SHA256=`52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`；smoke12 SHA256=`56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`。
- MultiLine CAL-Rest common：4990 rows/143 clusters；full SHA256=`e805489111e788e315bcf14838b077b09d0c23c44900ad8220d9e9afcbb95c48`；smoke12 SHA256=`dd25c5aef0eaffcf13186eb9797bb074b5d79678e52f9e6c3669bd447a15fab7`。
- evaluator：HumanEval-Infilling `88062ff9859c875d04db115b698ed4b0f0395170`。
- model：`GSAI-ML/LLaDA-8B-Base@0f2787f2d87eac5eed8a087d5ecd24277e6255b2`；config SHA256=`5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a`；weight-index SHA256=`28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408`。

Exact technical smoke：

```bash
bash scripts/manual_launch_daedal_fim_20260731.sh daedal_dynamic singleline smoke
bash scripts/manual_launch_daedal_fim_20260731.sh daedal_fixed8_control singleline smoke
```

SingleLine full（两项 smoke 和 resume no-op 均通过后）：

```bash
tmux new-session -d -s daedal_fim_sl838_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && bash scripts/manual_launch_daedal_fim_20260731.sh daedal_dynamic singleline full'
```

MultiLine 先重复 dynamic/control smoke，再分别 full。Output=`outputs_clean/daedal_fim_<dataset>_20260731_v1/`；logs=`logs/paper_agent/20260731_daedal_fim_<dataset>_<arm>.log`。

GPU：physical `0`；默认 launcher 检测到已有 compute PID即退出3。2026-07-31 用户追加授权允许并行时，必须显式设置 `ALLOW_SHARED_GPU=1` 并记录 pre-existing PIDs；仍不抢占、不 kill。

## Gates and budget

- smoke：12/12 exact unique；0 missing/duplicate/error/failure；resume no-op；evaluator正常；forward/token ledger守恒；frozen=`sealed/0`。
- dynamic ledger：initial/final length、net expansion、contraction=`0`、Stage1/Stage2 max flags、remaining masks、终止原因、wall、peak memory。
- full：SingleLine 838/143、MultiLine 4990/143 分开报告；paired help/harm 只在同一 candidate keys 的 dynamic/Fixed8 都完成后计算。
- smoke预算20分钟；SingleLine full预算约2小时；MultiLine full预算约10小时，实际 ETA 以 smoke观测更新。OOM/ECC、failure journal、source/config/hash偏差、剩余 mask/stall、accounting错误均为 kill/blocker。
- 运行中只读进度、ETA、OOM/ECC、failure journal和完整性，不读 partial accuracy。

frozen test=`sealed`，`test_evaluation_count=0`。
