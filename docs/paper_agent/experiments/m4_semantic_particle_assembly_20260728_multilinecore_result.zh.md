# M4 Semantic Particle Assembly V0 — MultiLine-Core 148-group 正式结果

分析器 `analysis/m4_semantic_particle_assembly.py` 的统计定义在 GPU run 前已由 commit `267dda5` 固定；本次 direct-CLI import fix 不改变 comparison、统计单位、bucket、success 定义或分析逻辑。primary estimand 是 148 个 base task 的 equal-weight macro accuracy；span-micro 仅 descriptive；CI 为 fixed-seed 10,000 task-cluster bootstrap，p 为 group-aware label-swap。

## 数据与完整性边界

mandatory candidate bank 是只读 `40,632 = 5,079 × 8` MultiLine rows（SHA-256 `f51728b445a68eac39ae9a051ab2b9865712a42339eaa65ce8fcf7be7c58537e`）。它与 RandomSpanLight visible prefix/suffix context 的匹配是 `0/164`，所以本 GPU run 不能诚实标为 RandomSpanLight。pre-launch committed manifest 按 `sha256('m4_multiline_core_v1|task_group|row_key')` 每个 non-frozen base task 选择一条 MultiLine span，得到 exactly `148` groups；source-context audit、frozen intersection 都是 `0`。

环境：`GSAI-ML/LLaDA-8B-Base`、H200 NVL GPU `0`、`CUDA_VISIBLE_DEVICES=0`、`TOKENIZERS_PARALLELISM=false`。命令：`bash scripts/manual_launch_m4_semantic_particle_assembly_sprint_v1.sh`。raw outputs：`outputs_clean/m4_{best,assembly,repair}_multilinecore_20260728_v1/`；compact analysis：`analysis_outputs/m4_semantic_particle_assembly_20260717_grouped_v1/`。

12-case smoke 后自动 full。best/assembly/repair 三臂均为 `148/148` unique rows，missing/duplicate/error=`0/0/0`，frozen=`sealed`、`test_evaluation_count=0`，resume no-op=`0`。best/assembly 各 `512` standalone forwards、`30,720` token-forwards；repair 为 `576` forwards、`34,816` token-forwards。full wall=`264.47s`，peak CUDA allocation=`16,553,162,240` bytes（约 `15.42 GiB`）；GPU ECC=`0`。

## 固定 grouped 结果

| Arm | passed/148 | Task macro (95% CI) | mean wall/case | Standalone cost |
|---|---:|---:|---:|---:|
| best-single | 44 | 29.73% [22.30%, 37.16%] | 37.20s | 512 forwards / 30,720 token-forwards |
| assembly-without-repair | 25 | 16.89% [10.81%, 22.97%] | 37.12s | 512 forwards / 30,720 token-forwards |
| assembly-with-repair | 21 | 14.19% [8.78%, 19.59%] | 38.54s | 576 forwards / 34,816 token-forwards |

| Comparison | Task-macro delta (95% CI) | wins/losses/ties | help/harm | group label-swap p |
|---|---:|---:|---:|---:|
| assembly-without-repair − best-single | −12.84pp [−18.24, −7.43]pp | 0/19/129 | 0/19 | 0.0001 |
| assembly-with-repair − best-single | −15.54pp [−24.32, −6.76]pp | 11/34/103 | 11/34 | 0.0002 |
| assembly-with-repair − assembly-without-repair | −2.70pp [−9.46, +4.05]pp | 11/15/122 | 11/15 | 0.5610 |

Descriptive length buckets (best / assembly / repair): short `47.83% / 30.43% / 13.04%`; medium `48.57% / 28.57% / 8.57%`; long `20.00% / 10.00% / 23.33%`; extreme `16.67% / 8.33% / 13.33%`.

## Activation and decision

Assembly was not a no-op: assembly candidate hash differed from best-single on `83/148` rows; repair differed from assembly on `148/148`; provider counts exceeded one on `6/148`; fragment counts exceeded one on `13/148`; assembly fallback was `104/148`. Nonetheless the fair equal-512 primary comparison is clearly negative and has help<harm. Repair has a separate additional 64-forward budget and is also worse than best-single; it cannot be relabeled as pure assembly benefit.

Verdict: `reviewed_not_promoted_v0_multilinecore_cross_source`. M4 does not enter 296/927/5079, retuning, or fusion. This M4 data source differs from M1/M3 RandomSpanLight, so cross-method delta ranking is not apples-to-apples; it does not create a paper primary.
