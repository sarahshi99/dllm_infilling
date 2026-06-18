# Route2 V5.1 Anchor GPU Action

Date: 2026-06-18 CST

## Action Name

Run CPU verification and full GPU experiments for V5.1 `anchor_len32_confidence`.

## Why This Action

V5 smoke showed that the first `consensus_confidence` selector is not safe enough: it can choose `len24_s64` and lose a known Route2 precision `len32` win.

V5.1 fixes that by using `len32_s64` as the anchor:

```text
default: len32_s64
diagnostic-only: len24_s64
optional switch: len32_s96, only if score clears margin
```

This keeps the already successful Route2 precision `len32` behavior as the baseline action while testing whether slower same-length decoding can add wins.

## CPU Verification

Run before GPU:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py tests/test_route2_trace_rescue.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --help
git diff --check
```

## GPU Experiments

Use both GPU `2` and GPU `3` only after `nvidia-smi` confirms they are free of other users' jobs.

Shared settings:

- model: `GSAI-ML/LLaDA-8B-Base`
- baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Route2 reference: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- candidate set: `cheap` = `len24_s64`, `len32_s64`, `len32_s96`
- selector: `anchor_len32_confidence`
- HF mirror/cache: `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME=/home/shx/.cache/huggingface`

### GPU 2: Main Anchor Selector

Switch margin: `0.02`.

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector anchor_len32_confidence --anchor-switch-margin 0.02 --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name full_route2_rescue_quality_v5_anchor_m002_gpu2
```

Log:

- `logs/paper_agent/20260618_route2_v5_anchor_m002_gpu2.log`

### GPU 3: Stricter Anchor Selector

Switch margin: `0.10`.

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector anchor_len32_confidence --anchor-switch-margin 0.10 --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name full_route2_rescue_quality_v5_anchor_m010_gpu3
```

Log:

- `logs/paper_agent/20260618_route2_v5_anchor_m010_gpu3.log`

## Success Criteria

Primary success requires the main selector to beat Route2 precision `len32`:

- pass count at least `803/1033`;
- losses vs `midcons` at most `2`;
- losses vs Route2 precision `len32` at most `3`;
- oracle `<=8` net loss at most `1`;
- oracle `17-24` plus `25+` net gain positive.

Diagnostic success:

- preserve Route2 precision `len32` wins;
- report how often `len32_s96` beats the anchor;
- report candidate oracle upper bound;
- separate generation failure from selector failure.

## Kill Criteria

Stop or keep as diagnostic-only if:

- selected pass count is below Route2 precision `len32` (`801/1033`);
- losses vs Route2 precision `len32` exceed `3`;
- all gains depend only on oracle upper bound;
- runtime overhead is too high for the gain;
- GPU `2/3` become occupied by other users before launch.

## Documentation Outputs

- update this action brief with CPU verification and GPU run status;
- update `docs/paper_agent/experiments/20260618_route2_rescue_quality_v5_implementation.md`;
- update `docs/paper_agent/experiment_results.zh.md` only after full results are complete and parsed.

## Run Status

Completed on 2026-06-18 CST.

Verification:

- CPU unit tests passed: `tests/test_route2_rescue_quality_v5.py` and `tests/test_route2_trace_rescue.py`, `Ran 12 tests`, `OK`.
- `py_compile` passed for `clean_scripts/run_route2_rescue_quality_v5.py`.
- `--help` command succeeded.
- `git diff --check` passed before GPU launch.
- GPU jobs exited with code `0`.
- Both full runs wrote `1033` result rows.

Outputs:

- GPU 2 / margin `0.02`: `outputs_clean/full_route2_rescue_quality_v5_anchor_m002_gpu2_20260618_175641`
- GPU 3 / margin `0.10`: `outputs_clean/full_route2_rescue_quality_v5_anchor_m010_gpu3_20260618_175642`

## Full Results

| Run | Pass | Rate | Avg sec incl. probe | Triggered | Selected candidates | Candidate upper bound on triggered | Pairwise vs `midcons` | Pairwise vs Route2 precision `len32` |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| V5.1 anchor margin `0.02` | `801/1033` | `77.54%` | `4.9208` | `57` | `primary=976`, `len32_s64=56`, `len32_s96=1` | `9/57 = 15.79%` | `6/0/795/232` | `0/0/801/232` |
| V5.1 anchor margin `0.10` | `801/1033` | `77.54%` | `4.9618` | `57` | `primary=976`, `len32_s64=57` | `9/57 = 15.79%` | `6/0/795/232` | `0/0/801/232` |

Oracle bucket pass rates:

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| V5.1 anchor margin `0.02` | `540/598 = 90.30%` | `184/232 = 79.31%` | `53/90 = 58.89%` | `19/82 = 23.17%` | `5/31 = 16.13%` |
| V5.1 anchor margin `0.10` | `540/598 = 90.30%` | `184/232 = 79.31%` | `53/90 = 58.89%` | `19/82 = 23.17%` | `5/31 = 16.13%` |

Triggered-row diagnostic:

| Bucket | Triggered | Selected pass | Oracle upper-bound pass |
|---|---:|---:|---:|
| `<=8` | `6` | `2` | `2` |
| `9-12` | `6` | `2` | `2` |
| `13-16` | `10` | `0` | `2` |
| `17-24` | `24` | `2` | `3` |
| `25+` | `11` | `0` | `0` |

The two margin settings produced identical pass/fail outcomes. The only policy-selection difference was `SingleLineInfilling/HumanEval/122/L0`: margin `0.02` switched to `len32_s96`, margin `0.10` stayed on `len32_s64`; both failed, and no candidate passed.

## Interpretation

Primary success was not met because V5.1 did not beat Route2 precision `len32`; it exactly matched it at `801/1033 = 77.54%`.

Diagnostic success was met:

- V5.1 preserved all Route2 precision `len32` wins and introduced no observed losses.
- The anchor rule prevented the unsafe `len24_s64` override seen in smoke.
- The candidate oracle upper bound was only `9/57` triggered rows, while the selected policy passed `6/57`; therefore the main remaining bottleneck is candidate generation/rescue quality, not only selector choice.

There are three upper-bound-only rows where `len24_s64` passed and the anchor failed:

- `SingleLineInfilling/HumanEval/7/L0`, oracle `14`
- `SingleLineInfilling/HumanEval/11/L6`, oracle `22`
- `SingleLineInfilling/HumanEval/128/L2`, oracle `15`

Those rows explain the residual selector upside, but choosing `len24_s64` globally is not reviewer-safe because the earlier targeted smoke showed that it can lose a known Route2 win. The next useful direction is therefore not another blind full run. It should be a CPU-first design for a conservative, verifiable shorter-candidate override or a better rescue-generation candidate family.
