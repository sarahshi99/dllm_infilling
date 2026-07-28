# DreamOn and rho-EOS CPU Protocol Audit

Timestamp: 2026-07-28 UTC

## Action

Perform a source-only audit of the official DreamOn and rho-EOS repositories after official CAL integrity completion and all M1--M4 V0 decisions. This is not a GPU experiment, model download, training run, or pass-rate evaluation.

## Reviewer Motivation

DreamOn is a training-based dynamic-canvas method and rho-EOS is described as training-free bidirectional variable-length control. A paper cannot label either as a protocol-matched baseline until its official source confirms the checkpoint/training status, FIM versus completion input format, dataset/evaluator path, decoding budget, and whether an infilling adaptation preserves the claimed algorithm.

## Hypotheses and Decision Rules

- DreamOn: determine whether the official repository exposes a reproducible code-infilling training/inference route with identifiable model/data/prompt/evaluator requirements. If training artifacts or a faithful HumanEval-Infilling evaluator route are absent, record `official_source_audited_but_full_not_launchable` rather than inventing an inference-only adapter.
- rho-EOS: determine whether the official quick-start operates on genuine bidirectional FIM context. If it only supports left-to-right completion or requires an algorithmic change to imitate FIM, record `completion_only_or_unfaithful_for_infilling` and do not launch GPU work.

## Exact CPU Commands

```bash
git ls-remote https://github.com/DreamLM/DreamOn.git HEAD
git ls-remote https://github.com/yjyddq/rho-EOS.git HEAD
git clone --depth 1 --no-tags https://github.com/DreamLM/DreamOn.git /tmp/dllm_infilling_protocol_audit_20260728/DreamOn
git clone --depth 1 --no-tags https://github.com/yjyddq/rho-EOS.git /tmp/dllm_infilling_protocol_audit_20260728/rho-EOS
```

Outputs: `analysis_outputs/p1_dreamon_rhoeos_protocol_audit_20260728_v1/` and this brief. No raw generated code, model weights, or cloned repository contents enter Git.

## Success / Kill Criteria

Success is a compact source map with official HEAD, README/requirements/launcher paths, explicit FIM/completion compatibility verdicts, and a no-GPU decision. Fail-stop if clone integrity fails, a repository is unavailable, or the source contains no auditable execution path; record the exact network/source blocker without substituting a new algorithm.

## Result

Both repositories cloned from their official remotes without modification: DreamOn=`8a0a54918412eda9402a327646f7f067f7160ec8`; rho-EOS=`69992caff275e5cd4c96f654aad633fca69f3f75`. DreamOn exposes a genuine prefix/masks/suffix HumanEval-Infilling route and a dynamic expansion evaluator, but is training-based and its official full launcher uses 8 GPUs; verdict=`official_source_audited_gpu_smoke_not_started`. rho-EOS exposes completion-style `generate_until` only: it appends masks after a left prompt and returns the continuation after that prompt, with no suffix/FIM or HumanEval-Infilling evaluator route; verdict=`completion_only_or_unfaithful_for_infilling`. Both source pycompile and launcher `bash -n` checks passed. Compact evidence: `analysis_outputs/p1_dreamon_rhoeos_protocol_audit_20260728_v1/`.
