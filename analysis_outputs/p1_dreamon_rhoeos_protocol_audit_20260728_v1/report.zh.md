# P1 DreamOn / rho-EOS Official Source Audit

Scope: CPU-only official source audit. No model weights were downloaded, no GPU process was started, and no pass-rate outcome was opened.

## DreamOn

- Official source: `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8`.
- The official README and demo construct a genuine FIM input from `prefix + masks + suffix`; the evaluator loads `human_eval_infilling` single-line problems and invokes an infilling functional-correctness command.
- The official dynamic launcher evaluates `Dream-org/DreamOn-v0-7B` with min lengths `4/8/16/32/64`, max length `64`, `256` steps, temperature `0.2`, top-p `0.9`, entropy remasking, and mask expansion under `torchrun --nproc_per_node 8`.
- DreamOn is training-based. The public evaluation route is auditable and relevant, but no local protocol-matched GPU smoke/full has been launched. A future smoke must pin the HumanEval-Infilling evaluator revision and state whether changing the official 8-GPU topology affects comparability.

Verdict: `official_source_audited_gpu_smoke_not_started`.

## rho-EOS

- Official source: `yjyddq/rho-EOS@69992caff275e5cd4c96f654aad633fca69f3f75`.
- The official launchers evaluate completion-style `gsm8k`, `math500`, `humaneval`, and `mbpp` tasks. Its generator allocates masks after a left prompt and returns generated tokens after `prompt_length`.
- No suffix/FIM parameter or HumanEval-Infilling evaluator route was found in the audited source/launchers. Replacing this with a custom `prefix + masks + suffix` procedure would change the official algorithmic protocol.

Verdict: `completion_only_or_unfaithful_for_infilling`; no rho-EOS GPU smoke/full is authorized without a separately specified faithful FIM adaptation.

Source file hashes and machine-readable audit fields are in `source_audit.json`.
