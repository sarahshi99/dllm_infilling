# Implementation audit

- Base commit: `5d5d5f2eb9c550e77327e35e827a9ed2ca27b34d`.
- Only global-confidence single-token refresh and fixed-left-frontier single-token refresh are run; C2/C4/L2/L4 are not rerun.
- Released logits shift, temperature 0.2, top-p 0.9, entropy confidence, sampling, expansion, deletion, EOS broadcast-delete, prompt, tokenizer and evaluator semantics are preserved.
- `raw_model_distribution` is softmax after released target alignment and protocol action masking but before temperature/top-p.
- `actual_decode_distribution` exactly repeats released temperature/top-p filtering without sampling; diagnostics do not consume RNG.
- Full-vocabulary TV is computed online. Only scalar metrics and top-5 audit summaries are serialized; no full logits/probability vectors are written.
- Reference tokens are passed only to oracle diagnostic code after official sampling/selection inputs are fixed. Poison-reference regression requires identical selected positions, sampled tokens, actions, canvas states and final output.
- Chains terminate on structural actions, coordinate changes, nonconsecutive commits, target resolution or case termination.
- Actual-decode top-p truncation is explicit through retained flags and extended-real log-prob delta kinds; it is never silently treated as an ordinary finite zero.
- Confidence ties use a stable position-order rank for the scalar rank field; source/fresh top-2 and top-4 membership are separately recorded using the released `torch.topk` operation.
- Fixed smoke12 completed 24/24, matched v2 C1/L1 outputs and all comparable step fields, and passed journal SHA-256 resume/dedup no-op.
- Fresh full alignment compared all 2,066 C1/L1 cases and 20,741 comparable step rows against v2 with zero output/action/canvas-accounting mismatches.
- Reference-poison regression passed: two different pseudo references produced identical selected positions, sampled tokens, actions, canvas states and final output.
- Full reproduction gates: C1=951/1033 and L1=942/1033.
- Reviewer gate: disabled by project policy; local diff review and fresh verification are required before push.
- Analyzer status: `completed`.
