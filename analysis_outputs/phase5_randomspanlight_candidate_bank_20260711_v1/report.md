# Phase 5 RandomSpanLight Candidate Bank

Smoke verdict: `smoke_passed`.
Smoke cases/candidates: `12` / `108`.
Canvas 128 uses the same fixed-mask vanilla decoder and 64-step schedule; no substitute length is allowed.
Raw generated code remains in ignored local output. Compact artifacts contain hashes and metrics only.
Frozen controller test remains sealed with `test_evaluation_count=0`.

Full verdict: `full_passed`.
Full cases/candidates: `148` / `1332`.
Base bank expected rows: `1332`.
Mean candidate latency: `2.5858` seconds.
Summed GPU decode time: `3268.3834` seconds.
Summed verification time: `175.9374` seconds.
Peak CUDA memory: `16545870336` bytes.

Alpha auxiliary verdict: `alpha_auxiliary_passed`.
Verified alpha tasks/candidates: `91` / `728`.
These rows are auxiliary F2 mirrors, not deployable bank candidates.
