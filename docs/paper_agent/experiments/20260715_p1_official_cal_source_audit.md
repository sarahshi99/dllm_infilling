# P1.1 Official CAL Source Audit — 2026-07-15

Pinned official sources:

- CAL: `NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`.
- evaluator/data: `openai/human-eval-infilling@88062ff9859c875d04db115b698ed4b0f0395170` (MIT).

The official MultiLine source and the current local MultiLine source both have `5815` unique task IDs and their intersection is exactly `5815`. After the project's sealed controller-test exclusion, the allowed exact-intersection manifest has `5079` rows. The compact audit writes a deterministic 12-case technical-smoke manifest only; it performs no model generation, correctness evaluation, or performance inspection.

Official LLaDA-CAL defaults are model `GSAI-ML/LLaDA-8B-Base`, initial length `8`, `span=1`, `max_gen_length=64`, `dstep=4`, temperature/CFG `0`, bias disabled, and no oracle. Formal decode steps default to the discovered length; total forwards are the CAL probe forwards plus those formal decode forwards. The source has no explicit seed CLI/set-seed path, so seed locking remains a required adapter audit item. The CAL checkout contains no license file. The current environment is missing `scipy` from the official requirements.

GPU smoke/full has not started: an existing M1 process predates this source audit, and this audit does not break the declared serial GPU order by inserting a concurrent CAL job. The required runtime manifest must capture wall time, GPU/memory, actual probe/decode forward counts, seed adapter, evaluator command, and resume/dedup/missing/duplicate/error audit.
