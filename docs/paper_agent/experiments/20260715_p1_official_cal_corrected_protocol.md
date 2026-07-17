# P1.1 Official CAL Corrected Protocol — 2026-07-15

Pinned sources are CAL `741e8418a88a732b4c92812424d4f03cab1f7b1f` and HumanEval-Infilling `88062ff9859c875d04db115b698ed4b0f0395170`. The audit verifies exact normalized hashes of prompt, suffix, canonical solution, test, and entry point for all `5815` MultiLine task IDs.

The official seed-42 bias-demo split is reproduced exactly: SingleLine full=`1033`, demo=`100`, Rest=`933`, project non-frozen Rest=`838`. The 100 demo rows map one-to-one by evaluation-field hash to 100 MultiLine rows: MultiLine full=`5815`, CAL-Rest=`5715`, project non-frozen CAL-Rest common=`4990`.

Primary official CAL uses `initial_gen_length=32`, `span=1`, `dstep=4`, `max_gen_length=128`, `use_bias=true`, temperature/CFG zero, no oracle, and upstream formal decoding with unspecified steps/block length. The adapter records search/formal/total forwards, exact input-token ledger, wall time, GPU memory, seed, and pinned evaluator provenance. It also supports an official fixed32 arm and labels project fixed64 as internal/non-same-compute.

The source checkout has no LICENSE file; that fact is recorded and no upstream code is copied to paper artifacts. On 2026-07-17, a normal SciPy installation was blocked by host-network policy and an escalated installation received approval-service `422`. The pinned import-closure audit then established that SciPy occurs only in upstream `length_bias.py`, a separate fitting utility: `llada_cal.llada_cal.generate`, the local adapter, and the pinned evaluator do not import it. No pinned upstream algorithm was changed. The launcher therefore removes only its artificial `import scipy` gate and requires the 12-case technical smoke to prove the actual decoder/evaluator path.

The adapter's canonical raw JSONL is success-only and append-only. Per-case exceptions go to a separate failure journal and cause immediate fail-stop; resume skips only existing successful keys. It atomically updates a smoke/full progress manifest every 25 completed cases (and on final completion), recording missing count, failure-journal count, rate/ETA, GPU, and memory. Full completion requires 4,990 unique successes with zero canonical missing/duplicate/error rows.

Artifacts: `analysis_outputs/official_cal_corrected_protocol_20260715_v1/`.
