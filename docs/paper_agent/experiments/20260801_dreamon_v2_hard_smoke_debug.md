# DreamOn V2-Hard Smoke Debug

Timestamp: 2026-08-01 UTC

## Observation

The first protocol-v1 smoke completed all five rows, but three rows reached the 256-forward cap with unresolved masks. The gate stopped before pilot/full.

## Evidence

- `MultiLineInfilling/HumanEval/0/L1_L3`: 133 expands, 106 deletes; final repeated cycle `31 -> 32 -> 31` in `HARD_SLOT_0`.
- `MultiLineInfilling/HumanEval/0/L2_L4`: final repeated cycle `8 -> 9 -> 8` in `HARD_SLOT_1`.
- `MultiLineInfilling/HumanEval/1/L0_L2`: final repeated cycle `6 -> 7 -> 6` in `HARD_SLOT_1`.
- Zero mask-token no-ops, zero future/prefix/suffix/separator mutation, and no runtime error.

Local output retained at `repro_results/dreamon_progressive_v2_hard_smoke5_v1/`.

## Root cause

The deterministic selected-position birth/death process can enter an exact expand/delete cycle. The protocol explicitly requires unresolved masks at the total-forward cap to be recorded as `protocol_error`; it does not authorize silently deleting them.

The implementation bug was in promotion auditing: it treated this explicit terminal model outcome as if it were a state/invariant protocol violation. The Stage B gate prohibits silent unresolved acceptance and structural mutations, not explicit cap failures. Pass@1 degradation is also not a stop condition.

## Fix

- Preserve `status=protocol_error`, empty completion, unresolved count, trace, and failed score for cap failures.
- Classify exactly `forward_cap_with_unresolved_masks` as `allowed_terminal_protocol_failure` for smoke/pilot promotion.
- Continue to reject every future/prefix/suffix/separator/PAD/invariant error.
- Preserve the old smoke and rerun from a new `_v2` smoke/pilot directory after a new runner commit.

No generation, attention, activation, cap, expand, delete, or sampling rule changed.
