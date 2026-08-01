# DreamOn V2-Hard-v2 Boundary Decoder

Timestamp: 2026-08-01 UTC

Status: pilot gate failed; stopped before full.

Decision: `iterate`.

## Scope

This round evaluates only `v2_hard_v2_boundary` on the frozen 642-row / 115-base-problem exact-three-line development/mechanism population. It is an oracle structural diagnostic, not a held-out test. V2-OpenTail-v2 and Joint-OpenTail-v2 are explicitly out of scope.

## Why V2-Hard-v1 is not the efficacy test

V2-Hard-v1 remains immutable at Pass@1 `7/642`, compile `46/642`, and 421 forward-cap failures. It is now labeled `V2-Hard-v1 decoder/protocol failure diagnostic; not a clean test of progressive-slot efficacy` because three decoder defects confound the result:

1. every newline-containing hard-slot proposal was suppressed at vocabulary-logit level;
2. EOS deleted only one selected mask, creating deterministic expand/delete loops;
3. repeated exact deterministic transitions were not detected before the 256-forward cap.

The partial V2-OpenTail-v1 run is abandoned rather than resumed.

## Protocol-v2 revisions

- Newline proposals are O(1)-lookup online `line_boundary` actions after a one-time tokenizer-vocabulary metadata scan.
- The first normalized newline terminates the proposal; left text is strictly retokenized and committed, internal right text and all current-slot positions to the right are discarded diagnostically, and the fixed separator remains unchanged.
- EOS deletes the selected and right-side unresolved masks within the active region while preserving right-side resolved tokens.
- Every action records a full pre/post state hash and transition signature. A second identical transition ends the row as `exact_deterministic_cycle`.

All population/model/attention/activation/entropy/cap/scorer conditions otherwise match V2-Hard-v1.

## Preregistered gates

Smoke 5 requires 5/5 completed, zero unresolved/runtime/protocol/cycle/invariant/cross-region-delete rows, strict sequential activation, complete traces, and direct output extraction.

Pilot 30 requires 30/30 completed, compile at least 24/30, Pass@1 at least 10/30, zero failures/cycles, and complete post-generation oracle discard/reference diagnostics.

Full 642 runs only after pilot promotion. It stops if the first 50 contain at least five non-completed rows, if the cumulative non-completed rate exceeds 5% after 100 rows, or if any runtime/invariant damage occurs.

## Paths

- Protocol: `repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json`
- Runner: `repro_scripts/run_dreamon_progressive_v2.py`
- Orchestration: `repro_scripts/run_dreamon_progressive_v2_hard_v2.sh`
- Tests: `tests/test_dreamon_hard_v2_boundary.py`
- Smoke: `repro_results/dreamon_progressive_v2_hard_v2_smoke5/`
- Pilot: `repro_results/dreamon_progressive_v2_hard_v2_pilot30/`
- Full: `repro_results/dreamon_progressive_v2_hard_v2_all642/`

## Smoke result

Smoke passed every preregistered protocol gate: 5/5 completed, zero unresolved/runtime/protocol/cycle/invariant/cross-region-delete rows, strict sequential activation, complete traces, and deterministic resume equality. Smoke Pass@1 was 2/5 and compile was 3/5; these were not smoke promotion thresholds.

## Pilot result and stop

The 30-row pilot failed its preregistered gate and full generation was not started.

| Metric | V2-Hard-v2 pilot |
|---|---:|
| Completed | `25/30 = 83.33%` |
| Pass@1 | `11/30 = 36.67%` |
| Compile | `19/30 = 63.33%` |
| Exact match | `5/30 = 16.67%` |
| Task-macro Pass@1 | `32.67%` |
| Exact deterministic cycles | `5` |
| Forward-cap failures | `0` |
| Runtime/invariant errors | `0` |

The pilot met only the Pass@1 floor (`11 >= 10`). It failed 30/30 completion, zero-cycle, zero-unresolved, zero-protocol-error, and compile-at-least-24 requirements. Full 642 is therefore forbidden under the frozen gate.

Against the historical first-30 one-shot control, V2-Hard-v2 has 2 wins / 10 losses, net `-8`, exact McNemar `p=0.0386`, and 0 compile helps / 11 compile harms. Against historical V1 progressive it has 1 win / 6 losses, net `-5`, exact McNemar `p=0.125`, and 2 compile helps / 8 compile harms.

## Mechanism findings

- All five exact cycles occur in `HARD_SLOT_2`; four are length-2 expand/local-EOS loops and one is a length-6 expand/boundary loop. Exact detection saves compute but does not remove the underlying instability.
- There are 45 boundary events: 39 mixed, 6 pure-newline, and 0 multiple-newline tokens. Boundary actions discard 165 masks and 47 resolved tokens across all slots.
- The oracle slot0/slot1 diagnostic covers 33 events. Fifteen of 30 rows have resolved discard. Only three events have gap-free complete discarded text; two exactly match the reference next line. Among 18 leading contiguous segments, 3 are reference prefixes and 10 are substrings.
- Resolved-discard rows pass 7/15 and compile 10/15, versus 4/15 and 9/15 without resolved discard. This suggests some next-line-like information is discarded, but the denominator and fragmentary cases are insufficient to authorize BoundaryShift.
- Region-local EOS occurs 10 times, deletes 17 masks, and never crosses a region. Nevertheless, four of five cycles still involve expand/local-EOS alternation.
- Completed compile failures comprise two missing-colon cases and four indentation/unclosed-suite cases; three of those six contain a blank slot.

Machine-readable analysis: `repro_results/dreamon_progressive_v2_hard_v2_pilot30/pilot_analysis.json`.

## Decision

`stop_before_full`. The decoder revisions materially improve termination relative to V2-Hard-v1—there are no forward-cap failures and completed-only compile reaches 19/25—but the method remains too unstable and syntactically weak to pass the minimum pilot gate. No V2-Hard-v2 full, V2-OpenTail-v2, or Joint-OpenTail-v2 run is authorized in this round.
