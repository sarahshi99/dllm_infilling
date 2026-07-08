# Official Second-Regime Hard-Tail Diagnostic

Verdict: `official_second_regime_mixed_stress_evidence`.

This is the approved smaller bounded hard-tail diagnostic sampled from the fixed first-pass labels. It does not modify first-pass labels and does not run the full 104-case hard-tail manifest.

## Overall

Cases: `48`.
Control fixed64 pass: `12/48`.
Best deployable cal-lite pass: `16/48`.
Oracle-sufficient canvas pass: `28/48`.
Genuinely canvas-recoverable now: `24`.
Rescue-limited/non-canvas now: `12`.
Deployable cal-lite helps now: `13`.
Deployable cal-lite harms control now: `9`.
Oracle canvas harms control now: `8`.
First-pass label changes under rerun: `0`.

## Answers

1. Genuinely canvas-recoverable cases are those where control fails and oracle passes in this hard-tail rerun: `24` cases. See `case_failure_notes.csv` rows with `genuinely_canvas_recoverable_now=True`.
2. Rescue-limited/non-canvas-limited cases are those where both control and oracle fail: `12` cases.
3. Deployable cal-lite helps on `13` cases and harms control on `9` cases.
4. Oracle canvas harms control on `8` cases.
5. Paper wording: official second-regime should be written as mixed stress evidence: it supports the diagnostic claim that canvas sufficiency matters on a real official regime, while also marking a scope boundary for deployable cal-lite and for rescue/non-canvas-limited random-span/extreme failures.

## Taxonomy Summary

| Sample group | Current taxonomy | Cases | Control | Deployable | Oracle | Canvas-recoverable | Rescue/non-canvas | Deployable help | Deployable harm | Oracle harm | Label changed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ALL` | `deployable_and_oracle_recover_control_failure` | `12` | `0` | `12` | `12` | `12` | `0` | `12` | `0` | `0` | `0` |
| `ALL` | `deployable_harm_vs_control` | `9` | `9` | `0` | `4` | `0` | `0` | `0` | `9` | `5` | `0` |
| `ALL` | `oracle_canvas_harm_vs_control` | `3` | `3` | `3` | `0` | `0` | `0` | `0` | `0` | `3` | `0` |
| `ALL` | `oracle_only_canvas_recoverable` | `12` | `0` | `0` | `12` | `12` | `0` | `0` | `0` | `0` | `0` |
| `ALL` | `rescue_limited_or_noncanvas_failure` | `12` | `0` | `1` | `0` | `0` | `12` | `1` | `0` | `0` | `0` |
| `deployable_and_oracle_recover_control_failure` | `ALL` | `12` | `0` | `12` | `12` | `12` | `0` | `12` | `0` | `0` | `0` |
| `deployable_and_oracle_recover_control_failure` | `deployable_and_oracle_recover_control_failure` | `12` | `0` | `12` | `12` | `12` | `0` | `12` | `0` | `0` | `0` |
| `harm_risk` | `ALL` | `12` | `12` | `3` | `4` | `0` | `0` | `0` | `9` | `8` | `0` |
| `harm_risk` | `deployable_harm_vs_control` | `9` | `9` | `0` | `4` | `0` | `0` | `0` | `9` | `5` | `0` |
| `harm_risk` | `oracle_canvas_harm_vs_control` | `3` | `3` | `3` | `0` | `0` | `0` | `0` | `0` | `3` | `0` |
| `oracle_only_canvas_recoverable` | `ALL` | `12` | `0` | `0` | `12` | `12` | `0` | `0` | `0` | `0` | `0` |
| `oracle_only_canvas_recoverable` | `oracle_only_canvas_recoverable` | `12` | `0` | `0` | `12` | `12` | `0` | `0` | `0` | `0` | `0` |
| `rescue_limited_or_noncanvas_failure` | `ALL` | `12` | `0` | `1` | `0` | `0` | `12` | `1` | `0` | `0` | `0` |
| `rescue_limited_or_noncanvas_failure` | `rescue_limited_or_noncanvas_failure` | `12` | `0` | `1` | `0` | `0` | `12` | `1` | `0` | `0` | `0` |

## Stop Rule

Stop second-regime GPU work here unless web/user identifies a concrete bug or approves a clearly preregistered follow-up. Do not run the full 104-case hard-tail manifest by default.

## Compact Outputs

- `manifest.csv`
- `results.csv`
- `taxonomy_summary.csv`
- `case_failure_notes.csv`
- `summary.json`
- `report.md`
