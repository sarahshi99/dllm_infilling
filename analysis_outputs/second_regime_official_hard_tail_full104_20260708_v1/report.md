# Official Second-Regime Hard-Tail Full104 Stress Diagnostic

Verdict: `official_second_regime_mixed_stress_evidence`.

This is a post-first-pass fixed hard-tail stress set: it uses all `104` rows from `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/`.
It is used for failure taxonomy and for robustness of the mixed-stress conclusion established by the prior 48-case diagnostic.
The official first-pass 120-case results remain the unbiased official diagnostic estimate.
Full104 does not authorize deployable controller claims and is not a benchmark aggregate or positive-method sweep.
First-pass labels are not modified, sampling is not changed after seeing results, and the frozen controller test remains sealed.

## Overall

Cases: `104`.
Control fixed64 pass: `18/104`.
Best deployable cal-lite pass: `20/104`.
Oracle-sufficient canvas pass: `33/104`.
Genuinely canvas-recoverable now: `26`.
Rescue-limited/non-canvas now: `60`.
Deployable cal-lite helps now: `17`.
Deployable cal-lite harms control now: `15`.
Oracle canvas harms control now: `11`.
First-pass label changes under full104 rerun: `0`.

## Interpretation

1. Genuinely canvas-recoverable cases are rows where control fails and oracle passes in this full104 hard-tail rerun. See `case_failure_notes.csv` rows with `genuinely_canvas_recoverable_now=True`.
2. Rescue-limited/non-canvas-limited cases are rows where both control and oracle fail.
3. Deployable cal-lite help/harm are recorded per row in `case_failure_notes.csv`.
4. Oracle canvas harm vs control is recorded per row in `case_failure_notes.csv`.
5. Paper wording: official second-regime should remain mixed stress evidence. The full104 run strengthens the taxonomy/robustness story, while the 120-case first pass remains the unbiased official diagnostic estimate.

## Comparison Vs Prior 48-Case Diagnostic

| Metric | Prior 48 | Prior 48 rate | Full104 | Full104 rate | Delta |
|---|---:|---:|---:|---:|---:|
| Cases | `48` | `` | `104` | `` | `56` |
| Control fixed64 pass | `12` | `25.00%` | `18` | `17.31%` | `6` |
| Best deployable cal-lite pass | `16` | `33.33%` | `20` | `19.23%` | `4` |
| Oracle-sufficient canvas pass | `28` | `58.33%` | `33` | `31.73%` | `5` |
| Genuinely canvas-recoverable | `24` | `50.00%` | `26` | `25.00%` | `2` |
| Rescue/non-canvas-limited | `12` | `25.00%` | `60` | `57.69%` | `48` |
| Deployable cal-lite help | `13` | `27.08%` | `17` | `16.35%` | `4` |
| Deployable cal-lite harm | `9` | `18.75%` | `15` | `14.42%` | `6` |
| Oracle canvas harm vs control | `8` | `16.67%` | `11` | `10.58%` | `3` |
| First-pass label changes | `0` | `0.00%` | `0` | `0.00%` | `0` |

Prior 48 verdict: `official_second_regime_mixed_stress_evidence`.
Full104 verdict: `official_second_regime_mixed_stress_evidence`.

## Taxonomy Summary

| Sample group | Current taxonomy | Cases | Control | Deployable | Oracle | Canvas-recoverable | Rescue/non-canvas | Deployable help | Deployable harm | Oracle harm | Label changed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ALL` | `deployable_and_oracle_recover_control_failure` | `13` | `0` | `13` | `13` | `13` | `0` | `13` | `0` | `0` | `0` |
| `ALL` | `deployable_harm_vs_control` | `15` | `15` | `0` | `7` | `0` | `0` | `0` | `15` | `8` | `0` |
| `ALL` | `oracle_canvas_harm_vs_control` | `3` | `3` | `3` | `0` | `0` | `0` | `0` | `0` | `3` | `0` |
| `ALL` | `oracle_only_canvas_recoverable` | `13` | `0` | `0` | `13` | `13` | `0` | `0` | `0` | `0` | `0` |
| `ALL` | `rescue_limited_or_noncanvas_failure` | `60` | `0` | `4` | `0` | `0` | `60` | `4` | `0` | `0` | `0` |
| `deployable_and_oracle_recover_control_failure` | `ALL` | `13` | `0` | `13` | `13` | `13` | `0` | `13` | `0` | `0` | `0` |
| `deployable_and_oracle_recover_control_failure` | `deployable_and_oracle_recover_control_failure` | `13` | `0` | `13` | `13` | `13` | `0` | `13` | `0` | `0` | `0` |
| `harm_risk` | `ALL` | `18` | `18` | `3` | `7` | `0` | `0` | `0` | `15` | `11` | `0` |
| `harm_risk` | `deployable_harm_vs_control` | `15` | `15` | `0` | `7` | `0` | `0` | `0` | `15` | `8` | `0` |
| `harm_risk` | `oracle_canvas_harm_vs_control` | `3` | `3` | `3` | `0` | `0` | `0` | `0` | `0` | `3` | `0` |
| `oracle_only_canvas_recoverable` | `ALL` | `13` | `0` | `0` | `13` | `13` | `0` | `0` | `0` | `0` | `0` |
| `oracle_only_canvas_recoverable` | `oracle_only_canvas_recoverable` | `13` | `0` | `0` | `13` | `13` | `0` | `0` | `0` | `0` | `0` |
| `rescue_limited_or_noncanvas_failure` | `ALL` | `60` | `0` | `4` | `0` | `0` | `60` | `4` | `0` | `0` | `0` |
| `rescue_limited_or_noncanvas_failure` | `rescue_limited_or_noncanvas_failure` | `60` | `0` | `4` | `0` | `0` | `60` | `4` | `0` | `0` | `0` |

## Stop Rule

Stop second-regime GPU work after this supplemental fixed-manifest stress/taxonomy completion unless web/user identifies a concrete bug or approves a clearly preregistered follow-up. Do not run synthetic stress or controller V4.

## Compact Outputs

- `manifest.csv`
- `results.csv`
- `taxonomy_summary.csv`
- `case_failure_notes.csv`
- `comparison_vs_48.csv`
- `summary.json`
- `report.md`
