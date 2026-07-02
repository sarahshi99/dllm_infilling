# Route2 V6 Short-Override Audit

Decision: `policy_candidate`.

## Reproduced Accounting

- rows: `1033`
- V5.1 pass count: `801`
- triggered rows: `57`
- selected triggered pass: `6`
- candidate upper bound: `9`
- pairwise vs Route2 precision len32: `{'win': 0, 'loss': 0, 'tie_pass': 801, 'tie_fail': 232}`

## Upper-Bound-Only `len24_s64` Rows

| task_id | oracle bucket | oracle length |
|---|---:|---:|
| `SingleLineInfilling/HumanEval/7/L0` | `13-16` | `14` |
| `SingleLineInfilling/HumanEval/11/L6` | `17-24` | `22` |
| `SingleLineInfilling/HumanEval/128/L2` | `13-16` | `15` |

## Anchor-Risk Rows

| task_id | oracle bucket | oracle length |
|---|---:|---:|
| `SingleLineInfilling/HumanEval/34/L0` | `<=8` | `8` |
| `SingleLineInfilling/HumanEval/60/L0` | `9-12` | `10` |
| `SingleLineInfilling/HumanEval/66/L1` | `17-24` | `21` |
| `SingleLineInfilling/HumanEval/116/L0` | `17-24` | `21` |

## Top Rules

| Rule | Decision | Overrides | Captured | Anchor losses | Short risk | Delta |
|---|---|---:|---:|---:|---:|---:|
| `anchor_text_chars_ge_136` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `delta_short_minus_anchor_trace_gap_median_ge_0p285156` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `delta_short_minus_anchor_trace_top1_median_ge_0p283203` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `delta_slow_minus_anchor_trace_top1_median_ge_0p0410156` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `short_trace_gap_median_ge_0p65625` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `short_trace_top1_median_ge_0p765625` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `slow_text_chars_ge_136` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `anchor_text_chars_ge_136_AND_delta_slow_minus_anchor_trace_top1_median_ge_0p0410156` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `anchor_text_chars_ge_136_AND_slow_text_chars_ge_136` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |
| `anchor_text_chars_ge_136_AND_delta_slow_minus_anchor_trace_gap_median_ge_0p0273438` | `policy_candidate` | `1` | `1` | `0` | `0` | `1` |

## Interpretation

This is a CPU-only offline selector audit. It does not prove a new pass-rate claim.
A rule is useful only if it captures `len24_s64`-only wins while preserving the `len32_s64` anchor.
