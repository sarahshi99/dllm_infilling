# Dream-Coder Full Allowed SingleLine Diagnostic

Verdict: `dreamcoder_full_allowed_singleline_mixed_second_backbone_diagnostic`.

This optional full-run branch uses all non-frozen allowed `HumanEval-SingleLineInfilling` rows only. Frozen-controller-test rows remain sealed and excluded. E/F/G actions are not used because Dream-Coder has no trace-remasking adapter.

Cases: `927`; frozen rows: `0`.
Primary/control pass: `735/927` (`79.29%`).
Best simple length policy pass: `744/927` (`80.26%`).
Oracle-sufficient canvas pass: `858/927` (`92.56%`).
Oracle gain vs primary: `137`.
Simple help/harm vs primary: `25` / `16`.
Oracle harm vs primary: `14`.
Rescue/non-canvas-limited: `55`.

## Interpretation

Use this as optional second-backbone SingleLine diagnostic evidence. It can strengthen or weaken Dream-Coder canvas-recoverability claims, but it must not be written as model-agnostic confirmation of LLaDA's missed-vs-triggered split.

## Compact Outputs

- `manifest.csv`
- `results.csv`
- `stratum_summary.csv`
- `taxonomy_summary.csv`
- `comparison_vs_expanded37.csv`
- `summary.json`
