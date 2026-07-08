# Official Second-Regime Manifest Gate

Verdict: `official_second_regime_manifest_gate_passed`.

Source dataset: `loubnabnl/humaneval_infilling`.
Frozen controller test: `sealed_not_touched`.

## Manifest

Cases: `120`.
Target: `40` cases per official config, `10` per length bucket.

| Source config | short | medium | long | extreme |
|---|---:|---:|---:|---:|
| `HumanEval-MultiLineInfilling` | `10` | `10` | `10` | `10` |
| `HumanEval-RandomSpanInfilling` | `10` | `10` | `10` | `10` |
| `HumanEval-RandomSpanInfillingLight` | `10` | `10` | `10` | `10` |

Bucket definition uses LLaDA tokenizer reference/middle token length: short `1-8`, medium `9-16`, long `17-24`, extreme `25+` selected from the longest available non-frozen rows with task-group diversity where possible.

## Frozen-Test Exclusion

No frozen-controller-test rows included: `True`.
Frozen-controller-test rows in manifest: `0`.

## Evaluator Smoke

Smoke set covers one row per `(source_config, length_bucket)`, i.e. `12` rows total.
Smoke rows: `12`.
All smoke rows passed tier1/tier2/tier3: `True`.

## Gate Decision

GPU diagnostic is authorized only if the manifest has exactly `120` rows, no frozen-controller-test rows, and all selected evaluator-smoke rows pass.

Compact outputs: `manifest.csv`, `summary.json`, `evaluator_smoke.csv`, `report.md`.
