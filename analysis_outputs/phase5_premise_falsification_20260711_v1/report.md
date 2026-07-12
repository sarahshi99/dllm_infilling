# Phase 5 Premise Falsification

All results use the shared full RandomSpanLight candidate bank. Reference code and functional outcomes are offline labels only; task IDs, split labels, oracle length, reference code, verifier outcomes, and error types are absent from deployable feature families.

## F1 Candidate/Fragment Diagnostic

Tasks/candidates: `148` / `1184`.
Mean unique candidate hashes: `4.0000`.
Mean unique parsable AST hashes: `3.3243`.
All-fail tasks with complementary reference-matching semantic units: `6/61`.
Mean consensus-landmark precision: `0.6827`.

## F2 Alpha-Renaming Equivariance

Verdict: `f2_equivariance_not_independently_predictive`.
Paired candidates/tasks: `728` / `91`.
Controlled baseline AUROC: `0.5031`.
Equivariance-augmented AUROC: `0.5174`.
Delta 95% grouped CI: `[-0.0550, 0.0829]`.
Equivariance/stability is not correctness.

## F3A Supervised Probe Diagnostic

These grouped-OOF logistic probes use functional pass/fail labels. They answer only whether feature families contain information and cannot authorize or supply scores to V0.

| Family | Pass AUROC | 95% CI | Horizon MAE | Horizon Spearman |
|---|---:|---:|---:|---:|
| `prefix_only` | `0.6417` | `[0.5825, 0.6978]` | `0.1859` | `0.4860` |
| `suffix_only` | `0.5162` | `[0.4600, 0.5745]` | `0.2760` | `0.0163` |
| `combined_bridge` | `0.6541` | `[0.6025, 0.7098]` | `0.1479` | `0.5911` |
| `token_length` | `0.4871` | `[0.4309, 0.5376]` | `0.2900` | `0.0529` |
| `ordinary_confidence` | `0.5436` | `[0.4938, 0.5877]` | `0.2925` | `0.0180` |

## F3B/F4 Deterministic AST/Def-Use Bridge Proxy

Verdict: `f3_corrected_deployable_gate_failed`.
Corrected gate passed: `False`.
Failed conditions: `['positive_primary_delta_vs_all_baselines']`.
Global AUROC is secondary diagnostic only.

| Family | Cross-canvas pairwise accuracy | 95% CI |
|---|---:|---:|
| `prefix_only` | `0.5941` | `[0.4735, 0.6993]` |
| `suffix_only` | `0.5055` | `[0.4034, 0.6205]` |
| `token_canvas` | `0.5018` | `[0.4837, 0.5232]` |
| `ordinary_confidence` | `0.4945` | `[0.3755, 0.6204]` |
| `combined` | `0.6273` | `[0.4990, 0.7366]` |

## Method Gate

AST/def-use bridge proxy V0 is authorized only by the corrected within-task/cross-canvas, deterministic-baseline, paired-selection, and short-safety gate.
Current authorization: `killed_for_this_round`.
This proxy is not full Semantic Bridge Projection or Abductive Program-State Bridge; genuine program-state analysis, backward obligations, bridge anchors, and denoising intervention remain future stages.
