# DreamOn V3-C Targeted Results

Date: 2026-08-03 UTC

## Identity

- Branch: `codex/dreamon-progressive-v3-budgeted`
- Population: 642 rows / 115 base problems, exact-three-line development/mechanism population; not held-out.
- Model: `Dream-org/DreamOn-v0-7B`, snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.
- Canonical manifest SHA256: `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`.
- Pure30 SHA256: `5946c7d6e6642861d2e4738616f5b39dcafa5feb5081983d9a8edf23dc55ad4f`.

## Gates

| Method | Smoke5 | Pure30 Pass | Pilot30 Pass | Full status |
|---|---:|---:|---:|---|
| C0 one-shot pure-newline veto | passed | 5/30 | 15/30 | not authorized |
| C nonconsuming blank line | passed | 14/30 | 17/30 | completed 642/642 |

Both methods passed engineering, deterministic, resume, and no-trigger isolation checks in all required pre-Full stages. The frozen Full quality threshold was Pure30 Pass at least 6/30; Pilot had no additional Pass threshold.

## Full result

| Method | Pass@1 | Compile | Exact | Task-macro Pass |
|---|---:|---:|---:|---:|
| One-shot | 301/642 (46.88%) | 638/642 (99.38%) | 83/642 (12.93%) | 43.95% |
| Historical V1 | 316/642 (49.22%) | 585/642 (91.12%) | 122/642 (19.00%) | 46.46% |
| V3-A Budgeted | 281/642 (43.77%) | 538/642 (83.80%) | 97/642 (15.11%) | 41.17% |
| V3-B Nonempty oracle | 280/642 (43.61%) | 471/642 (73.36%) | 106/642 (16.51%) | 42.46% |
| V3-C Nonconsuming blank line | 292/642 (45.48%) | 541/642 (84.27%) | 97/642 (15.11%) | 43.54% |

C completed all 642 rows with zero runtime errors, protocol errors, exact cycles, forward caps, unresolved masks, cross-region deletes, or invariant corruption.

## Paired results

- C vs A: 12 wins / 1 loss / 280 both-pass / 349 both-fail; exact McNemar `p=0.00341797`; row bootstrap 95% CI `[+0.62pp,+2.80pp]`; 115-cluster bootstrap 95% CI `[+0.50pp,+3.17pp]`; task-macro delta `+2.37pp`.
- C vs B: 17 wins / 5 losses; McNemar `p=0.0169005`; cluster CI `[0.00pp,+4.14pp]`.
- C vs one-shot: 96 wins / 105 losses; cluster CI `[-6.76pp,+3.95pp]`.
- C vs V1: 68 wins / 92 losses; cluster CI `[-8.20pp,+0.57pp]`.

The positive result is relative to A, the matched budgeted control. It does not prove superiority to one-shot or historical V1.

## Mechanism findings

- On Pure30, C0 vs A was 2 wins / 0 losses, while C vs A was 12 wins / 1 loss. C vs C0 was 10 wins / 1 loss with cluster CI `[+8.33pp,+53.57pp]`.
- Both C0 and C preserve the right-side slot canvas. Their difference is the inserted physical blank-line context. The result therefore supports blank-line reconditioning over simple veto on this targeted post-hoc diagnostic.
- C trigger rows: 30 rows / 31 events. No-trigger rows: 612/612 exactly match A in completion, compact action signature, score, forwards, and token-forwards.
- Slot effect: slot0 rows net +2; slot1 rows net +2; slot2 rows net +7. The strongest observed subgroup is slot2, but subgroup sizes are small and post-hoc.
- C reduces A blank-slot rows from 137 to 120 without a blanket nonempty constraint.
- Future frozen normal confidence is rare: 2/31 events by lower entropy and 2/31 by higher top1 probability. One occurs on a C win and one on the single C loss, so this observation does not explain the aggregate gain.

## Compute

- C: 35,796 forwards; 9,132,622 token-forwards; recorded wall 2,647.45 seconds; peak GPU memory 15,819,467,264 bytes.
- C minus A: +58 forwards and +5,959 token-forwards. Peak memory is unchanged. Wall time is observational because runs occurred at different times with an unrelated user GPU process present.
- Tokenizer newline-map preprocessing: 0.818 seconds once per run.

## Failure taxonomy

- 101 compile failures.
- 249 compiled functional failures.
- 120 rows retain at least one blank designated content slot after the one-shot guard opportunity is consumed or a different termination mechanism occurs.
- 136 budget-exhausted rows; 391 slot-cap-hit rows; 34 global-cap-hit rows. These are diagnostics, not implementation failures.

## Decision

`advance` to independent review and a separately preregistered held-out validation of unchanged V3-C. Do not deploy, do not claim one-shot/V1 superiority, and do not automatically start held-out evaluation or any new DreamOn method.

Canonical analysis: `repro_results/dreamon_v3c_targeted_analysis/analysis.json` and `repro_results/dreamon_v3c_targeted_analysis/report.zh.md`.
