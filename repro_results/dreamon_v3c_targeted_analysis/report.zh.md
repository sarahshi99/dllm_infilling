# DreamOn V3-C targeted results

Date: 2026-08-03 UTC

Population: 642 rows / 115 base problems, development/mechanism population, not held-out.

## Stage decisions

- C0 Smoke5 passed; Pure30 reached 5/30 and therefore did not authorize Full.
- C Smoke5 passed; Pure30 reached 14/30; Pilot30 engineering/isolation checks passed; Full642 was authorized and completed.

## Main Full642 result

| Method | Pass@1 | Compile | Exact | Task-macro Pass |
|---|---:|---:|---:|---:|
| One-shot | 301/642 (46.88%) | 638/642 (99.38%) | 83/642 (12.93%) | 43.95% |
| Historical V1 | 316/642 (49.22%) | 585/642 (91.12%) | 122/642 (19.00%) | 46.46% |
| V3-A Budgeted | 281/642 (43.77%) | 538/642 (83.80%) | 97/642 (15.11%) | 41.17% |
| V3-B Nonempty oracle | 280/642 (43.61%) | 471/642 (73.36%) | 106/642 (16.51%) | 42.46% |
| V3-C Nonconsuming blank line | 292/642 (45.48%) | 541/642 (84.27%) | 97/642 (15.11%) | 43.54% |

## Paired interpretation

- C vs A: 12 wins / 1 losses / 280 both-pass / 349 both-fail; McNemar p=0.00341797; 115-cluster bootstrap CI [0.50%, 3.17%].
- C vs C0 on Pure30: 10 wins / 1 losses; McNemar p=0.0117188; cluster CI [8.33%, 53.57%].
- No-trigger isolation: 612/612 rows matched A exactly in completion, compact action signature, score, forwards, and token-forwards.
- C does not establish superiority to one-shot or historical V1 on this development population. C vs one-shot is net -9 rows and C vs V1 is net -24 rows, with clustered intervals including zero.

## Mechanism

- C0's one-shot veto was insufficient: 5/30 on Pure30 versus A 3/30.
- C reached 14/30 on the same Pure30. Because both preserve the right-side content canvas, the additional benefit is consistent with reconditioning on the inserted physical blank line.
- Frozen future slots rarely had a more confident normal candidate: lower-entropy normal in 2/31 events and higher top1-probability normal in 2/31 events. This does not support a general future-slot-confidence explanation.
- C reduced blank-slot rows from 137 to 120 without a blanket nonempty guard.

## Compute

- C Full: 35796 forwards, 9132622 token-forwards, 2647.5s, peak 15819467264 bytes.
- C minus A: +58 forwards and +5959 token-forwards. Wall time is not used as the cleaner causal compute comparison.

## Decision

`advance`: send V3-C for independent review and preregistered held-out validation. Do not deploy it, do not claim it beats one-shot/V1, and do not restart C0, blanket nonempty, OpenTail, Joint, BoundaryShift, compile gate, or AST repair in this round.
