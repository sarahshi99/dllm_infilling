# M3 Birth--Death Canvas Diffusion V0 — RandomSpanLight grouped result

Primary estimand: equal-weight base-task macro accuracy over 148 task groups. Span-micro is descriptive only. The arms are equal-forward (256 forwards), not equal-token.

Paired task-macro delta (birth-death − uniform): `-0.0405` with 95% task-cluster CI [`-0.1014`, `0.0203`]; wins/losses/ties=`8/14/126`, help/harm=`8/14`, group-aware label-swap p=`0.2888`.

Activation audit: candidate hashes differ on `76/148` rows; events=`444` across `148` rows; uniform events=`0`. Birth-death token-forward min/mean/max=`8192/11532.11/24832`; uniform is 15,360 by contract.

See `method_summary.csv`, `paired_effects.csv`, `length_bucket_summary.csv`, `activation_audit.json`, and `accuracy_token_frontier.csv` for paper-ready compact data. No raw generated code is emitted.
