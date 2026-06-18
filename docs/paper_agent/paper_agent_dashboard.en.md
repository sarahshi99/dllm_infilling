# Paper Agent Dashboard

Updated: 2026-06-18 00:00 CST

## Current Research Goal

Turn the current DLLM code-infilling project into a competitive CCF-A paper by converting the existing LCAL/LCAS empirical progress into a principled length-control contribution with reproducible evidence.

## Current Central Claim

Inference-time length control for DLLM code infilling can safely recover medium-length under-selection by separating medium rescue from true-long detection; however, true-long infilling remains dominated by length underestimation and likely requires a stronger length-modeling signal than the current official-CAL gate family.

Terminology correction: previous/local baseline or local control rows in these docs are the user's earlier local methods or control versions from this project, not local reproductions of CAL, LR-DLLM, or DreamOn. Paper-reported numbers should be compared in the same table, but they must be labeled separately from previous local methods and the current method.

## Current Experiment Plan Version

`v4`: CPU-first Discovery signal-model implementation and full-log audit are complete. The current A6000 LLaDA-Base checkpoint is `midcons`; Route2 precision `len32` gives a clean low-risk signal at `801/1033 = 77.54%`, pairwise `6/0/795/232`, but the oracle `25+` bucket is still not solved. The 2026-06-18 Discovery V4 CPU audit found no GPU-ready low-risk rule after filtering oracle/pass/outcome leakage. Final decision: `route2_polish_only`.

## Completed This Session

- Read `AGENTS.md`, recent plans, result reports, run registry, literature notes, and core clean runner/analysis code.
- Created the `paper-agent-overnight` branch.
- Confirmed GPUs 0,1,2,3 were already occupied by other Python jobs; no heavy GPU experiment was launched.
- Initialized the bilingual `docs/paper_agent/` research tracking structure.
- Added a tested evidence snapshot builder and regenerated the current A6000 evidence snapshot from raw local outputs.
- Added a tested probe-curve signal audit; it found no strict viable single-feature threshold and confirmed current outputs have no saved stopping traces.
- Verified the probe-curve milestone with `8` focused tests, compile checks, audit regeneration, JSON assertions, and `git diff --check`.
- Entered graceful pause mode; added `docs/paper_agent/pause_checkpoint.current.md`; no new research or GPU experiment was started.
- Resumed in low-token mode and reconciled the stale checkpoint with the current strict-split probe diagnostic files.
- Added and ran a CPU-only strict-split probe-score diagnostic. It failed the held-out safety gate, so no GPU smoke run is justified from the current probe-curve fields alone.
- After resuming, reconciled dashboard/checkpoint verification status and completed fresh focused verification for the strict-split diagnostic: unit test, py_compile, audit regeneration, JSON assertions, and `git diff --check` all passed.
- Completed the user-approved `GSAI-ML/LLaDA-8B-Instruct + midcons` full cross-model run. The first direct HuggingFace launch was interrupted because the network was unreachable; the successful run used `HF_ENDPOINT=https://hf-mirror.com`.
- Inspected CAL, LR-DLLM, and DreamOn source PDFs and wrote the literature-backbone rerun matrix to `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`.
- After user approval for sandbox-outside execution, both DreamCoder Base and DreamCoder Instruct 2-sample smokes passed. Then completed two full `1033` sample runs: Base on GPU2 and Instruct on GPU3.
- Downloaded/probed `Dream-org/Dream-v0-Base-7B` through a local Git/LFS checkout under `/tmp`, passed a 2-sample smoke, then completed a full local same-backbone Dream-7B pair: cal_lite baseline on GPU2 and LCAL official bounded-repair candidate on GPU3.
- Downloaded/probed `apple/DiffuCoder-7B-Base` through the proxy-enabled Git/LFS path under `/tmp`, passed a 2-sample smoke, then completed a full local same-backbone DiffuCoder-Base pair: cal_lite baseline on GPU2 and LCAL official bounded-repair candidate on GPU3.
- Completed the `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe through the configured proxy path. Direct HuggingFace worked better than `hf-mirror.com` for this repo; all six weight shards were downloaded to `/tmp/llada15_probe_20260609` and byte-size checked.
- Completed the `GSAI-ML/LLaDA-1.5` 2-sample smoke pair and full local same-backbone pair on shared GPUs `2/3`. Baseline completed on GPU2 and candidate completed on GPU3; both logs exited `0` and both outputs have `1033` valid rows plus `summary.json`.
- Completed the `inclusionAI/LLaDA-MoE-7B-A1B-Base` download recovery, local API/weight validation, flash-attn environment root-cause investigation, 2-sample smoke gate, and full local same-backbone pair. Candidate is `801/1033 = 77.54%`; local `cal_lite` baseline is `777/1033 = 75.22%`; pairwise is `31` wins / `7` losses.
- Completed trace-long-rescue Task 1/2/3 without Superpowers skills, subagents, reviewer discovery, or Goal tools. The action brief/current action passed markdown hygiene; trace feature extraction and route analysis/report scripts were implemented; focused verification passed with `Ran 6 tests` / `OK`, py_compile, and `git diff --check`.
- Completed serial Task 4 full trace collection: previous local method trace run on GPU `2` produced `769/1033 = 74.44%` with `35257` trace rows; current `midcons` trace run on GPU `3` reproduced `795/1033 = 76.96%` with `35768` trace rows. Both logs ended with `COMMAND_EXIT_CODE="0"` and both trace files cover `1033` task ids.
- Completed Task 5 offline Route 1/2/3 analysis. Route 1 and Route 2 triggered `0` rows on both trace sources, and Route 3 stopped because single-canvas traces plus no Route 1/2 signal do not justify multi-canvas policy cost.
- Implemented and ran CPU-only `trace_feature_audit_v2` serially under `superpowers:executing-plans`. Final output: `analysis_outputs/trace_feature_audit_v2_20260613_204721`. Decision: `diagnostic_only`; the previous source has policy-level candidates, but the midcons source is diagnostic-only, so cross-source stability is insufficient.
- After the user confirmed continuing, completed two LLaDA-Base Route 2 trace-gated long-rescue full follow-up runs using the existing tracked runner `clean_scripts/run_route2_trace_rescue.py`. Broad plateau is `801/1033 = 77.54%`, pairwise `7/1/794/231`; precision top1/conf is `800/1033 = 77.44%`, pairwise `5/0/795/233`. Both logs exited `0`, and both outputs have `1033` valid rows plus `summary.json`.
- Per the user's urgent request, moved the precision `len32` follow-up to a clean GPU3-only full run; the earlier GPU1 partial run stopped around `405/1033` and is excluded from evidence. The GPU3-only run completed successfully: `801/1033 = 77.54%`, pairwise `6/0/795/232`, `57` triggers, `61.40%` trigger true-long precision, and `5.4622` seconds/sample including probe.
- Completed the CPU-only Route2 error analysis Discovery V3 under the Superpowers local fallback. Added `analysis/route2_error_analysis.py` and `tests/test_route2_error_analysis.py`; output is `analysis_outputs/route2_error_analysis_20260617_165806`. The diagnostic reproduces `1033` joined rows, pairwise `6/0/795/232`, `33` triggered failed-long rows, `56` missed failed-long rows, and `31/33` triggered failed-long rows with rescue length >= oracle; decision is `mixed_rescue_quality_and_gate_recall`. No GPU action was launched.
- Per the user's request, continued the true-long signal search and completed the Discovery V4 literature brainstorm plus executable plan. Added `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`, `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`, and `docs/paper_agent/experiments/20260617_discovery_v4_literature_brainstorm.md`. No GPU action was launched.
- Implemented and ran CPU-only `analysis/discovery_v4_signal_audit.py`; `tests/test_discovery_v4_signal_audit.py` passed with `Ran 5 tests` / `OK`. Output is `analysis_outputs/discovery_v4_signal_audit_20260618_000000`; result brief is `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`. Real-data dry runs exposed and fixed two leaked candidates, `true_long` and `triggered_rescue_failure_*`; the final V4 decision is `route2_polish_only`, so no GPU job was launched.

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` and `research_design.current.*.md` contain the research-community user, need, and minimum publishable contribution mapping; log entry `2026-05-31 12:39 CST` records paper-agent document initialization | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | No separate CLI transcript; completion is evidenced by the output research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` includes `CEO-Style Stress Review` covering novelty, importance, reviewer appeal, scope, central claim, weakest assumption, and CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | Current conclusion: credible foothold, not yet a CCF-A claim |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` includes engineering review, datasets, baselines, metrics, compute budget, reproducibility, failure modes, and kill criteria | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | Current version is `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | The current direction was carried by the gstack-style design docs; run before any future creative spec change |
| Superpowers `writing-plans` | completed | Current context records the skill was read; `experiment_plan.current.*.md` and history provide the executable plan | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | No separate Superpowers plan file was created |
| Superpowers `systematic-debugging` | completed | During 2026-06-04 verification, the first JSON assertion used the wrong schema path; inspecting the audit JSON and script showed the real path is `cross_validation.aggregate_heldout` | `docs/paper_agent/current_action.md` | Root cause was the assertion command, not changed audit metrics |
| Superpowers `verification-before-completion` | completed | Log entries `2026-05-31 21:47 CST` and `2026-05-31 22:11 CST` record earlier probe-curve audit verification; 2026-06-04 fresh strict-split verification records unit test, py_compile, audit regeneration, JSON assertions, and `git diff --check` | `docs/paper_agent/current_action.md`, `docs/paper_agent/paper_agent_dashboard.en.md`, `docs/paper_agent/probe_curve_split_score_audit.md` | Key verification: probe-curve audit `Ran 8 tests` / `OK`; strict-split diagnostic `Ran 3 tests` / `OK` with `strict_heldout_pass=False` |
| Superpowers `requesting-code-review` | blocked | Log entry `2026-05-31 21:47 CST` records that no independent Task/subagent reviewer tool was visible; local diff review plus fresh tests were used as fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | Independent reviewer was not completed; residual risk is documented |

## Latest Result Summary

- A6000 control: `787/1033 = 76.19%`.
- A6000 `midcons`: `795/1033 = 76.96%`, `+8` wins and `0` losses vs same-hardware control.
- Long buckets remain unchanged: `17-24 = 20.73%`, `25+ = 16.13%`.
- Offline long-underestimate sweep found no safe heuristic rule from current result fields.
- Fresh snapshot: `docs/paper_agent/evidence_snapshot.md`, generated by `analysis/build_paper_agent_evidence_snapshot.py`.
- Probe-curve audit: `4106` thresholds, `0` strict viable; best short-risk is `8.70%`, above the `5%` GPU gate.
- Strict-split probe-score audit: `5` deterministic task-id folds, `24` features, `63` aggregate held-out triggers, `47.62%` true-long precision, `32.97%` failed-long recall, `22.22%` short-risk, `7.94%` current-pass risk; `strict_heldout_pass=False`.
- Verification: the earlier probe-curve audit's `8` focused tests passed; the strict-split diagnostic passed fresh verification on 2026-06-04 with `Ran 3 tests` / `OK`, py_compile, audit regeneration, JSON assertions, and `git diff --check`.
- LLaDA-Instruct cross-model run: `815/1033 = 78.90%`, below the same-backbone historical LCAS-v3 baseline `817/1033 = 79.09%`. Pairwise: `17` wins, `19` losses, `798` tie-pass, `199` tie-fail. Runtime improved from `6.8661s` to `4.1766s` per sample including probe, but this is negative transfer evidence rather than a claim upgrade.
- Literature-backbone plan: `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`.
- DreamCoder Base smoke: valid sandbox-outside smoke passed, `2/2`, first two tasks were `2` tie-pass versus the same-backbone baseline, avg total sec including probe `3.1439`.
- DreamCoder Instruct smoke: valid sandbox-outside smoke passed, `2/2`, first two tasks were `2` tie-pass versus the same-backbone baseline, avg total sec including probe `3.0555`.
- DreamCoder Base full run: `832/1033 = 80.54%` versus local same-backbone official-canvas cal_lite baseline `825/1033 = 79.86%`; pairwise `27` wins, `20` losses, `805` tie-pass, `181` tie-fail; avg total sec including probe `3.7763` versus `3.7847`. This is a small local positive result, not yet a strong claim.
- DreamCoder Instruct full run: `834/1033 = 80.74%` versus local same-backbone official-canvas cal_lite baseline `848/1033 = 82.09%`; pairwise `21` wins, `35` losses, `813` tie-pass, `164` tie-fail; avg total sec including probe `3.8472` versus `3.8657`. This is negative transfer evidence.
- DreamCoder literature positioning: Base is above CAL DreamCoder-Base anchors (`70.2` average, `76.2` best shown) and below LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` is training-based. These are anchors, not protocol-matched claims.
- Dream-7B local pair: candidate `803/1033 = 77.73%` versus local same-backbone cal_lite baseline `802/1033 = 77.64%`; pairwise `28` wins, `27` losses, `775` tie-pass, `203` tie-fail; avg total sec including probe `3.7337` versus `3.6494`. This is a near-tie/slight local positive, not a strong claim.
- Dream-7B literature positioning: candidate is above the LR-DLLM Dream-7B single-line anchor `76.7`, but DreamOn Dream-7B `88.6` is training-based and much higher. These are anchors, not protocol-matched claims.
- DiffuCoder-Base local pair: candidate `839/1033 = 81.22%` versus local same-backbone cal_lite baseline `838/1033 = 81.12%`; pairwise `25` wins, `24` losses, `814` tie-pass, `170` tie-fail; avg total sec including probe `3.7538` versus `3.6562`. This is a near-tie/slight local positive, not a strong bounded-repair improvement claim.
- DiffuCoder literature positioning: both local rows are above CAL DiffuCoder-Base anchors (`68.0` average, `74.8` best shown), but DreamOn DiffuCoder-7B `92.2` is training-based and much higher. These are anchors, not protocol-matched claims.
- LLaDA-1.5 probe: local path `/tmp/llada15_probe_20260609`; architecture `LLaDAModelLM`; `model_type=llada`; config `mask_token_id=126336`; tokenizer `<|mdm_mask|>` resolves to `126336` while `tokenizer.mask_token` is `None`; all six shards match expected byte sizes with total `16,031,197,144` bytes. This is an API/download result only, not a pass-rate result.
- LLaDA-1.5 local pair: candidate `818/1033 = 79.19%` versus local same-backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`; pairwise `18` wins, `17` losses, `800` tie-pass, `198` tie-fail; avg total sec including probe `6.6453` versus `5.4224`. This is a near-tie/slight local positive, not a strong claim. It loses `6` net tasks in oracle `<=8`, gains `+7` total across oracle `>=13`, and official-repair true-long precision is only `10.91%`.
- LLaDA-MoE local pair: candidate `801/1033 = 77.54%` versus local same-backbone `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`; pairwise `31` wins, `7` losses, `770` tie-pass, `225` tie-fail; avg total sec including probe `10.6107` versus `8.7025`. All oracle buckets are positive or neutral: `<=8 +9`, `9-12 +5`, `13-16 +4`, `17-24 +6`, `25+ 0`. This is the strongest current local transfer result, but still not an external SOTA claim.
- LLaDA-Base full trace diagnostics: previous local method trace `769/1033 = 74.44%`, `35257` trace rows; current `midcons` trace `795/1033 = 76.96%`, `35768` trace rows. Route 1/2/3 all fail Gate A and Gate B under offline accounting; no route-specific GPU policy runner is justified.
- Trace feature audit v2: final output `analysis_outputs/trace_feature_audit_v2_20260613_204721`, decision `diagnostic_only`. Previous source: `1033` rows, `113` true-long, `96` failed-long, source decision `policy_candidate`; midcons source: `1033` rows, `113` true-long, `91` failed-long, source decision `diagnostic_only`. The strongest midcons candidate is `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`, with `18` held-out triggers, `9` failed-long, `2` short-risk, and `0` current-pass risk, but this is still not enough to justify a full GPU policy run.
- Route 2 trace-gated full follow-up: the `midcons` baseline is `795/1033 = 76.96%`. Broad plateau reaches `801/1033 = 77.54%`, net `+6`, but has `1` loss / short loss; it triggers `73` rows with `53.42%` trigger true-long precision. Precision top1/conf reaches `800/1033 = 77.44%`, net `+5`, with `0` losses; it triggers `57` rows with `61.40%` trigger true-long precision. The precision policy is the cleaner candidate, but oracle `17-24` improves by only `+1` and `25+` is unchanged, so true-long is not solved. Core diagnostic: among `91` baseline failed-long rows, Broad triggers `39` and rescues only `2`, while Precision triggers `35` and rescues only `1`.
- Route 2 precision len32 GPU3-only follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`; `801/1033 = 77.54%`, net `+6`, pairwise `6/0/795/232`, `57` triggers, `61.40%` trigger true-long precision, and `5.4622` seconds/sample including probe. Bucket net is `<=8 +2`, `9-12 +2`, `13-16 0`, `17-24 +2`, `25+ 0`; triggered oracle `25+` rows are `0/11` pass. Interpretation: clean low-risk incremental gain, but still no `25+` true-long solution.
- Route2 error analysis Discovery V3: output `analysis_outputs/route2_error_analysis_20260617_165806`; report `analysis_outputs/route2_error_analysis_20260617_165806/report.md`. It confirms that all `6` Route2 wins are triggered rescue cases, but `31/33` triggered failed-long rows already have rescue length >= oracle while `56` failed-long rows are missed entirely. Conclusion: do not blindly lengthen the rescue canvas; the next mechanism should jointly inspect rescue generation/selection quality and probe-trace fusion gate recall.
- Discovery V4 design: spec `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`; plan `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`. V4 treats the problem as risk-controlled action selection rather than single-feature enumeration: `MissedLongHead` searches probe-trace fusion signals for missed failed-long rows, `RescueQualityHead` explains triggered rows that still fail despite enough length, and Policy Distillation converts stable signals into reviewer-readable training-free rules/actions.
- Discovery V4 signal audit: output `analysis_outputs/discovery_v4_signal_audit_20260618_000000`; report `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`; decision `route2_polish_only`. Joined rows `1033`, true-long `113`, baseline failed-long `91`. Best non-leaking candidate `broad_len24_triggered >= 1` triggers `73` rows with `4` missed failed-long, `33` triggered rescue-failure, `10` short risk, `1` current-pass risk, `0.534` true-long precision, and `3/5` stable folds, so it is rejected. Conclusion: do not launch a GPU full run directly from this V4 audit.

## Key Plan Adjustments

- Treat LLaDA-Base `midcons` as a real short/medium checkpoint, not a full solution.
- Treat LLaDA-Instruct `midcons` as negative transfer evidence.
- Stop spending GPU on the current official-CAL true-long trigger family until a stronger signal is defined.
- Prioritize probe-curve multivariate/learned scoring from current outputs; trajectory diagnostics require a trace-enabled smoke run.
- Treat the first simple strict-split linear probe score as negative evidence, not as a candidate GPU policy.
- Restrict future GPU experiments to cards `2,3`, waiting rather than interrupting existing jobs.
- Promote trajectory features, learned length classification, DreamOn-style dynamic canvas control, or LR-DLLM-style length regularization as the next paper-level direction.
- Record the Route 2 precision policy as paper-cleaner incremental positive evidence, not as the final main method. Keep the broad policy as a more aggressive comparison with short-loss risk. Route2 error analysis V3 further shows a mixed bottleneck: rescue generation/selection quality plus gate recall, not just rescue length. V4 upgrades the next step from "find one feature" to "find stable slice/action signals and distill them into rules."

## Biggest Risk

The current improvement is too small and too heuristic for a CCF-A contribution unless the next phase produces either principled long-length control or strong cross-model/protocol-matched validation.

## Next Actions

1. Do not launch a GPU full run directly from Discovery V4.
2. Keep Route2 precision len32 as conservative polish: `801/1033 = 77.54%`, pairwise `6/0/795/232`.
3. If true-long recovery continues, design a rescue generation/selection quality mechanism rather than another blind canvas-length increase.
4. Before any GPU work, write a fresh action brief, success/kill criteria, and confirm GPUs `2/3` are free of other users' jobs.
5. Keep literature anchors, previous local methods, current methods, and trace diagnostics separated.

## User Decisions Needed

No Route 2 GPU experiment is currently running. Discovery V4 CPU audit is complete and did not pass the GPU gate; the next user decision is whether to start a new rescue generation/selection mechanism design.
