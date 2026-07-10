# Decision Log

Updated: 2026-07-10 UTC

- H200 accepted as the current evidence base. A6000 results remain historical reference only and must not be silently mixed into H200 main claims.
- Frozen test remains sealed unless a preregistered validation gate explicitly passes. Current `test_evaluation_count` remains `0`.
- Controller V4 is not a priority. Do not continue unbounded controller tuning on HumanEval validation without a new evidence source.
- Controller V1/V2/V3 are validation-only negative/weak evidence. V3 route decision remains `weak_validation_signal_test_sealed`; no frozen-test policy is authorized.
- The paper framing is `diagnostic-driven mixed paper`, not a pure positive controller or SOTA method paper.
- FSE is currently the preferred target venue over AAAI because the strongest story is diagnostic, empirical, and software-engineering oriented.
- Easy synthetic second-regime is insufficient. `synthetic_second_regime_minimal` at 18/18 for control/deployable/oracle is runner/data unblock evidence only, not an official benchmark.
- Official or stress second-regime evidence must be clearly labeled as official, synthetic stress, or unblock/sanity. Synthetic stress must not be presented as HumanEval MultiLine/RandomSpan official evidence.
- LR-DLLM remains blocked by missing protocol-matched algorithmic detail. Do not claim official reproduction or invent a local Stage I/II adapter under the LR-DLLM name.
- Dream-Coder is the selected second-backbone route for near-term diagnostics because local cache, runner compatibility, and existing full SingleLine outputs exist.
- Dream-Coder evidence is currently mixed: expanded37 oracle-sufficient canvas reaches `26/37`, but the LLaDA missed-vs-triggered split is not cleanly replicated because Dream-Coder recovers triggered_failed_long proxy cases under oracle canvas.
- Official second-regime data has been recovered locally from `loubnabnl/humaneval_infilling`; the 120-case source-labeled manifest gate passed with frozen-controller-test rows `0` and evaluator smoke `12/12`.
- Official second-regime first-pass GPU diagnostic is completed and is not near ceiling: control fixed64 `34/120`, deployable cal-lite `36/120`, oracle-sufficient canvas `49/120`, oracle gain vs control `26`, deployable harm vs control `15`, oracle harm vs control `11`.
- Because official second-regime has meaningful failures and oracle recovers a nonzero subset, the stop rule became hard-tail follow-up, not immediate benchmark-style expansion. CPU hard-tail manifest `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/` has `104` rows and frozen rows `0`.
- Approved 48-case official second-regime hard-tail diagnostic is completed and first-pass labels remained stable (`0` label changes): control `12/48`, deployable `16/48`, oracle `28/48`, genuine canvas-recoverable `24`, rescue/non-canvas `12`, deployable harm `9`, oracle harm vs control `8`.
- Reviewer-requested fixed full104 official second-regime hard-tail stress diagnostic is completed as supplemental taxonomy/robustness evidence, not as an unbiased benchmark aggregate: control `18/104`, deployable `20/104`, oracle `33/104`, genuine canvas-recoverable `26`, rescue/non-canvas `60`, deployable harm `15`, oracle harm vs control `11`, first-pass label changes `0`.
- User accepted HEAD `1547ed087e9e1eff68ddb14999e0fe40cb5b6d87` and revised execution strategy: when a run is scientifically useful, does not touch frozen test, and does not induce validation overfitting, prefer full runs over small bounded samples because GPU budget is not the limiting factor.
- Full allowed official second-regime diagnostic is completed on all non-frozen rows from the three recovered official configs: `6707` rows, frozen rows `0`, control fixed64 `2019/6707`, deployable cal-lite `1464/6707`, oracle-sufficient canvas `3180/6707`, oracle gain vs control `1633`, deployable harm `1175`, oracle harm vs control `472`, rescue/non-canvas-limited `3055`.
- Official second-regime should be written as `mixed stress evidence`: it supports the canvas-sufficiency diagnostic on official data, but it is also a scope boundary for deployable cal-lite and rescue/non-canvas-limited random-span/extreme failures. Full allowed strengthens the mixed diagnostic claim; the 120-case first pass remains the preregistered/unbiased official diagnostic estimate; full104 remains fixed hard-tail taxonomy/stress; none of these authorize deployable controller claims. Stop second-regime GPU work unless a concrete bug appears.
- Controller V1/V2/V3 route closure is now a real artifact at `analysis_outputs/controller_route_closure_20260708_v1/`; no Controller V4 is authorized.
- Negative results are paper evidence if they preserve split discipline, report costs/risks, and are tied to a scientific claim boundary.
- No checkpoint, cache, token, SSH key, large raw trace, or raw generated-code dump should be committed. Commit compact CSV/JSON/Markdown artifacts only.
- Every new experiment must have a stop condition, success/failure rule, input manifest, and expected report path before GPU execution.
- Current top immediate work is paper claim/table consolidation: official second-regime mixed stress write-up, Dream-Coder model-dependent write-up, and controller route-closure integration.
