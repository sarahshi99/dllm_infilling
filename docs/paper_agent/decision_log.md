# Decision Log

Updated: 2026-07-08 UTC

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
- Dream-Coder evidence is currently mixed: oracle-sufficient canvas is strong on the 15-case subset, but the LLaDA missed-vs-triggered split is not cleanly replicated.
- Negative results are paper evidence if they preserve split discipline, report costs/risks, and are tied to a scientific claim boundary.
- No checkpoint, cache, token, SSH key, large raw trace, or raw generated-code dump should be committed. Commit compact CSV/JSON/Markdown artifacts only.
- Every new experiment must have a stop condition, success/failure rule, input manifest, and expected report path before GPU execution.
- Current top immediate experiments are: second-regime stress gate, Dream-Coder expanded diagnostic, and controller/evidence consolidation.
