# Current action: DreamOn external Markov-head training v1

- Date: 2026-09-01 UTC
- Branch/base: `codex/dreamon-markov-head-training-v1` from `77f0572b1ca4fe031ab6bbf29b3a4d8740f38802`
- Decision: `advance_to_external_markov_training`
- Reviewer motivation: test whether a rank-256 first-order token head can approximate the fresh DreamOn right-neighbor distribution without changing the frozen DreamOn backbone.
- Scientific comparison: identical `TV-head` and `KL-head`; the only scientific difference is full-vocabulary L1/TV versus forward-KL distribution matching.
- Dataset: only `OpenCoder-LLM/opc-sft-stage2`, config `educational_instruct`, pinned revision `7d28f40d579edd7c24402d17d0c7639f991e6f8d`, MIT, advertised rows `118278`; actual rows and post-dedup rows will be audited.
- Isolation: normalize and deduplicate raw OpenCoder records, strictly remove HumanEval prompt/reference/test overlaps, group duplicates before deterministic hash split seed `20260901`, then freeze train/validation/external-test manifests before trajectory generation.
- Model: `Dream-org/DreamOn-v0-7B@8ccc74750e43177327f29dab9e91882ba759e194`, source `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8`, bf16 frozen inference; Markov losses and softmax in float32.
- Head: `Embedding(152064,256)` plus bias-free `Linear(256,152064)`, expected and programmatically checked parameters `77,856,768`; structural-token correction is zero.
- Shared bank: offset-1 only, global-confidence and fixed-left-frontier trajectories mixed as close to 1:1 as legal samples permit; no full-vocabulary logits are serialized.
- GPU/env: physical H200 index `0`; `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro`; one unrelated `lyx` process was observed using about 1.6 GiB and will not be interrupted.
- Local run root: `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/`
- Git result root: `analysis_outputs/dreamon_markov_head_training_20260901_v1/`
- Log: `logs/paper_agent/20260901_dreamon_markov_head_training_v1.log`
- Execution order: implementation/tests -> data manifest -> shared replay bank -> TV smoke/pilot/full/validation and process exit -> GPU release audit -> KL smoke/pilot/full/validation from the common initialization -> lambda calibration -> one conditional external-test opening -> analysis/docs/commits/push.
- Success criteria: all requested numerical, leakage, replay, frozen-backbone, gradient/update, structural-token, common-init, resume, and overfit tests pass; each pilot/full gate is applied exactly; external test remains unopened unless at least one full-validation gate passes.
- Kill criteria: leakage, reference poison changing trajectories, structural/coordinate transition admitted, DreamOn weight drift, NaN/Inf, irrecoverable checkpoint, protocol mismatch, or resource OOM not resolved by lowering the common micro-batch for both heads.
- Scope boundary: no HumanEval generation/results, no K=2/K=4 decoding, no controller/RNN/backbone modification/MultiLine/long-block experiment.
- Expected outputs: the user-required result files, compressed diagnostics, checkpoint registry with absolute local paths and sizes, research-record updates, five focused commit stages, and push to this branch.

<!-- dreamon-markov-head-training-20260901-v1 -->
## 2026-09-01 DreamOn external Markov-head training v1

Status: `completed_external_test_opened`. TV pilot/full=`True/True`; KL pilot/full=`True/True`; winner=`kl`. This is external OpenCoder training/validation evidence, not a HumanEval method result. Artifacts: `analysis_outputs/dreamon_markov_head_training_20260901_v1/`.
