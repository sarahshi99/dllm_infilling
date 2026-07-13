# P4.1 ExecRepoBench Preparation

This is the first external benchmark only; it is not an external-result run.
The fixed provenance is:

- dataset: `CSJianYang/ExecRepoBench` at `fa61028ce495c9ceff58398b8a7c47b5ae9f5276`;
- evaluator: `QwenLM/Qwen3-Coder` at `33bc6aabd7791ad7b32f7e92104f11f2359ba890`, entrypoint `qwencoder-eval/base/benchmarks/ExecRepoBench`.

`experiments/p4_execrepobench_audit.py` accepts only those exact checked-out revisions. It verifies the public dataset schema (`repo_name`, `file_name`, `prefix_code`, `suffix_code`, `middle_code`, `context_code`, `fill_type`), records dataset checksum, dataset/evaluator license-file checksums and declared-license hints, local Python/platform/git environment, repository/fill-type counts, and a six-case smoke plan. The plan is repository grouped, spans at least two repositories, includes every observed fill type, and contains only record indices/hashes/metadata—never code payloads or external scores.

The audit is intentionally smoke-plan-only. `final_external_results_opened` remains `false`, and no full ExecRepoBench result may be run until a candidate method configuration is frozen. The actual evaluator smoke must use the pinned Qwen checkout and the plan, then record environment, license, evaluator command, wall time, and any failures in a separate process/output directory.
