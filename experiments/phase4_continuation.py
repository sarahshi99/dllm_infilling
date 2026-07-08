#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "clean_scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "clean_scripts"))

from expvision_dllm_clean.dataset import CodeTask, infer_reference_middle_text, load_humaneval_infilling
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.decode import run_vanilla_decode
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack


DREAM_CAL = PROJECT_ROOT / "model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl"
DREAM_LCAL = PROJECT_ROOT / "outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327/results.jsonl"
PHASE4_BACKBONE_DIAG = REPO / "analysis_outputs/second_backbone_diagnostic_20260708_phase4_v4"
SINGLELINE_CONTROL = REPO / "outputs_clean/h200_rebaseline_control_20260706_tier1_20260706_031441/results.jsonl"
SINGLELINE_V6 = REPO / "outputs_clean/h200_rebaseline_v6_20260707_tier1_tmux_20260707_044307/results.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def task_group(task_id: str) -> str:
    parts = task_id.split("/")
    if "HumanEval" in parts:
        idx = parts.index("HumanEval")
        if idx + 1 < len(parts):
            return f"HumanEval/{parts[idx + 1]}"
    if len(parts) >= 3 and parts[-2].startswith("HumanEval"):
        return "/".join(parts[-2:])
    return task_id


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def pass_value(row: Mapping[str, Any]) -> bool:
    return boolish(metric(row, "passed", False))


def verification_error(row: Mapping[str, Any]) -> tuple[bool, str, str]:
    verification = row.get("verification", {})
    compile_tier = verification.get("tier1_parse_compile", {})
    unit_tier = verification.get("tier3_unit_tests", {})
    compile_passed = boolish(compile_tier.get("passed", False))
    error_type = unit_tier.get("error_type") or compile_tier.get("error_type") or ""
    error_message = unit_tier.get("error_message") or compile_tier.get("error_message") or ""
    return compile_passed, str(error_type), str(error_message)


def code_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(row.get("code", "")).encode("utf-8")).hexdigest()


def oracle_bucket(length: Any) -> str:
    try:
        value = int(float(length))
    except (TypeError, ValueError):
        return "unknown"
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def rows_by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def dream_args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="Dream-org/Dream-Coder-v0-Base-7B",
        split="test",
        dataset_subset="HumanEval-SingleLineInfilling",
        max_samples=None,
        seed=42,
        mask_length_source="oracle",
        fixed_mask_length=16,
        probe_lengths="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24",
        tie_break="shorter",
        score_mode="length_power",
        length_alpha=0.1,
        base_probe_lengths="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24",
        base_alpha=0.06,
        weak_probe_lengths="13,14,15,16",
        strong_probe_lengths="13,14,15,16,20,24,28,32,40",
        long_alpha=0.1,
        strong_min_len=13,
        weak_min_base_len=8,
        weak_max_base_len=12,
        ratio_trigger_threshold=0.97,
        long_score_floor=0.55,
        raw_ratio_threshold=0.97,
        support_count_threshold=2,
        cap_base_le8=14,
        cap_base_9_12=16,
        short_safe_policy="s3",
        correction_selection_rule="shortest_supported",
        shortest_supported_ratio=0.985,
        official_initial_length=8,
        official_span=1,
        official_max_length=64,
        official_dstep=4,
        official_no_bias=False,
        official_bias_params="1.0,1.77,0.56,0.06,0.24",
        official_eval_max_s3_len=12,
        repair_max_s3_len=5,
        repair_min_official_len=6,
        repair_max_official_len=9,
        repair_min_delta=1,
        repair_max_delta=8,
        suspicion_max_s3_len=5,
        suspicion_min_official_len=16,
        suspicion_max_official_len=64,
        suspicion_min_delta=1,
        mid_rescue_max_s3_len=12,
        mid_rescue_min_official_len=11,
        mid_rescue_max_official_len=13,
        mid_rescue_min_delta=3,
        mid_rescue_max_delta=7,
        mid_rescue_min_long_ratio=0.8,
        mid_rescue_source="base",
        dream_steps=64,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        alg="entropy",
        alg_temp=0.0,
        eos_penalty=3.0,
        right_pad_new_tokens=1,
        no_force_right_pad_eos=False,
        no_bos=False,
        no_eos=False,
        torch_dtype="bfloat16",
        device_map="auto",
        output_dir=str(output_dir),
        experiment_name="dreamcoder_phase4_fresh_oracle",
        baseline_results=None,
    )


def find_manifest(path: Path) -> Path:
    preferred = path / "case_manifest.csv"
    if preferred.exists():
        return preferred
    matches = sorted(path.glob("*manifest*.csv"))
    if not matches:
        raise FileNotFoundError(f"No manifest csv under {path}")
    return matches[0]


def run_dream_oracle(manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import run_dreamcoder_official_infilling as dream

    dream.ensure_modeling_rope_utils_available()
    args = dream_args(output_dir)
    cfg = dream.build_config(args)
    dream.set_global_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(
        "modeling_dream.DreamModel",
        cfg.model.model_path,
        local_files_only=bool(os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")),
    )
    if not getattr(model_cls.__init__, "_dllm_weights_only_compatible", False):
        original_init = model_cls.__init__

        def _weights_only_compatible_init(self, config, *init_args, **init_kwargs):
            init_kwargs.pop("weights_only", None)
            return original_init(self, config, *init_args, **init_kwargs)

        _weights_only_compatible_init._dllm_weights_only_compatible = True  # type: ignore[attr-defined]
        model_cls.__init__ = _weights_only_compatible_init
    model = model_cls.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        torch_dtype=dream.get_torch_dtype(cfg.model.torch_dtype),
        device_map=cfg.model.device_map,
    )
    model.eval()
    all_tasks = load_humaneval_infilling(split="test", dataset_subset="HumanEval-SingleLineInfilling")
    by_task = {task.task_id: task for task in all_tasks}
    rows: list[dict[str, Any]] = []
    for item in read_csv(manifest_path):
        task = by_task[str(item["task_id"])]
        rows.append(dream.run_task(task, tokenizer, model, cfg, args))
    return rows


def write_dream_oracle_outputs(manifest_path: Path, output_dir: Path, oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_csv(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_manifest.csv", manifest)
    with (output_dir / "oracle_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in oracle_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cal_by_task = rows_by_task(read_jsonl(DREAM_CAL))
    lcal_by_task = rows_by_task(read_jsonl(DREAM_LCAL))
    oracle_by_task = rows_by_task(oracle_rows)

    oracle_csv: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for item in manifest:
        task_id = str(item["task_id"])
        for policy, source in [
            ("primary_control", cal_by_task),
            ("best_simple_length_policy", lcal_by_task),
            ("oracle_sufficient_canvas", oracle_by_task),
        ]:
            row = source.get(task_id)
            if not row:
                combined.append({"task_id": task_id, "stratum": item["stratum"], "policy": policy, "status": "missing"})
                continue
            compile_passed, error_type, error_message = verification_error(row)
            record = {
                "task_id": task_id,
                "stratum": item["stratum"],
                "policy": policy,
                "status": "ok",
                "passed": pass_value(row),
                "oracle_length": metric(row, "oracle_mask_length"),
                "selected_length": metric(row, "selected_mask_length"),
                "selected_minus_oracle": metric(row, "selected_minus_oracle_length"),
                "candidate_sha256": code_hash(row),
                "compile_passed": compile_passed,
                "error_type": error_type,
                "error_message": error_message[:200],
                "total_sec_including_probe": metric(row, "total_sec_including_probe"),
                "decode_sec": metric(row, "decode_sec"),
                "verification_sec": metric(row, "verification_sec"),
            }
            combined.append(record)
            if policy == "oracle_sufficient_canvas":
                oracle_csv.append(record)
    write_csv(output_dir / "oracle_run_results.csv", oracle_csv)
    write_csv(output_dir / "combined_results.csv", combined)

    by_key = {(row["task_id"], row["policy"]): row for row in combined if row["status"] == "ok"}
    strata = {item["task_id"]: item["stratum"] for item in manifest}

    def count_pass(policy: str, rows: Sequence[Mapping[str, str]] = manifest) -> int:
        return sum(1 for item in rows if boolish(by_key.get((item["task_id"], policy), {}).get("passed")))

    missed = [item for item in manifest if item["stratum"] == "missed_failed_long"]
    triggered = [item for item in manifest if item["stratum"] == "triggered_failed_long"]
    short = [item for item in manifest if item["stratum"] == "short_primary_pass"]
    primary_pass = count_pass("primary_control")
    simple_pass = count_pass("best_simple_length_policy")
    oracle_pass = count_pass("oracle_sufficient_canvas")
    short_regressions = sum(
        1
        for item in short
        if boolish(by_key.get((item["task_id"], "primary_control"), {}).get("passed"))
        and not boolish(by_key.get((item["task_id"], "oracle_sufficient_canvas"), {}).get("passed"))
    )
    summary = {
        "verdict": "second_backbone_fresh_oracle_completed",
        "case_count": len(manifest),
        "primary_control_pass": primary_pass,
        "best_simple_policy_pass": simple_pass,
        "oracle_sufficient_canvas_pass": oracle_pass,
        "missed_long_oracle_recoveries": count_pass("oracle_sufficient_canvas", missed),
        "triggered_long_oracle_recoveries": count_pass("oracle_sufficient_canvas", triggered),
        "short_case_regressions_vs_primary": short_regressions,
        "oracle_error_type_histogram": dict(Counter(str(row["error_type"] or "None") for row in oracle_csv)),
        "avg_oracle_cost_sec": avg(row.get("total_sec_including_probe") for row in oracle_csv),
        "efg_status": "not_applicable_dreamcoder_no_trace_remask_adapter",
        "qualitative_agreement_with_llada_h200": qualitative_backbone_agreement(missed, triggered, by_key),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "working_directory": str(REPO),
            "manifest": str(manifest_path),
            "model": "Dream-org/Dream-Coder-v0-Base-7B",
            "action_set": ["primary_control_existing_full", "best_simple_length_policy_existing_full", "fresh_oracle_sufficient_canvas"],
            "efg_status": "not_applicable_dreamcoder_no_trace_remask_adapter",
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            },
        },
    )
    report = [
        "# Dream-Coder Fresh Oracle-Sufficient Diagnostic",
        "",
        "Verdict: `second_backbone_fresh_oracle_completed`.",
        "",
        f"Cases: `{len(manifest)}`.",
        f"Primary/control pass: `{primary_pass}/{len(manifest)}`.",
        f"Best simple length policy pass: `{simple_pass}/{len(manifest)}`.",
        f"Oracle-sufficient canvas pass: `{oracle_pass}/{len(manifest)}`.",
        f"Missed-long oracle recoveries: `{summary['missed_long_oracle_recoveries']}`.",
        f"Triggered-long oracle recoveries: `{summary['triggered_long_oracle_recoveries']}`.",
        f"Short-case regressions vs primary under oracle canvas: `{short_regressions}`.",
        f"Average oracle cost sec: `{summary['avg_oracle_cost_sec']}`.",
        "",
        f"Qualitative agreement: {summary['qualitative_agreement_with_llada_h200']}",
        "",
        "E/F/G actions are not applicable for Dream-Coder because the current repository has no Dream-Coder trace-remasking adapter.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def avg(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        try:
            if value not in {None, ""}:
                nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else None


def qualitative_backbone_agreement(
    missed: Sequence[Mapping[str, str]],
    triggered: Sequence[Mapping[str, str]],
    by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    missed_recoveries = sum(1 for item in missed if boolish(by_key.get((item["task_id"], "oracle_sufficient_canvas"), {}).get("passed")))
    triggered_recoveries = sum(1 for item in triggered if boolish(by_key.get((item["task_id"], "oracle_sufficient_canvas"), {}).get("passed")))
    if missed and missed_recoveries > 0 and triggered_recoveries == 0:
        return "supports LLaDA H200 central claim on this subset: missed-long recoverability appears while triggered-long remains rescue-limited"
    if missed_recoveries == 0 and triggered_recoveries == 0:
        return "partially supports rescue-limited caution but does not show missed-long recoverability on this small Dream-Coder subset"
    return "mixed: Dream-Coder subset does not cleanly match the LLaDA H200 missed-vs-triggered split"


def manual_handoff(output_dir: Path, intended_output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        "HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 "
        "HF_HOME=/home/shx/.cache/huggingface TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "
        "TOKENIZERS_PARALLELISM=false DLLM_DISABLE_FLASH_ATTN=1 "
        "/home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase4_continuation.py "
        f"--mode dream-oracle --timestamp {intended_output_dir.name.replace('second_backbone_oracle_diagnostic_', '')}"
    )
    text = f"""# Dream-Coder Oracle Diagnostic Manual Handoff

Status: `BLOCKED_GPU_APPROVAL -- MANUAL RUN REQUIRED`.

## Working Directory

`{REPO}`

## Conda Activation

```bash
source /home/shx/miniconda3/etc/profile.d/conda.sh
conda activate dllm_env
```

## CUDA Device

`CUDA_VISIBLE_DEVICES=0`

## Exact Command

```bash
cd {REPO}
{command}
```

## Required Input Files

- `analysis_outputs/second_backbone_diagnostic_20260708_phase4_v4/case_manifest.csv`
- `{DREAM_CAL}`
- `{DREAM_LCAL}`
- local Hugging Face cache for `Dream-org/Dream-Coder-v0-Base-7B`

## Expected Output Directory

`{intended_output_dir}`

## Success Criteria

- `case_manifest.csv` has 15 rows plus header.
- `oracle_run_results.csv` has 15 rows plus header.
- `combined_results.csv` has primary/control, best simple policy, and oracle-sufficient rows for all 15 cases.
- `summary.json` reports `verdict = second_backbone_fresh_oracle_completed`.
- `report.md` exists and records missed-long recoveries, triggered-long recoveries, short regressions, hashes, errors, and cost.

## Post-Run Validation Commands

```bash
python -m json.tool {intended_output_dir}/summary.json
test -f {intended_output_dir}/oracle_run_results.csv
test -f {intended_output_dir}/combined_results.csv
test -f {intended_output_dir}/report.md
```

## Files Codex Should Read After Manual Run

- `{intended_output_dir}/summary.json`
- `{intended_output_dir}/combined_results.csv`
- `{intended_output_dir}/report.md`
- `{intended_output_dir}/run_manifest.json`
"""
    (output_dir / "manual_commands.md").write_text(text, encoding="utf-8")


def data_search(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    roots = [REPO, PROJECT_ROOT, Path.home() / ".cache/huggingface", Path("/tmp")]
    patterns = ["*MultiLine*", "*RandomSpan*", "*humaneval*infilling*", "*HumanEval*Infilling*", "*.jsonl", "*.json", "*.parquet", "*.arrow"]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if len(candidates) >= 2000:
                    break
                if not path.is_file():
                    continue
                name = path.name.lower()
                text = str(path).lower()
                if any(token in text for token in ["multiline", "randomspan", "humaneval_infilling", "humaneval-infilling"]):
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = None
                    candidates.append({"path": str(path), "size": size, "suffix": path.suffix})
            if len(candidates) >= 2000:
                break
    write_csv(output_dir / "candidate_files.csv", candidates)
    return {"candidate_count": len(candidates), "paths": [row["path"] for row in candidates[:20]]}


def load_singleline_tasks() -> list[CodeTask]:
    return load_humaneval_infilling(split="test", dataset_subset="HumanEval-SingleLineInfilling")


def make_synthetic_regime(output_dir: Path, per_bucket: int = 6) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    tasks = load_singleline_tasks()
    selected: list[CodeTask] = []
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    bucket_line_counts = {"short": 1, "medium": 3, "long": 6}
    for task in tasks:
        reference = infer_reference_middle_text(task)
        if reference is None:
            continue
        group = task_group(task.task_id)
        if group in seen_groups:
            continue
        full_code = task.prefix + reference + task.suffix
        lines = full_code.splitlines(keepends=True)
        if len(lines) < 8:
            continue
        body_start = 0
        for idx, line in enumerate(lines):
            if line.startswith("def "):
                body_start = min(idx + 1, len(lines) - 1)
                break
        made_for_group = False
        for bucket, span_len in bucket_line_counts.items():
            if bucket_counts[bucket] >= per_bucket:
                continue
            if len(lines) <= body_start + span_len:
                continue
            start = min(body_start + 1, max(0, len(lines) - span_len - 1))
            end = min(len(lines), start + span_len)
            middle = "".join(lines[start:end])
            if not middle.strip():
                continue
            prefix = "".join(lines[:start])
            suffix = "".join(lines[end:])
            synthetic_id = task.task_id.replace("SingleLineInfilling", "SyntheticSecondRegimeMinimal") + f"/{bucket}"
            raw = dict(task.raw or {})
            raw["source_task_id"] = task.task_id
            raw["synthetic_second_regime_minimal"] = True
            raw["regime_construction"] = f"synthetic {span_len}-line span from reconstructed HumanEval solution"
            selected_task = CodeTask(
                task_id=synthetic_id,
                prefix=prefix,
                suffix=suffix,
                full_prompt=prefix + "<FILL_ME>" + suffix,
                test_code=task.test_code,
                entry_point=task.entry_point,
                canonical_solution=middle,
                raw=raw,
            )
            selected.append(selected_task)
            rows.append(
                {
                    "synthetic_task_id": synthetic_id,
                    "source_task_id": task.task_id,
                    "task_group": group,
                    "regime": "synthetic_second_regime_minimal",
                    "span_line_count": span_len,
                    "length_bucket": bucket,
                    "reference_chars": len(middle),
                    "entry_point": task.entry_point,
                    "span_start_line": start,
                    "span_end_line": end,
                }
            )
            bucket_counts[bucket] += 1
            made_for_group = True
            break
        if not made_for_group:
            continue
        seen_groups.add(group)
        if all(bucket_counts[bucket] >= per_bucket for bucket in bucket_line_counts):
            break
    missing = {bucket: per_bucket - bucket_counts[bucket] for bucket in bucket_line_counts if bucket_counts[bucket] < per_bucket}
    if missing:
        raise RuntimeError(f"Insufficient source rows for synthetic second regime: {missing}")
    write_csv(output_dir / "second_regime_manifest.csv", rows)
    with (output_dir / "synthetic_second_regime_minimal.jsonl").open("w", encoding="utf-8") as handle:
        for task, row in zip(selected, rows):
            payload = {
                "task_id": task.task_id,
                "source_task_id": row["source_task_id"],
                "prompt": task.prefix,
                "suffix": task.suffix,
                "canonical_solution": task.canonical_solution,
                "test": task.test_code,
                "entry_point": task.entry_point,
                "synthetic_second_regime_minimal": True,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return selected, rows


def second_regime_unblock(output_dir: Path) -> dict[str, Any]:
    search = data_search(output_dir)
    tasks, manifest_rows = make_synthetic_regime(output_dir)
    bucket_hist = dict(Counter(row["length_bucket"] for row in manifest_rows))
    decision = {
        "verdict": "synthetic_second_regime_minimal_constructed",
        "candidate_file_count": search["candidate_count"],
        "synthetic_data_file": str(output_dir / "synthetic_second_regime_minimal.jsonl"),
        "second_regime_manifest": str(output_dir / "second_regime_manifest.csv"),
        "case_count": len(tasks),
        "bucket_histogram": bucket_hist,
        "missing_official_files": [
            "data/HumanEval-MultiLineInfilling.jsonl",
            "data/HumanEval-RandomSpanInfilling.jsonl",
            "data/HumanEval-RandomSpanInfillingLight.jsonl",
        ],
    }
    write_json(output_dir / "unblock_decision.json", decision)
    (output_dir / "data_search_report.md").write_text(
        "\n".join(
            [
                "# Second-Regime Data Search Report",
                "",
                f"Candidate files found: `{search['candidate_count']}`.",
                "",
                "Official MultiLine/RandomSpan JSONL files were not found under the repository `data/` path.",
                "A minimal synthetic second regime was constructed from existing HumanEval SingleLine source fields by selecting multi-line target spans.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "synthetic_generation_report.md").write_text(
        "\n".join(
            [
                "# Synthetic Second-Regime Minimal Construction",
                "",
                "Verdict: `synthetic_second_regime_minimal_constructed`.",
                "",
                f"Cases: `{len(tasks)}`.",
                f"Bucket histogram: `{bucket_hist}`.",
                "",
                "Construction: reconstruct HumanEval solutions from existing SingleLine source fields, then cut synthetic 1-line, 3-line, and 6-line spans while preserving tests and entry point. This is explicitly synthetic and must not be reported as the official MultiLine or RandomSpan benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


def load_synthetic_tasks(unblock_dir: Path) -> list[CodeTask]:
    tasks: list[CodeTask] = []
    path = unblock_dir / "synthetic_second_regime_minimal.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                CodeTask(
                    task_id=row["task_id"],
                    prefix=row["prompt"],
                    suffix=row["suffix"],
                    full_prompt=row["prompt"] + "<FILL_ME>" + row["suffix"],
                    test_code=row["test"],
                    entry_point=row["entry_point"],
                    canonical_solution=row["canonical_solution"],
                    raw=dict(row),
                )
            )
    return tasks


def synthetic_cfg(mask_length_source: str) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = "GSAI-ML/LLaDA-8B-Base"
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = "synthetic_second_regime_minimal"
    cfg.decode.mask_length_source = mask_length_source
    cfg.decode.total_steps = 64
    cfg.decode.seed = 42
    cfg.decode.cal_lite_probe_lengths_csv = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
    cfg.decode.cal_lite_tie_break = "shorter"
    cfg.decode.cal_lite_score_mode = "length_power"
    cfg.decode.cal_lite_length_alpha = 0.06
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def run_synthetic_policy(tasks: Sequence[CodeTask], tokenizer: Any, model: Any, mask_length_source: str) -> list[dict[str, Any]]:
    cfg = synthetic_cfg(mask_length_source)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        result = run_vanilla_decode(task, tokenizer, model, cfg)
        result["dataset_subset"] = "synthetic_second_regime_minimal"
        rows.append(
            {
                "task_id": result["task_id"],
                "dataset_subset": result["dataset_subset"],
                "metrics": result["metrics"],
                "verification": result["verification"],
                "length_probe": result["length_probe"],
                "code": result["code"],
                "diagnostics": result["diagnostics"],
            }
        )
    return rows


def synthetic_results_from_existing(output_dir: Path, unblock_dir: Path, run_gpu: bool = False) -> dict[str, Any]:
    manifest = read_csv(unblock_dir / "second_regime_manifest.csv")
    write_csv(output_dir / "case_manifest.csv", manifest)
    control_by_task: dict[str, Mapping[str, Any]] = {}
    deployable_by_task: dict[str, Mapping[str, Any]] = {}
    oracle_by_task: dict[str, Mapping[str, Any]] = {}
    if run_gpu:
        set_global_seed(42)
        tokenizer, model = load_model_and_tokenizer(synthetic_cfg("fixed").model)
        tasks = load_synthetic_tasks(unblock_dir)
        control_rows = run_synthetic_policy(tasks, tokenizer, model, "fixed")
        deployable_rows = run_synthetic_policy(tasks, tokenizer, model, "cal_lite")
        oracle_rows = run_synthetic_policy(tasks, tokenizer, model, "oracle")
        with (output_dir / "control_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in control_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "deployable_cal_lite_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in deployable_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "oracle_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in oracle_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        control_by_task = rows_by_task(control_rows)
        deployable_by_task = rows_by_task(deployable_rows)
        oracle_by_task = rows_by_task(oracle_rows)
    combined: list[dict[str, Any]] = []
    for item in manifest:
        synthetic_id = item["synthetic_task_id"]
        for policy, source in [
            ("control_fixed_fresh", control_by_task),
            ("best_deployable_cal_lite_fresh", deployable_by_task),
            ("oracle_sufficient_canvas_fresh", oracle_by_task),
        ]:
            row = source.get(synthetic_id)
            if not row:
                combined.append({"synthetic_task_id": synthetic_id, "source_task_id": item["source_task_id"], "policy": policy, "status": "not_run_requires_gpu"})
                continue
            compile_passed, error_type, error_message = verification_error(row)
            combined.append(
                {
                    "synthetic_task_id": synthetic_id,
                    "source_task_id": item["source_task_id"],
                    "length_bucket": item["length_bucket"],
                    "policy": policy,
                    "status": "ok",
                    "passed": pass_value(row),
                    "selected_length": metric(row, "selected_mask_length"),
                    "oracle_length": metric(row, "oracle_mask_length"),
                    "candidate_sha256": code_hash(row),
                    "compile_passed": compile_passed,
                    "error_type": error_type,
                    "error_message": error_message[:200],
                    "total_sec_including_probe": metric(row, "total_sec_including_probe"),
                }
            )
    write_csv(output_dir / "results.csv", combined)
    bucket_hist = dict(Counter(item["length_bucket"] for item in manifest))
    control_pass = sum(1 for row in combined if row["policy"] == "control_fixed_fresh" and boolish(row.get("passed")))
    deployable_pass = sum(1 for row in combined if row["policy"] == "best_deployable_cal_lite_fresh" and boolish(row.get("passed")))
    oracle_pass = sum(1 for row in combined if row["policy"] == "oracle_sufficient_canvas_fresh" and boolish(row.get("passed")))

    def row_pass(synthetic_id: str, policy: str) -> bool:
        return boolish(next((row for row in combined if row["synthetic_task_id"] == synthetic_id and row["policy"] == policy), {}).get("passed"))

    control_fail_oracle_pass = sum(
        1
        for item in manifest
        if not row_pass(item["synthetic_task_id"], "control_fixed_fresh")
        and row_pass(item["synthetic_task_id"], "oracle_sufficient_canvas_fresh")
    )
    control_fail_oracle_fail = sum(
        1
        for item in manifest
        if not row_pass(item["synthetic_task_id"], "control_fixed_fresh")
        and not row_pass(item["synthetic_task_id"], "oracle_sufficient_canvas_fresh")
    ) if run_gpu else None
    short_regressions = sum(
        1
        for item in manifest
        if item["length_bucket"] == "short"
        and row_pass(item["synthetic_task_id"], "control_fixed_fresh")
        and not row_pass(item["synthetic_task_id"], "oracle_sufficient_canvas_fresh")
    ) if run_gpu else None
    avg_costs = {
        policy: avg(row.get("total_sec_including_probe") for row in combined if row["policy"] == policy)
        for policy in ["control_fixed_fresh", "best_deployable_cal_lite_fresh", "oracle_sufficient_canvas_fresh"]
    }
    summary = {
        "verdict": "second_regime_minimal_diagnostic_completed" if run_gpu else "second_regime_diagnostic_oracle_gpu_required",
        "regime": "synthetic_second_regime_minimal",
        "official_multiline_randomspan_status": "missing_local_jsonl_not_official_benchmark",
        "case_count": len(manifest),
        "short_medium_long_counts": bucket_hist,
        "control_pass": control_pass if run_gpu else None,
        "best_existing_deployable_policy": "cal_lite_length_policy",
        "best_existing_deployable_policy_pass": deployable_pass if run_gpu else None,
        "oracle_sufficient_canvas_pass": oracle_pass if run_gpu else None,
        "short_case_regressions": short_regressions,
        "canvas_limited_fraction": (control_fail_oracle_pass / len(manifest)) if run_gpu else None,
        "rescue_limited_fraction": (control_fail_oracle_fail / len(manifest)) if run_gpu else None,
        "sufficient_canvas_recoverability": (control_fail_oracle_pass / max(1, control_fail_oracle_pass + control_fail_oracle_fail)) if run_gpu else None,
        "deployable_policy_gap": (oracle_pass - deployable_pass) if run_gpu else None,
        "avg_cost_sec": avg_costs if run_gpu else None,
        "qualitative_agreement_with_singleline": (
            "assessable from fresh synthetic control/deployable/oracle diagnostic; compare canvas_limited_fraction with SingleLine oracle action ceiling"
            if run_gpu
            else "not assessable until oracle-sufficient canvas diagnostic is run on synthetic regime"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "working_directory": str(REPO),
            "unblock_dir": str(unblock_dir),
            "case_manifest": str(output_dir / "case_manifest.csv"),
            "model": "GSAI-ML/LLaDA-8B-Base",
            "regime": "synthetic_second_regime_minimal",
            "official_multiline_randomspan_status": "missing_local_jsonl_not_official_benchmark",
            "policies": ["control_fixed_fresh", "best_deployable_cal_lite_fresh", "oracle_sufficient_canvas_fresh"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            },
        },
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Second-Regime Minimal Diagnostic",
                "",
                f"Verdict: `{summary['verdict']}`.",
                "",
                "Regime: `synthetic_second_regime_minimal` (official MultiLine/RandomSpan JSONL files remain missing; this is not an official benchmark result).",
                "",
                f"Cases: `{len(manifest)}`.",
                f"Short/medium/long counts: `{bucket_hist}`.",
                f"Control fresh pass: `{summary['control_pass']}`.",
                f"Best deployable cal-lite fresh pass: `{summary['best_existing_deployable_policy_pass']}`.",
                f"Oracle-sufficient fresh pass: `{summary['oracle_sufficient_canvas_pass']}`.",
                f"Canvas-limited fraction: `{summary['canvas_limited_fraction']}`.",
                f"Rescue-limited fraction: `{summary['rescue_limited_fraction']}`.",
                f"Sufficient-canvas recoverability: `{summary['sufficient_canvas_recoverability']}`.",
                f"Deployable policy gap: `{summary['deployable_policy_gap']}`.",
                f"Average cost sec: `{summary['avg_cost_sec']}`.",
                "",
                "The deployable comparison uses the existing cal-lite length policy freshly on the synthetic tasks. Historical V6 rows are not reused because the synthetic task ids have no matching baseline/reference rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dream-oracle", "dream-manual-handoff", "second-regime-unblock", "second-regime-diagnostic"], required=True)
    parser.add_argument("--timestamp", default=datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--manifest", default=str(PHASE4_BACKBONE_DIAG / "case_manifest.csv"))
    parser.add_argument("--unblock-dir", default=None)
    parser.add_argument("--run-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "dream-oracle":
        out = REPO / "analysis_outputs" / f"second_backbone_oracle_diagnostic_{args.timestamp}"
        if out.exists():
            raise FileExistsError(out)
        manifest = find_manifest(Path(args.manifest).parent if Path(args.manifest).is_file() else Path(args.manifest))
        start = time.perf_counter()
        rows = run_dream_oracle(manifest, out)
        summary = write_dream_oracle_outputs(manifest, out, rows)
        summary["wall_clock_sec"] = time.perf_counter() - start
        write_json(out / "summary.json", summary)
        print(json.dumps({"output_dir": str(out), **summary}, indent=2, sort_keys=True))
        return
    if args.mode == "dream-manual-handoff":
        intended = REPO / "analysis_outputs" / f"second_backbone_oracle_diagnostic_{args.timestamp}"
        out = REPO / "analysis_outputs" / f"second_backbone_oracle_diagnostic_manual_handoff_{args.timestamp}"
        if out.exists():
            raise FileExistsError(out)
        manual_handoff(out, intended)
        print(json.dumps({"output_dir": str(out), "intended_output_dir": str(intended), "verdict": "manual_run_required"}, indent=2))
        return
    if args.mode == "second-regime-unblock":
        out = REPO / "analysis_outputs" / f"second_regime_unblock_{args.timestamp}"
        if out.exists():
            raise FileExistsError(out)
        decision = second_regime_unblock(out)
        print(json.dumps({"output_dir": str(out), **decision}, indent=2, sort_keys=True))
        return
    if args.mode == "second-regime-diagnostic":
        if not args.unblock_dir:
            raise ValueError("--unblock-dir is required for second-regime-diagnostic")
        out = REPO / "analysis_outputs" / f"second_regime_diagnostic_{args.timestamp}"
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        summary = synthetic_results_from_existing(out, Path(args.unblock_dir), run_gpu=args.run_gpu)
        print(json.dumps({"output_dir": str(out), **summary}, indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
