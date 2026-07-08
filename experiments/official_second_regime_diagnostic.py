#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.decode import run_vanilla_decode
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack


SOURCE_CONFIGS = {
    "HumanEval-MultiLineInfilling": REPO / "data/HumanEval-MultiLineInfilling.jsonl",
    "HumanEval-RandomSpanInfilling": REPO / "data/HumanEval-RandomSpanInfilling.jsonl",
    "HumanEval-RandomSpanInfillingLight": REPO / "data/HumanEval-RandomSpanInfillingLight.jsonl",
}
MANIFEST_DIR = REPO / "analysis_outputs/second_regime_official_manifest_20260708_v1"
DIAGNOSTIC_DIR = REPO / "analysis_outputs/second_regime_official_diagnostic_20260708_v1"
TEST_LOCK = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
GROUPED_TEST_TASKS = REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json"
PROBE_LENGTHS = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
POLICIES = [
    ("control_fixed64", "fixed"),
    ("best_deployable_cal_lite_alpha006", "cal_lite"),
    ("oracle_sufficient_canvas", "oracle"),
]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if match:
        return f"HumanEval/{match.group(1)}"
    return task_id


def prompt_id(task_id: str) -> str:
    parts = task_id.split("/")
    if "HumanEval" in parts:
        idx = parts.index("HumanEval")
        if idx + 2 < len(parts):
            return "/".join(parts[idx + 2 :])
    return parts[-1] if parts else task_id


def load_frozen_test_groups() -> set[str]:
    groups: set[str] = set()
    if GROUPED_TEST_TASKS.exists():
        groups.update(str(item) for item in json.loads(GROUPED_TEST_TASKS.read_text(encoding="utf-8")))
    if TEST_LOCK.exists():
        lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
        groups.update(str(item) for item in lock.get("test_task_ids", []))
    return groups


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def length_bucket(middle_tokens: int) -> str:
    if middle_tokens <= 0:
        return "invalid_empty"
    if middle_tokens <= 8:
        return "short"
    if middle_tokens <= 16:
        return "medium"
    if middle_tokens <= 24:
        return "long"
    return "extreme"


def code_task(row: Mapping[str, Any]) -> CodeTask:
    prefix = str(row["prompt"])
    suffix = str(row["suffix"])
    middle = str(row["canonical_solution"])
    return CodeTask(
        task_id=str(row["task_id"]),
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code=str(row["test"]),
        entry_point=str(row["entry_point"]),
        canonical_solution=middle,
        raw=dict(row),
    )


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def verification_status(row: Mapping[str, Any]) -> tuple[bool, bool, bool, str, str]:
    verification = row.get("verification", {})
    tier1 = verification.get("tier1_parse_compile", {})
    tier2 = verification.get("tier2_smoke_exec", {})
    tier3 = verification.get("tier3_unit_tests", {})
    compile_passed = boolish(tier1.get("passed", False))
    smoke_passed = boolish(tier2.get("passed", False))
    unit_passed = boolish(tier3.get("passed", False))
    error_type = tier3.get("error_type") or tier2.get("error_type") or tier1.get("error_type") or ""
    error_message = tier3.get("error_message") or tier2.get("error_message") or tier1.get("error_message") or ""
    return compile_passed, smoke_passed, unit_passed, str(error_type), str(error_message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_row_key(source_config: str, row_id: int, task_id: str) -> str:
    return f"{source_config}:{row_id}:{task_id}"


def load_source_rows(tokenizer: Any, frozen_groups: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    for source_config, path in SOURCE_CONFIGS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        for row_id, row in enumerate(read_jsonl(path)):
            group = task_group(str(row["task_id"]))
            is_frozen = group in frozen_groups
            if is_frozen:
                excluded_counts[source_config] += 1
                continue
            middle = str(row.get("canonical_solution", ""))
            middle_tokens = token_len(tokenizer, middle)
            bucket = length_bucket(middle_tokens)
            if bucket == "invalid_empty":
                continue
            enriched.append(
                {
                    **row,
                    "source_config": source_config,
                    "row_id": row_id,
                    "task_group": group,
                    "prompt_id": prompt_id(str(row["task_id"])),
                    "prefix_len_chars": len(str(row["prompt"])),
                    "middle_len_chars": len(middle),
                    "suffix_len_chars": len(str(row["suffix"])),
                    "prefix_len_tokens": token_len(tokenizer, str(row["prompt"])),
                    "middle_len_tokens": middle_tokens,
                    "suffix_len_tokens": token_len(tokenizer, str(row["suffix"])),
                    "length_bucket": bucket,
                    "frozen_controller_test_row": False,
                    "frozen_controller_test_exclusion_flag": "included_not_frozen_controller_test",
                    "row_key": stable_row_key(source_config, row_id, str(row["task_id"])),
                }
            )
    return enriched, dict(excluded_counts)


def select_bucket_rows(rows: Sequence[Mapping[str, Any]], config: str, bucket: str, used_groups: set[str]) -> list[Mapping[str, Any]]:
    candidates = [row for row in rows if row["source_config"] == config and row["length_bucket"] == bucket]
    if bucket == "extreme":
        candidates = sorted(candidates, key=lambda row: (-int(row["middle_len_tokens"]), int(row["row_id"])))
    else:
        candidates = sorted(candidates, key=lambda row: (int(row["middle_len_tokens"]), int(row["row_id"])))

    selected: list[Mapping[str, Any]] = []
    for row in candidates:
        if row["task_group"] in used_groups:
            continue
        selected.append(row)
        used_groups.add(str(row["task_group"]))
        if len(selected) == 10:
            return selected

    for row in candidates:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == 10:
            return selected

    raise RuntimeError(f"Need 10 rows for {config} {bucket}, found {len(selected)}")


def build_manifest_rows(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_groups = load_frozen_test_groups()
    source_rows, excluded_counts = load_source_rows(tokenizer, frozen_groups)
    selected: list[dict[str, Any]] = []
    for config in SOURCE_CONFIGS:
        used_groups: set[str] = set()
        for bucket in ["short", "medium", "long", "extreme"]:
            selected.extend(dict(row) for row in select_bucket_rows(source_rows, config, bucket, used_groups))

    for idx, row in enumerate(selected):
        row["manifest_index"] = idx
        row["selected_canvas_metadata_available"] = False
        row["control_fixed_canvas_tokens"] = 64
        row["deployable_policy"] = "cal_lite_length_power_alpha006"
        row["deployable_probe_lengths"] = PROBE_LENGTHS
        row["oracle_canvas_tokens"] = row["middle_len_tokens"]
        row["evaluator_smoke_selected"] = False
        row["evaluator_smoke_status"] = "not_smoked"
        row["evaluator_smoke_tier1"] = ""
        row["evaluator_smoke_tier2"] = ""
        row["evaluator_smoke_tier3"] = ""
        row["evaluator_smoke_error"] = ""

    summary = {
        "case_count": len(selected),
        "excluded_frozen_controller_test_rows": excluded_counts,
        "frozen_controller_test_groups": sorted(frozen_groups),
        "bucket_counts": {
            f"{config}:{bucket}": sum(1 for row in selected if row["source_config"] == config and row["length_bucket"] == bucket)
            for config in SOURCE_CONFIGS
            for bucket in ["short", "medium", "long", "extreme"]
        },
        "source_row_counts_after_filter": dict(Counter(row["source_config"] for row in source_rows)),
    }
    return selected, summary


def smoke_selection(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for config in SOURCE_CONFIGS:
        for bucket in ["short", "medium", "long", "extreme"]:
            candidates = [
                row
                for row in manifest
                if row["source_config"] == config and row["length_bucket"] == bucket
            ]
            if not candidates:
                raise RuntimeError(f"Missing smoke candidate for {config} {bucket}")
            selected.add(str(candidates[0]["row_key"]))
    return selected


def run_cpu_manifest_gate(tokenizer: Any) -> dict[str, Any]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest, manifest_summary = build_manifest_rows(tokenizer)
    smoke_keys = smoke_selection(manifest)
    smoke_rows: list[dict[str, Any]] = []
    all_smoke_passed = True
    for row in manifest:
        if str(row["row_key"]) not in smoke_keys:
            continue
        task = code_task(row)
        full_code = task.prefix + str(row["canonical_solution"]) + task.suffix
        verification = run_verifier_stack(task, full_code=full_code, completion_without_suffix=str(row["canonical_solution"]))
        tier1 = verification.get("tier1_parse_compile")
        tier2 = verification.get("tier2_smoke_exec")
        tier3 = verification.get("tier3_unit_tests")
        passed = bool(tier1 and tier1.passed and tier2 and tier2.passed and tier3 and tier3.passed)
        all_smoke_passed = all_smoke_passed and passed
        error = ""
        for item in [tier1, tier2, tier3]:
            if item and not item.passed:
                error = f"{item.error_type}: {item.error_message}"
                break
        row["evaluator_smoke_selected"] = True
        row["evaluator_smoke_status"] = "passed" if passed else "failed"
        row["evaluator_smoke_tier1"] = bool(tier1 and tier1.passed)
        row["evaluator_smoke_tier2"] = bool(tier2 and tier2.passed)
        row["evaluator_smoke_tier3"] = bool(tier3 and tier3.passed)
        row["evaluator_smoke_error"] = error[:240]
        smoke_rows.append(
            {
                "source_config": row["source_config"],
                "length_bucket": row["length_bucket"],
                "row_id": row["row_id"],
                "task_id": row["task_id"],
                "task_group": row["task_group"],
                "middle_len_tokens": row["middle_len_tokens"],
                "tier1_parse_compile": bool(tier1 and tier1.passed),
                "tier2_smoke_exec": bool(tier2 and tier2.passed),
                "tier3_unit_tests": bool(tier3 and tier3.passed),
                "passed": passed,
                "error": error[:240],
            }
        )

    no_frozen_rows = not any(boolish(row.get("frozen_controller_test_row")) for row in manifest)
    correct_size = len(manifest) == 120 and all(
        count == 10 for count in manifest_summary["bucket_counts"].values()
    )
    gate_passed = bool(correct_size and no_frozen_rows and all_smoke_passed)
    summary = {
        "verdict": "official_second_regime_manifest_gate_passed" if gate_passed else "official_second_regime_manifest_gate_failed",
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_dataset": "loubnabnl/humaneval_infilling",
        "requested_case_count": 120,
        "case_count": len(manifest),
        "per_config_bucket_target": 10,
        "bucket_definition": {
            "short": "1-8 reference/middle tokens",
            "medium": "9-16 reference/middle tokens",
            "long": "17-24 reference/middle tokens",
            "extreme": "25+ reference/middle tokens; selected from longest available rows with task-group diversity",
        },
        "manifest_selection_rule": "exclude frozen controller test task groups, then select 10 rows per source_config and bucket with task-group diversity where possible",
        "no_frozen_controller_test_rows_included": no_frozen_rows,
        "frozen_controller_test_row_count_in_manifest": sum(boolish(row.get("frozen_controller_test_row")) for row in manifest),
        "evaluator_smoke_rows": len(smoke_rows),
        "evaluator_smoke_all_passed": all_smoke_passed,
        **manifest_summary,
    }
    manifest_fields = [
        "manifest_index",
        "source_config",
        "task_id",
        "prompt_id",
        "row_id",
        "task_group",
        "prefix_len_chars",
        "middle_len_chars",
        "suffix_len_chars",
        "prefix_len_tokens",
        "middle_len_tokens",
        "suffix_len_tokens",
        "length_bucket",
        "selected_canvas_metadata_available",
        "control_fixed_canvas_tokens",
        "deployable_policy",
        "deployable_probe_lengths",
        "oracle_canvas_tokens",
        "frozen_controller_test_row",
        "frozen_controller_test_exclusion_flag",
        "evaluator_smoke_selected",
        "evaluator_smoke_status",
        "evaluator_smoke_tier1",
        "evaluator_smoke_tier2",
        "evaluator_smoke_tier3",
        "evaluator_smoke_error",
        "row_key",
    ]
    write_csv(MANIFEST_DIR / "manifest.csv", manifest, manifest_fields)
    write_csv(MANIFEST_DIR / "evaluator_smoke.csv", smoke_rows)
    write_json(MANIFEST_DIR / "summary.json", summary)
    write_manifest_report(summary)
    return summary


def write_manifest_report(summary: Mapping[str, Any]) -> None:
    lines = [
        "# Official Second-Regime Manifest Gate",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        "Source dataset: `loubnabnl/humaneval_infilling`.",
        "Frozen controller test: `sealed_not_touched`.",
        "",
        "## Manifest",
        "",
        f"Cases: `{summary['case_count']}`.",
        "Target: `40` cases per official config, `10` per length bucket.",
        "",
        "| Source config | short | medium | long | extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    counts = summary["bucket_counts"]
    for config in SOURCE_CONFIGS:
        lines.append(
            f"| `{config}` | `{counts[f'{config}:short']}` | `{counts[f'{config}:medium']}` | "
            f"`{counts[f'{config}:long']}` | `{counts[f'{config}:extreme']}` |"
        )
    lines.extend(
        [
            "",
            "Bucket definition uses LLaDA tokenizer reference/middle token length: short `1-8`, medium `9-16`, long `17-24`, extreme `25+` selected from the longest available non-frozen rows with task-group diversity where possible.",
            "",
            "## Frozen-Test Exclusion",
            "",
            f"No frozen-controller-test rows included: `{summary['no_frozen_controller_test_rows_included']}`.",
            f"Frozen-controller-test rows in manifest: `{summary['frozen_controller_test_row_count_in_manifest']}`.",
            "",
            "## Evaluator Smoke",
            "",
            "Smoke set covers one row per `(source_config, length_bucket)`, i.e. `12` rows total.",
            f"Smoke rows: `{summary['evaluator_smoke_rows']}`.",
            f"All smoke rows passed tier1/tier2/tier3: `{summary['evaluator_smoke_all_passed']}`.",
            "",
            "## Gate Decision",
            "",
            "GPU diagnostic is authorized only if the manifest has exactly `120` rows, no frozen-controller-test rows, and all selected evaluator-smoke rows pass.",
            "",
            "Compact outputs: `manifest.csv`, `summary.json`, `evaluator_smoke.csv`, `report.md`.",
        ]
    )
    (MANIFEST_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cfg_for_policy(mask_length_source: str) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = "GSAI-ML/LLaDA-8B-Base"
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = "official_second_regime_120"
    cfg.decode.mask_length_source = mask_length_source
    cfg.decode.fixed_mask_length = 64
    cfg.decode.total_steps = 64
    cfg.decode.seed = 42
    cfg.decode.cal_lite_probe_lengths_csv = PROBE_LENGTHS
    cfg.decode.cal_lite_tie_break = "shorter"
    cfg.decode.cal_lite_score_mode = "length_power"
    cfg.decode.cal_lite_length_alpha = 0.06
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def load_manifest_tasks(manifest_path: Path) -> tuple[list[dict[str, str]], list[CodeTask]]:
    manifest = read_csv(manifest_path)
    source_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for source_config, path in SOURCE_CONFIGS.items():
        for row_id, row in enumerate(read_jsonl(path)):
            source_by_key[(source_config, str(row_id))] = row

    tasks: list[CodeTask] = []
    for item in manifest:
        if boolish(item.get("frozen_controller_test_row")):
            raise RuntimeError(f"Refusing to run frozen-controller-test row: {item.get('task_id')}")
        source = source_by_key[(item["source_config"], item["row_id"])]
        if str(source["task_id"]) != item["task_id"]:
            raise RuntimeError(f"Manifest/source task mismatch for row {item['manifest_index']}")
        tasks.append(code_task(source))
    return manifest, tasks


def compact_result(
    manifest_row: Mapping[str, str],
    policy_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    compile_passed, smoke_passed, unit_passed, error_type, error_message = verification_status(result)
    middle_text = str(result.get("middle_text", ""))
    return {
        "manifest_index": manifest_row["manifest_index"],
        "source_config": manifest_row["source_config"],
        "task_id": manifest_row["task_id"],
        "prompt_id": manifest_row["prompt_id"],
        "row_id": manifest_row["row_id"],
        "task_group": manifest_row["task_group"],
        "length_bucket": manifest_row["length_bucket"],
        "policy": policy_name,
        "status": "ok",
        "passed": boolish(metric(result, "passed", False)),
        "prefix_len_tokens": manifest_row["prefix_len_tokens"],
        "middle_len_tokens": manifest_row["middle_len_tokens"],
        "suffix_len_tokens": manifest_row["suffix_len_tokens"],
        "oracle_canvas_tokens": metric(result, "oracle_mask_length"),
        "selected_canvas_tokens": metric(result, "selected_mask_length"),
        "selected_minus_oracle": metric(result, "selected_minus_oracle_length"),
        "abs_selected_minus_oracle": metric(result, "abs_selected_minus_oracle_length"),
        "mask_length_source": metric(result, "mask_length_source"),
        "selected_score": metric(result, "selected_score"),
        "selected_raw_score": metric(result, "selected_raw_score"),
        "selected_adjusted_score": metric(result, "selected_adjusted_score"),
        "compile_passed": compile_passed,
        "tier2_smoke_exec_passed": smoke_passed,
        "tier3_unit_tests_passed": unit_passed,
        "error_type": error_type,
        "error_message": error_message[:240],
        "candidate_full_code_sha256": sha256_text(str(result.get("code", ""))),
        "candidate_middle_sha256": sha256_text(middle_text),
        "total_sec_including_probe": metric(result, "total_sec_including_probe"),
        "decode_sec": metric(result, "decode_sec"),
        "verification_sec": metric(result, "verification_sec"),
        "length_probe_sec": metric(result, "length_probe_sec"),
        "mean_final_confidence": metric(result, "mean_final_confidence"),
    }


def run_policy(policy_name: str, mask_length_source: str, tasks: Sequence[CodeTask], manifest: Sequence[Mapping[str, str]], tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    cfg = cfg_for_policy(mask_length_source)
    rows: list[dict[str, Any]] = []
    set_global_seed(cfg.decode.seed)
    for item, task in zip(manifest, tasks):
        start = time.perf_counter()
        try:
            result = run_vanilla_decode(task, tokenizer, model, cfg)
            rows.append(compact_result(item, policy_name, result))
        except Exception as exc:
            rows.append(
                {
                    "manifest_index": item["manifest_index"],
                    "source_config": item["source_config"],
                    "task_id": item["task_id"],
                    "prompt_id": item["prompt_id"],
                    "row_id": item["row_id"],
                    "task_group": item["task_group"],
                    "length_bucket": item["length_bucket"],
                    "policy": policy_name,
                    "status": "error",
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:240],
                    "total_sec_including_probe": time.perf_counter() - start,
                }
            )
    return rows


def pass_for(by_case_policy: Mapping[tuple[str, str], Mapping[str, Any]], manifest_index: str, policy: str) -> bool:
    return boolish(by_case_policy.get((manifest_index, policy), {}).get("passed", False))


def build_failure_taxonomy(manifest: Sequence[Mapping[str, str]], results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case_policy = {(str(row["manifest_index"]), str(row["policy"])): row for row in results}
    taxonomy: list[dict[str, Any]] = []
    for item in manifest:
        idx = str(item["manifest_index"])
        control = pass_for(by_case_policy, idx, "control_fixed64")
        deployable = pass_for(by_case_policy, idx, "best_deployable_cal_lite_alpha006")
        oracle = pass_for(by_case_policy, idx, "oracle_sufficient_canvas")
        if control and deployable and oracle:
            category = "stable_all_pass"
        elif (not control) and oracle and deployable:
            category = "deployable_and_oracle_recover_control_failure"
        elif (not control) and oracle and not deployable:
            category = "oracle_only_canvas_recoverable"
        elif (not control) and not oracle:
            category = "rescue_limited_or_noncanvas_failure"
        elif control and not deployable:
            category = "deployable_harm_vs_control"
        elif control and deployable and not oracle:
            category = "oracle_canvas_harm_vs_control"
        else:
            category = "mixed_other"
        row = {
            "manifest_index": idx,
            "source_config": item["source_config"],
            "task_id": item["task_id"],
            "row_id": item["row_id"],
            "task_group": item["task_group"],
            "length_bucket": item["length_bucket"],
            "middle_len_tokens": item["middle_len_tokens"],
            "control_passed": control,
            "deployable_passed": deployable,
            "oracle_passed": oracle,
            "taxonomy": category,
            "hard_tail_candidate": category in {
                "oracle_only_canvas_recoverable",
                "deployable_and_oracle_recover_control_failure",
                "rescue_limited_or_noncanvas_failure",
                "deployable_harm_vs_control",
            },
        }
        for policy in ["control_fixed64", "best_deployable_cal_lite_alpha006", "oracle_sufficient_canvas"]:
            result = by_case_policy.get((idx, policy), {})
            row[f"{policy}_selected_canvas_tokens"] = result.get("selected_canvas_tokens", "")
            row[f"{policy}_error_type"] = result.get("error_type", "")
        taxonomy.append(row)
    return taxonomy


def avg(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        try:
            if value not in {None, ""}:
                nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return (sum(nums) / len(nums)) if nums else None


def summarize_strata(manifest: Sequence[Mapping[str, str]], taxonomy: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result_by_case_policy = {(str(row["manifest_index"]), str(row["policy"])): row for row in results}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in taxonomy:
        groups[(str(row["source_config"]), str(row["length_bucket"]))].append(row)
    for row in taxonomy:
        groups[(str(row["source_config"]), "ALL")].append(row)
        groups[("ALL", str(row["length_bucket"]))].append(row)
        groups[("ALL", "ALL")].append(row)

    out: list[dict[str, Any]] = []
    for (source_config, bucket), rows in sorted(groups.items()):
        cases = len(rows)
        control_pass = sum(boolish(row["control_passed"]) for row in rows)
        deployable_pass = sum(boolish(row["deployable_passed"]) for row in rows)
        oracle_pass = sum(boolish(row["oracle_passed"]) for row in rows)
        idxs = [str(row["manifest_index"]) for row in rows]
        out.append(
            {
                "source_config": source_config,
                "length_bucket": bucket,
                "cases": cases,
                "control_fixed64_pass": control_pass,
                "best_deployable_cal_lite_pass": deployable_pass,
                "oracle_sufficient_canvas_pass": oracle_pass,
                "control_fixed64_rate": control_pass / cases if cases else None,
                "best_deployable_cal_lite_rate": deployable_pass / cases if cases else None,
                "oracle_sufficient_canvas_rate": oracle_pass / cases if cases else None,
                "oracle_gain_vs_control_cases": sum((not boolish(row["control_passed"])) and boolish(row["oracle_passed"]) for row in rows),
                "deployable_gain_vs_control_cases": sum((not boolish(row["control_passed"])) and boolish(row["deployable_passed"]) for row in rows),
                "deployable_harm_vs_control_cases": sum(boolish(row["control_passed"]) and not boolish(row["deployable_passed"]) for row in rows),
                "oracle_harm_vs_control_cases": sum(boolish(row["control_passed"]) and not boolish(row["oracle_passed"]) for row in rows),
                "rescue_limited_or_noncanvas_failures": sum(row["taxonomy"] == "rescue_limited_or_noncanvas_failure" for row in rows),
                "oracle_only_canvas_recoverable": sum(row["taxonomy"] == "oracle_only_canvas_recoverable" for row in rows),
                "hard_tail_candidates": sum(boolish(row["hard_tail_candidate"]) for row in rows),
                "avg_control_selected_canvas": avg(result_by_case_policy.get((idx, "control_fixed64"), {}).get("selected_canvas_tokens") for idx in idxs),
                "avg_deployable_selected_canvas": avg(result_by_case_policy.get((idx, "best_deployable_cal_lite_alpha006"), {}).get("selected_canvas_tokens") for idx in idxs),
                "avg_oracle_selected_canvas": avg(result_by_case_policy.get((idx, "oracle_sufficient_canvas"), {}).get("selected_canvas_tokens") for idx in idxs),
            }
        )
    return out


def diagnostic_verdict(overall: Mapping[str, Any]) -> str:
    cases = int(overall["cases"])
    control_pass = int(overall["control_fixed64_pass"])
    oracle_pass = int(overall["oracle_sufficient_canvas_pass"])
    deployable_pass = int(overall["best_deployable_cal_lite_pass"])
    oracle_gain = int(overall["oracle_gain_vs_control_cases"])
    control_rate = control_pass / cases
    if control_rate >= 0.95 and oracle_pass - control_pass <= 2:
        return "official_second_regime_first_pass_too_easy_weak_stress"
    if (control_pass < cases or deployable_pass < cases) and oracle_gain > 0:
        return "official_second_regime_nontrivial_failures_oracle_recovers_subset_hard_tail_needed"
    if control_pass < cases and oracle_gain == 0:
        return "official_second_regime_failures_not_canvas_recoverable_in_first_pass"
    return "official_second_regime_mixed_first_pass"


def run_gpu_diagnostic() -> dict[str, Any]:
    manifest_summary_path = MANIFEST_DIR / "summary.json"
    manifest_path = MANIFEST_DIR / "manifest.csv"
    if not manifest_summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run --mode manifest before --mode diagnostic")
    manifest_summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))
    if manifest_summary.get("verdict") != "official_second_regime_manifest_gate_passed":
        raise RuntimeError(f"Manifest gate did not pass: {manifest_summary.get('verdict')}")

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    manifest, tasks = load_manifest_tasks(manifest_path)
    set_global_seed(42)
    tokenizer, model = load_model_and_tokenizer(cfg_for_policy("fixed").model)

    all_results: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for policy_name, source in POLICIES:
        policy_rows = run_policy(policy_name, source, tasks, manifest, tokenizer, model)
        all_results.extend(policy_rows)

    taxonomy = build_failure_taxonomy(manifest, all_results)
    stratum_summary = summarize_strata(manifest, taxonomy, all_results)
    overall = next(row for row in stratum_summary if row["source_config"] == "ALL" and row["length_bucket"] == "ALL")
    verdict = diagnostic_verdict(overall)
    hard_tail_count = sum(boolish(row["hard_tail_candidate"]) for row in taxonomy)
    summary = {
        "verdict": verdict,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_dataset": "loubnabnl/humaneval_infilling",
        "manifest_dir": str(MANIFEST_DIR.relative_to(REPO)),
        "diagnostic_dir": str(DIAGNOSTIC_DIR.relative_to(REPO)),
        "model": "GSAI-ML/LLaDA-8B-Base",
        "policies": [policy for policy, _ in POLICIES],
        "case_count": int(overall["cases"]),
        "control_fixed64_pass": int(overall["control_fixed64_pass"]),
        "best_deployable_cal_lite_pass": int(overall["best_deployable_cal_lite_pass"]),
        "oracle_sufficient_canvas_pass": int(overall["oracle_sufficient_canvas_pass"]),
        "oracle_gain_vs_control_cases": int(overall["oracle_gain_vs_control_cases"]),
        "deployable_gain_vs_control_cases": int(overall["deployable_gain_vs_control_cases"]),
        "deployable_harm_vs_control_cases": int(overall["deployable_harm_vs_control_cases"]),
        "oracle_harm_vs_control_cases": int(overall["oracle_harm_vs_control_cases"]),
        "hard_tail_candidate_count": hard_tail_count,
        "near_ceiling_stop_rule_triggered": verdict == "official_second_regime_first_pass_too_easy_weak_stress",
        "hard_tail_next_step_required": verdict == "official_second_regime_nontrivial_failures_oracle_recovers_subset_hard_tail_needed",
        "frozen_controller_test_status": "sealed_not_touched",
        "frozen_controller_test_rows": 0,
        "wall_clock_sec": time.perf_counter() - wall_start,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
    }

    requested_manifest_fields = [
        "manifest_index",
        "source_config",
        "task_id",
        "prompt_id",
        "row_id",
        "task_group",
        "prefix_len_chars",
        "middle_len_chars",
        "suffix_len_chars",
        "prefix_len_tokens",
        "middle_len_tokens",
        "suffix_len_tokens",
        "length_bucket",
        "selected_canvas_metadata_available",
        "control_fixed_canvas_tokens",
        "deployable_policy",
        "deployable_probe_lengths",
        "oracle_canvas_tokens",
        "frozen_controller_test_row",
        "frozen_controller_test_exclusion_flag",
        "evaluator_smoke_selected",
        "evaluator_smoke_status",
        "row_key",
    ]
    write_csv(DIAGNOSTIC_DIR / "manifest.csv", manifest, requested_manifest_fields)
    write_csv(DIAGNOSTIC_DIR / "results.csv", all_results)
    write_csv(DIAGNOSTIC_DIR / "stratum_summary.csv", stratum_summary)
    write_csv(DIAGNOSTIC_DIR / "failure_taxonomy.csv", taxonomy)
    write_json(DIAGNOSTIC_DIR / "summary.json", summary)
    write_diagnostic_report(summary, stratum_summary)
    return summary


def write_diagnostic_report(summary: Mapping[str, Any], stratum_summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Official Second-Regime Bounded Diagnostic",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        "Regime: official `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight` from `loubnabnl/humaneval_infilling`.",
        "Frozen controller test: `sealed_not_touched`; manifest frozen-test rows `0`.",
        "",
        "## Overall",
        "",
        f"Cases: `{summary['case_count']}`.",
        f"Control fixed64 pass: `{summary['control_fixed64_pass']}/{summary['case_count']}`.",
        f"Best deployable cal-lite pass: `{summary['best_deployable_cal_lite_pass']}/{summary['case_count']}`.",
        f"Oracle-sufficient canvas pass: `{summary['oracle_sufficient_canvas_pass']}/{summary['case_count']}`.",
        f"Oracle gain vs control cases: `{summary['oracle_gain_vs_control_cases']}`.",
        f"Deployable gain vs control cases: `{summary['deployable_gain_vs_control_cases']}`.",
        f"Deployable harm vs control cases: `{summary['deployable_harm_vs_control_cases']}`.",
        f"Oracle harm vs control cases: `{summary['oracle_harm_vs_control_cases']}`.",
        f"Hard-tail candidates flagged in taxonomy: `{summary['hard_tail_candidate_count']}`.",
        "",
        "## By Stratum",
        "",
        "| Source config | Bucket | Cases | Control | Deployable | Oracle | Oracle gain | Hard-tail |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stratum_summary:
        if row["source_config"] == "ALL" and row["length_bucket"] == "ALL":
            continue
        if row["source_config"] == "ALL" or row["length_bucket"] == "ALL":
            continue
        lines.append(
            f"| `{row['source_config']}` | `{row['length_bucket']}` | `{row['cases']}` | "
            f"`{row['control_fixed64_pass']}` | `{row['best_deployable_cal_lite_pass']}` | "
            f"`{row['oracle_sufficient_canvas_pass']}` | `{row['oracle_gain_vs_control_cases']}` | "
            f"`{row['hard_tail_candidates']}` |"
        )
    lines.extend(
        [
            "",
            "## Stop-Rule Interpretation",
            "",
            "- If all policies are near ceiling, this first pass is weak stress and should stop without stronger claims.",
            "- If control/deployable failures exist and oracle recovers a nonzero subset, do not immediately full-run; use `failure_taxonomy.csv` hard-tail flags to build the next bounded manifest.",
            "- If oracle does not recover failures, official second-regime should be written as a scope boundary for the canvas-rescue claim.",
            "",
            "## Compact Outputs",
            "",
            "- `manifest.csv`",
            "- `results.csv`",
            "- `stratum_summary.csv`",
            "- `failure_taxonomy.csv`",
            "- `summary.json`",
            "- `report.md`",
        ]
    )
    (DIAGNOSTIC_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manifest", "diagnostic", "all"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = None
    if args.mode in {"manifest", "all"}:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
        manifest_summary = run_cpu_manifest_gate(tokenizer)
        print(json.dumps({"manifest_dir": str(MANIFEST_DIR), **manifest_summary}, indent=2, sort_keys=True))
        if manifest_summary["verdict"] != "official_second_regime_manifest_gate_passed":
            return
    if args.mode in {"diagnostic", "all"}:
        summary = run_gpu_diagnostic()
        print(json.dumps({"diagnostic_dir": str(DIAGNOSTIC_DIR), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
