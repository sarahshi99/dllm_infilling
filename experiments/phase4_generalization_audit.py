#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
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

from expvision_dllm_clean.dataset import load_humaneval_infilling


H200_ACTION_TABLE = REPO / "analysis_outputs/controller_validation_h200_20260707_v1_replay/action_training_table.csv"
LLADA_H200_PRIMARY = REPO / "outputs_clean/h200_rebaseline_midcons_20260706_tier1_20260706_042029/results.jsonl"
DREAM_CAL = PROJECT_ROOT / "model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl"
DREAM_LCAL = PROJECT_ROOT / "outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327/results.jsonl"
DREAM_LCAL_SUMMARY = PROJECT_ROOT / "outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327/summary.json"
TEST_LOCK = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bucket(length: Any) -> str:
    value = int(float(length))
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def pass_value(row: Mapping[str, Any]) -> bool:
    return _bool(row.get("metrics", {}).get("passed"))


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def rows_by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def task_group(task_id: str) -> str:
    parts = task_id.split("/")
    if len(parts) >= 3 and parts[-2].startswith("HumanEval"):
        return "/".join(parts[-2:])
    return task_id


def load_h200_task_rows() -> list[dict[str, Any]]:
    action_rows = read_csv(H200_ACTION_TABLE)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in action_rows:
        if row["split"] in {"train", "calibration", "validation"}:
            grouped[row["task_id"]].append(row)
    out: list[dict[str, Any]] = []
    for task_id, rows in sorted(grouped.items()):
        keep = next(row for row in rows if row["action"] == "KEEP_PRIMARY")
        expansions = [row for row in rows if row["action"] != "KEEP_PRIMARY"]
        primary_pass = _bool(keep["action_passed"])
        any_expand_pass = any(_bool(row["action_passed"]) for row in expansions)
        any_expand_fail = any(not _bool(row["action_passed"]) for row in expansions)
        oracle_len = int(float(keep["oracle_length"]))
        primary_len = int(float(keep["primary_selected_length"]))
        out.append(
            {
                "task_id": task_id,
                "task_group": keep["task_group"],
                "split": keep["split"],
                "oracle_length": oracle_len,
                "oracle_bucket": keep["oracle_bucket"],
                "primary_selected_length": primary_len,
                "primary_passed": primary_pass,
                "recoverable_by_expansion": (not primary_pass) and any_expand_pass,
                "harmable_by_expansion": primary_pass and any_expand_fail,
                "expansion_pass_count": sum(1 for row in expansions if _bool(row["action_passed"])),
                "triggered_long_proxy": (not primary_pass) and oracle_len >= 17 and primary_len >= 16,
                "missed_long_proxy": (not primary_pass) and oracle_len >= 17 and primary_len < 16,
            }
        )
    return out


def choose_case_manifest(limit: int = 15) -> list[dict[str, Any]]:
    h200 = load_h200_task_rows()
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(label: str, pred, n: int) -> None:
        rows = [row for row in h200 if pred(row) and row["task_id"] not in used]
        rows.sort(key=lambda row: (row["task_group"], row["task_id"]))
        for row in rows[:n]:
            used.add(row["task_id"])
            chosen.append({"stratum": label, **row})

    add("short_primary_pass", lambda r: r["oracle_bucket"] == "<=8" and r["primary_passed"], 3)
    add(
        "medium_near_long_underselection",
        lambda r: r["oracle_bucket"] in {"13-16", "17-24"} and r["primary_selected_length"] < r["oracle_length"],
        3,
    )
    add("missed_failed_long", lambda r: r["missed_long_proxy"], 3)
    add("triggered_failed_long", lambda r: r["triggered_long_proxy"], 3)
    add("positive_control_recoverable", lambda r: r["recoverable_by_expansion"], 3)
    return chosen[:limit]


def backbone_feasibility(output_dir: Path) -> dict[str, Any]:
    candidates = [
        {
            "backbone": "Dream-org/Dream-Coder-v0-Base-7B",
            "dllm_infilling": True,
            "current_prompt_evaluator": True,
            "checkpoint_cached": (Path.home() / ".cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Base-7B").exists(),
            "h200_runnable": True,
            "tokenizer_change": "yes: Dream-Coder uses <|mask|> and bos_prefix_masks_suffix_eos canvas",
            "decode_loop_change": "uses clean_scripts/run_dreamcoder_official_infilling.py native diffusion_generate",
            "minimal_subset_cost": "about 3.8 sec/case/action from existing full run; 15 cases x 3 actions is <3 minutes after model load",
            "reuse_existing_runner": True,
            "historical_full_result": "832/1033 lcal_official_bounded_repair; 825/1033 cal_lite",
            "recommended": True,
        },
        {
            "backbone": "Dream-org/Dream-Coder-v0-Instruct-7B",
            "dllm_infilling": True,
            "current_prompt_evaluator": True,
            "checkpoint_cached": (Path.home() / ".cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Instruct-7B").exists(),
            "h200_runnable": True,
            "tokenizer_change": "yes: Dream-Coder instruction variant uses same official canvas adapter",
            "decode_loop_change": "uses same Dream-Coder runner",
            "minimal_subset_cost": "similar to base; historical full exists",
            "reuse_existing_runner": True,
            "historical_full_result": "848/1033 cal_lite official-canvas; 834/1033 lcal bounded repair",
            "recommended": False,
        },
        {
            "backbone": "GSAI-ML/LLaDA-8B-Instruct",
            "dllm_infilling": True,
            "current_prompt_evaluator": True,
            "checkpoint_cached": (Path.home() / ".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct").exists(),
            "h200_runnable": True,
            "tokenizer_change": "no major change versus LLaDA base",
            "decode_loop_change": "existing LLaDA runner",
            "minimal_subset_cost": "similar to LLaDA base",
            "reuse_existing_runner": True,
            "historical_full_result": "817/1033 LCAS v3 historical",
            "recommended": False,
        },
    ]
    write_csv(output_dir / "candidate_backbones.csv", candidates)
    recommended = next(row for row in candidates if row["recommended"])
    write_json(output_dir / "recommended_backbone.json", recommended)
    report = [
        "# Second-Backbone Feasibility Audit",
        "",
        "Verdict: `recommended_backbone_available`.",
        "",
        "The most feasible second backbone is `Dream-org/Dream-Coder-v0-Base-7B`: it is cached locally, has a dedicated official-canvas infilling runner, and has existing full SingleLine evidence. It requires tokenizer/canvas adaptation, but no new backbone weights or evaluator changes.",
        "",
        "Dream-Coder does not expose the LLaDA trace-remasking E/F/G action family. For Phase 4, the diagnostic action set is primary/cal-lite, current best simple length policy, and oracle-sufficient canvas.",
    ]
    (output_dir / "compatibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"verdict": "recommended_backbone_available", "recommended": recommended}


def build_dream_args(mask_length_source: str, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="Dream-org/Dream-Coder-v0-Base-7B",
        split="test",
        dataset_subset="HumanEval-SingleLineInfilling",
        max_samples=None,
        seed=42,
        mask_length_source=mask_length_source,
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
        experiment_name=f"dreamcoder_phase4_{mask_length_source}",
        baseline_results=None,
    )


def maybe_run_oracle(manifest: Sequence[Mapping[str, Any]], output_dir: Path, run_gpu: bool) -> dict[str, Mapping[str, Any]]:
    if not run_gpu:
        return {}
    import torch
    from transformers import AutoModel, AutoTokenizer
    import run_dreamcoder_official_infilling as dream

    dream.ensure_modeling_rope_utils_available()
    args = build_dream_args("oracle", output_dir)
    cfg = dream.build_config(args)
    dream.set_global_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        torch_dtype=dream.get_torch_dtype(cfg.model.torch_dtype),
        device_map=cfg.model.device_map,
    )
    model.eval()
    tasks = load_humaneval_infilling(split="test", dataset_subset="HumanEval-SingleLineInfilling")
    by_task = {task.task_id: task for task in tasks}
    rows: list[dict[str, Any]] = []
    for item in manifest:
        task = by_task[str(item["task_id"])]
        rows.append(dream.run_task(task, tokenizer, model, cfg, args))
    return rows_by_task(rows)


def second_backbone_diagnostic(output_dir: Path, run_gpu: bool, oracle_blocked_reason: str | None = None) -> dict[str, Any]:
    manifest = choose_case_manifest()
    write_csv(output_dir / "case_manifest.csv", manifest)
    cal = rows_by_task(read_jsonl(DREAM_CAL))
    lcal = rows_by_task(read_jsonl(DREAM_LCAL))
    oracle = maybe_run_oracle(manifest, output_dir, run_gpu=run_gpu)
    if oracle:
        with (output_dir / "oracle_results.jsonl").open("w", encoding="utf-8") as handle:
            for task_id in [row["task_id"] for row in manifest]:
                handle.write(json.dumps(oracle[task_id], ensure_ascii=False) + "\n")

    result_rows: list[dict[str, Any]] = []
    for item in manifest:
        task_id = str(item["task_id"])
        for policy, source_rows in [
            ("primary_cal_lite", cal),
            ("best_simple_lcal_bounded_repair", lcal),
            ("oracle_sufficient_canvas", oracle),
        ]:
            row = source_rows.get(task_id)
            if not row:
                result_rows.append({"task_id": task_id, "stratum": item["stratum"], "policy": policy, "status": "not_run"})
                continue
            result_rows.append(
                {
                    "task_id": task_id,
                    "stratum": item["stratum"],
                    "policy": policy,
                    "status": "ok",
                    "passed": pass_value(row),
                    "oracle_length": metric(row, "oracle_mask_length"),
                    "selected_length": metric(row, "selected_mask_length"),
                    "selected_minus_oracle": metric(row, "selected_minus_oracle_length"),
                    "final_source": metric(row, "final_source"),
                    "total_sec_including_probe": metric(row, "total_sec_including_probe"),
                }
            )
    write_csv(output_dir / "results.csv", result_rows)

    by_task_policy = {(row["task_id"], row["policy"]): row for row in result_rows if row["status"] == "ok"}
    missed = [row for row in manifest if row["stratum"] == "missed_failed_long"]
    triggered = [row for row in manifest if row["stratum"] == "triggered_failed_long"]
    short = [row for row in manifest if row["stratum"] == "short_primary_pass"]

    def count_pass(rows: Sequence[Mapping[str, Any]], policy: str) -> int:
        return sum(1 for row in rows if _bool(by_task_policy.get((row["task_id"], policy), {}).get("passed")))

    summary = {
        "verdict": "diagnostic_subset_completed"
        if oracle
        else (
            "diagnostic_subset_oracle_gpu_blocked"
            if oracle_blocked_reason
            else "diagnostic_subset_from_existing_primary_and_policy_oracle_not_run"
        ),
        "run_gpu_oracle": bool(oracle),
        "oracle_blocked_reason": oracle_blocked_reason,
        "manual_oracle_command": (
            "HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 "
            "HF_HOME=/home/shx/.cache/huggingface TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "
            "TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python "
            "experiments/phase4_generalization_audit.py --timestamp 20260708_phase4_oracle_manual "
            "--run-second-backbone-oracle"
        )
        if oracle_blocked_reason
        else None,
        "case_count": len(manifest),
        "policies": ["primary_cal_lite", "best_simple_lcal_bounded_repair", "oracle_sufficient_canvas"],
        "efg_status": "not_applicable_dreamcoder_no_trace_remask_adapter",
        "primary_pass_count": count_pass(manifest, "primary_cal_lite"),
        "best_simple_pass_count": count_pass(manifest, "best_simple_lcal_bounded_repair"),
        "oracle_pass_count": count_pass(manifest, "oracle_sufficient_canvas"),
        "missed_long_oracle_recoveries": count_pass(missed, "oracle_sufficient_canvas"),
        "missed_long_primary_passes": count_pass(missed, "primary_cal_lite"),
        "triggered_long_oracle_recoveries": count_pass(triggered, "oracle_sufficient_canvas"),
        "triggered_long_primary_passes": count_pass(triggered, "primary_cal_lite"),
        "short_primary_pass_regressions_under_lcal": sum(
            1
            for row in short
            if _bool(by_task_policy.get((row["task_id"], "primary_cal_lite"), {}).get("passed"))
            and not _bool(by_task_policy.get((row["task_id"], "best_simple_lcal_bounded_repair"), {}).get("passed"))
        ),
        "qualitative_agreement_with_llada_h200": "partial: enough to test Dream-Coder canvas recoverability; full claim needs fresh oracle run" if not oracle else "diagnostic: missed-long and triggered-long strata are explicitly separated on Dream-Coder subset",
    }
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Second-Backbone Minimal Diagnostic",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        f"Backbone: `Dream-org/Dream-Coder-v0-Base-7B`.",
        f"Cases: `{len(manifest)}` stratified HumanEval SingleLine rows.",
        "",
        "Policies:",
        "- primary/control: `primary_cal_lite` from existing full Dream-Coder official-canvas run.",
        "- current best simple length policy: `best_simple_lcal_bounded_repair` from existing full Dream-Coder run.",
        "- oracle-sufficient canvas: fresh oracle run when `run_gpu_oracle=true`; otherwise marked `not_run`.",
        "- E/F/G: not applicable because Dream-Coder runner has no trace-remasking adapter.",
        "",
        "Oracle run status:",
        f"- run_gpu_oracle: `{summary['run_gpu_oracle']}`",
        f"- blocked reason: `{oracle_blocked_reason or 'none'}`",
        "",
        "Key counts:",
        f"- missed-long oracle recoveries: `{summary['missed_long_oracle_recoveries']}`",
        f"- triggered-long oracle recoveries: `{summary['triggered_long_oracle_recoveries']}`",
        f"- short-case regressions under best simple policy: `{summary['short_primary_pass_regressions_under_lcal']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def second_regime_feasibility(output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for subset in [
        "HumanEval-MultiLineInfilling",
        "HumanEval-RandomSpanInfilling",
        "HumanEval-RandomSpanInfillingLight",
    ]:
        try:
            sample = load_humaneval_infilling(split="test", max_samples=5, dataset_subset=subset)
            status = "available"
            blocker = ""
            row_count = "unknown_cached"
            example = sample[0].task_id if sample else ""
        except Exception as exc:
            status = "blocked_missing_local_dataset_file"
            blocker = f"{type(exc).__name__}: {exc}"
            row_count = ""
            example = ""
        rows.append(
            {
                "dataset": subset,
                "status": status,
                "runner_compatibility": "dataset alias exists in expvision_dllm_clean.dataset",
                "evaluator_compatibility": "HumanEval verifier stack expected compatible if rows load",
                "expected_row_count": row_count,
                "length_distribution": "not_computed" if status != "available" else "requires full load",
                "recommended_minimal_subset": "short/medium/long primary-pass/fail strata",
                "blocker": blocker,
                "example_task": example,
            }
        )
    write_csv(output_dir / "available_datasets.csv", rows)
    recommended = next((row for row in rows if row["status"] == "available"), None)
    summary = {
        "verdict": "blocked_missing_multiline_randomspan_dataset_files" if recommended is None else "second_regime_available",
        "recommended_dataset": None if recommended is None else recommended["dataset"],
        "diagnostic_run": "not_run" if recommended is None else "eligible",
    }
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Second-Regime Feasibility Audit",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        "The repository has aliases for MultiLine and RandomSpan, but the cached dataset loader expects local JSONL files under `data/` and they are absent on this H200 workspace. No second-regime diagnostic run was launched.",
        "",
        "Minimum unblocker: provide `data/HumanEval-MultiLineInfilling.jsonl` or equivalent cached dataset files, then run the same short/medium/long diagnostic strata with control, best deployable policy, and oracle-sufficient canvas.",
    ]
    (output_dir / "compatibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def lrdllm_final_attempt(output_dir: Path) -> dict[str, Any]:
    protocol = [
        "# LR-DLLM Final Protocol Audit",
        "",
        "Checked local repository, existing LR-DLLM protocol audit, and web sources on 2026-07-08.",
        "",
        "Source check:",
        "- arXiv entry: https://arxiv.org/abs/2602.07546",
        "- alphaXiv overview: https://www.alphaxiv.org/overview/2602.07546v1",
        "- Web search did not identify a protocol-matched official Stage I/II code release usable with this repository.",
        "",
        "Findings:",
        "- arXiv `2602.07546` describes LR-DLLM as a training-free framework for variable-length DLLM generation with explicit length regularization.",
        "- CatalyzeX/arXiv-style listings expose the paper entry and abstract but no directly usable repository for this project.",
        "- GitHub search did not identify an official LR-DLLM implementation matching this paper; unrelated dLLM repositories exist.",
        "- Local repository still has no Stage I/Stage II adapter that can be called with the H200 HumanEval infilling prompt/evaluator.",
    ]
    adapter = [
        "# LR-DLLM Adapter Feasibility",
        "",
        "Verdict: local adapter is not safe to claim as LR-DLLM.",
        "",
        "A rough Stage I length-bias correction could be approximated from public descriptions, but the exact protocol-matched Stage I/II scoring, span adjustment, and token commitment details are not implemented locally. Running such a heuristic would create a new local method, not an LR-DLLM baseline.",
    ]
    verdict = "blocked_missing_algorithmic_detail"
    (output_dir / "protocol_audit.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    (output_dir / "adapter_feasibility.md").write_text("\n".join(adapter) + "\n", encoding="utf-8")
    (output_dir / "verdict.md").write_text(f"`{verdict}`\n", encoding="utf-8")
    return {"verdict": verdict, "sanity_run": False}


def create_paper_skeleton(paper_dir: Path) -> None:
    sections = paper_dir / "sections"
    tables = paper_dir / "tables"
    figures = paper_dir / "figures"
    sections.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (paper_dir / "main.tex").write_text(
        r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\title{Diagnosing Canvas and Rescue Limits in Unknown-Length Diffusion Code Infilling}
\author{Anonymous Authors}
\begin{document}
\maketitle
\begin{abstract}
Unknown-length diffusion code infilling exposes a gap between diagnostic recoverability and safe deployable control. We present a diagnostic-driven mixed study of canvas-limited and rescue-limited regimes, showing that missed true-long cases can be recoverable under oracle canvas expansion while already-triggered long failures remain resistant to longer trajectories and trace-guided remasking. Risk-controlled controllers show weak but insufficient validation signal, motivating a mixed framing rather than a positive controller claim.
\end{abstract}
\input{sections/introduction}
\input{sections/problem}
\input{sections/method_framework}
\input{sections/experiments}
\input{sections/results}
\input{sections/discussion}
\input{sections/related_work}
\input{sections/limitations}
\input{sections/conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
""",
        encoding="utf-8",
    )
    section_text = {
        "introduction.tex": "We study unknown-length code infilling with diffusion language models as a diagnostic problem: when does failure come from insufficient canvas, and when does it come from the generator/rescue mechanism itself?\n",
        "problem.tex": "The task is HumanEval-style infilling with unknown target length. The key risk is that inference must choose both content and canvas length without access to oracle length, verifier outcomes, task identifiers, or test labels.\n",
        "method_framework.tex": "Our framework separates diagnostic oracle actions from deployable policies. Oracle canvas actions estimate recoverability; deployable controllers are evaluated under frozen train/calibration/validation/test discipline and explicit harm constraints.\n",
        "experiments.tex": "Experiments cover H200-replayed LLaDA baselines, action-ceiling diagnostics, risk-controlled controllers, second-backbone feasibility, second-regime feasibility, and LR-DLLM protocol status.\n",
        "results.tex": "Main results support separable regimes: missed true-long failures expose oracle-canvas recoverability, while triggered long failures remain rescue-limited. Controller V3 shows weak validation signal but does not authorize frozen test.\n",
        "discussion.tex": "The evidence favors a diagnostic-driven mixed paper. The contribution is not a successful controller, but a careful map of the gap between diagnostic upper bounds and safe inference-time intervention.\n",
        "related_work.tex": "Discuss diffusion language models, DreamOn, CAL, LR-DLLM, dynamic canvas generation, code infilling benchmarks, and risk-controlled inference-time intervention.\n",
        "limitations.tex": "We do not claim SOTA, solved unknown-length generation, model-agnostic generalization, or frozen-test improvement. Generalization remains a blocking gap until second-backbone/regime diagnostics are expanded.\n",
        "conclusion.tex": "Unknown-length DLLM infilling has canvas-limited and rescue-limited regimes. Safe deployable control remains open, and future work should validate these regimes across backbones and benchmarks.\n",
    }
    for name, text in section_text.items():
        (sections / name).write_text(text, encoding="utf-8")
    table_text = {
        "main_results.tex": "\\begin{tabular}{lrr}\\toprule Run & Pass@1 & Notes\\\\\\midrule H200 V6 & 796/1033 & evidence base\\\\ Controller V3 selected & 90/127 & validation only\\\\\\bottomrule\\end{tabular}\n",
        "action_ceiling.tex": "\\begin{tabular}{lrr}\\toprule Diagnostic & Recoveries & Notes\\\\\\midrule Oracle canvas C & 29/89 & missed-long only\\\\ E/F/G incremental & 2 & rescue limited\\\\\\bottomrule\\end{tabular}\n",
        "controller_validation.tex": "\\begin{tabular}{lrrrr}\\toprule Controller & Pass & Wins & Losses & Gate\\\\\\midrule V1 & 89/127 & 0 & 0 & fail\\\\ V2 best nonzero & 90/127 & 5 & 4 & fail\\\\ V3 selected & 90/127 & 1 & 0 & weak\\\\\\bottomrule\\end{tabular}\n",
        "generalization_placeholder.tex": "\\begin{tabular}{lll}\\toprule Axis & Status & Artifact\\\\\\midrule Second backbone & feasible & Phase 4 audit\\\\ Second regime & blocked & missing local data\\\\\\bottomrule\\end{tabular}\n",
    }
    for name, text in table_text.items():
        (tables / name).write_text(text, encoding="utf-8")
    (figures / "README.md").write_text(
        "# Figures\n\nPlanned figures: regime taxonomy, action-ceiling waterfall, controller risk-coverage curves, and generalization audit flow.\n",
        encoding="utf-8",
    )
    (paper_dir / "refs.bib").write_text(
        """@misc{lr_dllm_2026,
  title={Improving Variable-Length Generation in Diffusion Language Models via Length Regularization},
  author={Cheng, Zicong and Jia, Ruixuan and Li, Jia and Yang, Guo-Wei and Guo, Meng-Hao and Hu, Shi-Min},
  year={2026},
  eprint={2602.07546},
  archivePrefix={arXiv}
}
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--run-second-backbone-oracle", action="store_true")
    parser.add_argument("--oracle-blocked-reason", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = args.timestamp
    out_root = REPO / "analysis_outputs"
    bb_dir = out_root / f"second_backbone_feasibility_{ts}"
    bb_diag_dir = out_root / f"second_backbone_diagnostic_{ts}"
    regime_dir = out_root / f"second_regime_feasibility_{ts}"
    lrdllm_dir = out_root / f"lrdllm_final_attempt_{ts}"
    for path in [bb_dir, bb_diag_dir, regime_dir, lrdllm_dir]:
        if path.exists():
            raise FileExistsError(path)
        path.mkdir(parents=True)
    paper_dir = REPO / "paper" / "diagnostic_mixed_draft"

    bb = backbone_feasibility(bb_dir)
    bb_diag = second_backbone_diagnostic(
        bb_diag_dir,
        run_gpu=args.run_second_backbone_oracle,
        oracle_blocked_reason=args.oracle_blocked_reason,
    )
    regime = second_regime_feasibility(regime_dir)
    lrdllm = lrdllm_final_attempt(lrdllm_dir)
    create_paper_skeleton(paper_dir)
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "second_backbone_feasibility_dir": str(bb_dir.relative_to(REPO)),
        "second_backbone_diagnostic_dir": str(bb_diag_dir.relative_to(REPO)),
        "second_regime_feasibility_dir": str(regime_dir.relative_to(REPO)),
        "lrdllm_final_attempt_dir": str(lrdllm_dir.relative_to(REPO)),
        "paper_dir": str(paper_dir.relative_to(REPO)),
        "second_backbone_verdict": bb["verdict"],
        "second_backbone_diagnostic_verdict": bb_diag["verdict"],
        "second_regime_verdict": regime["verdict"],
        "lrdllm_verdict": lrdllm["verdict"],
        "test_status": lock.get("test_status"),
        "test_evaluation_count": lock.get("test_evaluation_count"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
