#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
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

from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling


CONSOLIDATION_DIR = REPO / "analysis_outputs/paper_evidence_consolidation_20260710_v1"
DREAM_FULL_DIR = REPO / "analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1"
DREAM_EXPANDED_DIR = REPO / "analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1"
ATTRIBUTION_SUMMARY = REPO / "analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/summary.json"
OFFICIAL_120_SUMMARY = REPO / "analysis_outputs/second_regime_official_diagnostic_20260708_v1/summary.json"
OFFICIAL_FULL_SUMMARY = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/summary.json"
OFFICIAL_FULL_CONFIG = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/config_summary.csv"
OFFICIAL_FULL_LENGTH = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/length_bucket_summary.csv"
OFFICIAL_FULL_TAXONOMY = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/taxonomy_summary.csv"
OFFICIAL_FULL_HARM = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/harm_summary.csv"
HARD_TAIL_104_SUMMARY = REPO / "analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/summary.json"
CONTROLLER_ROUTE_SUMMARY = REPO / "analysis_outputs/controller_route_closure_20260708_v1/summary.json"
LRDLLM_VERDICT = REPO / "analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/verdict.md"
TEST_LOCK = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
GROUPED_TEST_TASKS = REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def append_csv_row(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()


def md_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    labels = [field.replace("_", " ") for field in fields]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("\n", "<br>") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pct(num: int | float, den: int | float) -> str:
    return f"{100.0 * float(num) / float(den):.2f}%" if den else "n/a"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def task_group(task_id: str) -> str:
    parts = task_id.split("/")
    if "HumanEval" in parts:
        idx = parts.index("HumanEval")
        if idx + 1 < len(parts):
            return f"HumanEval/{parts[idx + 1]}"
    return task_id


def prompt_id(task_id: str) -> str:
    parts = task_id.split("/")
    if "HumanEval" in parts:
        idx = parts.index("HumanEval")
        if idx + 2 < len(parts):
            return "/".join(parts[idx + 2 :])
    return parts[-1] if parts else task_id


def load_frozen_groups() -> set[str]:
    groups: set[str] = set()
    if GROUPED_TEST_TASKS.exists():
        groups.update(str(item) for item in json.loads(GROUPED_TEST_TASKS.read_text(encoding="utf-8")))
    if TEST_LOCK.exists():
        lock = read_json(TEST_LOCK)
        groups.update(str(item) for item in lock.get("test_task_ids", []))
    return groups


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


def length_bucket(length: Any) -> str:
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


def avg(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        try:
            if value not in {None, ""}:
                nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else None


def generate_claim_matrix() -> list[dict[str, str]]:
    return [
        {
            "evidence": "LLaDA/H200 SingleLine",
            "supports": "Strong mechanism evidence: oracle-sufficient canvas recovers 29/89 hard cases; missed failed-long recovers 29 while triggered failed-long recovers 0 under C/E/F/G.",
            "limits_contradicts": "Deployable selectors remain weak; H200 drift is accepted but must stay separated from A6000 history.",
            "allowed_claim": "Unknown-length infilling has separable canvas inadequacy and rescue inadequacy on H200 SingleLine.",
            "forbidden_claim": "Do not claim a deployed controller improves sealed test or that triggered-long is solved.",
            "paper_placement": "Main mechanism result and motivation figure.",
        },
        {
            "evidence": "Dream-Coder expanded37",
            "supports": "Second-backbone oracle canvas improves 11/37 to 26/37 with 0 short regressions.",
            "limits_contradicts": "Missed/triggered split is model-dependent; triggered_failed_long proxy recovers 4/4, unlike LLaDA H200 triggered-long.",
            "allowed_claim": "Oracle-canvas recoverability transfers as diagnostic evidence to Dream-Coder.",
            "forbidden_claim": "Do not claim model-agnostic confirmation of the LLaDA missed-vs-triggered split.",
            "paper_placement": "Generalization audit / external validity subsection.",
        },
        {
            "evidence": "official second-regime 120 first-pass",
            "supports": "Official first-pass is nontrivial: control 34/120, deployable 36/120, oracle 49/120; oracle gain 26.",
            "limits_contradicts": "Deployable harm 15 and oracle harm 11; hard-tail follow-up required.",
            "allowed_claim": "Official MultiLine/RandomSpan first-pass supports mixed stress evidence.",
            "forbidden_claim": "Do not present it as near-ceiling or as positive controller evidence.",
            "paper_placement": "Official second-regime first-pass estimate.",
        },
        {
            "evidence": "official second-regime full allowed 6707",
            "supports": "Major official milestone: control 2019/6707, deployable 1464/6707, oracle 3180/6707; oracle gain 1633.",
            "limits_contradicts": "Deployable harms 1175 cases; oracle harms 472; rescue/non-canvas-limited 3055.",
            "allowed_claim": "Full official allowed population strengthens the mixed diagnostic claim.",
            "forbidden_claim": "Do not authorize deployable controller claims from cal-lite; do not call this frozen-test performance.",
            "paper_placement": "Main official stress result table.",
        },
        {
            "evidence": "official hard-tail full104",
            "supports": "Fixed post-first-pass stress taxonomy: control 18/104, deployable 20/104, oracle 33/104; 26 genuine canvas-recoverable.",
            "limits_contradicts": "60 rescue/non-canvas-limited; deployable harm 15; oracle harm 11.",
            "allowed_claim": "Hard-tail confirms mixed stress taxonomy and robustness of conclusion.",
            "forbidden_claim": "Do not use full104 as an unbiased benchmark aggregate.",
            "paper_placement": "Failure taxonomy appendix or main stress table.",
        },
        {
            "evidence": "Controller V1/V2/V3 route closure",
            "supports": "Oracle action-bank headroom exists, but selected deployable policies do not clear validation/frozen-test gates.",
            "limits_contradicts": "V1 zero intervention; V2 best nonzero 5/4 wins/losses with 7.06% upper95 harm; V3 top-k too weak for frozen-test authorization.",
            "allowed_claim": "Risk-controlled deployable selection is unresolved under sealed-test discipline.",
            "forbidden_claim": "Do not claim positive controller success or run Controller V4 without new evidence.",
            "paper_placement": "Negative controller evidence / scope boundary.",
        },
        {
            "evidence": "LR-DLLM blocked",
            "supports": "Documents a baseline gap transparently.",
            "limits_contradicts": "No protocol-matched official/local Stage I/II adapter; generation not executed.",
            "allowed_claim": "LR-DLLM is blocked by missing reproducible protocol details in this repository.",
            "forbidden_claim": "Do not claim an official LR-DLLM reproduction or invented adapter result.",
            "paper_placement": "Baseline limitations and reproducibility paragraph.",
        },
    ]


def generate_main_results() -> list[dict[str, Any]]:
    attr = read_json(ATTRIBUTION_SUMMARY)
    dream37 = read_json(DREAM_EXPANDED_DIR / "summary.json")
    first120 = read_json(OFFICIAL_120_SUMMARY)
    full = read_json(OFFICIAL_FULL_SUMMARY)
    hard104 = read_json(HARD_TAIL_104_SUMMARY)
    route = read_json(CONTROLLER_ROUTE_SUMMARY)
    return [
        {
            "result": "LLaDA/H200 SingleLine baselines and attribution",
            "n": attr["hard_case_count"],
            "control_or_primary": "H200 Tier1: control 787/1033; midcons 794/1033; route2 795/1033; V6 796/1033",
            "deployable_or_simple": "Controller candidates fail frozen-test gate",
            "oracle_or_upper_bound": "C oracle canvas 29/89 hard recoveries; E/F/G any 31/89",
            "key_delta": "missed failed-long C recovery 29; triggered failed-long C/E/F/G recovery 0",
            "claim_status": "main mechanism evidence",
        },
        {
            "result": "Dream-Coder expanded37",
            "n": dream37["case_count"],
            "control_or_primary": f"{dream37['primary_control_pass']}/{dream37['case_count']}",
            "deployable_or_simple": f"{dream37['best_simple_policy_pass']}/{dream37['case_count']}",
            "oracle_or_upper_bound": f"{dream37['oracle_sufficient_canvas_pass']}/{dream37['case_count']}",
            "key_delta": "oracle +15 vs primary; 0 short regressions",
            "claim_status": "mixed second-backbone diagnostic",
        },
        {
            "result": "Official second-regime 120 first-pass",
            "n": first120["case_count"],
            "control_or_primary": f"{first120['control_fixed64_pass']}/{first120['case_count']}",
            "deployable_or_simple": f"{first120['best_deployable_cal_lite_pass']}/{first120['case_count']}",
            "oracle_or_upper_bound": f"{first120['oracle_sufficient_canvas_pass']}/{first120['case_count']}",
            "key_delta": f"oracle gain {first120['oracle_gain_vs_control_cases']}; deployable harm {first120['deployable_harm_vs_control_cases']}",
            "claim_status": "unbiased official first-pass diagnostic estimate",
        },
        {
            "result": "Official second-regime full allowed",
            "n": full["case_count"],
            "control_or_primary": f"{full['control_fixed64_pass']}/{full['case_count']} ({pct(full['control_fixed64_pass'], full['case_count'])})",
            "deployable_or_simple": f"{full['best_deployable_cal_lite_pass']}/{full['case_count']} ({pct(full['best_deployable_cal_lite_pass'], full['case_count'])})",
            "oracle_or_upper_bound": f"{full['oracle_sufficient_canvas_pass']}/{full['case_count']} ({pct(full['oracle_sufficient_canvas_pass'], full['case_count'])})",
            "key_delta": f"oracle gain {full['oracle_gain_vs_control_cases']}; deployable harm {full['deployable_harm_vs_control_cases']}; rescue/non-canvas {full['rescue_limited_or_noncanvas_cases']}",
            "claim_status": "major mixed-stress evidence milestone",
        },
        {
            "result": "Official hard-tail full104",
            "n": hard104["case_count"],
            "control_or_primary": f"{hard104['control_fixed64_pass']}/{hard104['case_count']}",
            "deployable_or_simple": f"{hard104['best_deployable_cal_lite_pass']}/{hard104['case_count']}",
            "oracle_or_upper_bound": f"{hard104['oracle_sufficient_canvas_pass']}/{hard104['case_count']}",
            "key_delta": f"canvas-recoverable {hard104['genuinely_canvas_recoverable_now']}; rescue/non-canvas {hard104['rescue_limited_or_noncanvas_now']}",
            "claim_status": "post-first-pass fixed stress taxonomy",
        },
        {
            "result": "Controller route closure",
            "n": route["rows"],
            "control_or_primary": "V1/V2/V3 validation-only",
            "deployable_or_simple": "No gate-passing frozen-test policy",
            "oracle_or_upper_bound": "H200 action-bank upper bound 106/127 validation",
            "key_delta": "V3 exploratory 1/0 wins/losses too weak for frozen-test gate",
            "claim_status": "negative/diagnostic controller evidence",
        },
    ]


def generate_failure_taxonomy() -> list[dict[str, Any]]:
    full = read_json(OFFICIAL_FULL_SUMMARY)
    config_rows = {row["source_config"]: row for row in read_csv(OFFICIAL_FULL_CONFIG)}
    length_rows = {row["length_bucket"]: row for row in read_csv(OFFICIAL_FULL_LENGTH)}
    tax_rows = read_csv(OFFICIAL_FULL_TAXONOMY)

    def tax_count(name: str) -> int:
        for row in tax_rows:
            if row["source_config"] == "ALL" and row["length_bucket"] == "ALL" and row["taxonomy"] == name:
                return int(row["cases"])
        return 0

    random_span = config_rows["HumanEval-RandomSpanInfilling"]
    extreme = length_rows["extreme"]
    return [
        {
            "category": "canvas-recoverable",
            "count_or_rate": f"{full['oracle_gain_vs_control_cases']} cases ({pct(full['oracle_gain_vs_control_cases'], full['case_count'])})",
            "evidence": "Full allowed official oracle pass exceeds control on failures.",
            "paper_use": "Supports canvas inadequacy as a real official-regime phenomenon.",
        },
        {
            "category": "rescue/non-canvas-limited",
            "count_or_rate": f"{full['rescue_limited_or_noncanvas_cases']} cases ({pct(full['rescue_limited_or_noncanvas_cases'], full['case_count'])})",
            "evidence": "Control fails and oracle canvas also fails.",
            "paper_use": "Bounds canvas-only claims and motivates rescue adequacy.",
        },
        {
            "category": "deployable help",
            "count_or_rate": f"{full['deployable_help_vs_control_cases']} cases ({pct(full['deployable_help_vs_control_cases'], full['case_count'])})",
            "evidence": "Cal-lite recovers some control failures.",
            "paper_use": "Shows deployable signal exists but is not reliable enough.",
        },
        {
            "category": "deployable harm",
            "count_or_rate": f"{full['deployable_harm_vs_control_cases']} cases ({pct(full['deployable_harm_vs_control_cases'], full['case_count'])})",
            "evidence": "Cal-lite loses many control-passing cases.",
            "paper_use": "Forbids positive deployable controller claims.",
        },
        {
            "category": "oracle harm",
            "count_or_rate": f"{full['oracle_harm_vs_control_cases']} cases ({pct(full['oracle_harm_vs_control_cases'], full['case_count'])})",
            "evidence": "Even oracle-length canvas can harm some control-passing cases.",
            "paper_use": "Marks rescue/generation fragility separate from length selection.",
        },
        {
            "category": "RandomSpan scope boundary",
            "count_or_rate": f"rescue/non-canvas {random_span['rescue_limited_or_noncanvas_cases']}/{random_span['cases']} ({pct(int(random_span['rescue_limited_or_noncanvas_cases']), int(random_span['cases']))})",
            "evidence": "RandomSpan has the highest rescue/non-canvas fraction among configs.",
            "paper_use": "Write RandomSpan as a harder semantic/rescue boundary.",
        },
        {
            "category": "extreme-length scope boundary",
            "count_or_rate": f"rescue/non-canvas {extreme['rescue_limited_or_noncanvas_cases']}/{extreme['cases']} ({pct(int(extreme['rescue_limited_or_noncanvas_cases']), int(extreme['cases']))})",
            "evidence": f"Extreme bucket control {extreme['control_fixed64_pass']}/{extreme['cases']}, deployable {extreme['best_deployable_cal_lite_pass']}/{extreme['cases']}, oracle {extreme['oracle_sufficient_canvas_pass']}/{extreme['cases']}.",
            "paper_use": "Use as length-tail limitation, not as controller success evidence.",
        },
        {
            "category": "taxonomy split",
            "count_or_rate": f"oracle_only {tax_count('oracle_only_canvas_recoverable')}; deployable_and_oracle {tax_count('deployable_and_oracle_recover_control_failure')}; rescue/non-canvas {tax_count('rescue_limited_or_noncanvas_failure')}",
            "evidence": "Full allowed taxonomy summary.",
            "paper_use": "Feeds failure taxonomy table/figure.",
        },
    ]


def write_official_second_regime_writeup() -> None:
    full = read_json(OFFICIAL_FULL_SUMMARY)
    first120 = read_json(OFFICIAL_120_SUMMARY)
    hard104 = read_json(HARD_TAIL_104_SUMMARY)
    config_rows = [row for row in read_csv(OFFICIAL_FULL_CONFIG) if row["source_config"] != "ALL"]
    lines = [
        "# Official Second-Regime Writeup",
        "",
        "The full allowed official second-regime diagnostic is the strongest official-regime evidence now available. It includes all non-frozen rows from `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight`: `6707` cases and `0` frozen-controller-test rows.",
        "",
        f"Overall, control fixed64 passes `{full['control_fixed64_pass']}/{full['case_count']}` (`{pct(full['control_fixed64_pass'], full['case_count'])}`), best deployable cal-lite passes `{full['best_deployable_cal_lite_pass']}/{full['case_count']}` (`{pct(full['best_deployable_cal_lite_pass'], full['case_count'])}`), and oracle-sufficient canvas passes `{full['oracle_sufficient_canvas_pass']}/{full['case_count']}` (`{pct(full['oracle_sufficient_canvas_pass'], full['case_count'])}`). Oracle canvas recovers `{full['oracle_gain_vs_control_cases']}` control failures, but deployable cal-lite harms `{full['deployable_harm_vs_control_cases']}` control-passing cases and oracle canvas harms `{full['oracle_harm_vs_control_cases']}` control-passing cases. `{full['rescue_limited_or_noncanvas_cases']}` cases remain rescue/non-canvas-limited.",
        "",
        "## Config-Level Analysis",
        "",
    ]
    for row in config_rows:
        cases = int(row["cases"])
        lines.append(
            f"- `{row['source_config']}`: control `{row['control_fixed64_pass']}/{cases}` (`{pct(int(row['control_fixed64_pass']), cases)}`), deployable `{row['best_deployable_cal_lite_pass']}/{cases}` (`{pct(int(row['best_deployable_cal_lite_pass']), cases)}`), oracle `{row['oracle_sufficient_canvas_pass']}/{cases}` (`{pct(int(row['oracle_sufficient_canvas_pass']), cases)}`), oracle gain `{row['oracle_gain_vs_control_cases']}`, rescue/non-canvas `{row['rescue_limited_or_noncanvas_cases']}`, deployable harm `{row['deployable_harm_vs_control_cases']}`, oracle harm `{row['oracle_harm_vs_control_cases']}`."
        )
    lines.extend(
        [
            "",
            "MultiLine supplies most absolute canvas recoveries. RandomSpanLight is small but has strong oracle gain. RandomSpan is the clearest scope boundary: its rescue/non-canvas-limited rate is highest, so random spans should be discussed as stressing semantic/rescue adequacy beyond canvas length.",
            "",
            "## Comparison To First-Pass And Hard-Tail",
            "",
            f"The 120-case first-pass remains the preregistered/unbiased official diagnostic estimate: control `{first120['control_fixed64_pass']}/120`, deployable `{first120['best_deployable_cal_lite_pass']}/120`, oracle `{first120['oracle_sufficient_canvas_pass']}/120`, oracle gain `{first120['oracle_gain_vs_control_cases']}`.",
            f"The full allowed population strengthens rather than reverses that estimate: control remains low (`{pct(full['control_fixed64_pass'], full['case_count'])}`), oracle is materially higher (`{pct(full['oracle_sufficient_canvas_pass'], full['case_count'])}`), and deployable cal-lite underperforms control.",
            f"The full104 hard-tail is a post-first-pass fixed stress/taxonomy set, not an unbiased benchmark aggregate: control `{hard104['control_fixed64_pass']}/104`, deployable `{hard104['best_deployable_cal_lite_pass']}/104`, oracle `{hard104['oracle_sufficient_canvas_pass']}/104`, genuine canvas-recoverable `{hard104['genuinely_canvas_recoverable_now']}`, rescue/non-canvas `{hard104['rescue_limited_or_noncanvas_now']}`.",
            "",
            "## Allowed Claims",
            "",
            "- Official second-regime contains real canvas-recoverable failures.",
            "- Full allowed official population strengthens a mixed diagnostic paper.",
            "- RandomSpan and extreme-length cases expose rescue/non-canvas limitations and harm risk.",
            "- Deployable cal-lite is a scope-boundary result, not a successful controller.",
            "",
            "## Forbidden Claims",
            "",
            "- Do not claim deployable controller success.",
            "- Do not claim frozen-test performance.",
            "- Do not treat hard-tail full104 as an unbiased official benchmark aggregate.",
            "- Do not hide oracle/deployable harm cases.",
        ]
    )
    (CONSOLIDATION_DIR / "official_second_regime_writeup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_claim_rewrite() -> None:
    text = """# Paper Claim Rewrite

## Final Central Claim

Unknown-length diffusion language model infilling exhibits a measurable gap between canvas adequacy and rescue adequacy. Oracle-sufficient canvas can recover many failures across H200 SingleLine, Dream-Coder, and official second-regime data, but deployable length/canvas selection remains harm-prone and rescue/non-canvas limitations dominate substantial hard-tail, RandomSpan, and extreme-length cases.

## Three Main Contributions

1. A diagnostic decomposition of unknown-length infilling failures into canvas inadequacy, rescue/non-canvas limitations, and selection harm.
2. A sealed-test-preserving evidence base spanning H200 SingleLine attribution, Dream-Coder second-backbone audit, and official MultiLine/RandomSpan full allowed stress evidence.
3. A negative but actionable controller route closure showing that oracle action-bank headroom does not yet translate into a safe deployable controller under validation harm gates.

## Abstract Draft

Diffusion language models can infill code when the missing span length is known, but practical infilling requires deciding how much canvas to allocate without seeing the reference. We present a diagnostic study of unknown-length code infilling that separates canvas inadequacy from rescue inadequacy. On H200 SingleLine HumanEval infilling, oracle-sufficient canvas recovers many missed long-span failures, while already-triggered long failures remain rescue-limited under longer-trajectory and trace-remasking actions. A second-backbone Dream-Coder audit confirms substantial oracle-canvas recoverability but shows that the missed/triggered boundary is model-dependent. On the full allowed official MultiLine/RandomSpan infilling population, control fixed64 passes 2019/6707 cases, deployable cal-lite passes 1464/6707, and oracle-sufficient canvas passes 3180/6707, strengthening the mixed diagnostic claim while exposing deployable harm and rescue/non-canvas limitations. We close the current controller route as negative evidence: validation-visible selectors cannot safely harvest oracle headroom under sealed-test discipline. The result is not a new positive controller, but a claim-bounded map of where canvas expansion helps, where it harms, and where future rescue mechanisms are needed.

## Intro Contribution Bullets

- We quantify the canvas-rescue split for unknown-length DLLM code infilling and show that oracle canvas is a strong diagnostic tool but not a deployable method.
- We provide official second-regime stress evidence over 6707 non-frozen MultiLine/RandomSpan rows, showing both large oracle recoverability and substantial rescue/harm boundaries.
- We audit generalization on Dream-Coder and find recoverability transfers but qualitative strata are model-dependent.
- We preserve frozen-test discipline and report Controller V1/V2/V3 as route-closure evidence rather than overstating a weak deployable selector.

## Forbidden Claims

- No positive deployable controller claim.
- No frozen-test performance claim.
- No model-agnostic claim that Dream-Coder replicates the LLaDA missed-vs-triggered split.
- No official LR-DLLM reproduction claim.
- No synthetic easy-regime benchmark claim.
- No claim that hard-tail full104 is an unbiased benchmark aggregate.
"""
    (CONSOLIDATION_DIR / "paper_claim_rewrite.md").write_text(text, encoding="utf-8")


def run_consolidation() -> dict[str, Any]:
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    claim_rows = generate_claim_matrix()
    claim_fields = ["evidence", "supports", "limits_contradicts", "allowed_claim", "forbidden_claim", "paper_placement"]
    write_csv(CONSOLIDATION_DIR / "claim_matrix.csv", claim_rows, claim_fields)
    (CONSOLIDATION_DIR / "claim_matrix.md").write_text("# Claim Matrix\n\n" + md_table(claim_rows, claim_fields), encoding="utf-8")

    result_rows = generate_main_results()
    result_fields = ["result", "n", "control_or_primary", "deployable_or_simple", "oracle_or_upper_bound", "key_delta", "claim_status"]
    write_csv(CONSOLIDATION_DIR / "main_results_table.csv", result_rows, result_fields)
    (CONSOLIDATION_DIR / "main_results_table.md").write_text("# Main Results Table\n\n" + md_table(result_rows, result_fields), encoding="utf-8")

    failure_rows = generate_failure_taxonomy()
    failure_fields = ["category", "count_or_rate", "evidence", "paper_use"]
    write_csv(CONSOLIDATION_DIR / "failure_taxonomy_table.csv", failure_rows, failure_fields)
    (CONSOLIDATION_DIR / "failure_taxonomy_table.md").write_text("# Failure Taxonomy Table\n\n" + md_table(failure_rows, failure_fields), encoding="utf-8")

    write_official_second_regime_writeup()
    write_paper_claim_rewrite()
    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_dir": str(CONSOLIDATION_DIR.relative_to(REPO)),
        "claim_matrix_rows": len(claim_rows),
        "main_results_rows": len(result_rows),
        "failure_taxonomy_rows": len(failure_rows),
        "verdict": "paper_evidence_consolidation_completed",
        "next_step": "web_review_claim_boundaries_and_table_placement",
    }
    write_json(CONSOLIDATION_DIR / "summary.json", summary)
    return summary


DREAM_RESULT_FIELDS = [
    "manifest_index",
    "task_id",
    "task_group",
    "prompt_id",
    "length_bucket",
    "policy",
    "status",
    "passed",
    "oracle_length",
    "selected_length",
    "selected_minus_oracle",
    "abs_selected_minus_oracle",
    "mask_length_source",
    "final_source",
    "trigger_reason",
    "compile_passed",
    "error_type",
    "error_message",
    "candidate_sha256",
    "decode_sec",
    "verification_sec",
    "length_probe_sec",
    "total_sec_including_probe",
]


def dream_args(output_dir: Path, policy: str) -> SimpleNamespace:
    mask_length_source = {
        "primary_control": "cal_lite",
        "best_simple_length_policy": "lcal_official_bounded_repair",
        "oracle_sufficient_canvas": "oracle",
    }[policy]
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
        experiment_name="dreamcoder_full_allowed_singleline",
        baseline_results=None,
    )


def load_dream_model(output_dir: Path):
    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import run_dreamcoder_official_infilling as dream

    dream.ensure_modeling_rope_utils_available()
    args = dream_args(output_dir, "oracle_sufficient_canvas")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(
        "modeling_dream.DreamModel",
        args.model_path,
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
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dream.get_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    return dream, tokenizer, model


def build_dream_manifest(tokenizer: Any) -> list[dict[str, Any]]:
    DREAM_FULL_DIR.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_groups()
    tasks = load_humaneval_infilling(split="test", dataset_subset="HumanEval-SingleLineInfilling")
    manifest: list[dict[str, Any]] = []
    for task in tasks:
        group = task_group(task.task_id)
        if group in frozen:
            continue
        oracle = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
        oracle_len = int(oracle) if oracle is not None else ""
        manifest.append(
            {
                "manifest_index": len(manifest),
                "task_id": task.task_id,
                "task_group": group,
                "prompt_id": prompt_id(task.task_id),
                "oracle_length": oracle_len,
                "length_bucket": length_bucket(oracle_len),
                "frozen_controller_test_row": False,
                "frozen_controller_test_exclusion_flag": "included_not_frozen_controller_test",
                "source_dataset": "HumanEval-SingleLineInfilling",
                "model": "Dream-org/Dream-Coder-v0-Base-7B",
            }
        )
    write_csv(DREAM_FULL_DIR / "manifest.csv", manifest)
    return manifest


def compact_dream_result(manifest_row: Mapping[str, Any], policy: str, row: Mapping[str, Any]) -> dict[str, Any]:
    compile_passed, error_type, error_message = verification_error(row)
    return {
        "manifest_index": manifest_row["manifest_index"],
        "task_id": manifest_row["task_id"],
        "task_group": manifest_row["task_group"],
        "prompt_id": manifest_row["prompt_id"],
        "length_bucket": manifest_row["length_bucket"],
        "policy": policy,
        "status": "ok",
        "passed": pass_value(row),
        "oracle_length": metric(row, "oracle_mask_length"),
        "selected_length": metric(row, "selected_mask_length"),
        "selected_minus_oracle": metric(row, "selected_minus_oracle_length"),
        "abs_selected_minus_oracle": metric(row, "abs_selected_minus_oracle_length"),
        "mask_length_source": metric(row, "mask_length_source"),
        "final_source": metric(row, "final_source"),
        "trigger_reason": metric(row, "trigger_reason"),
        "compile_passed": compile_passed,
        "error_type": error_type,
        "error_message": error_message[:200],
        "candidate_sha256": code_hash(row),
        "decode_sec": metric(row, "decode_sec"),
        "verification_sec": metric(row, "verification_sec"),
        "length_probe_sec": metric(row, "length_probe_sec"),
        "total_sec_including_probe": metric(row, "total_sec_including_probe"),
    }


def completed_dream_keys() -> set[tuple[str, str]]:
    path = DREAM_FULL_DIR / "results.csv"
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {(row["manifest_index"], row["policy"]) for row in read_csv(path)}


def run_dream_policies() -> None:
    import run_dreamcoder_official_infilling as dream

    dream_mod, tokenizer, model = load_dream_model(DREAM_FULL_DIR)
    manifest_path = DREAM_FULL_DIR / "manifest.csv"
    manifest = read_csv(manifest_path) if manifest_path.exists() else build_dream_manifest(tokenizer)
    if any(boolish(row.get("frozen_controller_test_row")) for row in manifest):
        raise RuntimeError("Refusing to run Dream-Coder full allowed manifest with frozen rows")
    tasks = load_humaneval_infilling(split="test", dataset_subset="HumanEval-SingleLineInfilling")
    by_task: dict[str, CodeTask] = {task.task_id: task for task in tasks}
    completed = completed_dream_keys()
    for policy in ["primary_control", "best_simple_length_policy", "oracle_sufficient_canvas"]:
        args = dream_args(DREAM_FULL_DIR, policy)
        cfg = dream_mod.build_config(args)
        dream_mod.set_global_seed(args.seed)
        written = 0
        for item in manifest:
            key = (str(item["manifest_index"]), policy)
            if key in completed:
                continue
            start = time.perf_counter()
            try:
                result = dream_mod.run_task(by_task[str(item["task_id"])], tokenizer, model, cfg, args)
                compact = compact_dream_result(item, policy, result)
            except Exception as exc:
                compact = {
                    "manifest_index": item["manifest_index"],
                    "task_id": item["task_id"],
                    "task_group": item["task_group"],
                    "prompt_id": item["prompt_id"],
                    "length_bucket": item["length_bucket"],
                    "policy": policy,
                    "status": "error",
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                    "total_sec_including_probe": time.perf_counter() - start,
                }
            append_csv_row(DREAM_FULL_DIR / "results.csv", compact, DREAM_RESULT_FIELDS)
            completed.add(key)
            written += 1
            if written == 1 or written % 50 == 0:
                print(json.dumps({"policy": policy, "new_rows_written": written, "total_completed": len(completed)}, sort_keys=True), flush=True)


def dream_pass(by_case: Mapping[tuple[str, str], Mapping[str, str]], idx: str, policy: str) -> bool:
    return boolish(by_case.get((idx, policy), {}).get("passed", False))


def finalize_dream_full() -> dict[str, Any]:
    manifest = read_csv(DREAM_FULL_DIR / "manifest.csv")
    results = read_csv(DREAM_FULL_DIR / "results.csv")
    expected = len(manifest) * 3
    if len(results) != expected:
        raise RuntimeError(f"Incomplete Dream-Coder full allowed run: expected {expected}, found {len(results)}")
    if len({(row["manifest_index"], row["policy"]) for row in results}) != len(results):
        raise RuntimeError("Duplicate Dream-Coder result rows")
    by_case = {(row["manifest_index"], row["policy"]): row for row in results}
    taxonomy_rows: list[dict[str, Any]] = []
    for item in manifest:
        idx = str(item["manifest_index"])
        primary = dream_pass(by_case, idx, "primary_control")
        simple = dream_pass(by_case, idx, "best_simple_length_policy")
        oracle = dream_pass(by_case, idx, "oracle_sufficient_canvas")
        if (not primary) and simple and oracle:
            taxonomy = "simple_and_oracle_recover_control_failure"
        elif (not primary) and (not simple) and oracle:
            taxonomy = "oracle_only_canvas_recoverable"
        elif (not primary) and (not oracle):
            taxonomy = "rescue_limited_or_noncanvas_failure"
        elif primary and (not simple):
            taxonomy = "simple_harm_vs_control"
        elif primary and (not oracle):
            taxonomy = "oracle_canvas_harm_vs_control"
        elif primary and simple and oracle:
            taxonomy = "stable_pass"
        else:
            taxonomy = "mixed_other"
        taxonomy_rows.append({**item, "primary_passed": primary, "simple_passed": simple, "oracle_passed": oracle, "taxonomy": taxonomy})

    stratum_rows: list[dict[str, Any]] = []
    for bucket in sorted({row["length_bucket"] for row in taxonomy_rows}):
        rows = [row for row in taxonomy_rows if row["length_bucket"] == bucket]
        stratum_rows.append(
            {
                "length_bucket": bucket,
                "cases": len(rows),
                "primary_control_pass": sum(row["primary_passed"] for row in rows),
                "best_simple_length_policy_pass": sum(row["simple_passed"] for row in rows),
                "oracle_sufficient_canvas_pass": sum(row["oracle_passed"] for row in rows),
                "oracle_gain_vs_primary": sum((not row["primary_passed"]) and row["oracle_passed"] for row in rows),
                "simple_help_vs_primary": sum((not row["primary_passed"]) and row["simple_passed"] for row in rows),
                "simple_harm_vs_primary": sum(row["primary_passed"] and (not row["simple_passed"]) for row in rows),
                "oracle_harm_vs_primary": sum(row["primary_passed"] and (not row["oracle_passed"]) for row in rows),
                "rescue_limited_or_noncanvas": sum(row["taxonomy"] == "rescue_limited_or_noncanvas_failure" for row in rows),
            }
        )
    stratum_rows.insert(
        0,
        {
            "length_bucket": "ALL",
            "cases": len(taxonomy_rows),
            "primary_control_pass": sum(row["primary_passed"] for row in taxonomy_rows),
            "best_simple_length_policy_pass": sum(row["simple_passed"] for row in taxonomy_rows),
            "oracle_sufficient_canvas_pass": sum(row["oracle_passed"] for row in taxonomy_rows),
            "oracle_gain_vs_primary": sum((not row["primary_passed"]) and row["oracle_passed"] for row in taxonomy_rows),
            "simple_help_vs_primary": sum((not row["primary_passed"]) and row["simple_passed"] for row in taxonomy_rows),
            "simple_harm_vs_primary": sum(row["primary_passed"] and (not row["simple_passed"]) for row in taxonomy_rows),
            "oracle_harm_vs_primary": sum(row["primary_passed"] and (not row["oracle_passed"]) for row in taxonomy_rows),
            "rescue_limited_or_noncanvas": sum(row["taxonomy"] == "rescue_limited_or_noncanvas_failure" for row in taxonomy_rows),
        },
    )
    taxonomy_summary = []
    for taxonomy, rows in sorted(defaultdict(list, {k: [r for r in taxonomy_rows if r["taxonomy"] == k] for k in {r["taxonomy"] for r in taxonomy_rows}}).items()):
        taxonomy_summary.append(
            {
                "taxonomy": taxonomy,
                "cases": len(rows),
                "primary_pass": sum(row["primary_passed"] for row in rows),
                "simple_pass": sum(row["simple_passed"] for row in rows),
                "oracle_pass": sum(row["oracle_passed"] for row in rows),
            }
        )
    expanded = read_json(DREAM_EXPANDED_DIR / "summary.json")
    comparison = [
        {"metric": "Cases", "expanded37": expanded["case_count"], "full_allowed": len(taxonomy_rows), "note": "full allowed excludes frozen-controller-test task groups"},
        {"metric": "Primary/control pass", "expanded37": expanded["primary_control_pass"], "full_allowed": stratum_rows[0]["primary_control_pass"], "note": ""},
        {"metric": "Best simple pass", "expanded37": expanded["best_simple_policy_pass"], "full_allowed": stratum_rows[0]["best_simple_length_policy_pass"], "note": ""},
        {"metric": "Oracle pass", "expanded37": expanded["oracle_sufficient_canvas_pass"], "full_allowed": stratum_rows[0]["oracle_sufficient_canvas_pass"], "note": ""},
        {"metric": "Oracle gain vs primary", "expanded37": expanded["oracle_sufficient_canvas_pass"] - expanded["primary_control_pass"], "full_allowed": stratum_rows[0]["oracle_gain_vs_primary"], "note": ""},
        {"metric": "Oracle harm vs primary", "expanded37": "0 short regressions reported", "full_allowed": stratum_rows[0]["oracle_harm_vs_primary"], "note": "full allowed counts all length buckets"},
    ]
    write_csv(DREAM_FULL_DIR / "stratum_summary.csv", stratum_rows)
    write_csv(DREAM_FULL_DIR / "taxonomy_summary.csv", taxonomy_summary)
    write_csv(DREAM_FULL_DIR / "comparison_vs_expanded37.csv", comparison)
    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "case_count": len(taxonomy_rows),
        "frozen_controller_test_rows": 0,
        "primary_control_pass": stratum_rows[0]["primary_control_pass"],
        "best_simple_length_policy_pass": stratum_rows[0]["best_simple_length_policy_pass"],
        "oracle_sufficient_canvas_pass": stratum_rows[0]["oracle_sufficient_canvas_pass"],
        "oracle_gain_vs_primary_cases": stratum_rows[0]["oracle_gain_vs_primary"],
        "simple_help_vs_primary_cases": stratum_rows[0]["simple_help_vs_primary"],
        "simple_harm_vs_primary_cases": stratum_rows[0]["simple_harm_vs_primary"],
        "oracle_harm_vs_primary_cases": stratum_rows[0]["oracle_harm_vs_primary"],
        "rescue_limited_or_noncanvas_cases": stratum_rows[0]["rescue_limited_or_noncanvas"],
        "avg_primary_sec": avg(row.get("total_sec_including_probe") for row in results if row["policy"] == "primary_control"),
        "avg_simple_sec": avg(row.get("total_sec_including_probe") for row in results if row["policy"] == "best_simple_length_policy"),
        "avg_oracle_sec": avg(row.get("total_sec_including_probe") for row in results if row["policy"] == "oracle_sufficient_canvas"),
        "efg_status": "not_applicable_dreamcoder_no_trace_remask_adapter",
        "frozen_controller_test_status": "sealed_not_touched",
        "interpretation": "optional full allowed Dream-Coder SingleLine diagnostic; second-backbone evidence only, not model-agnostic confirmation",
        "verdict": "dreamcoder_full_allowed_singleline_mixed_second_backbone_diagnostic",
    }
    write_json(DREAM_FULL_DIR / "summary.json", summary)
    report = [
        "# Dream-Coder Full Allowed SingleLine Diagnostic",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        "This optional full-run branch uses all non-frozen allowed `HumanEval-SingleLineInfilling` rows only. Frozen-controller-test rows remain sealed and excluded. E/F/G actions are not used because Dream-Coder has no trace-remasking adapter.",
        "",
        f"Cases: `{summary['case_count']}`; frozen rows: `0`.",
        f"Primary/control pass: `{summary['primary_control_pass']}/{summary['case_count']}` (`{pct(summary['primary_control_pass'], summary['case_count'])}`).",
        f"Best simple length policy pass: `{summary['best_simple_length_policy_pass']}/{summary['case_count']}` (`{pct(summary['best_simple_length_policy_pass'], summary['case_count'])}`).",
        f"Oracle-sufficient canvas pass: `{summary['oracle_sufficient_canvas_pass']}/{summary['case_count']}` (`{pct(summary['oracle_sufficient_canvas_pass'], summary['case_count'])}`).",
        f"Oracle gain vs primary: `{summary['oracle_gain_vs_primary_cases']}`.",
        f"Simple help/harm vs primary: `{summary['simple_help_vs_primary_cases']}` / `{summary['simple_harm_vs_primary_cases']}`.",
        f"Oracle harm vs primary: `{summary['oracle_harm_vs_primary_cases']}`.",
        f"Rescue/non-canvas-limited: `{summary['rescue_limited_or_noncanvas_cases']}`.",
        "",
        "## Interpretation",
        "",
        "Use this as optional second-backbone SingleLine diagnostic evidence. It can strengthen or weaken Dream-Coder canvas-recoverability claims, but it must not be written as model-agnostic confirmation of LLaDA's missed-vs-triggered split.",
        "",
        "## Compact Outputs",
        "",
        "- `manifest.csv`",
        "- `results.csv`",
        "- `stratum_summary.csv`",
        "- `taxonomy_summary.csv`",
        "- `comparison_vs_expanded37.csv`",
        "- `summary.json`",
    ]
    (DREAM_FULL_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def run_dream_full_allowed() -> dict[str, Any]:
    start = time.perf_counter()
    run_dream_policies()
    summary = finalize_dream_full()
    summary["wall_clock_sec"] = time.perf_counter() - start
    write_json(DREAM_FULL_DIR / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["consolidate", "dream-full-allowed", "all"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"consolidate", "all"}:
        print(json.dumps(run_consolidation(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.mode in {"dream-full-allowed", "all"}:
        print(json.dumps(run_dream_full_allowed(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
