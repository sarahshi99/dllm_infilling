from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .ast_utils import (
    StructuralUnit,
    choose_structural_unit_for_error,
    control_flow_fragment_span,
    diagnose_full_span_projection,
    full_span_to_middle_relative,
)
from .config import ExperimentConfig
from .data_loader import CodeTask
from .decode_vanilla import _segment_decode, build_reconstruction_diagnostics, prepare_model_inputs
from .policies import (
    linear_target_masks,
    merge_policy_indices,
    select_low_confidence_mask_positions,
    structural_units_to_canvas_indices,
)
from .verifier import (
    VerificationResult,
    run_verifier_stack,
    tier1_parse_and_compile,
    tier2_smoke_exec,
    tier3_unit_tests,
)

MIN_SEMANTIC_STRUCTURAL_TOKENS = 6
MIN_CONTROL_FLOW_STRUCTURAL_TOKENS = 8
TINY_SEMANTIC_AST_TYPES = {"Name", "Constant", "Load", "Store", "Tuple"}
SYNTAX_LIKE_PATTERNS = [
    "expected an indented block",
    "invalid syntax",
    "was never closed",
    "unindent does not match",
    "unexpected EOF",
    "expected ':'",
]

# =========================
# 第十轮新增：固定小常量
# 不先动 config/CLI 太多，先把机制验证清楚
# =========================
FRAGMENT_LOCAL_CONTENT_K = 2
POSITION_EXPAND_MIN_DISTINCT = 2
POSITION_EXPAND_REPEAT_TRIGGER = 3
POSITION_EXPAND_LINE_WINDOW = 2
POSITION_PROBE_TOPK = 3


def _resolve_ablation_mode(cfg: ExperimentConfig) -> Dict[str, Any]:
    mode = getattr(cfg.decode, "ablation_mode", "A")
    mode = str(mode).upper()
    if mode not in {"A", "B", "C", "D"}:
        mode = "A"
    return {
        "mode": mode,
        "enable_position_probe": True,
        "enable_content_candidates": mode in {"B", "D"},
        "enable_position_expansion": mode in {"C", "D"},
    }


def get_verifier_stage(step: int, total_steps: int, cfg: ExperimentConfig) -> str:
    schedule = cfg.verifier.schedule
    schedule.validate()
    ratio = (step + 1) / total_steps
    if ratio <= schedule.blind_ratio:
        return "blind"
    if ratio <= schedule.blind_ratio + schedule.structure_ratio:
        return "structure"
    return "semantic"


def verifier_score(results: Dict[str, VerificationResult]) -> int:
    if results.get("tier3_unit_tests") and results["tier3_unit_tests"].passed:
        return 3
    if results.get("tier2_smoke_exec") and results["tier2_smoke_exec"].passed:
        return 2
    if results.get("tier1_parse_compile") and results["tier1_parse_compile"].passed:
        return 1
    return 0


def _classify_verification(results: Dict[str, VerificationResult]) -> Tuple[int, str]:
    """
    class_rank 数值越小越好：
    0: tier3_pass
    1: feasible_but_tier3_fail
    2: infeasible_tier1_or_tier2_fail
    """
    tier1 = results.get("tier1_parse_compile")
    tier2 = results.get("tier2_smoke_exec")
    tier3 = results.get("tier3_unit_tests")

    if tier3 is not None and tier3.passed:
        return 0, "tier3_pass"

    if (
        tier1 is not None and tier1.passed and
        tier2 is not None and tier2.passed and
        tier3 is not None and not tier3.passed
    ):
        return 1, "feasible_but_tier3_fail"

    return 2, "infeasible_tier1_or_tier2_fail"


def _feedback_richness(results: Dict[str, VerificationResult]) -> Tuple[int, int]:
    tier3 = results.get("tier3_unit_tests")
    informative_bonus = 0
    line_bonus = 0
    if tier3 is not None:
        if tier3.error_message not in {None, "failed:", "failed: "}:
            informative_bonus = 1
        if tier3.line is not None:
            line_bonus = 1
    return informative_bonus, line_bonus


def run_stage_verifier(
    task: CodeTask,
    full_code: str,
    completion_without_suffix: str,
    stage: str,
    cfg: ExperimentConfig,
) -> Dict[str, VerificationResult]:
    results: Dict[str, VerificationResult] = {}
    if stage == "blind":
        return results

    tier1 = tier1_parse_and_compile(full_code, compile_mode=cfg.verifier.compile_mode)
    results[tier1.tier] = tier1
    if not tier1.passed:
        return results

    if cfg.verifier.enable_tier2_exec:
        tier2 = tier2_smoke_exec(full_code)
        results[tier2.tier] = tier2
        if not tier2.passed:
            return results

    if stage == "semantic" and cfg.verifier.enable_tier3_tests:
        tier3 = tier3_unit_tests(task, completion_without_suffix, timeout=cfg.verifier.timeout)
        results[tier3.tier] = tier3
    return results


def _mutable_region_local_fallback(
    middle_text: str,
    reason: str,
    metadata: Dict[str, Any],
    anchor_line_in_middle: Optional[int] = None,
) -> StructuralUnit:
    if not middle_text:
        return StructuralUnit(
            unit_type="statement",
            char_start=0,
            char_end=0,
            line_start=1,
            line_end=1,
            reason=reason,
            metadata=metadata,
        )

    lines = middle_text.splitlines(keepends=True)
    if not lines:
        return StructuralUnit(
            unit_type="statement",
            char_start=0,
            char_end=max(1, min(len(middle_text), 4)),
            line_start=1,
            line_end=1,
            reason=reason,
            metadata=metadata,
        )

    target_line = 1 if anchor_line_in_middle is None else min(max(anchor_line_in_middle, 1), len(lines))
    idx = target_line - 1
    start_idx = max(0, idx - 1)
    end_idx = min(len(lines) - 1, idx + 1)

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    char_start = offsets[start_idx]
    char_end = offsets[end_idx + 1]

    return StructuralUnit(
        unit_type="statement",
        char_start=char_start,
        char_end=max(char_start + 1, char_end),
        line_start=start_idx + 1,
        line_end=end_idx + 1,
        reason=reason,
        metadata=dict(metadata, anchor_line_in_middle=target_line),
    )


def _empty_repair_state() -> Dict[str, Any]:
    return {
        "last_issue_key": None,
        "same_issue_streak": 0,
        "escalation_level": 0,
        "recent_nonnull_issue_lines": [],
        "recent_fragment_ranges": [],
        "fragment_attempt_counts": {},
        "last_fragment_key": None,
        "same_fragment_streak": 0,
        "low_diversity_position_streak": 0,
        "last_distinct_position_count": None,
    }


def _normalize_error_message(text: Optional[str]) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:160]


def _active_failure(verifier_results: Dict[str, VerificationResult]) -> Optional[VerificationResult]:
    for tier_name in ["tier1_parse_compile", "tier2_smoke_exec", "tier3_unit_tests"]:
        vr = verifier_results.get(tier_name)
        if vr is not None and not vr.passed:
            return vr
    return None


def _build_issue_signature(stage: str, verifier_results: Dict[str, VerificationResult]) -> Optional[Dict[str, Any]]:
    failure = _active_failure(verifier_results)
    if failure is None:
        return None
    return {
        "stage": stage,
        "tier": failure.tier,
        "line": failure.line,
        "error_type": failure.error_type,
        "message": _normalize_error_message(failure.error_message or failure.traceback_text),
    }


def _update_repair_state(repair_state: Dict[str, Any], issue_signature: Optional[Dict[str, Any]], cfg: ExperimentConfig) -> Dict[str, Any]:
    state = dict(repair_state)
    if issue_signature is None:
        state.update({"last_issue_key": None, "same_issue_streak": 0, "escalation_level": 0})
        return state

    key = (
        issue_signature["stage"],
        issue_signature["tier"],
        issue_signature["line"],
        issue_signature["error_type"],
        issue_signature["message"],
    )
    if state.get("last_issue_key") == key:
        state["same_issue_streak"] = int(state.get("same_issue_streak", 0)) + 1
    else:
        state["last_issue_key"] = key
        state["same_issue_streak"] = 1

    if issue_signature.get("line") is not None:
        lines = list(state.get("recent_nonnull_issue_lines", []))
        if not lines or lines[-1] != issue_signature["line"]:
            lines.append(issue_signature["line"])
        state["recent_nonnull_issue_lines"] = lines[-4:]

    streak = state["same_issue_streak"]
    base = max(1, int(cfg.policy.escalate_after_failed_attempts))
    tier = issue_signature["tier"]

    if tier == "tier1_parse_compile":
        if streak >= base + 2:
            state["escalation_level"] = 2
        elif streak >= base:
            state["escalation_level"] = 1
        else:
            state["escalation_level"] = 0
    elif tier == "tier2_smoke_exec":
        state["escalation_level"] = 1 if streak >= base else 0
    else:
        state["escalation_level"] = 0
    return state


def _update_position_diversity_state(repair_state: Dict[str, Any], distinct_count: Optional[int]) -> Dict[str, Any]:
    state = dict(repair_state)
    if distinct_count is None:
        return state

    if distinct_count < POSITION_EXPAND_MIN_DISTINCT:
        state["low_diversity_position_streak"] = int(state.get("low_diversity_position_streak", 0)) + 1
    else:
        state["low_diversity_position_streak"] = 0
    state["last_distinct_position_count"] = distinct_count
    return state


def _should_trigger_position_expansion(
    deduped_candidates: Sequence[Dict[str, Any]],
    repair_state: Dict[str, Any],
) -> Tuple[bool, str]:
    if len(deduped_candidates) < POSITION_EXPAND_MIN_DISTINCT:
        return True, "distinct_lt_min"
    if int(repair_state.get("low_diversity_position_streak", 0)) >= POSITION_EXPAND_REPEAT_TRIGGER:
        return True, "low_diversity_streak"
    return False, "not_triggered"


def _fallback_mode_for_issue(issue_signature: Optional[Dict[str, Any]], repair_state: Dict[str, Any]) -> str:
    if issue_signature is None:
        return "line"
    tier = issue_signature["tier"]
    level = int(repair_state.get("escalation_level", 0))
    if tier == "tier1_parse_compile":
        return ["line", "window", "block"][min(level, 2)]
    if tier == "tier2_smoke_exec":
        return "window" if level == 0 else "block"
    return "window"


def _expand_indices_to_min_span(indices: Sequence[int], total_tokens: int, min_tokens: int) -> List[int]:
    unique = sorted(set(indices))
    if not unique or len(unique) >= min_tokens:
        return unique
    center = (unique[0] + unique[-1]) // 2
    half = max(0, min_tokens // 2)
    start = max(0, center - half)
    end = min(total_tokens, start + min_tokens)
    start = max(0, end - min_tokens)
    return list(range(start, end))


def _extract_line_numbers(text: Optional[str]) -> List[int]:
    if not text:
        return []
    hits = re.findall(r"line (\d+)", text)
    return [int(h) for h in hits]


def _looks_like_syntax_failure(vr: VerificationResult) -> bool:
    hay = ((vr.error_message or "") + " " + (vr.traceback_text or "")).lower()
    return any(pat in hay for pat in SYNTAX_LIKE_PATTERNS)


def _is_weak_tier3_failure(vr: Optional[VerificationResult]) -> bool:
    if vr is None or vr.passed:
        return False
    msg = _normalize_error_message(vr.error_message or vr.traceback_text).lower()
    return vr.line is None and (msg in {"failed:", "failed"} or len(msg) <= 12)


def _boundary_lines(prefix_text: str, middle_text: str) -> List[int]:
    boundary_line = (prefix_text + middle_text).count("\n") + 1
    return [max(1, boundary_line - 1), boundary_line, boundary_line + 1]


def _dedup_keep_order(values: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _fragment_key(unit: StructuralUnit) -> str:
    meta = unit.metadata or {}
    reason = meta.get("origin_reason", unit.reason)
    line_start = meta.get("origin_line_start", unit.line_start)
    line_end = meta.get("origin_line_end", unit.line_end)
    header_keyword = meta.get("origin_header_keyword", meta.get("header_keyword", "NA"))
    unit_type = meta.get("origin_unit_type", unit.unit_type)
    return f"{unit_type}:{reason}:{line_start}:{line_end}:{header_keyword}"


def _fragment_identity(unit: StructuralUnit) -> Tuple[Any, ...]:
    meta = unit.metadata or {}
    header_line = meta.get("header_line", unit.line_start)
    header_keyword = meta.get("header_keyword", "NA")
    line_end = unit.line_end
    return (header_line, header_keyword, line_end)


def _line_interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    if union <= 0:
        return 0.0
    return inter / union


def _same_fragment_for_probe(a: StructuralUnit, b: StructuralUnit) -> bool:
    if _fragment_identity(a) == _fragment_identity(b):
        return True

    meta_a = a.metadata or {}
    meta_b = b.metadata or {}
    kw_a = meta_a.get("header_keyword")
    kw_b = meta_b.get("header_keyword")
    if kw_a is not None and kw_a == kw_b:
        iou = _line_interval_iou(a.line_start, a.line_end, b.line_start, b.line_end)
        if iou >= 0.8:
            return True
    return False


def _project_unit_to_middle_or_local_fallback(
    unit: StructuralUnit,
    prefix_text: str,
    middle_text: str,
) -> Tuple[StructuralUnit, Dict[str, Any]]:
    projection = diagnose_full_span_projection(prefix_text, middle_text, unit)

    origin_meta = {
        "origin_reason": unit.reason,
        "origin_line_start": unit.line_start,
        "origin_line_end": unit.line_end,
        "origin_unit_type": unit.unit_type,
        "origin_header_keyword": (unit.metadata or {}).get("header_keyword"),
    }

    middle_relative = full_span_to_middle_relative(prefix_text, middle_text, unit)
    if middle_relative is not None:
        new_meta = dict(middle_relative.metadata or {})
        new_meta.update(origin_meta)
        middle_relative.metadata = new_meta
        return middle_relative, {
            "projection": projection,
            "projection_result": "mapped_to_middle",
        }

    middle_anchor_line = None
    if unit.line_start is not None:
        prefix_line_count = prefix_text.count("\n") + 1
        middle_anchor_line = max(1, unit.line_start - prefix_line_count + 1)

    local_fallback = _mutable_region_local_fallback(
        middle_text=middle_text,
        reason="mutable_region_local_fallback",
        metadata={
            "original_reason": unit.reason,
            "projection": projection,
            **origin_meta,
        },
        anchor_line_in_middle=middle_anchor_line,
    )
    return local_fallback, {
        "projection": projection,
        "projection_result": "local_mutable_fallback",
        "fallback_unit": local_fallback.to_dict(),
    }


def _pick_boundary_control_flow_unit(
    full_text: str,
    prefix_text: str,
    middle_text: str,
    tier3: VerificationResult,
) -> Tuple[Optional[StructuralUnit], Dict[str, Any]]:
    candidate_lines = []
    if tier3.line is not None:
        candidate_lines.append(tier3.line)
    candidate_lines.extend(_extract_line_numbers(tier3.error_message))
    candidate_lines.extend(_extract_line_numbers(tier3.traceback_text))
    candidate_lines.extend(_boundary_lines(prefix_text, middle_text))
    candidate_lines = _dedup_keep_order([ln for ln in candidate_lines if ln is not None and ln > 0])

    debug: Dict[str, Any] = {
        "syntax_like_tier3": _looks_like_syntax_failure(tier3),
        "tier3_line_candidates": candidate_lines,
        "boundary_lines": _boundary_lines(prefix_text, middle_text),
        "selected_boundary_control_flow": None,
    }

    best_unit: Optional[StructuralUnit] = None
    best_score: Optional[Tuple[int, int]] = None
    boundary_center = _boundary_lines(prefix_text, middle_text)[1]

    for line in candidate_lines:
        unit = control_flow_fragment_span(full_text, anchor_line=line, lookaround=1, max_body_lines=3)
        if unit is None:
            continue

        projection = diagnose_full_span_projection(prefix_text, middle_text, unit)
        projection_rank = 0 if projection["projection_case"] == "overlap" else 1
        line_distance = abs(unit.line_start - boundary_center)
        score = (projection_rank, line_distance)

        if best_unit is None or score < (best_score or (10**9, 10**9)):
            best_unit = unit
            best_score = score
            debug["selected_boundary_control_flow"] = {
                "candidate_line": line,
                "unit": unit.to_dict(),
                "projection": projection,
                "score": score,
            }

    return best_unit, debug


def _build_weak_tier3_candidate_lines(
    prefix_text: str,
    middle_text: str,
    tier3: VerificationResult,
    repair_state: Dict[str, Any],
) -> List[Tuple[str, int]]:
    candidates: List[Tuple[str, int]] = []

    for line in _extract_line_numbers(tier3.error_message) + _extract_line_numbers(tier3.traceback_text):
        if line > 0:
            candidates.append(("tier3_text", line))

    for line in repair_state.get("recent_nonnull_issue_lines", []):
        if line > 0:
            candidates.append(("recent_issue", line))

    for line_start, line_end in repair_state.get("recent_fragment_ranges", []):
        if line_start is not None:
            candidates.append(("recent_fragment", line_start))
        if line_end is not None and line_end != line_start:
            candidates.append(("recent_fragment", line_end))

    for line in _boundary_lines(prefix_text, middle_text):
        candidates.append(("boundary", line))

    ordered: List[Tuple[str, int]] = []
    seen = set()
    for source, line in candidates:
        key = (source, line)
        if line <= 0 or key in seen:
            continue
        seen.add(key)
        ordered.append((source, line))
    return ordered


def _make_ranked_fragment_candidate(
    unit: StructuralUnit,
    source: str,
    line: int,
    prefix_text: str,
    middle_text: str,
    repair_state: Dict[str, Any],
) -> Dict[str, Any]:
    boundary_center = _boundary_lines(prefix_text, middle_text)[1]
    attempt_counts: Dict[str, int] = repair_state.get("fragment_attempt_counts", {})
    last_fragment_key = repair_state.get("last_fragment_key")
    same_fragment_streak = int(repair_state.get("same_fragment_streak", 0))

    projection = diagnose_full_span_projection(prefix_text, middle_text, unit)
    projection_rank = 0 if projection["projection_case"] == "overlap" else 1
    key = _fragment_key(unit)

    source_rank = {
        "tier3_text": 0,
        "recent_issue": 1,
        "recent_fragment": 2,
        "boundary": 3,
        "expanded_context": 4,
    }

    repeat_penalty = same_fragment_streak * 4 if key == last_fragment_key else 0
    history_penalty = attempt_counts.get(key, 0) * 4 + repeat_penalty

    distance_rank = abs(unit.line_start - boundary_center)
    span_lines = max(1, unit.line_end - unit.line_start + 1)
    span_penalty = abs(span_lines - 3)

    score = (
        projection_rank,
        history_penalty,
        source_rank.get(source, 9),
        distance_rank,
        span_penalty,
    )

    return {
        "source": source,
        "line": line,
        "unit": unit,
        "unit_dict": unit.to_dict(),
        "projection": projection,
        "score": score,
        "history_penalty": history_penalty,
        "repeat_penalty": repeat_penalty,
        "fragment_key": key,
    }


def _rank_weak_location_control_flow_candidates(
    full_text: str,
    prefix_text: str,
    middle_text: str,
    tier3: VerificationResult,
    repair_state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_lines = _build_weak_tier3_candidate_lines(prefix_text, middle_text, tier3, repair_state)

    debug: Dict[str, Any] = {
        "weak_location_candidate_lines": candidate_lines,
        "selected_weak_location_candidate": None,
        "weak_location_candidates_scored": [],
    }

    ranked: List[Dict[str, Any]] = []
    for source, line in candidate_lines:
        unit = control_flow_fragment_span(full_text, anchor_line=line, lookaround=1, max_body_lines=3)
        if unit is None:
            continue
        item = _make_ranked_fragment_candidate(
            unit=unit,
            source=source,
            line=line,
            prefix_text=prefix_text,
            middle_text=middle_text,
            repair_state=repair_state,
        )
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"])
    debug["weak_location_candidates_scored"] = [
        {
            "source": item["source"],
            "line": item["line"],
            "unit": item["unit_dict"],
            "projection": item["projection"],
            "score": item["score"],
            "history_penalty": item["history_penalty"],
            "repeat_penalty": item["repeat_penalty"],
            "fragment_key": item["fragment_key"],
        }
        for item in ranked[:6]
    ]
    if ranked:
        best = ranked[0]
        debug["selected_weak_location_candidate"] = {
            "source": best["source"],
            "line": best["line"],
            "unit": best["unit_dict"],
            "projection": best["projection"],
            "score": best["score"],
            "fragment_key": best["fragment_key"],
        }
    return ranked, debug


def _dedup_ranked_probe_candidates(
    ranked_candidates: Sequence[Dict[str, Any]],
    max_keep: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    for item in ranked_candidates:
        unit = item["unit"]
        is_dup = False
        for kept in deduped:
            if _same_fragment_for_probe(unit, kept["unit"]):
                duplicates.append({
                    "source": item["source"],
                    "line": item["line"],
                    "unit": item["unit_dict"],
                    "fragment_key": item["fragment_key"],
                })
                is_dup = True
                break
        if is_dup:
            continue
        deduped.append(item)
        if len(deduped) >= max_keep:
            break

    debug = {
        "max_keep": max_keep,
        "deduped_count": len(deduped),
        "deduped_units": [
            {
                "source": item["source"],
                "line": item["line"],
                "unit": item["unit_dict"],
                "fragment_key": item["fragment_key"],
            }
            for item in deduped
        ],
        "duplicates_dropped": duplicates,
    }
    return deduped, debug


def _maybe_expand_position_candidates(
    full_text: str,
    prefix_text: str,
    middle_text: str,
    deduped_candidates: Sequence[Dict[str, Any]],
    repair_state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not deduped_candidates:
        return list(deduped_candidates), {
            "triggered": False,
            "reason": "empty_input",
            "added_anchor_lines": [],
            "added_units": [],
            "after_expand_count": 0,
        }

    added_anchor_lines: List[int] = []

    best_unit = deduped_candidates[0]["unit"]
    for line in range(
        max(1, best_unit.line_start - POSITION_EXPAND_LINE_WINDOW),
        best_unit.line_end + POSITION_EXPAND_LINE_WINDOW + 1,
    ):
        added_anchor_lines.append(line)

    boundary_center = _boundary_lines(prefix_text, middle_text)[1]
    for line in range(
        max(1, boundary_center - POSITION_EXPAND_LINE_WINDOW),
        boundary_center + POSITION_EXPAND_LINE_WINDOW + 1,
    ):
        added_anchor_lines.append(line)

    for line in repair_state.get("recent_nonnull_issue_lines", []):
        for delta in range(-1, 2):
            if line + delta > 0:
                added_anchor_lines.append(line + delta)

    for line_start, line_end in repair_state.get("recent_fragment_ranges", []):
        if line_start is not None:
            added_anchor_lines.append(line_start)
        if line_end is not None:
            added_anchor_lines.append(line_end)

    added_anchor_lines = _dedup_keep_order([ln for ln in added_anchor_lines if ln > 0])

    extra_items: List[Dict[str, Any]] = []
    for line in added_anchor_lines:
        unit = control_flow_fragment_span(
            full_text,
            anchor_line=line,
            lookaround=POSITION_EXPAND_LINE_WINDOW,
            max_body_lines=4,
        )
        if unit is None:
            continue
        item = _make_ranked_fragment_candidate(
            unit=unit,
            source="expanded_context",
            line=line,
            prefix_text=prefix_text,
            middle_text=middle_text,
            repair_state=repair_state,
        )
        extra_items.append(item)

    combined = list(deduped_candidates) + extra_items
    combined.sort(key=lambda x: x["score"])
    rededuped, rededup_debug = _dedup_ranked_probe_candidates(combined, max_keep=POSITION_PROBE_TOPK)

    debug = {
        "triggered": True,
        "reason": "expanded_context_window",
        "added_anchor_lines": added_anchor_lines,
        "added_units": [
            {
                "source": item["source"],
                "line": item["line"],
                "unit": item["unit_dict"],
                "fragment_key": item["fragment_key"],
            }
            for item in extra_items
        ],
        "after_expand_count": len(rededuped),
        "after_expand_dedup": rededup_debug,
    }
    return rededuped, debug


def _build_fragment_content_indices(
    tokenizer,
    middle_text: str,
    middle_length_tokens: int,
    selected_unit: StructuralUnit,
) -> List[int]:
    indices = structural_units_to_canvas_indices(
        tokenizer=tokenizer,
        middle_text=middle_text,
        middle_length_tokens=middle_length_tokens,
        units=[selected_unit],
    )
    indices = _expand_indices_to_min_span(
        indices,
        total_tokens=middle_length_tokens,
        min_tokens=MIN_CONTROL_FLOW_STRUCTURAL_TOKENS,
    )
    return indices


def _generate_fragment_content_candidates(
    tokenizer,
    logits: torch.Tensor,
    x_filled: torch.Tensor,
    prepared: Dict[str, Any],
    selected_unit: StructuralUnit,
    incumbent_segments: Dict[str, str],
    incumbent_verifier_results: Dict[str, VerificationResult],
    task: CodeTask,
    cfg: ExperimentConfig,
    k: int = FRAGMENT_LOCAL_CONTENT_K,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "applied": True,
        "content_indices": [],
        "candidate_count": 0,
        "generation_reason": None,
    }

    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]
    middle_length_tokens = prepared["middle_end"] - prepared["middle_start"]

    content_indices = _build_fragment_content_indices(
        tokenizer=tokenizer,
        middle_text=incumbent_segments["middle_text"],
        middle_length_tokens=middle_length_tokens,
        selected_unit=selected_unit,
    )
    debug["content_indices"] = content_indices

    incumbent_class_rank, incumbent_class_label = _classify_verification(incumbent_verifier_results)
    candidates: List[Dict[str, Any]] = [{
        "candidate_type": "incumbent",
        "candidate_id": 0,
        "tensor": x_filled.clone(),
        "segments": incumbent_segments,
        "verification_obj": incumbent_verifier_results,
        "verification": {k_: v.to_dict() for k_, v in incumbent_verifier_results.items()},
        "class_rank": incumbent_class_rank,
        "class_label": incumbent_class_label,
    }]

    if not content_indices:
        debug["candidate_count"] = len(candidates)
        debug["generation_reason"] = "empty_content_indices"
        return candidates, debug

    local_positions = [middle_start + idx for idx in content_indices]
    probs = torch.softmax(logits[0, local_positions, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(16, probs.shape[-1]), dim=-1)

    for cand_id in range(1, k + 1):
        candidate_tensor = x_filled.clone()
        sampled_token_ids: List[int] = []

        for pos_idx, abs_pos in enumerate(local_positions):
            dist = top_probs[pos_idx] / top_probs[pos_idx].sum()
            sampled_offset = torch.multinomial(dist, num_samples=1).item()
            sampled_token = top_ids[pos_idx, sampled_offset].item()
            candidate_tensor[0, abs_pos] = sampled_token
            sampled_token_ids.append(sampled_token)

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=candidate_tensor[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        verify = run_verifier_stack(
            task=task,
            full_code=segments["full_text"],
            completion_without_suffix=segments["middle_text"],
            timeout=cfg.verifier.timeout,
        )
        class_rank, class_label = _classify_verification(verify)

        candidates.append({
            "candidate_type": "sampled",
            "candidate_id": cand_id,
            "sampled_token_ids": sampled_token_ids,
            "tensor": candidate_tensor,
            "segments": segments,
            "verification_obj": verify,
            "verification": {k_: v.to_dict() for k_, v in verify.items()},
            "class_rank": class_rank,
            "class_label": class_label,
        })

    debug["candidate_count"] = len(candidates)
    debug["generation_reason"] = "ok"
    return candidates, debug


def _rank_fragment_content_candidates(
    candidate_entries: Sequence[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "applied": True,
        "candidate_count": len(candidate_entries),
        "candidates": [],
        "chosen_candidate_id": None,
        "chosen_candidate_type": None,
        "kept_incumbent": False,
    }

    best_entry: Optional[Dict[str, Any]] = None
    best_choice_key: Optional[Tuple[int, int, int, int, int]] = None

    for entry in candidate_entries:
        informative_bonus, line_bonus = _feedback_richness(entry["verification_obj"])
        incumbent_penalty = 0 if entry["candidate_type"] == "incumbent" else 1

        # class_rank 越小越好
        # 同 class 下：
        # 1) 更具体错误信息优先
        # 2) 有 line 优先
        # 3) 仍然优先 incumbent，避免无证据替换
        choice_key = (
            entry["class_rank"],
            -informative_bonus,
            -line_bonus,
            incumbent_penalty,
            entry["candidate_id"],
        )

        debug["candidates"].append({
            "candidate_id": entry["candidate_id"],
            "candidate_type": entry["candidate_type"],
            "class_rank": entry["class_rank"],
            "class_label": entry["class_label"],
            "choice_key": choice_key,
            "verification": entry["verification"],
        })

        if best_choice_key is None or choice_key < best_choice_key:
            best_choice_key = choice_key
            best_entry = entry

    if best_entry is not None:
        debug["chosen_candidate_id"] = best_entry["candidate_id"]
        debug["chosen_candidate_type"] = best_entry["candidate_type"]
        debug["kept_incumbent"] = best_entry["candidate_type"] == "incumbent"

    return best_entry, debug


# =========================
# 旧实现（保留对照，不再直接使用）：
# 旧版 top-3 probe 只负责位置，不负责任何内容候选；
# 它的主要问题是：
# 1) 即使位置排序正确，选中位置内部仍然只有一个填法
# 2) 因而 L2/L4 这类“位置可行但语义仍错”的样本，无法再进一步比较内容质量
# =========================
def _probe_topk_weak_location_candidates(
    tokenizer,
    model,
    x_filled: torch.Tensor,
    prepared: Dict[str, Any],
    prefix_text: str,
    middle_text: str,
    task: CodeTask,
    cfg: ExperimentConfig,
    ranked_candidates: Sequence[Dict[str, Any]],
    max_probe_k: int = POSITION_PROBE_TOPK,
) -> Tuple[Optional[StructuralUnit], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "probe_applied": False,
        "max_probe_k": max_probe_k,
        "probe_candidates": [],
        "probe_chosen": None,
        "kept_incumbent_due_to_no_feasible_candidate": False,
    }

    if not ranked_candidates:
        return None, debug

    debug["probe_applied"] = True
    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]
    mask_token_id = prepared["mask_token_id"]
    num_mask_tokens = cfg.decode.num_mask_tokens

    best_unit: Optional[StructuralUnit] = None
    best_choice_key: Optional[Tuple[int, int, int, int]] = None
    best_class_rank: Optional[int] = None

    for cand_rank, item in enumerate(ranked_candidates[:max_probe_k]):
        unit_full: StructuralUnit = item["unit"]
        projected_unit, projection_debug = _project_unit_to_middle_or_local_fallback(
            unit=unit_full,
            prefix_text=prefix_text,
            middle_text=middle_text,
        )
        structural_indices = structural_units_to_canvas_indices(
            tokenizer=tokenizer,
            middle_text=middle_text,
            middle_length_tokens=num_mask_tokens,
            units=[projected_unit],
        )
        structural_indices = _expand_indices_to_min_span(
            structural_indices,
            total_tokens=num_mask_tokens,
            min_tokens=MIN_CONTROL_FLOW_STRUCTURAL_TOKENS,
        )

        if not structural_indices:
            debug["probe_candidates"].append({
                "candidate_rank": cand_rank,
                "unit": item["unit_dict"],
                "projection_debug": projection_debug,
                "skipped": True,
                "reason": "empty_structural_indices",
            })
            continue

        probe_tensor = x_filled.clone()
        abs_positions = [middle_start + idx for idx in structural_indices]
        probe_tensor[0, abs_positions] = mask_token_id

        with torch.no_grad():
            outputs = model(probe_tensor)
            probe_logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probe_preds = torch.argmax(probe_logits, dim=-1)
        probe_mask_idx = (probe_tensor == mask_token_id)
        probe_filled = probe_tensor.clone()
        probe_filled[probe_mask_idx] = probe_preds[probe_mask_idx]

        probe_segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=probe_filled[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        probe_verify = run_verifier_stack(
            task=task,
            full_code=probe_segments["full_text"],
            completion_without_suffix=probe_segments["middle_text"],
            timeout=cfg.verifier.timeout,
        )

        class_rank, class_label = _classify_verification(probe_verify)
        informative_bonus, line_bonus = _feedback_richness(probe_verify)

        # 可行优先：
        # 0 tier3_pass
        # 1 feasible_but_tier3_fail
        # 2 infeasible_tier1_or_tier2_fail
        choice_key = (
            class_rank,
            -informative_bonus,
            -line_bonus,
            cand_rank,
        )

        debug["probe_candidates"].append({
            "candidate_rank": cand_rank,
            "unit": item["unit_dict"],
            "projection_debug": projection_debug,
            "structural_indices": structural_indices,
            "probe_verification": {k: v.to_dict() for k, v in probe_verify.items()},
            "probe_choice_key": choice_key,
            "probe_class_rank": class_rank,
            "probe_class_label": class_label,
            "fragment_key": _fragment_key(projected_unit),
        })

        if best_choice_key is None or choice_key < best_choice_key:
            best_choice_key = choice_key
            best_unit = projected_unit
            best_class_rank = class_rank
            debug["probe_chosen"] = {
                "candidate_rank": cand_rank,
                "unit": item["unit_dict"],
                "projection_debug": projection_debug,
                "probe_choice_key": choice_key,
                "probe_class_rank": class_rank,
                "probe_class_label": class_label,
                "fragment_key": _fragment_key(projected_unit),
            }

    if best_class_rank is None or best_class_rank >= 2:
        debug["kept_incumbent_due_to_no_feasible_candidate"] = True
        return None, debug

    return best_unit, debug


def _update_fragment_history(
    repair_state: Dict[str, Any],
    structural_units: Sequence[StructuralUnit],
    verifier_results: Dict[str, VerificationResult],
) -> Dict[str, Any]:
    state = dict(repair_state)

    if not structural_units:
        return state

    unit = structural_units[0]
    meta = unit.metadata or {}
    origin_unit_type = meta.get("origin_unit_type", unit.unit_type)
    if origin_unit_type != "control_flow_fragment":
        state["last_fragment_key"] = None
        state["same_fragment_streak"] = 0
        return state

    key = _fragment_key(unit)

    origin_line_start = meta.get("origin_line_start", unit.line_start)
    origin_line_end = meta.get("origin_line_end", unit.line_end)
    ranges = list(state.get("recent_fragment_ranges", []))
    ranges.append((origin_line_start, origin_line_end))
    state["recent_fragment_ranges"] = ranges[-4:]

    last_key = state.get("last_fragment_key")
    if last_key == key:
        state["same_fragment_streak"] = int(state.get("same_fragment_streak", 0)) + 1
    else:
        state["last_fragment_key"] = key
        state["same_fragment_streak"] = 1

    failure = _active_failure(verifier_results)
    if failure is not None and failure.tier == "tier3_unit_tests":
        counts = dict(state.get("fragment_attempt_counts", {}))
        counts[key] = counts.get(key, 0) + 1
        state["fragment_attempt_counts"] = counts

    return state


def choose_units_from_verifier(
    full_text: str,
    prefix_text: str,
    middle_text: str,
    verifier_results: Dict[str, VerificationResult],
    cfg: ExperimentConfig,
    repair_state: Dict[str, Any],
    issue_signature: Optional[Dict[str, Any]],
) -> Tuple[List[StructuralUnit], Dict[str, Any], List[Dict[str, Any]]]:
    tier1 = verifier_results.get("tier1_parse_compile")
    tier2 = verifier_results.get("tier2_smoke_exec")
    tier3 = verifier_results.get("tier3_unit_tests")

    chosen: Optional[StructuralUnit] = None
    weak_ranked_candidates: List[Dict[str, Any]] = []
    fallback_mode = _fallback_mode_for_issue(issue_signature, repair_state)
    selection_debug: Dict[str, Any] = {
        "selected_tier": None,
        "requested_prefer": None,
        "fallback_mode": fallback_mode,
        "repair_state_before_select": dict(repair_state),
        "issue_signature": issue_signature,
    }

    if tier1 is not None and not tier1.passed:
        selection_debug.update({"selected_tier": tier1.tier, "requested_prefer": cfg.policy.default_unit_for_parse_error})
        chosen = choose_structural_unit_for_error(
            code=full_text,
            error_line=tier1.line,
            error_col=tier1.col,
            prefer=cfg.policy.default_unit_for_parse_error,
            fallback_reason="tier1_parse_failure_fallback",
            fallback_mode=fallback_mode,
        )

    elif tier2 is not None and not tier2.passed:
        selection_debug.update({"selected_tier": tier2.tier, "requested_prefer": cfg.policy.default_unit_for_parse_error})
        chosen = choose_structural_unit_for_error(
            code=full_text,
            error_line=tier2.line,
            error_col=tier2.col,
            prefer=cfg.policy.default_unit_for_parse_error,
            fallback_reason="tier2_exec_failure_fallback",
            fallback_mode=fallback_mode,
        )

    elif tier3 is not None and not tier3.passed:
        selection_debug.update({"selected_tier": tier3.tier, "requested_prefer": cfg.policy.default_unit_for_test_error})

        boundary_unit, boundary_debug = _pick_boundary_control_flow_unit(
            full_text=full_text,
            prefix_text=prefix_text,
            middle_text=middle_text,
            tier3=tier3,
        )
        selection_debug.update(boundary_debug)

        if boundary_unit is not None and _looks_like_syntax_failure(tier3):
            chosen = boundary_unit
            selection_debug["tier3_selection_strategy"] = "boundary_control_flow_fragment"

        elif _is_weak_tier3_failure(tier3):
            weak_ranked_candidates, weak_debug = _rank_weak_location_control_flow_candidates(
                full_text=full_text,
                prefix_text=prefix_text,
                middle_text=middle_text,
                tier3=tier3,
                repair_state=repair_state,
            )
            selection_debug.update(weak_debug)
            if weak_ranked_candidates:
                chosen = weak_ranked_candidates[0]["unit"]
                selection_debug["tier3_selection_strategy"] = "weak_location_control_flow_candidates"

        if chosen is None:
            chosen = choose_structural_unit_for_error(
                code=full_text,
                error_line=tier3.line,
                error_col=tier3.col,
                prefer=cfg.policy.default_unit_for_test_error,
                fallback_reason="tier3_test_failure_fallback",
                fallback_mode="window",
            )
            selection_debug["tier3_selection_strategy"] = "ast_or_window_fallback"

    if chosen is None:
        selection_debug["selected_unit"] = None
        return [], selection_debug, weak_ranked_candidates

    selection_debug["selected_unit"] = chosen.to_dict()
    projected_unit, projection_debug = _project_unit_to_middle_or_local_fallback(
        unit=chosen,
        prefix_text=prefix_text,
        middle_text=middle_text,
    )
    selection_debug.update(projection_debug)
    return [projected_unit], selection_debug, weak_ranked_candidates


def sample_local_candidates(
    tokenizer,
    x_filled: torch.Tensor,
    logits: torch.Tensor,
    middle_start: int,
    middle_end: int,
    structural_indices: Sequence[int],
    task: CodeTask,
    prepared: Dict[str, Any],
    cfg: ExperimentConfig,
    k: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    if k <= 1 or not structural_indices:
        return x_filled, []

    candidates: List[Tuple[int, torch.Tensor, Dict[str, Any]]] = []
    local_positions = [middle_start + idx for idx in structural_indices]

    probs = torch.softmax(logits[0, local_positions, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(16, probs.shape[-1]), dim=-1)

    for cand_id in range(k):
        candidate = x_filled.clone()
        sampled_token_ids = []
        for pos_idx, abs_pos in enumerate(local_positions):
            dist = top_probs[pos_idx] / top_probs[pos_idx].sum()
            sampled_offset = torch.multinomial(dist, num_samples=1).item()
            sampled_token = top_ids[pos_idx, sampled_offset].item()
            candidate[0, abs_pos] = sampled_token
            sampled_token_ids.append(sampled_token)

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=candidate[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        verify = run_verifier_stack(
            task=task,
            full_code=segments["full_text"],
            completion_without_suffix=segments["middle_text"],
            timeout=cfg.verifier.timeout,
        )
        score = verifier_score(verify)
        meta = {
            "candidate_id": cand_id,
            "score": score,
            "sampled_token_ids": sampled_token_ids,
            "verification": {k_: v.to_dict() for k_, v in verify.items()},
            "full_text": segments["full_text"],
            "middle_text": segments["middle_text"],
        }
        candidates.append((score, candidate, meta))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_tensor, _ = candidates[0]
    return best_tensor, [meta for _, _, meta in candidates]


def run_structured_decode(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    ablation = _resolve_ablation_mode(cfg)

    prepared = prepare_model_inputs(task, tokenizer, cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    mask_token_id = prepared["mask_token_id"]
    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]
    num_mask_tokens = cfg.decode.num_mask_tokens
    total_steps = cfg.decode.total_steps

    step_traces: List[Dict[str, Any]] = []
    decode_start = time.perf_counter()

    first_pass_step: Optional[int] = None
    first_pass_code: Optional[str] = None
    early_stopped = False
    repair_state = _empty_repair_state()

    for step in range(total_steps):
        stage = get_verifier_stage(step, total_steps, cfg)

        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        current_mask_idx = (x_t == mask_token_id)
        x_filled = x_t.clone()
        x_filled[current_mask_idx] = preds[current_mask_idx]

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_filled[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )

        verifier_results = run_stage_verifier(
            task=task,
            full_code=segments["full_text"],
            completion_without_suffix=segments["middle_text"],
            stage=stage,
            cfg=cfg,
        )

        issue_signature = _build_issue_signature(stage, verifier_results)

        target_masks = linear_target_masks(num_mask_tokens, total_steps, step)
        low_conf_indices = select_low_confidence_mask_positions(
            max_probs=max_probs[:, middle_start:middle_end],
            current_mask_idx=current_mask_idx[:, middle_start:middle_end],
            target_masks=target_masks,
        )

        structural_units, selection_debug, weak_ranked_candidates = choose_units_from_verifier(
            full_text=segments["full_text"],
            prefix_text=segments["prefix_text"],
            middle_text=segments["middle_text"],
            verifier_results=verifier_results,
            cfg=cfg,
            repair_state=repair_state,
            issue_signature=issue_signature,
        )
        selection_debug["ablation_mode"] = ablation["mode"]

        distinct_candidate_count: Optional[int] = None

        # =========================
        # 位置候选：去重 -> 可选扩展 -> top3 probe
        # =========================
        if (
            stage == "semantic"
            and ablation["enable_position_probe"]
            and selection_debug.get("tier3_selection_strategy") == "weak_location_control_flow_candidates"
            and weak_ranked_candidates
        ):
            deduped_candidates, dedup_debug = _dedup_ranked_probe_candidates(
                weak_ranked_candidates,
                max_keep=POSITION_PROBE_TOPK,
            )
            selection_debug["weak_location_dedup"] = dedup_debug
            distinct_candidate_count = len(deduped_candidates)

            if ablation["enable_position_expansion"]:
                should_expand, expand_reason = _should_trigger_position_expansion(deduped_candidates, repair_state)
                if should_expand:
                    expanded_candidates, expand_debug = _maybe_expand_position_candidates(
                        full_text=segments["full_text"],
                        prefix_text=segments["prefix_text"],
                        middle_text=segments["middle_text"],
                        deduped_candidates=deduped_candidates,
                        repair_state=repair_state,
                    )
                    selection_debug["position_expand"] = dict(expand_debug, trigger_reason=expand_reason)
                    deduped_candidates = expanded_candidates
                    distinct_candidate_count = len(deduped_candidates)
                else:
                    selection_debug["position_expand"] = {
                        "triggered": False,
                        "reason": expand_reason,
                        "after_expand_count": len(deduped_candidates),
                    }
            else:
                selection_debug["position_expand"] = {
                    "triggered": False,
                    "reason": "ablation_mode_disabled",
                    "after_expand_count": len(deduped_candidates),
                }

            if len(deduped_candidates) >= 2:
                probed_unit, probe_debug = _probe_topk_weak_location_candidates(
                    tokenizer=tokenizer,
                    model=model,
                    x_filled=x_filled,
                    prepared=prepared,
                    prefix_text=segments["prefix_text"],
                    middle_text=segments["middle_text"],
                    task=task,
                    cfg=cfg,
                    ranked_candidates=deduped_candidates,
                    max_probe_k=POSITION_PROBE_TOPK,
                )
                selection_debug["weak_location_probe"] = probe_debug

                if probed_unit is not None:
                    structural_units = [probed_unit]
                    selection_debug["selected_unit"] = probed_unit.to_dict()
                    selection_debug["tier3_selection_strategy"] = "weak_location_top3_probe"
            else:
                selection_debug["weak_location_probe"] = {
                    "probe_applied": False,
                    "reason": "deduped_candidates_less_than_2",
                }
        else:
            selection_debug["position_expand"] = {
                "triggered": False,
                "reason": "not_applicable",
            }
            selection_debug["weak_location_probe"] = {
                "probe_applied": False,
                "reason": "not_applicable",
            }

        structural_indices = structural_units_to_canvas_indices(
            tokenizer=tokenizer,
            middle_text=segments["middle_text"],
            middle_length_tokens=num_mask_tokens,
            units=structural_units,
        )

        semantic_min_keep = 0
        if stage == "semantic" and structural_units:
            selected_meta = structural_units[0].metadata or {}
            selected_ast_type = selected_meta.get("ast_type")
            selected_unit_type = structural_units[0].unit_type
            origin_unit_type = selected_meta.get("origin_unit_type", selected_unit_type)

            if origin_unit_type == "control_flow_fragment" or selected_unit_type == "control_flow_fragment":
                structural_indices = _expand_indices_to_min_span(
                    structural_indices,
                    total_tokens=num_mask_tokens,
                    min_tokens=MIN_CONTROL_FLOW_STRUCTURAL_TOKENS,
                )
                semantic_min_keep = min(len(structural_indices), MIN_CONTROL_FLOW_STRUCTURAL_TOKENS)
                selection_debug["control_flow_min_span_applied"] = True

            elif selected_unit_type == "subtree" and (
                selected_ast_type in TINY_SEMANTIC_AST_TYPES or len(structural_indices) < MIN_SEMANTIC_STRUCTURAL_TOKENS
            ):
                structural_indices = _expand_indices_to_min_span(
                    structural_indices,
                    total_tokens=num_mask_tokens,
                    min_tokens=MIN_SEMANTIC_STRUCTURAL_TOKENS,
                )
                semantic_min_keep = min(len(structural_indices), MIN_SEMANTIC_STRUCTURAL_TOKENS)
                selection_debug["semantic_min_span_applied"] = True
                selection_debug["semantic_min_span_ast_type"] = selected_ast_type

            else:
                semantic_min_keep = min(len(structural_indices), MIN_SEMANTIC_STRUCTURAL_TOKENS)

        # =========================
        # 第十轮主线：fragment-local content candidates
        # =========================
        effective_segments = segments
        effective_verifier_results = verifier_results

        selected_origin_type = None
        if structural_units:
            selected_origin_type = (structural_units[0].metadata or {}).get("origin_unit_type", structural_units[0].unit_type)

        current_class_rank, current_class_label = _classify_verification(verifier_results)

        if (
            stage == "semantic"
            and ablation["enable_content_candidates"]
            and structural_units
            and selected_origin_type == "control_flow_fragment"
            and current_class_label == "feasible_but_tier3_fail"
        ):
            content_entries, content_generation_debug = _generate_fragment_content_candidates(
                tokenizer=tokenizer,
                logits=logits,
                x_filled=x_filled,
                prepared=prepared,
                selected_unit=structural_units[0],
                incumbent_segments=segments,
                incumbent_verifier_results=verifier_results,
                task=task,
                cfg=cfg,
                k=FRAGMENT_LOCAL_CONTENT_K,
            )
            best_content, content_rank_debug = _rank_fragment_content_candidates(content_entries)
            selection_debug["content_probe"] = {
                "applied": True,
                "selected_fragment_key": _fragment_key(structural_units[0]),
                "generation": content_generation_debug,
                **content_rank_debug,
            }

            if best_content is not None:
                x_filled = best_content["tensor"].clone()
                effective_segments = best_content["segments"]
                effective_verifier_results = best_content["verification_obj"]
        else:
            selection_debug["content_probe"] = {
                "applied": False,
                "reason": "disabled_or_not_feasible_tier3_fail",
                "current_class_rank": current_class_rank,
                "current_class_label": current_class_label,
            }

        merged_indices = merge_policy_indices(
            low_conf_indices,
            structural_indices,
            max_keep=target_masks,
            structural_min_keep=semantic_min_keep,
        )
        local_candidate_logs: List[Dict[str, Any]] = []

        if distinct_candidate_count is not None:
            repair_state = _update_position_diversity_state(repair_state, distinct_candidate_count)

        effective_issue_signature = _build_issue_signature(stage, effective_verifier_results)
        repair_state = _update_repair_state(repair_state, effective_issue_signature, cfg)
        repair_state = _update_fragment_history(repair_state, structural_units, effective_verifier_results)

        reconstruction_diag = build_reconstruction_diagnostics(task, effective_segments)
        reconstruction_diag["ablation_mode"] = ablation["mode"]
        reconstruction_diag["content_candidates_enabled"] = ablation["enable_content_candidates"]
        reconstruction_diag["position_expansion_enabled"] = ablation["enable_position_expansion"]

        tier3_effective = effective_verifier_results.get("tier3_unit_tests")
        if tier3_effective is not None and tier3_effective.passed:
            if first_pass_step is None:
                first_pass_step = step
                first_pass_code = effective_segments["full_text"]
            if not cfg.decode.shadow_mode:
                x_t = x_filled
                early_stopped = True
                step_traces.append(
                    {
                        "task_id": task.task_id,
                        "step": step,
                        "stage": stage,
                        "target_masks": target_masks,
                        "remaining_masks": 0,
                        "low_confidence_indices": low_conf_indices,
                        "structural_units": [u.to_dict() for u in structural_units],
                        "structural_indices": structural_indices,
                        "merged_indices": merged_indices,
                        "verifier_results": {k: v.to_dict() for k, v in effective_verifier_results.items()},
                        "verifier_called": True,
                        "early_stop_triggered": True,
                        "full_text": effective_segments["full_text"],
                        "middle_text": effective_segments["middle_text"],
                        "candidate_logs": local_candidate_logs,
                        "selection_debug": selection_debug,
                        "issue_signature": effective_issue_signature,
                        "repair_state_snapshot": dict(repair_state),
                        "reconstruction_diagnostics": reconstruction_diag,
                        "ablation_mode": ablation["mode"],
                    }
                )
                break

        if stage == "semantic" and structural_indices and cfg.decode.late_stage_candidate_k > 1 and (tier3_effective is None or not tier3_effective.passed):
            x_filled, local_candidate_logs = sample_local_candidates(
                tokenizer=tokenizer,
                x_filled=x_filled,
                logits=logits,
                middle_start=middle_start,
                middle_end=middle_end,
                structural_indices=structural_indices,
                task=task,
                prepared=prepared,
                cfg=cfg,
                k=cfg.decode.late_stage_candidate_k,
            )

        x_t = x_filled.clone()
        if merged_indices:
            x_t[0, [middle_start + idx for idx in merged_indices]] = mask_token_id

        step_traces.append(
            {
                "task_id": task.task_id,
                "step": step,
                "stage": stage,
                "target_masks": target_masks,
                "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                "low_confidence_indices": low_conf_indices,
                "structural_units": [u.to_dict() for u in structural_units],
                "structural_indices": structural_indices,
                "merged_indices": merged_indices,
                "verifier_results": {k: v.to_dict() for k, v in effective_verifier_results.items()},
                "verifier_called": stage != "blind",
                "early_stop_triggered": False,
                "full_text": effective_segments["full_text"],
                "middle_text": effective_segments["middle_text"],
                "candidate_logs": local_candidate_logs,
                "selection_debug": selection_debug,
                "issue_signature": effective_issue_signature,
                "repair_state_snapshot": dict(repair_state),
                "reconstruction_diagnostics": reconstruction_diag,
                "ablation_mode": ablation["mode"],
            }
        )

    total_decode_sec = time.perf_counter() - decode_start
    final_segments = _segment_decode(
        tokenizer=tokenizer,
        prefix_ids=prepared["prefix_ids"],
        middle_ids=x_t[0, middle_start:middle_end].tolist(),
        suffix_ids=prepared["suffix_ids"],
    )

    final_verification = run_verifier_stack(
        task=task,
        full_code=final_segments["full_text"],
        completion_without_suffix=final_segments["middle_text"],
        timeout=cfg.verifier.timeout,
    )

    backslide = False
    rescue = False
    wasted_steps_after_first_pass = None
    if first_pass_step is not None:
        final_tier3 = final_verification.get("tier3_unit_tests")
        if final_tier3 is not None:
            backslide = (first_pass_code is not None) and (not final_tier3.passed)
            wasted_steps_after_first_pass = total_steps - 1 - first_pass_step
    else:
        final_tier3 = final_verification.get("tier3_unit_tests")
        if final_tier3 is not None and final_tier3.passed:
            rescue = True

    diagnostics = build_reconstruction_diagnostics(task, final_segments)
    diagnostics["ablation_mode"] = ablation["mode"]
    diagnostics["content_candidates_enabled"] = ablation["enable_content_candidates"]
    diagnostics["position_expansion_enabled"] = ablation["enable_position_expansion"]

    return {
        "task_id": task.task_id,
        "task": task,
        "code": final_segments["full_text"],
        "prefix_text": final_segments["prefix_text"],
        "middle_text": final_segments["middle_text"],
        "suffix_text": final_segments["suffix_text"],
        "step_traces": step_traces,
        "metrics": {
            "passed": final_verification.get("tier3_unit_tests").passed if final_verification.get("tier3_unit_tests") else False,
            "decode_sec": total_decode_sec,
            "verification_sec": sum(v.duration_sec for v in final_verification.values()),
            "total_sec": total_decode_sec + sum(v.duration_sec for v in final_verification.values()),
            "early_stop": early_stopped,
            "first_pass_step": first_pass_step,
            "final_pass_step": total_steps - 1 if not early_stopped else first_pass_step,
            "backslide": backslide,
            "rescue": rescue,
            "wasted_steps_after_first_pass": wasted_steps_after_first_pass,
        },
        "verification": {k: v.to_dict() for k, v in final_verification.items()},
        "diagnostics": diagnostics,
        "prepared": prepared,
        "ablation_mode": ablation["mode"],
    }