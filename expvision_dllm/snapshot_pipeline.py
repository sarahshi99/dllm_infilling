from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .data_loader import CodeTask


# 新增文件原因：
# 1. 当前实验组织方式已切换到“统一 snapshot dump -> 离线 selector replay -> Tier3 稳定性检查”；
# 2. 旧版 S 系列代码把采样、排序、评测耦合在同一个运行脚本里，不适合这条新流水线；
# 3. 因此单独新增一个公共模块，集中放 schema、采样策略、selector 逻辑、轨迹摘要和通用读写函数。


FEATURE_SCHEMA_VERSION = "snapshot_schema_v3"
SAMPLING_POLICY_VERSION = "late_uniform_5points_v1"
SELECTOR_FAMILY_VERSION = "offline_selector_v1"
VERIFIER_CACHE_VERSION = "tier3_cache_v1"
BASELINE_CHAIN_VERSION = "snapshot_dump_chain_v1"


@dataclass
class SnapshotRecord:
    run_id: str
    task_id: str
    dataset_subset: str
    split: str
    dump_max_samples: Optional[int]
    seed: int
    total_steps: int
    sampling_policy_version: str
    feature_schema_version: str
    verifier_cache_version: str
    baseline_chain_version: str

    step: int
    full_text: str
    middle_text: str
    raw_code_hash: str
    normalized_code_hash: str

    mean_confidence: float
    min_confidence: float
    low_conf_count: int

    tier1_pass: bool
    tier2_pass: bool
    tier3_pass: Optional[bool]
    tier1_result: Dict[str, Any]
    tier2_result: Dict[str, Any]
    tier3_result: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotFeatureRecord:
    run_id: str
    task_id: str
    step: int
    normalized_code_hash: str
    tier1_pass: bool
    tier2_pass: bool
    tier3_pass: Optional[bool]
    mean_confidence: float
    min_confidence: float
    low_conf_count: int

    edit_distance_to_prev: int
    changed_token_count_to_prev: int
    same_as_prev: bool
    same_as_prev2: bool
    repeat_count_recent: int
    stable_run_length: int

    tier1_streak: int
    tier2_streak: int
    tier1_recent_consistency: float
    tier2_recent_consistency: float
    recent_drop_flag: bool

    conf_delta_last1: float
    conf_delta_last2: float
    conf_var_recent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectorDecision:
    run_id: str
    task_id: str
    selector_name: str
    selector_version: str
    candidate_steps: List[int]
    eligible_steps: List[int]
    excluded_reason_by_step: Dict[int, str]
    selected_step: int
    selected_score: float
    tie_break_reason: str
    per_step_scores: Dict[int, float]
    per_step_breakdown: Dict[int, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "selector_name": self.selector_name,
            "selector_version": self.selector_version,
            "candidate_steps": self.candidate_steps,
            "eligible_steps": self.eligible_steps,
            "excluded_reason_by_step": self.excluded_reason_by_step,
            "selected_step": self.selected_step,
            "selected_score": self.selected_score,
            "tie_break_reason": self.tie_break_reason,
            "per_step_scores": self.per_step_scores,
            "per_step_breakdown": self.per_step_breakdown,
        }


@dataclass
class TrajectorySummary:
    run_id: str
    task_id: str
    dataset_subset: str
    split: str
    dump_max_samples: Optional[int]
    seed: int
    total_steps: int
    sampling_policy_version: str
    feature_schema_version: str
    verifier_cache_version: str
    baseline_chain_version: str

    first_tier1_step: Optional[int]
    first_tier2_step: Optional[int]
    first_tier3_step: Optional[int]
    final_tier3_pass: Optional[bool]
    backslide_happened: Optional[bool]
    backslide_from_step: Optional[int]
    oracle_first_pass_step: Optional[int]
    num_distinct_late_candidates: int
    tier3_snapshot_summary_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_S3_WEIGHTS = {
    "same_as_prev": 2.0,
    "same_as_prev2": 1.0,
    "repeat_count_recent": 0.75,
    "stable_run_length": 0.25,
    "edit_distance_to_prev": 1.25,
    "changed_token_count_to_prev": 1.0,
}

_S4_WEIGHTS = {
    "tier1_streak": 0.50,
    "tier2_streak": 1.00,
    "tier1_recent_consistency": 0.50,
    "tier2_recent_consistency": 1.00,
    "recent_drop_flag": 1.00,
}

_S5_WEIGHTS = {
    "mean_confidence": 0.25,
    "conf_delta_last1": 0.50,
    "conf_delta_last2": 0.25,
    "conf_var_recent": 0.50,
    "low_conf_count": 0.25,
}


def make_run_id(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{timestamp}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_code_text(text: str) -> str:
    normalized_lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(normalized_lines).rstrip("\n")


def make_code_hash(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def resolve_sampling_steps(total_steps: int, sampling_policy_version: str) -> List[int]:
    safe_total_steps = max(1, int(total_steps))
    if sampling_policy_version != SAMPLING_POLICY_VERSION:
        raise ValueError(f"Unsupported sampling policy: {sampling_policy_version}")
    if safe_total_steps == 1:
        return [0]
    if safe_total_steps == 64:
        return [48, 52, 56, 60, 63]

    ratio_positions = [48 / 64.0, 52 / 64.0, 56 / 64.0, 60 / 64.0, 63 / 64.0]
    positions = [min(safe_total_steps - 1, max(0, int(round((safe_total_steps - 1) * ratio)))) for ratio in ratio_positions]
    positions[-1] = safe_total_steps - 1
    return sorted(set(positions))


def should_capture_step(step: int, capture_steps: Sequence[int]) -> bool:
    return int(step) in {int(item) for item in capture_steps}


def build_snapshot_record(
    *,
    run_id: str,
    task: "CodeTask",
    dataset_subset: str,
    split: str,
    dump_max_samples: Optional[int],
    seed: int,
    total_steps: int,
    step: int,
    full_text: str,
    middle_text: str,
    mean_confidence: float,
    min_confidence: float,
    low_conf_count: int,
    tier1_result: Dict[str, Any],
    tier2_result: Dict[str, Any],
    tier3_result: Optional[Dict[str, Any]],
) -> SnapshotRecord:
    normalized_text = normalize_code_text(full_text)
    return SnapshotRecord(
        run_id=run_id,
        task_id=task.task_id,
        dataset_subset=dataset_subset,
        split=split,
        dump_max_samples=dump_max_samples,
        seed=seed,
        total_steps=total_steps,
        sampling_policy_version=SAMPLING_POLICY_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        verifier_cache_version=VERIFIER_CACHE_VERSION,
        baseline_chain_version=BASELINE_CHAIN_VERSION,
        step=int(step),
        full_text=full_text,
        middle_text=middle_text,
        raw_code_hash=make_code_hash(full_text),
        normalized_code_hash=make_code_hash(normalized_text),
        mean_confidence=float(mean_confidence),
        min_confidence=float(min_confidence),
        low_conf_count=int(low_conf_count),
        tier1_pass=bool(tier1_result.get("passed", False)),
        tier2_pass=bool(tier2_result.get("passed", False)) if tier2_result else False,
        tier3_pass=bool(tier3_result.get("passed", False)) if tier3_result is not None else None,
        tier1_result=copy.deepcopy(tier1_result),
        tier2_result=copy.deepcopy(tier2_result),
        tier3_result=copy.deepcopy(tier3_result) if tier3_result is not None else None,
    )


def group_snapshot_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(str(item["task_id"]), []).append(item)
    for task_id in grouped:
        grouped[task_id] = sorted(grouped[task_id], key=lambda x: int(x["step"]))
    return grouped


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return float(sum((v - mu) ** 2 for v in values) / len(values))


def _normalized_penalty(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(float(value) / float(scale), 1.0)


def _char_edit_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    len_a = len(text_a)
    len_b = len(text_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        char_a = text_a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if char_a == text_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _changed_token_count(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    matcher = SequenceMatcher(a=tokens_a, b=tokens_b)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def build_snapshot_feature_records(records: Sequence[Dict[str, Any]], recent_window: int) -> List[SnapshotFeatureRecord]:
    if not records:
        return []
    safe_window = max(2, int(recent_window))
    out: List[SnapshotFeatureRecord] = []
    for index, record in enumerate(records):
        prev = records[index - 1] if index >= 1 else None
        prev2 = records[index - 2] if index >= 2 else None
        current_hash = str(record["normalized_code_hash"])
        same_as_prev = bool(prev is not None and current_hash == str(prev["normalized_code_hash"]))
        same_as_prev2 = bool(prev2 is not None and current_hash == str(prev2["normalized_code_hash"]))

        if prev is None:
            edit_distance_to_prev = 0
            changed_token_count_to_prev = 0
            conf_delta_last1 = 0.0
        else:
            edit_distance_to_prev = _char_edit_distance(str(record["full_text"]), str(prev["full_text"]))
            changed_token_count_to_prev = _changed_token_count(str(record["middle_text"]), str(prev["middle_text"]))
            conf_delta_last1 = abs(float(record.get("mean_confidence", 0.0)) - float(prev.get("mean_confidence", 0.0)))

        if prev2 is None:
            conf_delta_last2 = 0.0
        else:
            conf_delta_last2 = abs(float(record.get("mean_confidence", 0.0)) - float(prev2.get("mean_confidence", 0.0)))

        recent_start = max(0, index - safe_window + 1)
        recent = records[recent_start:index + 1]
        recent_hashes = [str(item["normalized_code_hash"]) for item in recent]
        recent_mean_conf = [float(item.get("mean_confidence", 0.0)) for item in recent]
        repeat_count_recent = sum(1 for item in recent_hashes if item == current_hash)

        stable_run_length = 1
        for back_idx in range(index - 1, -1, -1):
            if str(records[back_idx]["normalized_code_hash"]) == current_hash:
                stable_run_length += 1
            else:
                break

        tier1_streak = 0
        for back_idx in range(index, -1, -1):
            if bool(records[back_idx].get("tier1_pass", False)):
                tier1_streak += 1
            else:
                break

        tier2_streak = 0
        for back_idx in range(index, -1, -1):
            if bool(records[back_idx].get("tier2_pass", False)):
                tier2_streak += 1
            else:
                break

        tier1_recent_consistency = _mean([1.0 if bool(item.get("tier1_pass", False)) else 0.0 for item in recent])
        tier2_recent_consistency = _mean([1.0 if bool(item.get("tier2_pass", False)) else 0.0 for item in recent])

        recent_drop_flag = False
        if prev is not None:
            recent_drop_flag = (bool(prev.get("tier1_pass", False)) and not bool(record.get("tier1_pass", False))) or (
                bool(prev.get("tier2_pass", False)) and not bool(record.get("tier2_pass", False))
            )

        out.append(
            SnapshotFeatureRecord(
                run_id=str(record["run_id"]),
                task_id=str(record["task_id"]),
                step=int(record["step"]),
                normalized_code_hash=current_hash,
                tier1_pass=bool(record.get("tier1_pass", False)),
                tier2_pass=bool(record.get("tier2_pass", False)),
                tier3_pass=record.get("tier3_pass"),
                mean_confidence=float(record.get("mean_confidence", 0.0)),
                min_confidence=float(record.get("min_confidence", 0.0)),
                low_conf_count=int(record.get("low_conf_count", 0)),
                edit_distance_to_prev=edit_distance_to_prev,
                changed_token_count_to_prev=changed_token_count_to_prev,
                same_as_prev=same_as_prev,
                same_as_prev2=same_as_prev2,
                repeat_count_recent=repeat_count_recent,
                stable_run_length=stable_run_length,
                tier1_streak=tier1_streak,
                tier2_streak=tier2_streak,
                tier1_recent_consistency=tier1_recent_consistency,
                tier2_recent_consistency=tier2_recent_consistency,
                recent_drop_flag=recent_drop_flag,
                conf_delta_last1=conf_delta_last1,
                conf_delta_last2=conf_delta_last2,
                conf_var_recent=_variance(recent_mean_conf),
            )
        )
    return out


def score_feature_record(feature: SnapshotFeatureRecord, selector_name: str) -> Dict[str, Any]:
    selector = str(selector_name).upper()
    breakdown: Dict[str, Any] = {
        "stability_score": 0.0,
        "consistency_score": 0.0,
        "confidence_score": 0.0,
    }

    if selector in {"S3", "S4", "S5"}:
        breakdown["same_as_prev_bonus"] = _S3_WEIGHTS["same_as_prev"] * int(feature.same_as_prev)
        breakdown["same_as_prev2_bonus"] = _S3_WEIGHTS["same_as_prev2"] * int(feature.same_as_prev2)
        breakdown["repeat_bonus"] = _S3_WEIGHTS["repeat_count_recent"] * max(0, feature.repeat_count_recent - 1)
        breakdown["stable_run_bonus"] = _S3_WEIGHTS["stable_run_length"] * max(0, feature.stable_run_length - 1)
        breakdown["edit_distance_penalty"] = _S3_WEIGHTS["edit_distance_to_prev"] * _normalized_penalty(feature.edit_distance_to_prev, 32.0)
        breakdown["changed_token_penalty"] = _S3_WEIGHTS["changed_token_count_to_prev"] * _normalized_penalty(feature.changed_token_count_to_prev, 16.0)
        breakdown["stability_score"] = (
            breakdown["same_as_prev_bonus"]
            + breakdown["same_as_prev2_bonus"]
            + breakdown["repeat_bonus"]
            + breakdown["stable_run_bonus"]
            - breakdown["edit_distance_penalty"]
            - breakdown["changed_token_penalty"]
        )

    if selector in {"S4", "S5"}:
        breakdown["tier1_streak_bonus"] = _S4_WEIGHTS["tier1_streak"] * _normalized_penalty(feature.tier1_streak, 4.0)
        breakdown["tier2_streak_bonus"] = _S4_WEIGHTS["tier2_streak"] * _normalized_penalty(feature.tier2_streak, 4.0)
        breakdown["tier1_consistency_bonus"] = _S4_WEIGHTS["tier1_recent_consistency"] * feature.tier1_recent_consistency
        breakdown["tier2_consistency_bonus"] = _S4_WEIGHTS["tier2_recent_consistency"] * feature.tier2_recent_consistency
        breakdown["recent_drop_penalty"] = _S4_WEIGHTS["recent_drop_flag"] * int(feature.recent_drop_flag)
        breakdown["consistency_score"] = (
            breakdown["tier1_streak_bonus"]
            + breakdown["tier2_streak_bonus"]
            + breakdown["tier1_consistency_bonus"]
            + breakdown["tier2_consistency_bonus"]
            - breakdown["recent_drop_penalty"]
        )

    if selector == "S5":
        breakdown["mean_conf_bonus"] = _S5_WEIGHTS["mean_confidence"] * feature.mean_confidence
        breakdown["conf_delta_last1_penalty"] = _S5_WEIGHTS["conf_delta_last1"] * feature.conf_delta_last1
        breakdown["conf_delta_last2_penalty"] = _S5_WEIGHTS["conf_delta_last2"] * feature.conf_delta_last2
        breakdown["conf_var_penalty"] = _S5_WEIGHTS["conf_var_recent"] * feature.conf_var_recent
        breakdown["low_conf_penalty"] = _S5_WEIGHTS["low_conf_count"] * _normalized_penalty(feature.low_conf_count, 16.0)
        breakdown["confidence_score"] = (
            breakdown["mean_conf_bonus"]
            - breakdown["conf_delta_last1_penalty"]
            - breakdown["conf_delta_last2_penalty"]
            - breakdown["conf_var_penalty"]
            - breakdown["low_conf_penalty"]
        )

    breakdown["total_score"] = breakdown["stability_score"] + breakdown["consistency_score"] + breakdown["confidence_score"]
    return breakdown


def replay_selector(
    *,
    run_id: str,
    task_id: str,
    records: Sequence[Dict[str, Any]],
    selector_name: str,
    recent_window: int,
    consistency_window: int,
) -> Tuple[Dict[str, Any], SelectorDecision, List[Dict[str, Any]], TrajectorySummary]:
    features = build_snapshot_feature_records(records, recent_window=recent_window)
    grouped_meta = records[0]
    trajectory_summary = build_trajectory_summary(records)

    selector = str(selector_name).upper()
    candidate_steps = [int(r["step"]) for r in records]
    eligible_steps: List[int] = []
    excluded_reason_by_step: Dict[int, str] = {}
    per_step_scores: Dict[int, float] = {}
    per_step_breakdown: Dict[int, Dict[str, Any]] = {}

    if selector == "S0":
        selected_record = records[-1]
        for record in records:
            per_step_scores[int(record["step"])] = 0.0
            per_step_breakdown[int(record["step"])] = {"total_score": 0.0}
        decision = SelectorDecision(
            run_id=run_id,
            task_id=task_id,
            selector_name=selector,
            selector_version=SELECTOR_FAMILY_VERSION,
            candidate_steps=candidate_steps,
            eligible_steps=candidate_steps,
            excluded_reason_by_step={},
            selected_step=int(selected_record["step"]),
            selected_score=0.0,
            tie_break_reason="final_snapshot_passthrough",
            per_step_scores=per_step_scores,
            per_step_breakdown=per_step_breakdown,
        )

    elif selector == "S1":
        ranked = sorted(
            features,
            key=lambda f: (int(f.tier2_pass), int(f.tier1_pass), f.stable_run_length, -f.step, f.mean_confidence),
            reverse=True,
        )
        selected_feature = ranked[0]
        selected_record = next(record for record in records if int(record["step"]) == int(selected_feature.step))
        for record in records:
            per_step_scores[int(record["step"])] = 0.0
            per_step_breakdown[int(record["step"])] = {"total_score": 0.0}
        decision = SelectorDecision(
            run_id=run_id,
            task_id=task_id,
            selector_name=selector,
            selector_version=SELECTOR_FAMILY_VERSION,
            candidate_steps=candidate_steps,
            eligible_steps=candidate_steps,
            excluded_reason_by_step={},
            selected_step=int(selected_record["step"]),
            selected_score=0.0,
            tie_break_reason="legacy_proxy_rank_tier2_tier1_stable_run_earlier_step_mean_confidence",
            per_step_scores=per_step_scores,
            per_step_breakdown=per_step_breakdown,
        )

    elif selector == "S2":
        selected_record = None
        for feature in features:
            if feature.tier1_pass and feature.tier2_pass and feature.stable_run_length >= max(2, consistency_window):
                eligible_steps.append(int(feature.step))
                if selected_record is None:
                    selected_record = next(record for record in records if int(record["step"]) == int(feature.step))
            else:
                excluded_reason_by_step[int(feature.step)] = "fails_cheap_consistency_gate"

        if selected_record is None:
            ranked = sorted(
                features,
                key=lambda f: (int(f.tier2_pass), int(f.tier1_pass), f.stable_run_length, -f.step, f.mean_confidence),
                reverse=True,
            )
            selected_feature = ranked[0]
            selected_record = next(record for record in records if int(record["step"]) == int(selected_feature.step))
            tie_break_reason = "cheap_consistency_fallback_to_proxy"
            if not eligible_steps:
                eligible_steps = candidate_steps[:]
        else:
            tie_break_reason = "first_snapshot_passing_cheap_consistency_gate"

        for record in records:
            per_step_scores[int(record["step"])] = 0.0
            per_step_breakdown[int(record["step"])] = {"total_score": 0.0}

        decision = SelectorDecision(
            run_id=run_id,
            task_id=task_id,
            selector_name=selector,
            selector_version=SELECTOR_FAMILY_VERSION,
            candidate_steps=candidate_steps,
            eligible_steps=eligible_steps,
            excluded_reason_by_step=excluded_reason_by_step,
            selected_step=int(selected_record["step"]),
            selected_score=0.0,
            tie_break_reason=tie_break_reason,
            per_step_scores=per_step_scores,
            per_step_breakdown=per_step_breakdown,
        )

    elif selector in {"S3", "S4", "S5"}:
        scored_items: List[Tuple[SnapshotFeatureRecord, Dict[str, Any]]] = []
        for feature in features:
            breakdown = score_feature_record(feature, selector)
            scored_items.append((feature, breakdown))
            per_step_scores[int(feature.step)] = float(breakdown["total_score"])
            per_step_breakdown[int(feature.step)] = breakdown

        ranked = sorted(
            scored_items,
            key=lambda item: (item[1]["total_score"], -item[0].step, item[0].mean_confidence),
            reverse=True,
        )
        selected_feature, selected_breakdown = ranked[0]
        selected_record = next(record for record in records if int(record["step"]) == int(selected_feature.step))

        decision = SelectorDecision(
            run_id=run_id,
            task_id=task_id,
            selector_name=selector,
            selector_version=SELECTOR_FAMILY_VERSION,
            candidate_steps=candidate_steps,
            eligible_steps=candidate_steps,
            excluded_reason_by_step={},
            selected_step=int(selected_record["step"]),
            selected_score=float(selected_breakdown["total_score"]),
            tie_break_reason="score_desc_then_earlier_step_then_mean_confidence",
            per_step_scores=per_step_scores,
            per_step_breakdown=per_step_breakdown,
        )

    else:
        raise ValueError(f"Unsupported selector_name: {selector_name}")

    trajectory = TrajectorySummary(
        run_id=run_id,
        task_id=task_id,
        dataset_subset=str(grouped_meta["dataset_subset"]),
        split=str(grouped_meta["split"]),
        dump_max_samples=grouped_meta.get("dump_max_samples"),
        seed=int(grouped_meta["seed"]),
        total_steps=int(grouped_meta["total_steps"]),
        sampling_policy_version=str(grouped_meta["sampling_policy_version"]),
        feature_schema_version=str(grouped_meta["feature_schema_version"]),
        verifier_cache_version=str(grouped_meta["verifier_cache_version"]),
        baseline_chain_version=str(grouped_meta["baseline_chain_version"]),
        **trajectory_summary,
    )
    selected_payload = copy.deepcopy(selected_record)
    return selected_payload, decision, [feature.to_dict() for feature in features], trajectory


def build_trajectory_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_tier1_step = next((int(record["step"]) for record in records if bool(record.get("tier1_pass", False))), None)
    first_tier2_step = next((int(record["step"]) for record in records if bool(record.get("tier2_pass", False))), None)
    tier3_available = all(record.get("tier3_pass") is not None for record in records)

    if not tier3_available:
        return {
            "first_tier1_step": first_tier1_step,
            "first_tier2_step": first_tier2_step,
            "first_tier3_step": None,
            "final_tier3_pass": None,
            "backslide_happened": None,
            "backslide_from_step": None,
            "oracle_first_pass_step": None,
            "num_distinct_late_candidates": len({str(record["normalized_code_hash"]) for record in records}),
            "tier3_snapshot_summary_available": False,
        }

    first_tier3_step = next((int(record["step"]) for record in records if bool(record.get("tier3_pass", False))), None)
    final_tier3_pass = bool(records[-1].get("tier3_pass", False))
    backslide_happened = first_tier3_step is not None and not final_tier3_pass

    backslide_from_step = None
    if first_tier3_step is not None:
        for record in records:
            if int(record["step"]) > int(first_tier3_step) and not bool(record.get("tier3_pass", False)):
                backslide_from_step = int(record["step"])
                break

    return {
        "first_tier1_step": first_tier1_step,
        "first_tier2_step": first_tier2_step,
        "first_tier3_step": first_tier3_step,
        "final_tier3_pass": final_tier3_pass,
        "backslide_happened": backslide_happened,
        "backslide_from_step": backslide_from_step,
        "oracle_first_pass_step": first_tier3_step,
        "num_distinct_late_candidates": len({str(record["normalized_code_hash"]) for record in records}),
        "tier3_snapshot_summary_available": True,
    }


def summarize_replay_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}
    num_samples = len(results)
    passed = sum(1 for item in results if bool(item["passed"]))
    selected_steps = [float(item["selected_step"]) for item in results]
    return {
        "num_samples": num_samples,
        "pass_rate": passed / num_samples,
        "avg_selected_snapshot_step": _mean(selected_steps),
        "selector_names": sorted({str(item["selector_name"]) for item in results}),
    }


def compare_selector_outcomes(base_results: Sequence[Dict[str, Any]], other_results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    base_map = {str(item["task_id"]): bool(item["passed"]) for item in base_results}
    other_map = {str(item["task_id"]): bool(item["passed"]) for item in other_results}
    wins = losses = ties = 0
    for task_id, base_pass in base_map.items():
        other_pass = other_map[task_id]
        if other_pass and not base_pass:
            wins += 1
        elif base_pass and not other_pass:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def rerun_tier3_for_snapshot(task: "CodeTask", snapshot_record: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    # 延迟导入原因：
    # 1. 离线 replay 本身不需要 human_eval / human_eval_infilling；
    # 2. 如果在模块顶层直接导入 verifier，会让只做 replay 的脚本也强依赖完整评测环境。
    from .verifier import tier3_unit_tests

    rerun = tier3_unit_tests(task, str(snapshot_record["middle_text"]), timeout=timeout).to_dict()
    return rerun