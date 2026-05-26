from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# 新增文件原因：
# 1. 当前接口已经冻结，S 系列需要显式区分原始 snapshot 表、selector 决策表、sample 轨迹摘要表；
# 2. 旧版 decode_s_series.py 把采样、特征、排序、决策日志混在一起，无法继续保持 schema 干净；
# 3. 因此把“采样策略”“snapshot 特征构造”“selector 打分/决策”“轨迹摘要”拆到独立模块。


@dataclass
class SnapshotRecord:
    run_id: str
    sample_id: str
    dataset_subset: str
    split: str
    max_samples: Optional[int]
    seed: int
    total_steps: int
    sampling_policy_version: str
    feature_schema_version: str
    selector_version: str
    verifier_version: str
    baseline_chain_version: str

    step: int
    full_text: str
    middle_text: str
    raw_code_hash: str
    normalized_code_hash: str

    tier1_pass: bool
    tier2_pass: bool

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

    proxy_verification: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectorScoreBreakdown:
    total_score: float
    stability_score: float
    consistency_score: float
    confidence_score: float

    same_as_prev_bonus: float
    same_as_prev2_bonus: float
    repeat_bonus: float
    stable_run_bonus: float
    edit_distance_penalty: float
    changed_token_penalty: float

    tier1_streak_bonus: float
    tier2_streak_bonus: float
    tier1_consistency_bonus: float
    tier2_consistency_bonus: float
    recent_drop_penalty: float

    mean_conf_bonus: float
    conf_delta_last1_penalty: float
    conf_delta_last2_penalty: float
    conf_var_penalty: float
    low_conf_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectorDecision:
    run_id: str
    sample_id: str
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
            "sample_id": self.sample_id,
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
class SampleTrajectorySummary:
    run_id: str
    sample_id: str
    dataset_subset: str
    split: str
    max_samples: Optional[int]
    seed: int
    total_steps: int
    sampling_policy_version: str
    feature_schema_version: str
    selector_version: str
    verifier_version: str
    baseline_chain_version: str

    first_tier1_step: Optional[int]
    first_tier2_step: Optional[int]
    first_tier3_step: Optional[int]
    final_tier3_pass: bool
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


def normalize_code_text(text: str) -> str:
    normalized_lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = "\n".join(normalized_lines).rstrip("\n")
    return normalized


def make_code_hash(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def resolve_sampling_steps(total_steps: int, sampling_policy_version: str) -> List[int]:
    safe_total_steps = max(1, int(total_steps))
    if sampling_policy_version != "late_uniform_5points_v1":
        raise ValueError(f"Unsupported sampling policy: {sampling_policy_version}")

    # 当前冻结的 policy：
    # - total_steps=64 时，显式对齐到 {48, 52, 56, 60, 63}；
    # - 其他 total_steps 时，按这组相对位置做映射，并强制包含最终 step。
    if safe_total_steps == 1:
        return [0]
    if safe_total_steps == 64:
        return [48, 52, 56, 60, 63]

    ratio_positions = [48 / 64.0, 52 / 64.0, 56 / 64.0, 60 / 64.0, 63 / 64.0]
    positions = [min(safe_total_steps - 1, max(0, int(round((safe_total_steps - 1) * ratio)))) for ratio in ratio_positions]
    positions[-1] = safe_total_steps - 1
    return sorted(set(positions))


def should_capture_step(step: int, capture_steps: Set[int]) -> bool:
    return int(step) in capture_steps


def _normalized_penalty(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(float(value) / float(scale), 1.0)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return float(sum((v - mean_value) ** 2 for v in values) / len(values))


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
    if not tokens_a and not tokens_b:
        return 0
    matcher = SequenceMatcher(a=tokens_a, b=tokens_b)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _extract_proxy_pass(snapshot: Dict[str, Any], tier_name: str) -> bool:
    proxy = snapshot.get("proxy_verification", {})
    tier_result = proxy.get(tier_name, {})
    return bool(tier_result.get("passed", False))


def build_raw_snapshot(
    *,
    full_text: str,
    middle_text: str,
    step: int,
    mean_confidence: float,
    min_confidence: float,
    low_conf_count: int,
    proxy_verification: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_text = normalize_code_text(full_text)
    return {
        "step": int(step),
        "full_text": full_text,
        "middle_text": middle_text,
        "raw_code_hash": make_code_hash(full_text),
        "normalized_code_hash": make_code_hash(normalized_text),
        "mean_confidence": float(mean_confidence),
        "min_confidence": float(min_confidence),
        "low_conf_count": int(low_conf_count),
        "proxy_verification": copy.deepcopy(proxy_verification),
    }


def build_snapshot_records(
    *,
    run_id: str,
    sample_id: str,
    dataset_subset: str,
    split: str,
    max_samples: Optional[int],
    seed: int,
    total_steps: int,
    sampling_policy_version: str,
    feature_schema_version: str,
    selector_version: str,
    verifier_version: str,
    baseline_chain_version: str,
    raw_snapshots: Sequence[Dict[str, Any]],
    recent_window: int,
) -> List[SnapshotRecord]:
    if not raw_snapshots:
        return []

    records: List[SnapshotRecord] = []
    safe_window = max(2, int(recent_window))

    for index, snapshot in enumerate(raw_snapshots):
        prev_snapshot = raw_snapshots[index - 1] if index >= 1 else None
        prev2_snapshot = raw_snapshots[index - 2] if index >= 2 else None

        current_full_text = str(snapshot["full_text"])
        current_middle_text = str(snapshot["middle_text"])
        current_normalized_hash = str(snapshot["normalized_code_hash"])

        tier1_pass = _extract_proxy_pass(snapshot, "tier1_parse_compile")
        tier2_pass = _extract_proxy_pass(snapshot, "tier2_smoke_exec")

        same_as_prev = bool(prev_snapshot is not None and current_normalized_hash == str(prev_snapshot["normalized_code_hash"]))
        same_as_prev2 = bool(prev2_snapshot is not None and current_normalized_hash == str(prev2_snapshot["normalized_code_hash"]))

        if prev_snapshot is None:
            edit_distance_to_prev = 0
            changed_token_count_to_prev = 0
            conf_delta_last1 = 0.0
        else:
            prev_full_text = str(prev_snapshot["full_text"])
            edit_distance_to_prev = _char_edit_distance(current_full_text, prev_full_text)
            changed_token_count_to_prev = _changed_token_count(current_middle_text, str(prev_snapshot["middle_text"]))
            conf_delta_last1 = abs(float(snapshot.get("mean_confidence", 0.0)) - float(prev_snapshot.get("mean_confidence", 0.0)))

        if prev2_snapshot is None:
            conf_delta_last2 = 0.0
        else:
            conf_delta_last2 = abs(float(snapshot.get("mean_confidence", 0.0)) - float(prev2_snapshot.get("mean_confidence", 0.0)))

        recent_start = max(0, index - safe_window + 1)
        recent_snaps = raw_snapshots[recent_start:index + 1]
        recent_mean_confidences = [float(item.get("mean_confidence", 0.0)) for item in recent_snaps]
        recent_hashes = [str(item["normalized_code_hash"]) for item in recent_snaps]

        tier1_recent_consistency = _mean([1.0 if _extract_proxy_pass(item, "tier1_parse_compile") else 0.0 for item in recent_snaps])
        tier2_recent_consistency = _mean([1.0 if _extract_proxy_pass(item, "tier2_smoke_exec") else 0.0 for item in recent_snaps])

        tier1_streak = 0
        for back_idx in range(index, -1, -1):
            if _extract_proxy_pass(raw_snapshots[back_idx], "tier1_parse_compile"):
                tier1_streak += 1
            else:
                break

        tier2_streak = 0
        for back_idx in range(index, -1, -1):
            if _extract_proxy_pass(raw_snapshots[back_idx], "tier2_smoke_exec"):
                tier2_streak += 1
            else:
                break

        recent_drop_flag = False
        if prev_snapshot is not None:
            prev_tier1_pass = _extract_proxy_pass(prev_snapshot, "tier1_parse_compile")
            prev_tier2_pass = _extract_proxy_pass(prev_snapshot, "tier2_smoke_exec")
            recent_drop_flag = (prev_tier1_pass and not tier1_pass) or (prev_tier2_pass and not tier2_pass)

        repeat_count_recent = sum(1 for item in recent_hashes if item == current_normalized_hash)
        stable_run_length = 1
        for back_idx in range(index - 1, -1, -1):
            if str(raw_snapshots[back_idx]["normalized_code_hash"]) == current_normalized_hash:
                stable_run_length += 1
            else:
                break

        records.append(
            SnapshotRecord(
                run_id=run_id,
                sample_id=sample_id,
                dataset_subset=dataset_subset,
                split=split,
                max_samples=max_samples,
                seed=seed,
                total_steps=total_steps,
                sampling_policy_version=sampling_policy_version,
                feature_schema_version=feature_schema_version,
                selector_version=selector_version,
                verifier_version=verifier_version,
                baseline_chain_version=baseline_chain_version,
                step=int(snapshot["step"]),
                full_text=current_full_text,
                middle_text=current_middle_text,
                raw_code_hash=str(snapshot["raw_code_hash"]),
                normalized_code_hash=current_normalized_hash,
                tier1_pass=tier1_pass,
                tier2_pass=tier2_pass,
                mean_confidence=float(snapshot.get("mean_confidence", 0.0)),
                min_confidence=float(snapshot.get("min_confidence", 0.0)),
                low_conf_count=int(snapshot.get("low_conf_count", 0)),
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
                conf_var_recent=_variance(recent_mean_confidences),
                proxy_verification=copy.deepcopy(snapshot.get("proxy_verification", {})),
            )
        )
    return records


def _empty_breakdown() -> SelectorScoreBreakdown:
    return SelectorScoreBreakdown(
        total_score=0.0,
        stability_score=0.0,
        consistency_score=0.0,
        confidence_score=0.0,
        same_as_prev_bonus=0.0,
        same_as_prev2_bonus=0.0,
        repeat_bonus=0.0,
        stable_run_bonus=0.0,
        edit_distance_penalty=0.0,
        changed_token_penalty=0.0,
        tier1_streak_bonus=0.0,
        tier2_streak_bonus=0.0,
        tier1_consistency_bonus=0.0,
        tier2_consistency_bonus=0.0,
        recent_drop_penalty=0.0,
        mean_conf_bonus=0.0,
        conf_delta_last1_penalty=0.0,
        conf_delta_last2_penalty=0.0,
        conf_var_penalty=0.0,
        low_conf_penalty=0.0,
    )


def score_snapshot_s3(record: SnapshotRecord) -> SelectorScoreBreakdown:
    same_as_prev_bonus = _S3_WEIGHTS["same_as_prev"] * int(record.same_as_prev)
    same_as_prev2_bonus = _S3_WEIGHTS["same_as_prev2"] * int(record.same_as_prev2)
    repeat_bonus = _S3_WEIGHTS["repeat_count_recent"] * max(0, record.repeat_count_recent - 1)
    stable_run_bonus = _S3_WEIGHTS["stable_run_length"] * max(0, record.stable_run_length - 1)
    edit_distance_penalty = _S3_WEIGHTS["edit_distance_to_prev"] * _normalized_penalty(record.edit_distance_to_prev, scale=32.0)
    changed_token_penalty = _S3_WEIGHTS["changed_token_count_to_prev"] * _normalized_penalty(record.changed_token_count_to_prev, scale=16.0)

    stability_score = same_as_prev_bonus + same_as_prev2_bonus + repeat_bonus + stable_run_bonus - edit_distance_penalty - changed_token_penalty
    breakdown = _empty_breakdown()
    breakdown.total_score = stability_score
    breakdown.stability_score = stability_score
    breakdown.same_as_prev_bonus = same_as_prev_bonus
    breakdown.same_as_prev2_bonus = same_as_prev2_bonus
    breakdown.repeat_bonus = repeat_bonus
    breakdown.stable_run_bonus = stable_run_bonus
    breakdown.edit_distance_penalty = edit_distance_penalty
    breakdown.changed_token_penalty = changed_token_penalty
    return breakdown


def score_snapshot_s4(record: SnapshotRecord) -> SelectorScoreBreakdown:
    breakdown = score_snapshot_s3(record)
    tier1_streak_bonus = _S4_WEIGHTS["tier1_streak"] * _normalized_penalty(record.tier1_streak, scale=4.0)
    tier2_streak_bonus = _S4_WEIGHTS["tier2_streak"] * _normalized_penalty(record.tier2_streak, scale=4.0)
    tier1_consistency_bonus = _S4_WEIGHTS["tier1_recent_consistency"] * record.tier1_recent_consistency
    tier2_consistency_bonus = _S4_WEIGHTS["tier2_recent_consistency"] * record.tier2_recent_consistency
    recent_drop_penalty = _S4_WEIGHTS["recent_drop_flag"] * int(record.recent_drop_flag)

    consistency_score = tier1_streak_bonus + tier2_streak_bonus + tier1_consistency_bonus + tier2_consistency_bonus - recent_drop_penalty
    breakdown.consistency_score = consistency_score
    breakdown.tier1_streak_bonus = tier1_streak_bonus
    breakdown.tier2_streak_bonus = tier2_streak_bonus
    breakdown.tier1_consistency_bonus = tier1_consistency_bonus
    breakdown.tier2_consistency_bonus = tier2_consistency_bonus
    breakdown.recent_drop_penalty = recent_drop_penalty
    breakdown.total_score = breakdown.stability_score + consistency_score
    return breakdown


def score_snapshot_s5(record: SnapshotRecord) -> SelectorScoreBreakdown:
    breakdown = score_snapshot_s4(record)
    mean_conf_bonus = _S5_WEIGHTS["mean_confidence"] * record.mean_confidence
    conf_delta_last1_penalty = _S5_WEIGHTS["conf_delta_last1"] * record.conf_delta_last1
    conf_delta_last2_penalty = _S5_WEIGHTS["conf_delta_last2"] * record.conf_delta_last2
    conf_var_penalty = _S5_WEIGHTS["conf_var_recent"] * record.conf_var_recent
    low_conf_penalty = _S5_WEIGHTS["low_conf_count"] * _normalized_penalty(record.low_conf_count, scale=16.0)

    confidence_score = mean_conf_bonus - conf_delta_last1_penalty - conf_delta_last2_penalty - conf_var_penalty - low_conf_penalty
    breakdown.confidence_score = confidence_score
    breakdown.mean_conf_bonus = mean_conf_bonus
    breakdown.conf_delta_last1_penalty = conf_delta_last1_penalty
    breakdown.conf_delta_last2_penalty = conf_delta_last2_penalty
    breakdown.conf_var_penalty = conf_var_penalty
    breakdown.low_conf_penalty = low_conf_penalty
    breakdown.total_score = breakdown.stability_score + breakdown.consistency_score + confidence_score
    return breakdown


def _build_proxy_decision(
    *,
    run_id: str,
    sample_id: str,
    selector_version: str,
    records: Sequence[SnapshotRecord],
) -> Tuple[SnapshotRecord, SelectorDecision]:
    ranked = sorted(
        records,
        key=lambda record: (
            int(record.tier2_pass),
            int(record.tier1_pass),
            record.stable_run_length,
            -record.step,
            record.mean_confidence,
        ),
        reverse=True,
    )
    selected = ranked[0]
    empty = _empty_breakdown().to_dict()
    decision = SelectorDecision(
        run_id=run_id,
        sample_id=sample_id,
        selector_version=selector_version,
        candidate_steps=[record.step for record in records],
        eligible_steps=[record.step for record in records],
        excluded_reason_by_step={},
        selected_step=selected.step,
        selected_score=0.0,
        tie_break_reason="legacy_proxy_rank_tier2_tier1_stable_run_earlier_step_mean_confidence",
        per_step_scores={record.step: 0.0 for record in records},
        per_step_breakdown={record.step: empty for record in records},
    )
    return selected, decision


def _build_consistency_decision(
    *,
    run_id: str,
    sample_id: str,
    selector_version: str,
    records: Sequence[SnapshotRecord],
    consistency_window: int,
) -> Tuple[SnapshotRecord, SelectorDecision]:
    excluded_reason_by_step: Dict[int, str] = {}
    eligible_steps: List[int] = []
    selected: Optional[SnapshotRecord] = None
    for record in records:
        if record.tier1_pass and record.tier2_pass and record.stable_run_length >= max(2, consistency_window):
            eligible_steps.append(record.step)
            if selected is None:
                selected = record
        else:
            excluded_reason_by_step[record.step] = "fails_cheap_consistency_gate"
    if selected is None:
        selected, fallback = _build_proxy_decision(
            run_id=run_id,
            sample_id=sample_id,
            selector_version=selector_version,
            records=records,
        )
        fallback.eligible_steps = eligible_steps if eligible_steps else fallback.eligible_steps
        fallback.excluded_reason_by_step = excluded_reason_by_step
        fallback.tie_break_reason = "cheap_consistency_fallback_to_proxy"
        return selected, fallback

    empty = _empty_breakdown().to_dict()
    decision = SelectorDecision(
        run_id=run_id,
        sample_id=sample_id,
        selector_version=selector_version,
        candidate_steps=[record.step for record in records],
        eligible_steps=eligible_steps,
        excluded_reason_by_step=excluded_reason_by_step,
        selected_step=selected.step,
        selected_score=0.0,
        tie_break_reason="first_snapshot_passing_cheap_consistency_gate",
        per_step_scores={record.step: 0.0 for record in records},
        per_step_breakdown={record.step: empty for record in records},
    )
    return selected, decision


def select_best_snapshot(
    *,
    run_id: str,
    sample_id: str,
    records: Sequence[SnapshotRecord],
    mode: str,
    selector_version: str,
    consistency_window: int,
) -> Tuple[SnapshotRecord, SelectorDecision]:
    if not records:
        raise ValueError("records must not be empty")

    normalized_mode = str(mode).upper()
    if normalized_mode == "S0":
        selected = records[-1]
        empty = _empty_breakdown().to_dict()
        decision = SelectorDecision(
            run_id=run_id,
            sample_id=sample_id,
            selector_version=selector_version,
            candidate_steps=[record.step for record in records],
            eligible_steps=[record.step for record in records],
            excluded_reason_by_step={},
            selected_step=selected.step,
            selected_score=0.0,
            tie_break_reason="final_snapshot_passthrough",
            per_step_scores={record.step: 0.0 for record in records},
            per_step_breakdown={record.step: empty for record in records},
        )
        return selected, decision

    if normalized_mode == "S1":
        return _build_proxy_decision(
            run_id=run_id,
            sample_id=sample_id,
            selector_version=selector_version,
            records=records,
        )

    if normalized_mode == "S2":
        return _build_consistency_decision(
            run_id=run_id,
            sample_id=sample_id,
            selector_version=selector_version,
            records=records,
            consistency_window=consistency_window,
        )

    if normalized_mode == "S3":
        scorer = score_snapshot_s3
    elif normalized_mode == "S4":
        scorer = score_snapshot_s4
    elif normalized_mode == "S5":
        scorer = score_snapshot_s5
    else:
        raise ValueError(f"Unsupported selector mode: {mode}")

    scored_items: List[Tuple[SnapshotRecord, SelectorScoreBreakdown]] = [(record, scorer(record)) for record in records]
    sorted_items = sorted(
        scored_items,
        key=lambda item: (item[1].total_score, -item[0].step, item[0].mean_confidence),
        reverse=True,
    )
    selected_record, selected_breakdown = sorted_items[0]
    decision = SelectorDecision(
        run_id=run_id,
        sample_id=sample_id,
        selector_version=selector_version,
        candidate_steps=[record.step for record, _ in scored_items],
        eligible_steps=[record.step for record, _ in scored_items],
        excluded_reason_by_step={},
        selected_step=selected_record.step,
        selected_score=float(selected_breakdown.total_score),
        tie_break_reason="score_desc_then_earlier_step_then_mean_confidence",
        per_step_scores={record.step: float(breakdown.total_score) for record, breakdown in scored_items},
        per_step_breakdown={record.step: breakdown.to_dict() for record, breakdown in scored_items},
    )
    return selected_record, decision


def build_sample_trajectory_summary(
    *,
    run_id: str,
    sample_id: str,
    dataset_subset: str,
    split: str,
    max_samples: Optional[int],
    seed: int,
    total_steps: int,
    sampling_policy_version: str,
    feature_schema_version: str,
    selector_version: str,
    verifier_version: str,
    baseline_chain_version: str,
    records: Sequence[SnapshotRecord],
    final_tier3_pass: bool,
    first_tier3_step: Optional[int] = None,
    backslide_happened: Optional[bool] = None,
    backslide_from_step: Optional[int] = None,
    oracle_first_pass_step: Optional[int] = None,
    tier3_snapshot_summary_available: bool = False,
) -> SampleTrajectorySummary:
    first_tier1_step = next((record.step for record in records if record.tier1_pass), None)
    first_tier2_step = next((record.step for record in records if record.tier2_pass), None)
    distinct_candidates = len({record.normalized_code_hash for record in records})
    return SampleTrajectorySummary(
        run_id=run_id,
        sample_id=sample_id,
        dataset_subset=dataset_subset,
        split=split,
        max_samples=max_samples,
        seed=seed,
        total_steps=total_steps,
        sampling_policy_version=sampling_policy_version,
        feature_schema_version=feature_schema_version,
        selector_version=selector_version,
        verifier_version=verifier_version,
        baseline_chain_version=baseline_chain_version,
        first_tier1_step=first_tier1_step,
        first_tier2_step=first_tier2_step,
        first_tier3_step=first_tier3_step,
        final_tier3_pass=final_tier3_pass,
        backslide_happened=backslide_happened,
        backslide_from_step=backslide_from_step,
        oracle_first_pass_step=oracle_first_pass_step,
        num_distinct_late_candidates=distinct_candidates,
        tier3_snapshot_summary_available=tier3_snapshot_summary_available,
    )