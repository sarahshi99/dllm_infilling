#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


PROBE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_GEN = 128
TIE_EPSILON = 1e-12
FORBIDDEN_SELECTION_KEYS = {
    "canonical_solution",
    "evaluator_outcome",
    "oracle_length",
    "oracle_mask_length",
    "passed",
    "reference_middle",
    "test",
    "tests",
}


@dataclass(frozen=True)
class LogLengthFit:
    alpha: float
    slope: float


@dataclass(frozen=True)
class LocalSearchResult:
    selected_length: int
    raw_confidence: float
    adjusted_confidence: float
    moves: tuple[int, ...]
    cache_hits: int
    cache_misses: int
    termination_reason: str


@dataclass(frozen=True)
class CommitState:
    committed_tokens: tuple[int, ...]
    remaining_length: int


@dataclass
class ForwardTokenLedger:
    forward_calls: dict[str, int] = field(default_factory=dict)
    token_forwards: dict[str, int] = field(default_factory=dict)
    cache_hits: dict[str, int] = field(default_factory=dict)

    def record_forward(self, stage: str, input_tokens: int, batch_size: int = 1) -> None:
        if not stage:
            raise ValueError("stage must be non-empty")
        if input_tokens <= 0 or batch_size <= 0:
            raise ValueError("input_tokens and batch_size must be positive")
        self.forward_calls[stage] = self.forward_calls.get(stage, 0) + 1
        self.token_forwards[stage] = self.token_forwards.get(stage, 0) + input_tokens * batch_size

    def record_cache_hit(self, stage: str) -> None:
        if not stage:
            raise ValueError("stage must be non-empty")
        self.cache_hits[stage] = self.cache_hits.get(stage, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        search_stages = ("stage_i", "stage_ii")
        search_forwards = sum(self.forward_calls.get(stage, 0) for stage in search_stages)
        decode_forwards = self.forward_calls.get("commit", 0)
        total_forwards = sum(self.forward_calls.values())
        total_token_forwards = sum(self.token_forwards.values())
        return {
            "forward_calls_by_stage": dict(sorted(self.forward_calls.items())),
            "token_forwards_by_stage": dict(sorted(self.token_forwards.items())),
            "cache_hits_by_stage": dict(sorted(self.cache_hits.items())),
            "search_forward_calls": search_forwards,
            "decode_forward_calls": decode_forwards,
            "total_forward_calls": total_forwards,
            "total_token_forwards": total_token_forwards,
        }


class LocalSearchLimitError(RuntimeError):
    pass


def validate_probe_grid(lengths: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(length) for length in lengths)
    if normalized != PROBE_GRID:
        raise ValueError(f"LR-DLLM primary probe grid must be exactly {PROBE_GRID}")
    return normalized


def negative_entropy(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("probability distribution must be non-empty")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(probabilities)
    if total <= 0:
        raise ValueError("probability mass must be positive")
    normalized = (value / total for value in probabilities)
    return math.fsum(value * math.log(value) for value in normalized if value > 0)


def mean_negative_entropy(masked_position_probabilities: Sequence[Sequence[float]]) -> float:
    if not masked_position_probabilities:
        raise ValueError("at least one masked position is required")
    values = [negative_entropy(distribution) for distribution in masked_position_probabilities]
    return math.fsum(values) / len(values)


def fit_log_length_confidence(lengths: Sequence[int], confidences: Sequence[float]) -> LogLengthFit:
    if len(lengths) != len(confidences) or len(lengths) < 2:
        raise ValueError("lengths and confidences must have equal size >= 2")
    if any(length <= 0 for length in lengths):
        raise ValueError("lengths must be positive")
    if any(not math.isfinite(value) for value in confidences):
        raise ValueError("confidences must be finite")
    x = [math.log(length) for length in lengths]
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(confidences) / len(confidences)
    denominator = math.fsum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("log lengths must not all be equal")
    slope = math.fsum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, confidences)) / denominator
    return LogLengthFit(alpha=y_mean - slope * x_mean, slope=slope)


def adjusted_confidence(raw_confidence: float, length: int, slope: float) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    return float(raw_confidence) - float(slope) * math.log(length)


def select_initial_length(probe_confidences: Mapping[int, float]) -> tuple[int, LogLengthFit, dict[int, float]]:
    if {int(length) for length in probe_confidences} != set(PROBE_GRID) or len(probe_confidences) != len(PROBE_GRID):
        raise ValueError(f"LR-DLLM Stage I requires exactly the probe keys {PROBE_GRID}")
    lengths = PROBE_GRID
    raw = [float(probe_confidences[length]) for length in lengths]
    fit = fit_log_length_confidence(lengths, raw)
    adjusted = {length: adjusted_confidence(probe_confidences[length], length, fit.slope) for length in lengths}
    best = min(lengths)
    for length in lengths:
        if adjusted[length] > adjusted[best] + TIE_EPSILON:
            best = length
        elif abs(adjusted[length] - adjusted[best]) <= TIE_EPSILON and length < best:
            best = length
    return best, fit, adjusted


def local_neighbors(length: int, minimum: int = 1, maximum: int = MAX_GEN) -> tuple[int, ...]:
    if not minimum <= length <= maximum:
        raise ValueError("current length is outside bounds")
    return tuple(candidate for candidate in (length - 1, length, length + 1) if minimum <= candidate <= maximum)


def choose_local_length(current_length: int, raw_confidences: Mapping[int, float], slope: float) -> int:
    neighbors = local_neighbors(current_length)
    if set(raw_confidences) != set(neighbors):
        raise ValueError("raw_confidences must contain exactly the feasible local neighborhood")
    adjusted = {length: adjusted_confidence(raw_confidences[length], length, slope) for length in neighbors}
    current_score = adjusted[current_length]
    improving = [length for length in neighbors if adjusted[length] > current_score + TIE_EPSILON]
    if not improving:
        return current_length
    return sorted(improving, key=lambda length: (-adjusted[length], length))[0]


def run_local_search(
    initial_length: int,
    confidence_fn: Callable[[int], float],
    slope: float,
    max_moves: int = MAX_GEN,
) -> LocalSearchResult:
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")
    cache: dict[int, float] = {}
    cache_hits = 0
    cache_misses = 0

    def confidence(length: int) -> float:
        nonlocal cache_hits, cache_misses
        if length in cache:
            cache_hits += 1
            return cache[length]
        value = float(confidence_fn(length))
        if not math.isfinite(value):
            raise ValueError("confidence_fn returned a non-finite value")
        cache[length] = value
        cache_misses += 1
        return value

    current = int(initial_length)
    moves: list[int] = []
    while True:
        scores = {length: confidence(length) for length in local_neighbors(current)}
        selected = choose_local_length(current, scores, slope)
        if selected == current:
            raw = confidence(current)
            return LocalSearchResult(
                selected_length=current,
                raw_confidence=raw,
                adjusted_confidence=adjusted_confidence(raw, current, slope),
                moves=tuple(moves),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                termination_reason="strict_local_optimum",
            )
        if len(moves) >= max_moves:
            raise LocalSearchLimitError(f"local search exceeded {max_moves} moves")
        current = selected
        moves.append(current)


def derive_row_seed(global_seed: int, candidate_key: str) -> int:
    if not candidate_key:
        raise ValueError("candidate_key must be non-empty")
    digest = hashlib.sha256(f"{int(global_seed)}|{candidate_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[-4:], byteorder="big", signed=False)


def sample_top_p(
    logits: Sequence[float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
    forbidden_token_ids: Iterable[int] = (),
) -> int:
    if not logits:
        raise ValueError("logits must be non-empty")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    forbidden = {int(token_id) for token_id in forbidden_token_ids}
    allowed = [(token_id, float(logit) / temperature) for token_id, logit in enumerate(logits) if token_id not in forbidden and math.isfinite(logit)]
    if not allowed:
        raise ValueError("no finite non-special token remains for sampling")
    maximum = max(logit for _, logit in allowed)
    weighted = [(token_id, math.exp(logit - maximum)) for token_id, logit in allowed]
    total = math.fsum(weight for _, weight in weighted)
    ranked = sorted(((token_id, weight / total) for token_id, weight in weighted), key=lambda item: (-item[1], item[0]))
    nucleus: list[tuple[int, float]] = []
    cumulative = 0.0
    for item in ranked:
        nucleus.append(item)
        cumulative += item[1]
        if cumulative >= top_p:
            break
    nucleus_total = math.fsum(probability for _, probability in nucleus)
    draw = rng.random()
    cumulative = 0.0
    for token_id, probability in nucleus:
        cumulative += probability / nucleus_total
        if draw <= cumulative:
            return token_id
    return nucleus[-1][0]


def commit_left_token(state: CommitState, token_id: int, max_gen: int = MAX_GEN) -> CommitState:
    if state.remaining_length <= 0:
        raise ValueError("cannot commit after remaining length reached zero")
    if len(state.committed_tokens) >= max_gen:
        raise ValueError("MAX_GEN exceeded")
    return CommitState(
        committed_tokens=(*state.committed_tokens, int(token_id)),
        remaining_length=state.remaining_length - 1,
    )


def assert_selection_payload_safe(payload: Any) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_SELECTION_KEYS:
                raise ValueError(f"forbidden selection input: {key}")
            assert_selection_payload_safe(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_selection_payload_safe(value)


def canonical_success_index(rows: Sequence[Mapping[str, Any]], expected_keys: set[str]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("candidate_key") or "")
        if not key or key not in expected_keys:
            raise ValueError("canonical row has an unknown candidate key")
        if str(row.get("status") or "") != "ok":
            raise ValueError("canonical raw may contain success rows only")
        if key in indexed:
            raise ValueError(f"duplicate canonical candidate key: {key}")
        indexed[key] = row
    return indexed
