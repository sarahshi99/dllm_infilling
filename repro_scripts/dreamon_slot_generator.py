from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Iterable, Mapping, Sequence

import torch


class ProtocolError(RuntimeError):
    pass


class NonemptyGuardNoValidAction(ProtocolError):
    def __init__(self, rejection_events: Sequence[Mapping[str, Any]]) -> None:
        super().__init__("nonempty_guard_no_valid_action")
        self.rejection_events = [dict(event) for event in rejection_events]


class Method(str, Enum):
    V2_HARD = "v2_hard"
    V2_OPENTAIL = "v2_opentail"
    JOINT_OPENTAIL = "joint_opentail"
    V2_HARD_V2_BOUNDARY = "v2_hard_v2_boundary"
    V3_HARD_BUDGETED = "v3_hard_budgeted"
    V3_HARD_BUDGETED_NONEMPTY_ORACLE = "v3_hard_budgeted_nonempty_oracle"
    V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO = (
        "v3_c0_budgeted_oneshot_pure_newline_veto"
    )
    V3_C_BUDGETED_NONCONSUMING_BLANKLINE = (
        "v3_c_budgeted_nonconsuming_blankline"
    )


class Region(IntEnum):
    CONTEXT_PREFIX = 0
    HARD_SLOT_0 = 1
    HARD_SLOT_1 = 2
    HARD_SLOT_2 = 3
    OPEN_TAIL = 4
    LOCKED_NEWLINE = 5
    CONTEXT_SUFFIX = 6
    PAD = 7
    INSERTED_BLANK_NEWLINE = 8


HARD_REGIONS = (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2)
GENERATION_REGIONS = (*HARD_REGIONS, Region.OPEN_TAIL)


@dataclass(frozen=True)
class NewlineTokenInfo:
    token_id: int
    decoded_text: str
    normalized_text: str
    left_text: str
    left_token_ids: tuple[int, ...]
    right_text: str
    newline_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "decoded_text": self.decoded_text,
            "normalized_text": self.normalized_text,
            "left_text": self.left_text,
            "left_token_ids": list(self.left_token_ids),
            "right_text": self.right_text,
            "newline_count": self.newline_count,
        }


@dataclass(frozen=True)
class TokenizerSpec:
    bos_id: int
    eos_id: int
    pad_id: int
    mask_id: int
    expand_id: int
    newline_token_ids: frozenset[int]
    newline_token_map: Mapping[int, NewlineTokenInfo]
    literal_newline_ids: tuple[int, ...]


@dataclass(frozen=True)
class SlotGeneratorConfig:
    initial_masks_per_region: int = 4
    max_context_tokens: int = 2048
    max_total_forwards: int = 256
    max_global_middle_tokens: int = 64
    max_hard_slot_tokens: int = 32
    temperature: float = 0.0
    top_p: float | None = 0.9
    top_k: int | None = None
    initial_expand_budget: int | None = None
    nonempty_guard: bool = False
    pure_newline_guard_budget_per_slot: int = 0
    pure_newline_guard_global_budget: int = 0

    def __post_init__(self) -> None:
        if self.initial_masks_per_region != 4:
            raise ValueError("Protocol requires exactly four initial masks per region")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_total_forwards <= 0:
            raise ValueError("max_total_forwards must be positive")
        if self.max_global_middle_tokens <= 0:
            raise ValueError("max_global_middle_tokens must be positive")
        if self.max_hard_slot_tokens <= 0:
            raise ValueError("max_hard_slot_tokens must be positive")
        if self.initial_expand_budget is not None and self.initial_expand_budget < 0:
            raise ValueError("initial_expand_budget must be nonnegative")
        if self.pure_newline_guard_budget_per_slot not in {0, 1}:
            raise ValueError("pure_newline_guard_budget_per_slot must be 0 or 1")
        if self.pure_newline_guard_global_budget not in {0, 3}:
            raise ValueError("pure_newline_guard_global_budget must be 0 or 3")


@dataclass(frozen=True)
class SelectedUpdate:
    position: int
    region: Region
    proposal_token_id: int
    unconstrained_proposal_token_id: int
    confidence: float


@dataclass(frozen=True)
class ActionResult:
    action: str
    details: dict[str, Any]


class ExactTransitionCycleDetector:
    def __init__(self) -> None:
        self._seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []

    def observe(
        self,
        signature: tuple[Any, ...],
        forward_index: int,
        region_lengths: Mapping[str, int],
    ) -> dict[str, Any] | None:
        record = {
            "signature": signature,
            "forward_index": int(forward_index),
            "region_lengths": dict(region_lengths),
            "action": signature[4],
        }
        previous = self._seen.get(signature)
        self._history.append(record)
        if previous is None:
            self._seen[signature] = record
            return None
        first_index = int(previous["forward_index"])
        actions = [
            item["action"]
            for item in self._history
            if first_index <= int(item["forward_index"]) <= int(forward_index)
        ]
        return {
            "cycle_length": int(forward_index) - first_index,
            "first_forward_index": first_index,
            "repeat_forward_index": int(forward_index),
            "first_region_lengths": previous["region_lengths"],
            "repeat_region_lengths": dict(region_lengths),
            "action_sequence": actions,
            "transition_signature": list(signature),
        }


class BudgetedTransitionCycleDetector:
    def __init__(self) -> None:
        self._full_seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._canvas_seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []

    def observe(
        self,
        *,
        full_signature: tuple[Any, ...],
        canvas_signature: tuple[Any, ...],
        pre_remaining_budget: int,
        post_remaining_budget: int,
        forward_index: int,
        region_lengths: Mapping[str, int],
    ) -> dict[str, Any] | None:
        record = {
            "full_signature": full_signature,
            "canvas_signature": canvas_signature,
            "forward_index": int(forward_index),
            "region_lengths": dict(region_lengths),
            "action": full_signature[4],
            "pre_remaining_budget": int(pre_remaining_budget),
            "post_remaining_budget": int(post_remaining_budget),
        }
        previous_full = self._full_seen.get(full_signature)
        previous_canvas = self._canvas_seen.get(canvas_signature)
        self._history.append(record)
        if previous_full is not None:
            first_index = int(previous_full["forward_index"])
            return {
                "classification": "exact_deterministic_cycle",
                "terminal": True,
                "cycle_length": int(forward_index) - first_index,
                "first_forward_index": first_index,
                "repeat_forward_index": int(forward_index),
                "first_region_lengths": previous_full["region_lengths"],
                "repeat_region_lengths": dict(region_lengths),
                "first_pre_remaining_budget": int(
                    previous_full["pre_remaining_budget"]
                ),
                "first_post_remaining_budget": int(
                    previous_full["post_remaining_budget"]
                ),
                "repeat_pre_remaining_budget": int(pre_remaining_budget),
                "repeat_post_remaining_budget": int(post_remaining_budget),
                "action_sequence": self._actions_since(first_index, forward_index),
                "transition_signature": list(full_signature),
            }
        self._full_seen[full_signature] = record
        if previous_canvas is None:
            self._canvas_seen[canvas_signature] = record
            return None
        if (
            int(pre_remaining_budget)
            < int(previous_canvas["pre_remaining_budget"])
            and int(post_remaining_budget)
            < int(previous_canvas["post_remaining_budget"])
        ):
            first_index = int(previous_canvas["forward_index"])
            return {
                "classification": "budget_draining_loop",
                "terminal": False,
                "cycle_length": int(forward_index) - first_index,
                "first_forward_index": first_index,
                "repeat_forward_index": int(forward_index),
                "first_region_lengths": previous_canvas["region_lengths"],
                "repeat_region_lengths": dict(region_lengths),
                "first_pre_remaining_budget": int(
                    previous_canvas["pre_remaining_budget"]
                ),
                "first_post_remaining_budget": int(
                    previous_canvas["post_remaining_budget"]
                ),
                "repeat_pre_remaining_budget": int(pre_remaining_budget),
                "repeat_post_remaining_budget": int(post_remaining_budget),
                "budget_delta": int(pre_remaining_budget)
                - int(previous_canvas["pre_remaining_budget"]),
                "action_sequence": self._actions_since(first_index, forward_index),
                "canvas_transition_signature": list(canvas_signature),
            }
        return None

    def _actions_since(self, first_index: int, repeat_index: int) -> list[str]:
        return [
            item["action"]
            for item in self._history
            if first_index <= int(item["forward_index"]) <= repeat_index
        ]


@dataclass
class CanvasState:
    input_ids: torch.Tensor
    region_id: torch.Tensor
    attention_mask: torch.Tensor
    real_length: int
    method: Method
    tokenizer_spec: TokenizerSpec
    initial_prefix_ids: tuple[int, ...]
    initial_suffix_ids: tuple[int, ...]
    initial_locked_newline_ids: tuple[int, ...]
    active_region: Region | None
    initial_expand_budget: int | None
    remaining_expand_budget: int | None
    successful_expand_count: int = 0
    pure_newline_guard_remaining: dict[str, int] = field(default_factory=dict)
    pure_newline_guard_global_remaining: int = 0
    pending_pure_newline_veto: dict[str, Any] | None = None
    inserted_blankline_count: int = 0

    @property
    def capacity(self) -> int:
        return int(self.input_ids.numel())

    def positions_for_region(self, region: Region) -> list[int]:
        values = self.region_id[: self.real_length]
        return [
            int(position)
            for position in (values == int(region)).nonzero(as_tuple=False).flatten().tolist()
        ]

    def tokens_for_region(self, region: Region) -> list[int]:
        return [int(self.input_ids[position]) for position in self.positions_for_region(region)]

    def unresolved_positions(self, region: Region | None = None) -> list[int]:
        positions = range(self.real_length) if region is None else self.positions_for_region(region)
        return [
            int(position)
            for position in positions
            if bool(self.attention_mask[position])
            and int(self.input_ids[position]) == self.tokenizer_spec.mask_id
            and Region(int(self.region_id[position])) in GENERATION_REGIONS
        ]

    def region_length(self, region: Region) -> int:
        return len(self.positions_for_region(region))

    def global_middle_length(self) -> int:
        generation_or_separator = set(GENERATION_REGIONS) | {
            Region.LOCKED_NEWLINE,
            Region.INSERTED_BLANK_NEWLINE,
        }
        return sum(
            Region(int(region)) in generation_or_separator
            for region in self.region_id[: self.real_length]
        )

    def replace_token(self, position: int, token_id: int) -> None:
        self._assert_real_generation_position(position)
        if int(self.input_ids[position]) != self.tokenizer_spec.mask_id:
            raise ProtocolError("Only an unresolved generation position may be updated")
        self.input_ids[position] = int(token_id)

    def expand_at(self, position: int) -> None:
        self._assert_real_generation_position(position)
        if int(self.input_ids[position]) != self.tokenizer_spec.mask_id:
            raise ProtocolError("Only an unresolved generation position may expand")
        if self.real_length >= self.capacity:
            raise ProtocolError("context_capacity_exceeded")
        region = int(self.region_id[position])
        tail_ids = self.input_ids[position + 1 : self.real_length].clone()
        tail_regions = self.region_id[position + 1 : self.real_length].clone()
        tail_attention = self.attention_mask[position + 1 : self.real_length].clone()
        if tail_ids.numel():
            self.input_ids[position + 2 : self.real_length + 1] = tail_ids
            self.region_id[position + 2 : self.real_length + 1] = tail_regions
            self.attention_mask[position + 2 : self.real_length + 1] = tail_attention
        self.input_ids[position : position + 2] = self.tokenizer_spec.mask_id
        self.region_id[position : position + 2] = region
        self.attention_mask[position : position + 2] = True
        self.real_length += 1

    def delete_at(self, position: int) -> None:
        self._assert_real_generation_position(position)
        if int(self.input_ids[position]) != self.tokenizer_spec.mask_id:
            raise ProtocolError("Only an unresolved generation position may delete")
        tail_ids = self.input_ids[position + 1 : self.real_length].clone()
        tail_regions = self.region_id[position + 1 : self.real_length].clone()
        tail_attention = self.attention_mask[position + 1 : self.real_length].clone()
        if tail_ids.numel():
            self.input_ids[position : self.real_length - 1] = tail_ids
            self.region_id[position : self.real_length - 1] = tail_regions
            self.attention_mask[position : self.real_length - 1] = tail_attention
        last = self.real_length - 1
        self.input_ids[last] = self.tokenizer_spec.pad_id
        self.region_id[last] = int(Region.PAD)
        self.attention_mask[last] = False
        self.real_length -= 1

    def replace_real_sequence(
        self, token_ids: Sequence[int], region_ids: Sequence[int]
    ) -> None:
        if len(token_ids) != len(region_ids):
            raise ProtocolError("replacement_parallel_lengths_differ")
        if len(token_ids) > self.capacity:
            raise ProtocolError("context_capacity_exceeded")
        new_input_ids = torch.full_like(self.input_ids, self.tokenizer_spec.pad_id)
        new_region_ids = torch.full_like(self.region_id, int(Region.PAD))
        new_attention = torch.zeros_like(self.attention_mask)
        if token_ids:
            length = len(token_ids)
            new_input_ids[:length] = torch.tensor(
                list(map(int, token_ids)), dtype=torch.long, device=self.input_ids.device
            )
            new_region_ids[:length] = torch.tensor(
                list(map(int, region_ids)), dtype=torch.long, device=self.region_id.device
            )
            new_attention[:length] = True
        self.input_ids.copy_(new_input_ids)
        self.region_id.copy_(new_region_ids)
        self.attention_mask.copy_(new_attention)
        self.real_length = len(token_ids)

    def insert_locked_tokens(
        self, position: int, token_ids: Sequence[int], region: Region
    ) -> None:
        if not 0 <= position <= self.real_length:
            raise ProtocolError("insert_position_is_not_real_boundary")
        if region != Region.INSERTED_BLANK_NEWLINE:
            raise ProtocolError("inserted_tokens_require_blank_newline_region")
        old_ids = [int(item) for item in self.input_ids[: self.real_length]]
        old_regions = [int(item) for item in self.region_id[: self.real_length]]
        new_ids = old_ids[:position] + list(map(int, token_ids)) + old_ids[position:]
        new_regions = (
            old_regions[:position]
            + [int(region)] * len(token_ids)
            + old_regions[position:]
        )
        self.replace_real_sequence(new_ids, new_regions)

    def _assert_real_generation_position(self, position: int) -> None:
        if not 0 <= position < self.real_length or not bool(self.attention_mask[position]):
            raise ProtocolError("selected_position_is_not_real")
        region = Region(int(self.region_id[position]))
        if region not in GENERATION_REGIONS or region == Region.PAD:
            raise ProtocolError("selected_region_is_not_generation_eligible")


def clone_canvas_state(state: CanvasState) -> CanvasState:
    return CanvasState(
        input_ids=state.input_ids.clone(),
        region_id=state.region_id.clone(),
        attention_mask=state.attention_mask.clone(),
        real_length=int(state.real_length),
        method=state.method,
        tokenizer_spec=state.tokenizer_spec,
        initial_prefix_ids=state.initial_prefix_ids,
        initial_suffix_ids=state.initial_suffix_ids,
        initial_locked_newline_ids=state.initial_locked_newline_ids,
        active_region=state.active_region,
        initial_expand_budget=state.initial_expand_budget,
        remaining_expand_budget=state.remaining_expand_budget,
        successful_expand_count=int(state.successful_expand_count),
        pure_newline_guard_remaining=dict(state.pure_newline_guard_remaining),
        pure_newline_guard_global_remaining=int(
            state.pure_newline_guard_global_remaining
        ),
        pending_pure_newline_veto=(
            None
            if state.pending_pure_newline_veto is None
            else dict(state.pending_pure_newline_veto)
        ),
        inserted_blankline_count=int(state.inserted_blankline_count),
    )


def canvas_checkpoint_payload(state: CanvasState) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": state.method.value,
        "capacity": state.capacity,
        "input_ids": [int(item) for item in state.input_ids[: state.real_length]],
        "region_id": [int(item) for item in state.region_id[: state.real_length]],
        "real_length": int(state.real_length),
        "initial_prefix_ids": list(state.initial_prefix_ids),
        "initial_suffix_ids": list(state.initial_suffix_ids),
        "initial_locked_newline_ids": list(state.initial_locked_newline_ids),
        "active_region": (
            None if state.active_region is None else state.active_region.name
        ),
        "initial_expand_budget": state.initial_expand_budget,
        "remaining_expand_budget": state.remaining_expand_budget,
        "successful_expand_count": int(state.successful_expand_count),
        "pure_newline_guard_remaining": dict(
            sorted(state.pure_newline_guard_remaining.items())
        ),
        "pure_newline_guard_global_remaining": int(
            state.pure_newline_guard_global_remaining
        ),
        "pending_pure_newline_veto": state.pending_pure_newline_veto,
        "inserted_blankline_count": int(state.inserted_blankline_count),
    }


def restore_canvas_checkpoint(
    payload: Mapping[str, Any],
    tokenizer_spec: TokenizerSpec,
    device: torch.device,
) -> CanvasState:
    if int(payload.get("schema_version", -1)) != 1:
        raise ProtocolError("unsupported_canvas_checkpoint_schema")
    real_ids = list(map(int, payload["input_ids"]))
    real_regions = list(map(int, payload["region_id"]))
    if len(real_ids) != len(real_regions) or len(real_ids) != int(payload["real_length"]):
        raise ProtocolError("checkpoint_parallel_lengths_differ")
    capacity = int(payload["capacity"])
    if len(real_ids) > capacity:
        raise ProtocolError("checkpoint_real_length_exceeds_capacity")
    input_ids = torch.full(
        (capacity,), tokenizer_spec.pad_id, dtype=torch.long, device=device
    )
    region_id = torch.full(
        (capacity,), int(Region.PAD), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
    if real_ids:
        input_ids[: len(real_ids)] = torch.tensor(real_ids, dtype=torch.long, device=device)
        region_id[: len(real_regions)] = torch.tensor(
            real_regions, dtype=torch.long, device=device
        )
        attention_mask[: len(real_ids)] = True
    active_name = payload.get("active_region")
    state = CanvasState(
        input_ids=input_ids,
        region_id=region_id,
        attention_mask=attention_mask,
        real_length=len(real_ids),
        method=Method(str(payload["method"])),
        tokenizer_spec=tokenizer_spec,
        initial_prefix_ids=tuple(map(int, payload["initial_prefix_ids"])),
        initial_suffix_ids=tuple(map(int, payload["initial_suffix_ids"])),
        initial_locked_newline_ids=tuple(
            map(int, payload["initial_locked_newline_ids"])
        ),
        active_region=None if active_name is None else Region[str(active_name)],
        initial_expand_budget=payload.get("initial_expand_budget"),
        remaining_expand_budget=payload.get("remaining_expand_budget"),
        successful_expand_count=int(payload.get("successful_expand_count", 0)),
        pure_newline_guard_remaining={
            str(key): int(value)
            for key, value in payload.get(
                "pure_newline_guard_remaining", {}
            ).items()
        },
        pure_newline_guard_global_remaining=int(
            payload.get("pure_newline_guard_global_remaining", 0)
        ),
        pending_pure_newline_veto=(
            None
            if payload.get("pending_pure_newline_veto") is None
            else dict(payload["pending_pure_newline_veto"])
        ),
        inserted_blankline_count=int(payload.get("inserted_blankline_count", 0)),
    )
    validate_state(state, state.method)
    return state


def _state_hash_payload(state: CanvasState) -> dict[str, Any]:
    return {
        "method": state.method.value,
        "input_ids": [int(item) for item in state.input_ids[: state.real_length]],
        "region_id": [int(item) for item in state.region_id[: state.real_length]],
        "attention_visible_real_length": int(
            state.attention_mask[: state.real_length].sum().item()
        ),
        "real_length": int(state.real_length),
        "active_region": None if state.active_region is None else state.active_region.name,
    }


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canvas_only_state_hash(state: CanvasState) -> str:
    return _hash_payload(_state_hash_payload(state))


def full_state_hash(state: CanvasState) -> str:
    payload = _state_hash_payload(state)
    if state.remaining_expand_budget is not None:
        payload.update(
            {
                "initial_expand_budget": int(state.initial_expand_budget),
                "remaining_expand_budget": int(state.remaining_expand_budget),
                "successful_expand_count": int(state.successful_expand_count),
            }
        )
    if method_uses_pure_newline_guard(state.method):
        payload.update(
            {
                "pure_newline_guard_remaining": dict(
                    sorted(state.pure_newline_guard_remaining.items())
                ),
                "pure_newline_guard_global_remaining": int(
                    state.pure_newline_guard_global_remaining
                ),
                "pending_pure_newline_veto": state.pending_pure_newline_veto,
                "inserted_blankline_count": int(state.inserted_blankline_count),
            }
        )
    return _hash_payload(payload)


def state_hash(state: CanvasState) -> str:
    """Backward-compatible V2 canvas hash."""
    return canvas_only_state_hash(state)


def method_uses_boundary_decoder(method: Method) -> bool:
    return method in {
        Method.V2_HARD_V2_BOUNDARY,
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    }


def method_uses_expand_budget(method: Method) -> bool:
    return method in {
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    }


def method_uses_nonempty_guard(method: Method) -> bool:
    return method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE


def method_uses_pure_newline_guard(method: Method) -> bool:
    return method in {
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    }


def pure_newline_guard_mode(method: Method) -> str | None:
    if method == Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO:
        return "oneshot_veto"
    if method == Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE:
        return "nonconsuming_blankline"
    return None


def transition_signature(
    pre_state_hash: str,
    selected_position: int,
    selected_region: Region,
    proposal_token_id: int,
    action: str,
    post_state_hash: str,
) -> tuple[Any, ...]:
    return (
        pre_state_hash,
        int(selected_position),
        selected_region.name,
        int(proposal_token_id),
        action,
        post_state_hash,
    )


def sequential_regions(method: Method) -> tuple[Region, ...]:
    if method in (
        Method.V2_HARD,
        Method.V2_HARD_V2_BOUNDARY,
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    ):
        return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2)
    if method == Method.V2_OPENTAIL:
        return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.OPEN_TAIL)
    if method == Method.JOINT_OPENTAIL:
        return ()
    raise ValueError(f"Unknown method: {method}")


def method_generation_regions(method: Method) -> tuple[Region, ...]:
    if method in (
        Method.V2_HARD,
        Method.V2_HARD_V2_BOUNDARY,
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    ):
        return sequential_regions(method)
    return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.OPEN_TAIL)


def scan_newline_token_metadata(
    tokenizer,
) -> tuple[dict[int, NewlineTokenInfo], dict[str, Any]]:
    started = time.perf_counter()
    mapping: dict[int, NewlineTokenInfo] = {}
    examples: list[dict[str, Any]] = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" in decoded or "\r" in decoded:
            normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
            boundary_index = normalized.index("\n")
            left_text = normalized[:boundary_index]
            right_text = normalized[boundary_index + 1 :]
            left_token_ids = tuple(
                int(item)
                for item in tokenizer.encode(left_text, add_special_tokens=False)
            )
            round_trip = tokenizer.decode(
                list(left_token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if round_trip != left_text:
                raise ProtocolError(
                    f"newline_left_retokenization_mismatch:{token_id}"
                )
            info = NewlineTokenInfo(
                token_id=int(token_id),
                decoded_text=decoded,
                normalized_text=normalized,
                left_text=left_text,
                left_token_ids=left_token_ids,
                right_text=right_text,
                newline_count=normalized.count("\n"),
            )
            mapping[int(token_id)] = info
            if len(examples) < 20:
                examples.append(info.as_dict())
    records = [mapping[token_id].as_dict() for token_id in sorted(mapping)]
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    token_ids = sorted(mapping)
    id_digest = hashlib.sha256("\n".join(map(str, token_ids)).encode("utf-8")).hexdigest()
    mapping_digest = hashlib.sha256(encoded).hexdigest()
    elapsed = time.perf_counter() - started
    return mapping, {
        "vocab_size": len(tokenizer),
        "newline_token_count": len(token_ids),
        "newline_token_ids_sha256": id_digest,
        "newline_token_mapping_count": len(mapping),
        "newline_token_mapping_sha256": mapping_digest,
        "preprocessing_seconds": elapsed,
        "examples": examples,
    }


def scan_newline_token_ids(tokenizer) -> tuple[list[int], dict[str, Any]]:
    mapping, metadata = scan_newline_token_metadata(tokenizer)
    return sorted(mapping), metadata


def _initial_middle(method: Method, spec: TokenizerSpec, masks: int) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    regions: list[int] = []

    def add_region(region: Region) -> None:
        ids.extend([spec.mask_id] * masks)
        regions.extend([int(region)] * masks)

    def add_newline() -> None:
        ids.extend(spec.literal_newline_ids)
        regions.extend([int(Region.LOCKED_NEWLINE)] * len(spec.literal_newline_ids))

    add_region(Region.HARD_SLOT_0)
    add_newline()
    add_region(Region.HARD_SLOT_1)
    add_newline()
    if method in (
        Method.V2_HARD,
        Method.V2_HARD_V2_BOUNDARY,
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    ):
        add_region(Region.HARD_SLOT_2)
        add_newline()
    else:
        add_region(Region.OPEN_TAIL)
    return ids, regions


def build_canvas(
    method: Method,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    tokenizer_spec: TokenizerSpec,
    config: SlotGeneratorConfig,
    device: torch.device,
) -> CanvasState:
    if method_uses_expand_budget(method):
        if config.initial_expand_budget != 64:
            raise ProtocolError("v3_initial_expand_budget_must_equal_64")
    elif config.initial_expand_budget is not None:
        raise ProtocolError("expand_budget_configured_for_non_v3_method")
    if config.nonempty_guard != method_uses_nonempty_guard(method):
        raise ProtocolError("nonempty_guard_method_config_mismatch")
    expected_guard_budget = 1 if method_uses_pure_newline_guard(method) else 0
    expected_global_guard_budget = 3 if method_uses_pure_newline_guard(method) else 0
    if config.pure_newline_guard_budget_per_slot != expected_guard_budget:
        raise ProtocolError("pure_newline_guard_per_slot_config_mismatch")
    if config.pure_newline_guard_global_budget != expected_global_guard_budget:
        raise ProtocolError("pure_newline_guard_global_config_mismatch")
    middle_ids, middle_regions = _initial_middle(
        method, tokenizer_spec, config.initial_masks_per_region
    )
    real_ids = list(map(int, prefix_ids)) + middle_ids + list(map(int, suffix_ids))
    real_regions = (
        [int(Region.CONTEXT_PREFIX)] * len(prefix_ids)
        + middle_regions
        + [int(Region.CONTEXT_SUFFIX)] * len(suffix_ids)
    )
    if len(real_ids) > config.max_context_tokens:
        raise ProtocolError("initial_context_exceeds_max_context")
    if len(middle_ids) > config.max_global_middle_tokens:
        raise ProtocolError("initial_middle_exceeds_global_cap")

    input_ids = torch.full(
        (config.max_context_tokens,), tokenizer_spec.pad_id, dtype=torch.long, device=device
    )
    region_id = torch.full(
        (config.max_context_tokens,), int(Region.PAD), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(config.max_context_tokens, dtype=torch.bool, device=device)
    input_ids[: len(real_ids)] = torch.tensor(real_ids, dtype=torch.long, device=device)
    region_id[: len(real_ids)] = torch.tensor(real_regions, dtype=torch.long, device=device)
    attention_mask[: len(real_ids)] = True
    locked = [
        token_id
        for token_id, region in zip(real_ids, real_regions)
        if region == int(Region.LOCKED_NEWLINE)
    ]
    state = CanvasState(
        input_ids=input_ids,
        region_id=region_id,
        attention_mask=attention_mask,
        real_length=len(real_ids),
        method=method,
        tokenizer_spec=tokenizer_spec,
        initial_prefix_ids=tuple(map(int, prefix_ids)),
        initial_suffix_ids=tuple(map(int, suffix_ids)),
        initial_locked_newline_ids=tuple(locked),
        active_region=None,
        initial_expand_budget=config.initial_expand_budget,
        remaining_expand_budget=config.initial_expand_budget,
        successful_expand_count=0,
        pure_newline_guard_remaining=(
            {
                region.name: config.pure_newline_guard_budget_per_slot
                for region in HARD_REGIONS
            }
            if method_uses_pure_newline_guard(method)
            else {}
        ),
        pure_newline_guard_global_remaining=(
            config.pure_newline_guard_global_budget
            if method_uses_pure_newline_guard(method)
            else 0
        ),
        pending_pure_newline_veto=None,
        inserted_blankline_count=0,
    )
    advance_active_region(state, method)
    validate_state(state, method)
    return state


def advance_active_region(state: CanvasState, method: Method) -> Region | None:
    if method == Method.JOINT_OPENTAIL:
        state.active_region = None
        return None
    order = sequential_regions(method)
    active: Region | None = None
    for region in order:
        if state.unresolved_positions(region):
            active = region
            break
    if active is not None:
        active_index = order.index(active)
        for completed in order[:active_index]:
            if state.unresolved_positions(completed):
                raise ProtocolError("completed_region_contains_mask")
        for future in order[active_index + 1 :]:
            if any(
                token_id != state.tokenizer_spec.mask_id
                for token_id in state.tokens_for_region(future)
            ):
                raise ProtocolError("future_region_mutated_before_activation")
    state.active_region = active
    return active


def eligible_positions(state: CanvasState, method: Method) -> list[int]:
    if method == Method.JOINT_OPENTAIL:
        allowed = set(method_generation_regions(method))
        return [
            position
            for position in state.unresolved_positions()
            if Region(int(state.region_id[position])) in allowed
        ]
    active = advance_active_region(state, method)
    return [] if active is None else state.unresolved_positions(active)


def forward_inputs(state: CanvasState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = state.input_ids[: state.real_length].unsqueeze(0)
    real_attention = state.attention_mask[: state.real_length]
    attention = torch.logical_and(
        real_attention.view(1, 1, 1, -1),
        real_attention.view(1, 1, -1, 1),
    )
    position_ids = real_attention.long().cumsum(-1) - 1
    position_ids.masked_fill_(~real_attention, 1)
    return input_ids, attention, position_ids.unsqueeze(0)


def shifted_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _top_p_logits(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, float("-inf"))


def _top_k_logits(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _entropy_proposals(
    logits: torch.Tensor, config: SlotGeneratorConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    filtered = _top_k_logits(_top_p_logits(logits, config.top_p), config.top_k)
    valid_rows = torch.isfinite(filtered).any(dim=-1)
    probabilities = torch.zeros_like(filtered)
    proposals = torch.zeros(
        filtered.shape[:-1], dtype=torch.long, device=filtered.device
    )
    negative_entropy = torch.full(
        filtered.shape[:-1], float("-inf"), dtype=filtered.dtype, device=filtered.device
    )
    if valid_rows.any():
        valid_filtered = filtered[valid_rows]
        valid_probabilities = torch.softmax(valid_filtered, dim=-1)
        if not torch.isfinite(valid_probabilities).all():
            raise ProtocolError("nan_logits")
        probabilities[valid_rows] = valid_probabilities
        proposals[valid_rows] = torch.argmax(valid_filtered, dim=-1)
        negative_entropy[valid_rows] = torch.sum(
            valid_probabilities
            * torch.log(valid_probabilities.clamp_min(1e-10)),
            dim=-1,
        )
    return negative_entropy, proposals


def _expand_allowed(
    state: CanvasState, region: Region, config: SlotGeneratorConfig
) -> bool:
    if state.remaining_expand_budget is not None and state.remaining_expand_budget == 0:
        return False
    if state.real_length >= config.max_context_tokens:
        return False
    if state.global_middle_length() >= config.max_global_middle_tokens:
        return False
    if region in HARD_REGIONS and state.region_length(region) >= config.max_hard_slot_tokens:
        return False
    return True


def constrained_active_logits(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
) -> tuple[torch.Tensor, list[int]]:
    positions = eligible_positions(state, method)
    if not positions:
        raise ProtocolError("topk_requested_without_active_mask")
    position_tensor = torch.tensor(positions, dtype=torch.long, device=full_logits.device)
    active_logits = full_logits[position_tensor].float().clone()
    for row, position in enumerate(positions):
        region = Region(int(state.region_id[position]))
        if not _expand_allowed(state, region, config):
            active_logits[row, state.tokenizer_spec.expand_id] = float("-inf")
    return active_logits, positions


def select_update(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
    rejected_candidates: set[tuple[int, int]] | None = None,
) -> SelectedUpdate:
    active_logits, positions = constrained_active_logits(state, method, full_logits, config)
    position_rows = {position: row for row, position in enumerate(positions)}
    for position, token_id in rejected_candidates or set():
        row = position_rows.get(int(position))
        if row is not None and 0 <= int(token_id) < active_logits.shape[-1]:
            active_logits[row, int(token_id)] = float("-inf")
    confidence, proposals = _entropy_proposals(active_logits, config)
    if confidence.numel() == 0 or not torch.isfinite(confidence).any():
        raise ProtocolError("no_valid_proposal")
    selected_local = int(torch.topk(confidence, k=1, largest=True).indices[0].item())
    selected_position = positions[selected_local]
    unconstrained_logits = full_logits[selected_position].float().unsqueeze(0)
    _, unconstrained = _entropy_proposals(unconstrained_logits, config)
    return SelectedUpdate(
        position=selected_position,
        region=Region(int(state.region_id[selected_position])),
        proposal_token_id=int(proposals[selected_local].item()),
        unconstrained_proposal_token_id=int(unconstrained[0].item()),
        confidence=float(confidence[selected_local].item()),
    )


def select_update_with_pending_veto(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
) -> tuple[SelectedUpdate, bool, bool]:
    pending = state.pending_pure_newline_veto
    if pending is None:
        return select_update(state, method, full_logits, config), False, False
    if method != Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO:
        raise ProtocolError("pending_pure_newline_veto_on_wrong_method")
    if str(pending["pre_canvas_hash"]) != canvas_only_state_hash(state):
        raise ProtocolError("pending_pure_newline_veto_canvas_mismatch")
    position = int(pending["selected_position"])
    token_id = int(pending["proposal_token_id"])
    try:
        selected = select_update(
            state,
            method,
            full_logits,
            config,
            rejected_candidates={(position, token_id)},
        )
        fallback = False
    except ProtocolError as error:
        if str(error) != "no_valid_proposal":
            raise
        if (
            position not in state.unresolved_positions(state.active_region)
            or int(state.input_ids[position]) != state.tokenizer_spec.mask_id
        ):
            raise ProtocolError("pending_pure_newline_veto_position_invalid") from error
        selected = SelectedUpdate(
            position=position,
            region=Region(int(state.region_id[position])),
            proposal_token_id=token_id,
            unconstrained_proposal_token_id=token_id,
            confidence=float("-inf"),
        )
        fallback = True
    state.pending_pure_newline_veto = None
    return selected, True, fallback


def classify_action_token(token_id: int, spec: TokenizerSpec) -> str:
    token_id = int(token_id)
    if token_id == spec.expand_id:
        return "expand"
    if token_id == spec.eos_id:
        return "EOS"
    newline = spec.newline_token_map.get(token_id)
    if newline is not None:
        if (
            newline.normalized_text == "\n"
            and newline.left_text == ""
            and newline.right_text == ""
        ):
            return "pure_newline"
        return "other_newline"
    if token_id in {spec.bos_id, spec.pad_id, spec.mask_id}:
        return "special/sentinel"
    return "normal"


def _position_logit_diagnostic(
    state: CanvasState,
    position: int,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
    tokenizer,
    *,
    eligible: bool,
    region_offset: int,
) -> dict[str, Any]:
    region = Region(int(state.region_id[position]))
    logits = full_logits[position].float().clone()
    if not _expand_allowed(state, region, config):
        logits[state.tokenizer_spec.expand_id] = float("-inf")
    filtered = _top_k_logits(_top_p_logits(logits.unsqueeze(0), config.top_p), config.top_k)[0]
    finite = torch.isfinite(filtered)
    if not finite.any():
        raise ProtocolError("future_diagnostic_no_finite_proposal")
    probabilities = torch.softmax(filtered, dim=-1)
    entropy = float(
        -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-10))).item()
    )
    top1 = int(torch.argmax(filtered).item())
    finite_indices = finite.nonzero(as_tuple=False).flatten()
    k = min(5, int(finite_indices.numel()))
    finite_logits = filtered[finite_indices]
    top_local = torch.topk(finite_logits, k=k, largest=True).indices
    top_ids = [int(finite_indices[index].item()) for index in top_local]
    return {
        "region": region.name,
        "position": int(position),
        "region_offset": int(region_offset),
        "eligible": bool(eligible),
        "entropy": entropy,
        "top1_probability": float(probabilities[top1].item()),
        "top5_token_ids": top_ids,
        "top5_decoded_texts": [
            _decode(tokenizer, [token_id]) for token_id in top_ids
        ],
        "top5_probabilities": [
            float(probabilities[token_id].item()) for token_id in top_ids
        ],
        "top5_logits": [float(filtered[token_id].item()) for token_id in top_ids],
        "top1_action_class": classify_action_token(top1, state.tokenizer_spec),
    }


def collect_future_slot_diagnostics(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
    tokenizer,
    selected: SelectedUpdate,
) -> dict[str, Any]:
    if state.active_region is None or method == Method.JOINT_OPENTAIL:
        raise ProtocolError("future_diagnostic_requires_sequential_active_region")
    order = sequential_regions(method)
    active_index = order.index(state.active_region)
    inspected = order[active_index:]
    position_rows: list[dict[str, Any]] = []
    regions: dict[str, Any] = {}
    for region in inspected:
        region_positions = state.positions_for_region(region)
        unresolved = state.unresolved_positions(region)
        details = [
            _position_logit_diagnostic(
                state,
                position,
                full_logits,
                config,
                tokenizer,
                eligible=region == state.active_region,
                region_offset=region_positions.index(position),
            )
            for position in unresolved
        ]
        position_rows.extend(details)
        normal = [row for row in details if row["top1_action_class"] == "normal"]
        terminating = [
            row
            for row in details
            if row["top1_action_class"]
            in {"EOS", "pure_newline", "other_newline", "special/sentinel"}
        ]
        most_confident = min(details, key=lambda row: row["entropy"])
        regions[region.name] = {
            "eligible": region == state.active_region,
            "positions": details,
            "minimum_entropy": min(row["entropy"] for row in details),
            "maximum_top1_probability": max(
                row["top1_probability"] for row in details
            ),
            "normal_top1_position_count": len(normal),
            "termination_top1_position_count": len(terminating),
            "most_confident_position": most_confident,
        }
    selected_row = next(
        row for row in position_rows if row["position"] == selected.position
    )
    future_normal = [
        row
        for row in position_rows
        if not row["eligible"] and row["top1_action_class"] == "normal"
    ]
    return {
        "active_region": state.active_region.name,
        "selected_position": selected.position,
        "selected_region_offset": selected_row["region_offset"],
        "selected_entropy": selected_row["entropy"],
        "selected_top1_probability": selected_row["top1_probability"],
        "selected_top1_action_class": selected_row["top1_action_class"],
        "regions": regions,
        "future_has_lower_entropy_normal_candidate": any(
            row["entropy"] < selected_row["entropy"] for row in future_normal
        ),
        "future_has_higher_top1_probability_normal_candidate": any(
            row["top1_probability"] > selected_row["top1_probability"]
            for row in future_normal
        ),
    }


def action_completes_blank_hard_slot(
    candidate_state: CanvasState, region: Region, tokenizer
) -> bool:
    if region not in HARD_REGIONS or candidate_state.unresolved_positions(region):
        return False
    return not bool(_decode(tokenizer, candidate_state.tokens_for_region(region)).strip())


def select_nonempty_guarded_update(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
    tokenizer,
    *,
    task_id: str | None,
    forward_index: int,
) -> tuple[SelectedUpdate, list[dict[str, Any]]]:
    if not method_uses_nonempty_guard(method) or not config.nonempty_guard:
        raise ProtocolError("nonempty_guard_not_enabled")
    rejected: set[tuple[int, int]] = set()
    rejection_events: list[dict[str, Any]] = []
    pending_event: dict[str, Any] | None = None
    while True:
        try:
            choice = select_update(
                state,
                method,
                full_logits,
                config,
                rejected_candidates=rejected,
            )
        except ProtocolError as error:
            if str(error) != "no_valid_proposal":
                raise
            if pending_event is not None:
                pending_event.update(
                    {
                        "reselected_position": None,
                        "reselected_proposal_token_id": None,
                        "reselected_action": None,
                    }
                )
                rejection_events.append(pending_event)
            raise NonemptyGuardNoValidAction(rejection_events) from error

        candidate = clone_canvas_state(state)
        action_result = apply_selected_action(
            candidate,
            choice.position,
            choice.proposal_token_id,
            config,
            tokenizer=tokenizer,
        )
        if candidate.method != Method.JOINT_OPENTAIL:
            advance_active_region(candidate, method)
        validate_state(candidate, method)
        if pending_event is not None:
            pending_event.update(
                {
                    "reselected_position": choice.position,
                    "reselected_proposal_token_id": choice.proposal_token_id,
                    "reselected_action": action_result.action,
                }
            )
            rejection_events.append(pending_event)
            pending_event = None
        if not action_completes_blank_hard_slot(candidate, choice.region, tokenizer):
            return choice, rejection_events

        rejected.add((choice.position, choice.proposal_token_id))
        pending_event = {
            "task_id": task_id,
            "slot": choice.region.name,
            "forward_index": int(forward_index),
            "position": choice.position,
            "proposal_token_id": choice.proposal_token_id,
            "decoded_proposal": _decode(tokenizer, [choice.proposal_token_id]),
            "hypothetical_action": action_result.action,
            "pre_state_hash": full_state_hash(state),
            "remaining_budget": state.remaining_expand_budget,
            "hypothetical_region_text": _decode(
                tokenizer, candidate.tokens_for_region(choice.region)
            ),
            "hypothetical_unresolved_count": len(
                candidate.unresolved_positions(choice.region)
            ),
        }


def _candidate_middle_length(region_ids: Sequence[int]) -> int:
    middle_regions = {int(region) for region in GENERATION_REGIONS} | {
        int(Region.LOCKED_NEWLINE),
        int(Region.INSERTED_BLANK_NEWLINE),
    }
    return sum(int(region_id) in middle_regions for region_id in region_ids)


def _validate_candidate_caps(
    token_ids: Sequence[int],
    region_ids: Sequence[int],
    region: Region,
    config: SlotGeneratorConfig,
) -> None:
    if len(token_ids) > config.max_context_tokens:
        raise ProtocolError("boundary_context_cap_exceeded")
    if _candidate_middle_length(region_ids) > config.max_global_middle_tokens:
        raise ProtocolError("boundary_global_cap_exceeded")
    if region in HARD_REGIONS:
        slot_length = sum(int(item) == int(region) for item in region_ids)
        if slot_length > config.max_hard_slot_tokens:
            raise ProtocolError("boundary_slot_cap_exceeded")


def _resolved_segments(
    state: CanvasState, positions: Sequence[int], tokenizer
) -> list[dict[str, Any]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for position in positions:
        token_id = int(state.input_ids[position])
        if token_id == state.tokenizer_spec.mask_id:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token_id)
    if current:
        segments.append(current)
    return [
        {"token_ids": segment, "text": _decode(tokenizer, segment)}
        for segment in segments
    ]


def _discarded_contiguous_segments(
    state: CanvasState,
    right_text: str,
    positions: Sequence[int],
    tokenizer,
) -> list[str]:
    segments: list[str] = []
    current = right_text
    for position in positions:
        token_id = int(state.input_ids[position])
        if token_id == state.tokenizer_spec.mask_id:
            if current:
                segments.append(current)
                current = ""
        else:
            current += _decode(tokenizer, [token_id])
    if current:
        segments.append(current)
    return segments


def _apply_line_boundary(
    state: CanvasState,
    position: int,
    region: Region,
    info: NewlineTokenInfo,
    config: SlotGeneratorConfig,
    tokenizer,
) -> ActionResult:
    if tokenizer is None:
        raise ProtocolError("line_boundary_requires_tokenizer")
    region_positions = state.positions_for_region(region)
    right_positions = [item for item in region_positions if item > position]
    discarded_masks = sum(
        int(state.input_ids[item]) == state.tokenizer_spec.mask_id
        for item in right_positions
    )
    discarded_resolved_ids = [
        int(state.input_ids[item])
        for item in right_positions
        if int(state.input_ids[item]) != state.tokenizer_spec.mask_id
    ]
    resolved_segments = _resolved_segments(state, right_positions, tokenizer)
    contiguous_segments = _discarded_contiguous_segments(
        state, info.right_text, right_positions, tokenizer
    )
    removed = set(right_positions)
    old_ids = [int(item) for item in state.input_ids[: state.real_length]]
    old_regions = [int(item) for item in state.region_id[: state.real_length]]
    new_ids: list[int] = []
    new_regions: list[int] = []
    for index, (token_id, region_id) in enumerate(zip(old_ids, old_regions)):
        if index == position:
            new_ids.extend(info.left_token_ids)
            new_regions.extend([int(region)] * len(info.left_token_ids))
        elif index in removed:
            continue
        else:
            new_ids.append(token_id)
            new_regions.append(region_id)
    _validate_candidate_caps(new_ids, new_regions, region, config)
    before_length = len(region_positions)
    state.replace_real_sequence(new_ids, new_regions)
    left_masks_remaining = bool(state.unresolved_positions(region))
    return ActionResult(
        action="line_boundary",
        details={
            "proposal_token_id": info.token_id,
            "decoded_text": info.decoded_text,
            "normalized_text": info.normalized_text,
            "left_text": info.left_text,
            "left_token_ids": list(info.left_token_ids),
            "right_text": info.right_text,
            "newline_count": info.newline_count,
            "boundary_retokenized_left_token_count": len(info.left_token_ids),
            "discarded_internal_right_text": info.right_text,
            "discarded_masks_after_boundary": int(discarded_masks),
            "discarded_resolved_token_ids_after_boundary": discarded_resolved_ids,
            "discarded_resolved_segments_after_boundary": resolved_segments,
            "discarded_resolved_tokens_after_boundary": len(discarded_resolved_ids),
            "discarded_contiguous_segments": contiguous_segments,
            "discarded_complete_text": (
                "".join(contiguous_segments) if discarded_masks == 0 else None
            ),
            "boundary_event_with_left_masks_remaining": left_masks_remaining,
            "region_length_before": before_length,
            "region_length_after": state.region_length(region),
        },
    )


def _strict_pure_newline_blank_request(
    state: CanvasState,
    selected: SelectedUpdate,
    config: SlotGeneratorConfig,
    tokenizer,
    *,
    require_budget: bool,
) -> bool:
    if not method_uses_pure_newline_guard(state.method):
        return False
    if selected.region not in HARD_REGIONS or selected.region != state.active_region:
        return False
    info = state.tokenizer_spec.newline_token_map.get(selected.proposal_token_id)
    if info is None:
        return False
    if not (
        info.normalized_text == "\n"
        and info.left_text == ""
        and info.right_text == ""
    ):
        return False
    region_positions = state.positions_for_region(selected.region)
    if not region_positions or selected.position != region_positions[0]:
        return False
    if require_budget:
        if state.pure_newline_guard_global_remaining <= 0:
            return False
        if state.pure_newline_guard_remaining.get(selected.region.name, 0) <= 0:
            return False
    candidate = clone_canvas_state(state)
    _apply_line_boundary(
        candidate,
        selected.position,
        selected.region,
        info,
        config,
        tokenizer,
    )
    return (
        candidate.region_length(selected.region) == 0
        and not candidate.unresolved_positions(selected.region)
        and not _decode(tokenizer, candidate.tokens_for_region(selected.region)).strip()
    )


def is_strict_pure_newline_blank_trigger(
    state: CanvasState,
    selected: SelectedUpdate,
    config: SlotGeneratorConfig,
    tokenizer,
) -> bool:
    return _strict_pure_newline_blank_request(
        state, selected, config, tokenizer, require_budget=True
    )


def _consume_pure_newline_guard(state: CanvasState, region: Region) -> None:
    remaining = state.pure_newline_guard_remaining.get(region.name)
    if remaining is None or remaining <= 0:
        raise ProtocolError("pure_newline_guard_slot_budget_exhausted")
    if state.pure_newline_guard_global_remaining <= 0:
        raise ProtocolError("pure_newline_guard_global_budget_exhausted")
    state.pure_newline_guard_remaining[region.name] = remaining - 1
    state.pure_newline_guard_global_remaining -= 1


def apply_pure_newline_intervention(
    state: CanvasState,
    selected: SelectedUpdate,
    config: SlotGeneratorConfig,
    tokenizer,
) -> ActionResult:
    if not is_strict_pure_newline_blank_trigger(
        state, selected, config, tokenizer
    ):
        raise ProtocolError("pure_newline_intervention_without_trigger")
    pre_canvas_hash = canvas_only_state_hash(state)
    pre_full_hash = full_state_hash(state)
    info = state.tokenizer_spec.newline_token_map[selected.proposal_token_id]
    pre_slot_budget = state.pure_newline_guard_remaining[selected.region.name]
    pre_global_budget = state.pure_newline_guard_global_remaining

    if state.method == Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO:
        _consume_pure_newline_guard(state, selected.region)
        state.pending_pure_newline_veto = {
            "pre_canvas_hash": pre_canvas_hash,
            "selected_position": int(selected.position),
            "proposal_token_id": int(selected.proposal_token_id),
            "action": "line_boundary",
        }
        return ActionResult(
            "pure_newline_veto",
            {
                "slot": selected.region.name,
                "pre_canvas_hash": pre_canvas_hash,
                "post_canvas_hash": canvas_only_state_hash(state),
                "pre_state_hash": pre_full_hash,
                "post_state_hash": full_state_hash(state),
                "pre_slot_guard_budget": pre_slot_budget,
                "post_slot_guard_budget": state.pure_newline_guard_remaining[
                    selected.region.name
                ],
                "pre_global_guard_budget": pre_global_budget,
                "post_global_guard_budget": state.pure_newline_guard_global_remaining,
                "pending_signature": dict(state.pending_pure_newline_veto),
            },
        )

    if state.method != Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE:
        raise ProtocolError("unknown_pure_newline_guard_method")

    insert_ids = list(state.tokenizer_spec.literal_newline_ids)
    if _decode(tokenizer, insert_ids) != "\n":
        raise ProtocolError("literal_newline_ids_do_not_decode_to_one_newline")
    insert_position = state.positions_for_region(selected.region)[0]
    old_ids = [int(item) for item in state.input_ids[: state.real_length]]
    old_regions = [int(item) for item in state.region_id[: state.real_length]]
    candidate_ids = old_ids[:insert_position] + insert_ids + old_ids[insert_position:]
    candidate_regions = (
        old_regions[:insert_position]
        + [int(Region.INSERTED_BLANK_NEWLINE)] * len(insert_ids)
        + old_regions[insert_position:]
    )
    cap_fallback = False
    try:
        _validate_candidate_caps(
            candidate_ids, candidate_regions, selected.region, config
        )
    except ProtocolError as error:
        if str(error) not in {
            "boundary_context_cap_exceeded",
            "boundary_global_cap_exceeded",
        }:
            raise
        cap_fallback = True

    _consume_pure_newline_guard(state, selected.region)
    if cap_fallback:
        fallback = _apply_line_boundary(
            state,
            selected.position,
            selected.region,
            info,
            config,
            tokenizer,
        )
        return ActionResult(
            fallback.action,
            {
                **fallback.details,
                "blankline_insert_cap_fallback": True,
                "blankline_insert_slot": selected.region.name,
                "pre_insert_state_hash": pre_full_hash,
                "post_insert_state_hash": full_state_hash(state),
                "inserted_newline_positions": [],
                "pre_slot_guard_budget": pre_slot_budget,
                "post_slot_guard_budget": state.pure_newline_guard_remaining[
                    selected.region.name
                ],
                "pre_global_guard_budget": pre_global_budget,
                "post_global_guard_budget": state.pure_newline_guard_global_remaining,
            },
        )

    state.insert_locked_tokens(
        insert_position, insert_ids, Region.INSERTED_BLANK_NEWLINE
    )
    state.inserted_blankline_count += 1
    return ActionResult(
        "insert_locked_blank_newline",
        {
            "blankline_insert_cap_fallback": False,
            "blankline_insert_slot": selected.region.name,
            "inserted_newline_positions": list(
                range(insert_position, insert_position + len(insert_ids))
            ),
            "pre_insert_state_hash": pre_full_hash,
            "post_insert_state_hash": full_state_hash(state),
            "pre_slot_guard_budget": pre_slot_budget,
            "post_slot_guard_budget": state.pure_newline_guard_remaining[
                selected.region.name
            ],
            "pre_global_guard_budget": pre_global_budget,
            "post_global_guard_budget": state.pure_newline_guard_global_remaining,
        },
    )


def _apply_region_local_eos(
    state: CanvasState, position: int, region: Region
) -> ActionResult:
    region_positions = state.positions_for_region(region)
    delete_positions = [
        item
        for item in region_positions
        if item >= position
        and int(state.input_ids[item]) == state.tokenizer_spec.mask_id
    ]
    if position not in delete_positions:
        raise ProtocolError("region_local_eos_selected_position_not_deleted")
    delete_set = set(delete_positions)
    old_ids = [int(item) for item in state.input_ids[: state.real_length]]
    old_regions = [int(item) for item in state.region_id[: state.real_length]]
    new_ids = [token_id for index, token_id in enumerate(old_ids) if index not in delete_set]
    new_regions = [
        region_id for index, region_id in enumerate(old_regions) if index not in delete_set
    ]
    before_length = len(region_positions)
    state.replace_real_sequence(new_ids, new_regions)
    return ActionResult(
        action="region_local_eos",
        details={
            "deleted_positions": len(delete_positions),
            "deleted_masks": len(delete_positions),
            "deleted_original_positions": delete_positions,
            "region_length_before": before_length,
            "region_length_after": state.region_length(region),
            "cross_region_delete_attempts": 0,
        },
    )


def apply_selected_action(
    state: CanvasState,
    position: int,
    proposal_token_id: int,
    config: SlotGeneratorConfig,
    tokenizer=None,
) -> ActionResult:
    state._assert_real_generation_position(position)
    if int(state.input_ids[position]) != state.tokenizer_spec.mask_id:
        raise ProtocolError("selected_position_is_not_unresolved")
    region = Region(int(state.region_id[position]))
    if state.method != Method.JOINT_OPENTAIL and region != state.active_region:
        raise ProtocolError("selected_region_is_not_active_region")
    newline_info = state.tokenizer_spec.newline_token_map.get(int(proposal_token_id))
    if region in HARD_REGIONS and newline_info is not None:
        return _apply_line_boundary(
            state, position, region, newline_info, config, tokenizer
        )
    if proposal_token_id == state.tokenizer_spec.expand_id:
        if not _expand_allowed(state, region, config):
            raise ProtocolError("expand_selected_at_cap")
        pre_budget = state.remaining_expand_budget
        state.expand_at(position)
        if pre_budget is not None:
            if pre_budget <= 0:
                raise ProtocolError("expand_budget_underflow")
            state.remaining_expand_budget = pre_budget - 1
            state.successful_expand_count += 1
        return ActionResult(
            "expand",
            {
                "pre_expand_budget": pre_budget,
                "post_expand_budget": state.remaining_expand_budget,
            },
        )
    if proposal_token_id == state.tokenizer_spec.eos_id:
        return _apply_region_local_eos(state, position, region)
    if proposal_token_id == state.tokenizer_spec.mask_id:
        return ActionResult("mask_noop", {})
    state.replace_token(position, proposal_token_id)
    return ActionResult("normal", {})


def validate_state(state: CanvasState, method: Method) -> None:
    if not (
        len(state.input_ids) == len(state.region_id) == len(state.attention_mask)
    ):
        raise ProtocolError("parallel_state_lengths_differ")
    if not 0 <= state.real_length <= state.capacity:
        raise ProtocolError("invalid_real_length")
    if not torch.all(state.attention_mask[: state.real_length]):
        raise ProtocolError("real_position_missing_attention")
    if torch.any(state.attention_mask[state.real_length :]):
        raise ProtocolError("pad_position_has_attention")
    if torch.any(state.region_id[: state.real_length] == int(Region.PAD)):
        raise ProtocolError("pad_region_inside_real_sequence")
    if torch.any(state.region_id[state.real_length :] != int(Region.PAD)):
        raise ProtocolError("non_pad_region_in_right_padding")
    if state.tokens_for_region(Region.CONTEXT_PREFIX) != list(state.initial_prefix_ids):
        raise ProtocolError("prefix_tokens_mutated")
    if state.tokens_for_region(Region.CONTEXT_SUFFIX) != list(state.initial_suffix_ids):
        raise ProtocolError("suffix_tokens_mutated")
    if state.tokens_for_region(Region.LOCKED_NEWLINE) != list(
        state.initial_locked_newline_ids
    ):
        raise ProtocolError("locked_separator_tokens_mutated")
    inserted_ids = state.tokens_for_region(Region.INSERTED_BLANK_NEWLINE)
    expected_inserted_ids = list(state.tokenizer_spec.literal_newline_ids) * int(
        state.inserted_blankline_count
    )
    if inserted_ids != expected_inserted_ids:
        raise ProtocolError("inserted_blank_newline_tokens_mutated")
    if state.global_middle_length() > state.capacity:
        raise ProtocolError("middle_length_exceeds_context_capacity")
    if method_uses_expand_budget(method):
        if state.initial_expand_budget != 64:
            raise ProtocolError("invalid_initial_expand_budget")
        if state.remaining_expand_budget is None:
            raise ProtocolError("missing_remaining_expand_budget")
        if not 0 <= state.remaining_expand_budget <= state.initial_expand_budget:
            raise ProtocolError("invalid_remaining_expand_budget")
        if (
            state.initial_expand_budget - state.remaining_expand_budget
            != state.successful_expand_count
        ):
            raise ProtocolError("expand_budget_conservation_failed")
    elif (
        state.initial_expand_budget is not None
        or state.remaining_expand_budget is not None
        or state.successful_expand_count != 0
    ):
        raise ProtocolError("unexpected_expand_budget_state")
    if method_uses_pure_newline_guard(method):
        if set(state.pure_newline_guard_remaining) != {
            region.name for region in HARD_REGIONS
        }:
            raise ProtocolError("pure_newline_guard_region_state_invalid")
        if any(
            value not in {0, 1}
            for value in state.pure_newline_guard_remaining.values()
        ):
            raise ProtocolError("pure_newline_guard_slot_budget_invalid")
        consumed = sum(
            1 - int(state.pure_newline_guard_remaining[region.name])
            for region in HARD_REGIONS
        )
        if state.pure_newline_guard_global_remaining != 3 - consumed:
            raise ProtocolError("pure_newline_guard_budget_conservation_failed")
        if not 0 <= state.pure_newline_guard_global_remaining <= 3:
            raise ProtocolError("pure_newline_guard_global_budget_invalid")
        if state.pending_pure_newline_veto is not None:
            if method != Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO:
                raise ProtocolError("pending_pure_newline_veto_on_wrong_method")
            if (
                state.pending_pure_newline_veto.get("pre_canvas_hash")
                != canvas_only_state_hash(state)
            ):
                raise ProtocolError("pending_pure_newline_veto_canvas_mismatch")
        if (
            method == Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO
            and state.inserted_blankline_count != 0
        ):
            raise ProtocolError("c0_contains_inserted_blankline")
        if (
            method == Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE
            and state.pending_pure_newline_veto is not None
        ):
            raise ProtocolError("c_contains_pending_veto")
    elif (
        state.pure_newline_guard_remaining
        or state.pure_newline_guard_global_remaining != 0
        or state.pending_pure_newline_veto is not None
        or state.inserted_blankline_count != 0
        or inserted_ids
    ):
        raise ProtocolError("unexpected_pure_newline_guard_state")
    if method != Method.JOINT_OPENTAIL:
        order = sequential_regions(method)
        active = state.active_region
        if active is not None:
            active_index = order.index(active)
            for completed in order[:active_index]:
                if state.unresolved_positions(completed):
                    raise ProtocolError("completed_region_contains_mask")
            for future in order[active_index + 1 :]:
                if any(
                    token != state.tokenizer_spec.mask_id
                    for token in state.tokens_for_region(future)
                ):
                    raise ProtocolError("future_region_mutated_before_activation")


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(map(int, token_ids)),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def extract_completion(state: CanvasState, method: Method, tokenizer) -> dict[str, Any]:
    generation_regions = method_generation_regions(method)
    unresolved = [
        position
        for region in generation_regions
        for position in state.unresolved_positions(region)
    ]
    if unresolved:
        raise ProtocolError("completion_requested_with_unresolved_masks")
    completion_positions = [
        position
        for position in range(state.real_length)
        if Region(int(state.region_id[position]))
        in set(generation_regions)
        | {Region.LOCKED_NEWLINE, Region.INSERTED_BLANK_NEWLINE}
    ]
    completion_ids = [int(state.input_ids[position]) for position in completion_positions]
    forbidden_final = {
        state.tokenizer_spec.mask_id,
        state.tokenizer_spec.expand_id,
        state.tokenizer_spec.pad_id,
    }
    if any(token_id in forbidden_final for token_id in completion_ids):
        raise ProtocolError("special_sentinel_or_pad_in_completion")

    region_token_ids = {
        region.name: state.tokens_for_region(region) for region in generation_regions
    }
    region_text = {
        name: _decode(tokenizer, token_ids) for name, token_ids in region_token_ids.items()
    }
    for region in generation_regions:
        if region in HARD_REGIONS:
            text = region_text[region.name]
            if "\n" in text or "\r" in text:
                raise ProtocolError("decoded_hard_slot_contains_newline")
    completion = _decode(tokenizer, completion_ids)
    return {
        "completion": completion,
        "completion_token_ids": completion_ids,
        "region_token_ids": region_token_ids,
        "region_text": region_text,
        "blank_region_flags": {
            name: not bool(text.strip()) for name, text in region_text.items()
        },
        "tail_newline_count": region_text.get(Region.OPEN_TAIL.name, "").count("\n"),
        "completion_physical_line_count": len(completion.splitlines()),
    }


def _empty_region_counter(method: Method) -> dict[str, int]:
    return {region.name: 0 for region in method_generation_regions(method)}


def _partial_regions(state: CanvasState, method: Method, tokenizer) -> dict[str, Any]:
    region_ids = {
        region.name: state.tokens_for_region(region)
        for region in method_generation_regions(method)
    }
    return {
        "region_token_ids": region_ids,
        "region_text": {name: _decode(tokenizer, ids) for name, ids in region_ids.items()},
    }


def _record_nonempty_rejections(
    events: Sequence[Mapping[str, Any]],
    all_events: list[dict[str, Any]],
    by_slot: dict[str, int],
    by_action: dict[str, int],
) -> None:
    for event in events:
        record = dict(event)
        all_events.append(record)
        by_slot[str(record["slot"])] += 1
        action = str(record["hypothetical_action"])
        by_action[action] = by_action.get(action, 0) + 1


def run_slot_generation(
    model,
    tokenizer,
    tokenizer_spec: TokenizerSpec,
    method: Method,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    config: SlotGeneratorConfig,
    save_trace: bool,
    task_id: str | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    state = build_canvas(method, prefix_ids, suffix_ids, tokenizer_spec, config, device)
    initial_lengths = {
        region.name: state.region_length(region) for region in method_generation_regions(method)
    }
    region_forwards = _empty_region_counter(method)
    selected_updates = _empty_region_counter(method)
    normal_updates = _empty_region_counter(method)
    expand_counts = _empty_region_counter(method)
    delete_counts = _empty_region_counter(method)
    mask_noops = _empty_region_counter(method)
    forward_lengths: list[int] = []
    trace: list[dict[str, Any]] = []
    protocol_flags: list[str] = []
    hard_forbidden_attempts = 0
    slot_cap_hits = 0
    global_cap_hits = 0
    joint_updates_before_slot0_complete = 0
    first_selected_region: str | None = None
    active_region_history: list[str] = []
    newline_boundary_event_details: list[dict[str, Any]] = []
    region_local_eos_event_details: list[dict[str, Any]] = []
    exact_cycle_events: list[dict[str, Any]] = []
    budget_draining_loop_events: list[dict[str, Any]] = []
    transition_detector = (
        BudgetedTransitionCycleDetector()
        if method_uses_expand_budget(method)
        else ExactTransitionCycleDetector()
    )
    mixed_newline_token_events = 0
    pure_newline_token_events = 0
    multiple_newline_token_events = 0
    boundary_retokenized_left_token_count = 0
    discarded_internal_right_text: list[str] = []
    discarded_masks_after_boundary = 0
    discarded_resolved_token_ids_after_boundary: list[int] = []
    discarded_resolved_segments_after_boundary: list[dict[str, Any]] = []
    discarded_resolved_tokens_after_boundary = 0
    boundary_events_with_left_masks_remaining = 0
    region_local_eos_deleted_masks = 0
    max_masks_deleted_by_one_eos = 0
    cross_region_delete_attempts = 0
    expand_logit_blocked_by_budget_count = 0
    budget_exhausted_forward_index: int | None = None
    nonempty_guard_rejection_events: list[dict[str, Any]] = []
    nonempty_guard_rejections_by_slot = _empty_region_counter(method)
    nonempty_guard_rejections_by_action: dict[str, int] = {}
    nonempty_guard_no_valid_action_count = 0
    pure_newline_guard_events: list[dict[str, Any]] = []
    future_slot_diagnostic_events: list[dict[str, Any]] = []
    pure_newline_veto_triggers = 0
    pure_newline_veto_fallbacks = 0
    pure_newline_veto_extra_forwards = 0
    pure_newline_veto_reselected_action: list[str] = []
    blankline_insert_triggers = 0
    blankline_insert_slots: list[str] = []
    blankline_insert_cap_fallbacks = 0
    inserted_newline_positions: list[list[int]] = []
    pre_insert_state_hashes: list[str] = []
    post_insert_state_hashes: list[str] = []
    repeated_blankline_requests = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    status = "completed"
    try:
        while True:
            validate_state(state, method)
            positions = eligible_positions(state, method)
            if not positions:
                break
            if len(forward_lengths) >= config.max_total_forwards:
                raise ProtocolError("forward_cap_with_unresolved_masks")

            eligible_regions = {
                Region(int(state.region_id[position])) for position in positions
            }
            for region in eligible_regions:
                region_forwards[region.name] += 1
            if state.active_region is not None:
                active_region_history.append(state.active_region.name)

            pending_retry_forward = state.pending_pure_newline_veto is not None
            pre_canvas_hash = canvas_only_state_hash(state)
            pre_full_hash = full_state_hash(state)
            pre_action_hash = (
                pre_full_hash if method_uses_expand_budget(method) else pre_canvas_hash
            )
            pre_expand_budget = state.remaining_expand_budget
            pre_pure_newline_guard_remaining = dict(
                state.pure_newline_guard_remaining
            )
            pre_pure_newline_guard_global_remaining = int(
                state.pure_newline_guard_global_remaining
            )
            pre_pending_pure_newline_veto = (
                None
                if state.pending_pure_newline_veto is None
                else dict(state.pending_pure_newline_veto)
            )
            pre_region_lengths = {
                region.name: state.region_length(region)
                for region in method_generation_regions(method)
            }

            input_ids, attention_mask, position_ids = forward_inputs(state)
            real_length = state.real_length
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            forward_lengths.append(real_length)
            if pending_retry_forward:
                pure_newline_veto_extra_forwards += 1
            logits = shifted_logits(output.logits)[0]
            expand_logit_blocked_by_budget = bool(
                state.remaining_expand_budget == 0
            )
            if expand_logit_blocked_by_budget:
                expand_logit_blocked_by_budget_count += len(positions)
            guard_rejections_this_forward: list[dict[str, Any]] = []
            pending_veto_used = False
            pending_veto_fallback = False
            if method_uses_nonempty_guard(method):
                try:
                    selected, guard_rejections_this_forward = (
                        select_nonempty_guarded_update(
                            state,
                            method,
                            logits,
                            config,
                            tokenizer,
                            task_id=task_id,
                            forward_index=len(forward_lengths),
                        )
                    )
                except NonemptyGuardNoValidAction as error:
                    guard_rejections_this_forward = error.rejection_events
                    _record_nonempty_rejections(
                        guard_rejections_this_forward,
                        nonempty_guard_rejection_events,
                        nonempty_guard_rejections_by_slot,
                        nonempty_guard_rejections_by_action,
                    )
                    nonempty_guard_no_valid_action_count += 1
                    raise
                _record_nonempty_rejections(
                    guard_rejections_this_forward,
                    nonempty_guard_rejection_events,
                    nonempty_guard_rejections_by_slot,
                    nonempty_guard_rejections_by_action,
                )
            elif (
                method == Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO
                and state.pending_pure_newline_veto is not None
            ):
                selected, pending_veto_used, pending_veto_fallback = (
                    select_update_with_pending_veto(
                        state, method, logits, config
                    )
                )
            else:
                selected = select_update(state, method, logits, config)
            if first_selected_region is None:
                first_selected_region = selected.region.name
            if (
                method == Method.JOINT_OPENTAIL
                and state.unresolved_positions(Region.HARD_SLOT_0)
                and selected.region != Region.HARD_SLOT_0
            ):
                joint_updates_before_slot0_complete += 1
            if (
                not method_uses_boundary_decoder(method)
                and
                selected.region in HARD_REGIONS
                and selected.unconstrained_proposal_token_id
                in tokenizer_spec.newline_token_ids
            ):
                hard_forbidden_attempts += 1
            if (
                selected.region in HARD_REGIONS
                and state.region_length(selected.region) >= config.max_hard_slot_tokens
            ):
                slot_cap_hits += 1
            if state.global_middle_length() >= config.max_global_middle_tokens:
                global_cap_hits += 1

            trigger = (
                method_uses_pure_newline_guard(method)
                and is_strict_pure_newline_blank_trigger(
                    state, selected, config, tokenizer
                )
            )
            trigger_event: dict[str, Any] | None = None
            if trigger:
                right_positions = [
                    position
                    for position in state.positions_for_region(selected.region)
                    if position > selected.position
                ]
                resolved_segments = _resolved_segments(
                    state, right_positions, tokenizer
                )
                diagnostic = collect_future_slot_diagnostics(
                    state,
                    method,
                    logits,
                    config,
                    tokenizer,
                    selected,
                )
                diagnostic_event = {
                    "task_id": task_id,
                    "forward_index": len(forward_lengths),
                    "slot": selected.region.name,
                    **diagnostic,
                }
                future_slot_diagnostic_events.append(diagnostic_event)
                trigger_event = {
                    "task_id": task_id,
                    "forward_index": len(forward_lengths),
                    "slot": selected.region.name,
                    "selected_position": selected.position,
                    "proposal_token_id": selected.proposal_token_id,
                    "decoded_proposal": _decode(
                        tokenizer, [selected.proposal_token_id]
                    ),
                    "pre_canvas_hash": pre_canvas_hash,
                    "pre_full_state_hash": pre_full_hash,
                    "resolved_right_token_ids": [
                        int(state.input_ids[position])
                        for position in right_positions
                        if int(state.input_ids[position])
                        != state.tokenizer_spec.mask_id
                    ],
                    "resolved_right_segments": resolved_segments,
                    "has_resolved_right_token": bool(resolved_segments),
                    "has_nonwhitespace_resolved_right_token": any(
                        segment["text"].strip() for segment in resolved_segments
                    ),
                    "future_slot_diagnostic_index": len(
                        future_slot_diagnostic_events
                    )
                    - 1,
                }
                action_result = apply_pure_newline_intervention(
                    state, selected, config, tokenizer
                )
                if method == Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO:
                    pure_newline_veto_triggers += 1
                else:
                    blankline_insert_triggers += 1
                    blankline_insert_slots.append(selected.region.name)
            else:
                if method_uses_pure_newline_guard(method) and _strict_pure_newline_blank_request(
                    state,
                    selected,
                    config,
                    tokenizer,
                    require_budget=False,
                ):
                    repeated_blankline_requests += 1
                action_result = apply_selected_action(
                    state,
                    selected.position,
                    selected.proposal_token_id,
                    config,
                    tokenizer=tokenizer,
                )
            if pending_veto_used:
                pure_newline_veto_reselected_action.append(action_result.action)
                if pending_veto_fallback:
                    pure_newline_veto_fallbacks += 1
            if method != Method.JOINT_OPENTAIL:
                advance_active_region(state, method)
            validate_state(state, method)
            post_canvas_hash = canvas_only_state_hash(state)
            post_full_hash = full_state_hash(state)
            post_action_hash = (
                post_full_hash if method_uses_expand_budget(method) else post_canvas_hash
            )
            post_expand_budget = state.remaining_expand_budget
            if (
                pre_expand_budget is not None
                and pre_expand_budget > 0
                and post_expand_budget == 0
                and budget_exhausted_forward_index is None
            ):
                budget_exhausted_forward_index = len(forward_lengths)
            post_region_lengths = {
                region.name: state.region_length(region)
                for region in method_generation_regions(method)
            }
            action = action_result.action
            if trigger_event is not None:
                trigger_event.update(
                    {
                        "action": action,
                        "action_details": action_result.details,
                        "post_canvas_hash": post_canvas_hash,
                        "post_full_state_hash": post_full_hash,
                    }
                )
                pure_newline_guard_events.append(trigger_event)
            signature = transition_signature(
                pre_action_hash,
                selected.position,
                selected.region,
                selected.proposal_token_id,
                action,
                post_action_hash,
            )
            canvas_signature = transition_signature(
                pre_canvas_hash,
                selected.position,
                selected.region,
                selected.proposal_token_id,
                action,
                post_canvas_hash,
            )
            selected_updates[selected.region.name] += 1
            if action == "normal":
                normal_updates[selected.region.name] += 1
            elif action == "expand":
                expand_counts[selected.region.name] += 1
            elif action == "pure_newline_veto":
                pass
            elif action == "insert_locked_blank_newline":
                inserted_newline_positions.append(
                    list(action_result.details["inserted_newline_positions"])
                )
                pre_insert_state_hashes.append(
                    str(action_result.details["pre_insert_state_hash"])
                )
                post_insert_state_hashes.append(
                    str(action_result.details["post_insert_state_hash"])
                )
            elif action == "region_local_eos":
                delete_counts[selected.region.name] += 1
                eos_event = {
                    "task_id": task_id,
                    "slot": selected.region.name,
                    "original_position": selected.position,
                    "proposal_token_id": selected.proposal_token_id,
                    "pre_state_hash": pre_action_hash,
                    "post_state_hash": post_action_hash,
                    "region_lengths_before": pre_region_lengths,
                    "region_lengths_after": post_region_lengths,
                    **action_result.details,
                }
                region_local_eos_event_details.append(eos_event)
                deleted_masks = int(action_result.details["deleted_masks"])
                region_local_eos_deleted_masks += deleted_masks
                max_masks_deleted_by_one_eos = max(
                    max_masks_deleted_by_one_eos, deleted_masks
                )
                cross_region_delete_attempts += int(
                    action_result.details["cross_region_delete_attempts"]
                )
            elif action == "line_boundary":
                if bool(action_result.details.get("blankline_insert_cap_fallback")):
                    blankline_insert_cap_fallbacks += 1
                    inserted_newline_positions.append([])
                    pre_insert_state_hashes.append(
                        str(action_result.details["pre_insert_state_hash"])
                    )
                    post_insert_state_hashes.append(
                        str(action_result.details["post_insert_state_hash"])
                    )
                boundary_event = {
                    "task_id": task_id,
                    "slot": selected.region.name,
                    "original_position": selected.position,
                    "proposal_token_id": selected.proposal_token_id,
                    "pre_state_hash": pre_action_hash,
                    "post_state_hash": post_action_hash,
                    "region_lengths_before": pre_region_lengths,
                    "region_lengths_after": post_region_lengths,
                    **action_result.details,
                }
                newline_boundary_event_details.append(boundary_event)
                normalized = str(action_result.details["normalized_text"])
                if normalized.strip("\n"):
                    mixed_newline_token_events += 1
                else:
                    pure_newline_token_events += 1
                if int(action_result.details["newline_count"]) > 1:
                    multiple_newline_token_events += 1
                boundary_retokenized_left_token_count += int(
                    action_result.details["boundary_retokenized_left_token_count"]
                )
                discarded_internal_right_text.append(
                    str(action_result.details["discarded_internal_right_text"])
                )
                discarded_masks_after_boundary += int(
                    action_result.details["discarded_masks_after_boundary"]
                )
                discarded_resolved_token_ids_after_boundary.extend(
                    action_result.details[
                        "discarded_resolved_token_ids_after_boundary"
                    ]
                )
                discarded_resolved_segments_after_boundary.extend(
                    action_result.details[
                        "discarded_resolved_segments_after_boundary"
                    ]
                )
                discarded_resolved_tokens_after_boundary += int(
                    action_result.details[
                        "discarded_resolved_tokens_after_boundary"
                    ]
                )
                boundary_events_with_left_masks_remaining += int(
                    action_result.details[
                        "boundary_event_with_left_masks_remaining"
                    ]
                )
            elif action == "mask_noop":
                mask_noops[selected.region.name] += 1
            else:
                raise AssertionError(f"Unknown action: {action}")

            terminal_cycle = False
            if method_uses_expand_budget(method):
                cycle = transition_detector.observe(
                    full_signature=signature,
                    canvas_signature=canvas_signature,
                    pre_remaining_budget=int(pre_expand_budget),
                    post_remaining_budget=int(post_expand_budget),
                    forward_index=len(forward_lengths),
                    region_lengths=post_region_lengths,
                )
                if cycle is not None:
                    event = {
                        "task_id": task_id,
                        "slot": selected.region.name,
                        "canvas_only_pre_state_hash": pre_canvas_hash,
                        "canvas_only_post_state_hash": post_canvas_hash,
                        "full_pre_state_hash": pre_full_hash,
                        "full_post_state_hash": post_full_hash,
                        **cycle,
                    }
                    if cycle["classification"] == "budget_draining_loop":
                        budget_draining_loop_events.append(event)
                    else:
                        exact_cycle_events.append(event)
                        terminal_cycle = True
            else:
                cycle = transition_detector.observe(
                    signature, len(forward_lengths), post_region_lengths
                )
                if cycle is not None:
                    exact_cycle_events.append(cycle)
                    terminal_cycle = True
            if save_trace:
                next_positions = eligible_positions(state, method)
                next_active_region = (
                    None if state.active_region is None else state.active_region.name
                )
                trace.append(
                    {
                        "forward_index": len(forward_lengths),
                        "sequence_length": real_length,
                        "selected_position": selected.position,
                        "selected_region": selected.region.name,
                        "proposal_token_id": selected.proposal_token_id,
                        "unconstrained_proposal_token_id": selected.unconstrained_proposal_token_id,
                        "confidence": selected.confidence,
                        "action": action,
                        "action_details": action_result.details,
                        "pre_state_hash": pre_action_hash,
                        "post_state_hash": post_action_hash,
                        "pre_expand_budget": pre_expand_budget,
                        "post_expand_budget": post_expand_budget,
                        "canvas_only_pre_state_hash": pre_canvas_hash,
                        "canvas_only_post_state_hash": post_canvas_hash,
                        "full_pre_state_hash": pre_full_hash,
                        "full_post_state_hash": post_full_hash,
                        "expand_logit_blocked_by_budget": expand_logit_blocked_by_budget,
                        "nonempty_guard_rejection_events": guard_rejections_this_forward,
                        "pre_pure_newline_guard_remaining": pre_pure_newline_guard_remaining,
                        "post_pure_newline_guard_remaining": dict(
                            state.pure_newline_guard_remaining
                        ),
                        "pre_pure_newline_guard_global_remaining": pre_pure_newline_guard_global_remaining,
                        "post_pure_newline_guard_global_remaining": state.pure_newline_guard_global_remaining,
                        "pre_pending_pure_newline_veto": pre_pending_pure_newline_veto,
                        "post_pending_pure_newline_veto": (
                            None
                            if state.pending_pure_newline_veto is None
                            else dict(state.pending_pure_newline_veto)
                        ),
                        "pending_veto_retry_forward": pending_retry_forward,
                        "pending_veto_used": pending_veto_used,
                        "pending_veto_fallback": pending_veto_fallback,
                        "pure_newline_triggered": trigger,
                        "transition_signature": list(signature),
                        "active_region": next_active_region,
                        "active_mask_count": len(next_positions),
                        "global_middle_length": state.global_middle_length(),
                        "region_lengths": {
                            region.name: state.region_length(region)
                            for region in method_generation_regions(method)
                        },
                        "future_regions_all_mask": (
                            method == Method.JOINT_OPENTAIL
                            or _future_regions_all_mask(state, method)
                        ),
                    }
                )
            if terminal_cycle:
                raise ProtocolError("exact_deterministic_cycle")

        if state.unresolved_positions():
            raise ProtocolError("generation_ended_with_unresolved_masks")
        validate_state(state, method)
        extracted = extract_completion(state, method, tokenizer)
        if method_uses_nonempty_guard(method) and any(
            extracted["blank_region_flags"].values()
        ):
            raise ProtocolError("completed_nonempty_invariant_failed")
    except ProtocolError as error:
        status = "protocol_error"
        protocol_flags.append(str(error))
        extracted = {
            "completion": "",
            "completion_token_ids": [],
            **_partial_regions(state, method, tokenizer),
            "blank_region_flags": {},
            "tail_newline_count": 0,
            "completion_physical_line_count": 0,
        }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - start
    final_lengths = {
        region.name: state.region_length(region) for region in method_generation_regions(method)
    }
    initial_expand_budget = state.initial_expand_budget
    remaining_expand_budget = state.remaining_expand_budget
    expand_budget_consumed = (
        None
        if initial_expand_budget is None or remaining_expand_budget is None
        else initial_expand_budget - remaining_expand_budget
    )
    expand_budget_conservation_passed = (
        True
        if initial_expand_budget is None
        else expand_budget_consumed == state.successful_expand_count
    )
    final_nonempty_invariant_passed = bool(
        status == "completed"
        and not any(extracted.get("blank_region_flags", {}).values())
    )
    return {
        **extracted,
        "method": method.value,
        "regions": {
            name: {"token_ids": extracted["region_token_ids"][name], "text": text}
            for name, text in extracted["region_text"].items()
        },
        "region_lengths": {"initial": initial_lengths, "final": final_lengths},
        "region_forward_counts": region_forwards,
        "selected_update_counts": selected_updates,
        "total_forwards": len(forward_lengths),
        "forward_sequence_lengths": forward_lengths,
        "token_forwards": sum(forward_lengths),
        "normal_update_counts": normal_updates,
        "expand_counts": expand_counts,
        "delete_counts": delete_counts,
        "mask_noop_counts": mask_noops,
        "hard_forbidden_newline_attempts": hard_forbidden_attempts,
        "newline_boundary_events": len(newline_boundary_event_details),
        "newline_boundary_event_details": newline_boundary_event_details,
        "mixed_newline_token_events": mixed_newline_token_events,
        "pure_newline_token_events": pure_newline_token_events,
        "multiple_newline_token_events": multiple_newline_token_events,
        "boundary_retokenized_left_token_count": boundary_retokenized_left_token_count,
        "discarded_internal_right_text": discarded_internal_right_text,
        "discarded_masks_after_boundary": discarded_masks_after_boundary,
        "discarded_resolved_token_ids_after_boundary": discarded_resolved_token_ids_after_boundary,
        "discarded_resolved_segments_after_boundary": discarded_resolved_segments_after_boundary,
        "discarded_resolved_tokens_after_boundary": discarded_resolved_tokens_after_boundary,
        "boundary_events_with_left_masks_remaining": boundary_events_with_left_masks_remaining,
        "region_local_eos_events": len(region_local_eos_event_details),
        "region_local_eos_event_details": region_local_eos_event_details,
        "region_local_eos_deleted_masks": region_local_eos_deleted_masks,
        "max_masks_deleted_by_one_eos": max_masks_deleted_by_one_eos,
        "cross_region_delete_attempts": cross_region_delete_attempts,
        "exact_cycle_events": exact_cycle_events,
        "initial_expand_budget": initial_expand_budget,
        "remaining_expand_budget": remaining_expand_budget,
        "expand_budget_consumed": expand_budget_consumed,
        "expand_budget_conservation_passed": expand_budget_conservation_passed,
        "budget_exhausted": remaining_expand_budget == 0,
        "budget_exhausted_forward_index": budget_exhausted_forward_index,
        "expand_logit_blocked_by_budget_count": expand_logit_blocked_by_budget_count,
        "budget_draining_loop_count": len(budget_draining_loop_events),
        "budget_draining_loop_events": budget_draining_loop_events,
        "nonempty_guard_rejection_count": len(nonempty_guard_rejection_events),
        "nonempty_guard_rejections_by_slot": nonempty_guard_rejections_by_slot,
        "nonempty_guard_rejections_by_action": nonempty_guard_rejections_by_action,
        "nonempty_guard_rejection_events": nonempty_guard_rejection_events,
        "nonempty_guard_no_valid_action_count": nonempty_guard_no_valid_action_count,
        "final_nonempty_invariant_passed": final_nonempty_invariant_passed,
        "pure_newline_guard_trigger_count": len(pure_newline_guard_events),
        "pure_newline_guard_events": pure_newline_guard_events,
        "future_slot_diagnostic_events": future_slot_diagnostic_events,
        "pure_newline_guard_budget_remaining": dict(
            state.pure_newline_guard_remaining
        ),
        "pure_newline_guard_global_budget_remaining": state.pure_newline_guard_global_remaining,
        "pure_newline_veto_triggers": pure_newline_veto_triggers,
        "pure_newline_veto_budget_remaining": dict(
            state.pure_newline_guard_remaining
        ),
        "pure_newline_veto_pending_signature": state.pending_pure_newline_veto,
        "pure_newline_veto_reselected_action": pure_newline_veto_reselected_action,
        "pure_newline_veto_fallbacks": pure_newline_veto_fallbacks,
        "pure_newline_veto_extra_forwards": pure_newline_veto_extra_forwards,
        "blankline_insert_triggers": blankline_insert_triggers,
        "blankline_inserted_count": state.inserted_blankline_count,
        "blankline_insert_slot": blankline_insert_slots,
        "blankline_guard_budget_remaining": dict(
            state.pure_newline_guard_remaining
        ),
        "repeated_blankline_requests": repeated_blankline_requests,
        "blankline_insert_cap_fallbacks": blankline_insert_cap_fallbacks,
        "inserted_newline_positions": inserted_newline_positions,
        "final_inserted_newline_positions": state.positions_for_region(
            Region.INSERTED_BLANK_NEWLINE
        ),
        "pre_insert_state_hash": pre_insert_state_hashes,
        "post_insert_state_hash": post_insert_state_hashes,
        "slot_expand_cap_hits": slot_cap_hits,
        "global_expand_cap_hits": global_cap_hits,
        "unresolved_mask_count": len(state.unresolved_positions()),
        "protocol_flags": protocol_flags,
        "context_token_count": len(prefix_ids) + len(suffix_ids),
        "final_real_sequence_length": state.real_length,
        "wall_time_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "status": status,
        "first_selected_region": first_selected_region,
        "joint_updates_before_slot0_complete": joint_updates_before_slot0_complete,
        "active_region_history": active_region_history,
        "step_trace": trace if save_trace else None,
    }


def _future_regions_all_mask(state: CanvasState, method: Method) -> bool:
    if method == Method.JOINT_OPENTAIL or state.active_region is None:
        return True
    order = sequential_regions(method)
    active_index = order.index(state.active_region)
    return all(
        all(token == state.tokenizer_spec.mask_id for token in state.tokens_for_region(region))
        for region in order[active_index + 1 :]
    )
