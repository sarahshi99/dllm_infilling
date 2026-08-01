from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable, Sequence

import torch


class ProtocolError(RuntimeError):
    pass


class Method(str, Enum):
    V2_HARD = "v2_hard"
    V2_OPENTAIL = "v2_opentail"
    JOINT_OPENTAIL = "joint_opentail"


class Region(IntEnum):
    CONTEXT_PREFIX = 0
    HARD_SLOT_0 = 1
    HARD_SLOT_1 = 2
    HARD_SLOT_2 = 3
    OPEN_TAIL = 4
    LOCKED_NEWLINE = 5
    CONTEXT_SUFFIX = 6
    PAD = 7


HARD_REGIONS = (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2)
GENERATION_REGIONS = (*HARD_REGIONS, Region.OPEN_TAIL)


@dataclass(frozen=True)
class TokenizerSpec:
    bos_id: int
    eos_id: int
    pad_id: int
    mask_id: int
    expand_id: int
    newline_token_ids: frozenset[int]
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


@dataclass(frozen=True)
class SelectedUpdate:
    position: int
    region: Region
    proposal_token_id: int
    unconstrained_proposal_token_id: int
    confidence: float


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
        generation_or_separator = set(GENERATION_REGIONS) | {Region.LOCKED_NEWLINE}
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

    def _assert_real_generation_position(self, position: int) -> None:
        if not 0 <= position < self.real_length or not bool(self.attention_mask[position]):
            raise ProtocolError("selected_position_is_not_real")
        region = Region(int(self.region_id[position]))
        if region not in GENERATION_REGIONS or region == Region.PAD:
            raise ProtocolError("selected_region_is_not_generation_eligible")


def sequential_regions(method: Method) -> tuple[Region, ...]:
    if method == Method.V2_HARD:
        return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2)
    if method == Method.V2_OPENTAIL:
        return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.OPEN_TAIL)
    if method == Method.JOINT_OPENTAIL:
        return ()
    raise ValueError(f"Unknown method: {method}")


def method_generation_regions(method: Method) -> tuple[Region, ...]:
    if method == Method.V2_HARD:
        return sequential_regions(method)
    return (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.OPEN_TAIL)


def scan_newline_token_ids(tokenizer) -> tuple[list[int], dict[str, Any]]:
    token_ids: list[int] = []
    examples: list[dict[str, Any]] = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" in decoded or "\r" in decoded:
            token_ids.append(token_id)
            if len(examples) < 20:
                examples.append({"token_id": token_id, "decoded": decoded})
    digest = hashlib.sha256("\n".join(map(str, token_ids)).encode("utf-8")).hexdigest()
    return token_ids, {
        "vocab_size": len(tokenizer),
        "newline_token_count": len(token_ids),
        "newline_token_ids_sha256": digest,
        "examples": examples,
    }


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
    if method == Method.V2_HARD:
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
    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ProtocolError("nan_or_all_invalid_logits")
    proposals = torch.argmax(filtered, dim=-1)
    negative_entropy = torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-10)), dim=-1
    )
    return negative_entropy, proposals


def _expand_allowed(
    state: CanvasState, region: Region, config: SlotGeneratorConfig
) -> bool:
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
    vocab_size = active_logits.shape[-1]
    newline_ids = [
        token_id for token_id in state.tokenizer_spec.newline_token_ids if token_id < vocab_size
    ]
    for row, position in enumerate(positions):
        region = Region(int(state.region_id[position]))
        if region in HARD_REGIONS and newline_ids:
            active_logits[row, newline_ids] = float("-inf")
        if not _expand_allowed(state, region, config):
            active_logits[row, state.tokenizer_spec.expand_id] = float("-inf")
    return active_logits, positions


def select_update(
    state: CanvasState,
    method: Method,
    full_logits: torch.Tensor,
    config: SlotGeneratorConfig,
) -> SelectedUpdate:
    active_logits, positions = constrained_active_logits(state, method, full_logits, config)
    confidence, proposals = _entropy_proposals(active_logits, config)
    if confidence.numel() == 0:
        raise ProtocolError("topk_requested_without_active_mask")
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


def apply_selected_action(
    state: CanvasState,
    position: int,
    proposal_token_id: int,
    config: SlotGeneratorConfig,
) -> str:
    state._assert_real_generation_position(position)
    if int(state.input_ids[position]) != state.tokenizer_spec.mask_id:
        raise ProtocolError("selected_position_is_not_unresolved")
    region = Region(int(state.region_id[position]))
    if state.method != Method.JOINT_OPENTAIL and region != state.active_region:
        raise ProtocolError("selected_region_is_not_active_region")
    if region in HARD_REGIONS and proposal_token_id in state.tokenizer_spec.newline_token_ids:
        raise ProtocolError("forbidden_newline_selected_in_hard_slot")
    if proposal_token_id == state.tokenizer_spec.expand_id:
        if not _expand_allowed(state, region, config):
            raise ProtocolError("expand_selected_at_cap")
        state.expand_at(position)
        return "expand"
    if proposal_token_id == state.tokenizer_spec.eos_id:
        state.delete_at(position)
        return "delete"
    if proposal_token_id == state.tokenizer_spec.mask_id:
        return "mask_noop"
    state.replace_token(position, proposal_token_id)
    return "normal"


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
    if state.global_middle_length() > state.capacity:
        raise ProtocolError("middle_length_exceeds_context_capacity")
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
        in set(generation_regions) | {Region.LOCKED_NEWLINE}
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


def run_slot_generation(
    model,
    tokenizer,
    tokenizer_spec: TokenizerSpec,
    method: Method,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    config: SlotGeneratorConfig,
    save_trace: bool,
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
            logits = shifted_logits(output.logits)[0]
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

            action = apply_selected_action(
                state, selected.position, selected.proposal_token_id, config
            )
            selected_updates[selected.region.name] += 1
            if action == "normal":
                normal_updates[selected.region.name] += 1
            elif action == "expand":
                expand_counts[selected.region.name] += 1
            elif action == "delete":
                delete_counts[selected.region.name] += 1
            elif action == "mask_noop":
                mask_noops[selected.region.name] += 1
            else:
                raise AssertionError(f"Unknown action: {action}")

            validate_state(state, method)
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

        if state.unresolved_positions():
            raise ProtocolError("generation_ended_with_unresolved_masks")
        validate_state(state, method)
        extracted = extract_completion(state, method, tokenizer)
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
