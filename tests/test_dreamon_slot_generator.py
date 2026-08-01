from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from repro_scripts.dreamon_slot_generator import (
    Method,
    ProtocolError,
    Region,
    SlotGeneratorConfig,
    TokenizerSpec,
    advance_active_region,
    apply_selected_action,
    build_canvas,
    constrained_active_logits,
    eligible_positions,
    extract_completion,
    forward_inputs,
    run_slot_generation,
    scan_newline_token_ids,
    select_update,
    validate_state,
)


class FakeTokenizer:
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 11
    mask_token_id = 12

    _pieces = {
        0: "a",
        1: "b",
        2: "c",
        3: " ",
        4: "\n",
        5: "\r",
        6: ":\n",
        7: "xy",
        8: "z",
        9: "q",
        10: "<bos>",
        11: "<eos>",
        12: "<mask>",
        13: "<expand>",
        14: "r",
        15: "s",
    }

    def __len__(self) -> int:
        return len(self._pieces)

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        if text == "\n":
            return [4]
        raise AssertionError(f"Unexpected fake encode input: {text!r}")


@pytest.fixture()
def tokenizer():
    return FakeTokenizer()


@pytest.fixture()
def spec(tokenizer):
    newline_ids, _ = scan_newline_token_ids(tokenizer)
    return TokenizerSpec(
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        expand_id=13,
        newline_token_ids=frozenset(newline_ids),
        literal_newline_ids=(4,),
    )


@pytest.fixture()
def config():
    return SlotGeneratorConfig(
        initial_masks_per_region=4,
        max_context_tokens=64,
        max_total_forwards=256,
        max_global_middle_tokens=64,
        max_hard_slot_tokens=32,
        temperature=0.0,
        top_p=0.9,
    )


def make_state(method: Method, spec: TokenizerSpec, config: SlotGeneratorConfig):
    return build_canvas(
        method=method,
        prefix_ids=[spec.bos_id, 0, 1],
        suffix_ids=[2, spec.eos_id],
        tokenizer_spec=spec,
        config=config,
        device=torch.device("cpu"),
    )


def resolve_region(state, region: Region, token_id: int = 0):
    positions = state.positions_for_region(region)
    for position in positions:
        if int(state.input_ids[position]) == state.tokenizer_spec.mask_id:
            state.replace_token(position, token_id)


def logits_for_state(state, vocab_size: int = 16, default_token: int = 0):
    logits = torch.full((state.real_length, vocab_size), -8.0)
    logits[:, default_token] = 8.0
    return logits


def test_newline_scan_finds_every_single_token_with_newline_or_carriage_return(tokenizer):
    newline_ids, metadata = scan_newline_token_ids(tokenizer)
    assert newline_ids == [4, 5, 6]
    assert metadata["vocab_size"] == 16
    assert metadata["newline_token_count"] == 3


def test_full_forward_contains_prefix_future_masks_locked_newlines_and_suffix(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    input_ids, attention, position_ids = forward_inputs(state)
    assert input_ids.shape == (1, state.real_length)
    assert attention.shape == (1, 1, state.real_length, state.real_length)
    assert torch.all(attention)
    assert position_ids.tolist() == [list(range(state.real_length))]
    assert set(state.region_id[: state.real_length].tolist()) >= {
        Region.CONTEXT_PREFIX,
        Region.HARD_SLOT_0,
        Region.HARD_SLOT_1,
        Region.HARD_SLOT_2,
        Region.LOCKED_NEWLINE,
        Region.CONTEXT_SUFFIX,
    }


def test_active_selection_only_returns_current_region(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    positions = eligible_positions(state, Method.V2_HARD)
    assert positions
    assert {Region(int(state.region_id[p])) for p in positions} == {Region.HARD_SLOT_0}


def test_future_higher_confidence_still_cannot_be_selected(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    logits = logits_for_state(state)
    future = state.positions_for_region(Region.HARD_SLOT_1)[0]
    logits[future] = -20
    logits[future, 1] = 20
    selected = select_update(state, Method.V2_HARD, logits, config)
    assert selected.region == Region.HARD_SLOT_0


def test_progressive_action_rejects_future_region_even_if_called_directly(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    future = state.positions_for_region(Region.HARD_SLOT_1)[0]
    with pytest.raises(ProtocolError, match="selected_region_is_not_active_region"):
        apply_selected_action(state, future, 0, config)


def test_slot_zero_completion_activates_slot_one(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    resolve_region(state, Region.HARD_SLOT_0)
    assert advance_active_region(state, Method.V2_HARD) == Region.HARD_SLOT_1


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (Method.V2_HARD, Region.HARD_SLOT_2),
        (Method.V2_OPENTAIL, Region.OPEN_TAIL),
    ],
)
def test_slot_one_completion_activates_expected_final_region(method, expected, spec, config):
    state = make_state(method, spec, config)
    resolve_region(state, Region.HARD_SLOT_0)
    resolve_region(state, Region.HARD_SLOT_1)
    assert advance_active_region(state, method) == expected


def test_completed_active_region_switches_before_topk(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    resolve_region(state, Region.HARD_SLOT_0)
    assert advance_active_region(state, Method.V2_HARD) == Region.HARD_SLOT_1
    selected = select_update(state, Method.V2_HARD, logits_for_state(state), config)
    assert selected.region == Region.HARD_SLOT_1


def test_normal_update_preserves_region_id(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    position = eligible_positions(state, Method.V2_HARD)[0]
    before = int(state.region_id[position])
    apply_selected_action(state, position, 0, config)
    assert int(state.region_id[position]) == before
    assert int(state.input_ids[position]) == 0


def test_expand_in_middle_inherits_region_and_shifts_separator(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    position = positions[1]
    separator_before = state.positions_for_region(Region.LOCKED_NEWLINE)[0]
    apply_selected_action(state, position, spec.expand_id, config)
    assert state.positions_for_region(Region.HARD_SLOT_0) == [positions[0], position, position + 1, positions[2] + 1, positions[3] + 1]
    assert state.positions_for_region(Region.LOCKED_NEWLINE)[0] == separator_before + 1
    assert int(state.input_ids[position]) == spec.mask_id
    assert int(state.input_ids[position + 1]) == spec.mask_id


def test_expand_immediately_before_separator_does_not_cross_it(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    position = state.positions_for_region(Region.HARD_SLOT_0)[-1]
    apply_selected_action(state, position, spec.expand_id, config)
    new_slot = state.positions_for_region(Region.HARD_SLOT_0)
    separator = state.positions_for_region(Region.LOCKED_NEWLINE)[0]
    assert new_slot[-1] + 1 == separator
    assert int(state.input_ids[separator]) == 4


def test_expand_shifts_future_region_without_mutating_it(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    future_before = state.positions_for_region(Region.HARD_SLOT_1)
    apply_selected_action(state, state.positions_for_region(Region.HARD_SLOT_0)[0], spec.expand_id, config)
    future_after = state.positions_for_region(Region.HARD_SLOT_1)
    assert future_after == [position + 1 for position in future_before]
    assert all(int(state.input_ids[p]) == spec.mask_id for p in future_after)


def test_delete_in_middle_only_removes_selected_position(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    state.replace_token(positions[0], 0)
    state.replace_token(positions[1], 1)
    apply_selected_action(state, positions[2], spec.eos_id, config)
    assert state.region_length(Region.HARD_SLOT_0) == 3
    assert [int(state.input_ids[p]) for p in state.positions_for_region(Region.HARD_SLOT_0)[:2]] == [0, 1]


def test_delete_last_mask_preserves_separator(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    for position in state.positions_for_region(Region.HARD_SLOT_0)[:-1]:
        state.replace_token(position, 0)
    last = state.positions_for_region(Region.HARD_SLOT_0)[-1]
    apply_selected_action(state, last, spec.eos_id, config)
    separator = state.positions_for_region(Region.LOCKED_NEWLINE)[0]
    assert int(state.input_ids[separator]) == 4
    assert not state.unresolved_positions(Region.HARD_SLOT_0)


def test_delete_entire_slot_keeps_locked_newline(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    while state.unresolved_positions(Region.HARD_SLOT_0):
        apply_selected_action(
            state,
            state.unresolved_positions(Region.HARD_SLOT_0)[0],
            spec.eos_id,
            config,
        )
    assert state.region_length(Region.HARD_SLOT_0) == 0
    separator = state.positions_for_region(Region.LOCKED_NEWLINE)[0]
    assert int(state.input_ids[separator]) == 4


def test_delete_does_not_mutate_future_region(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    apply_selected_action(state, state.positions_for_region(Region.HARD_SLOT_0)[0], spec.eos_id, config)
    assert all(
        int(state.input_ids[p]) == spec.mask_id
        for p in state.positions_for_region(Region.HARD_SLOT_1)
    )


def test_pad_mask_id_is_never_eligible(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    pad_position = state.real_length
    state.input_ids[pad_position] = spec.mask_id
    state.region_id[pad_position] = Region.PAD
    state.attention_mask[pad_position] = False
    assert pad_position not in eligible_positions(state, Method.V2_HARD)


@pytest.mark.parametrize("method", [Method.V2_HARD, Method.V2_OPENTAIL, Method.JOINT_OPENTAIL])
def test_every_hard_slot_masks_all_newline_tokens(method, spec, config):
    state = make_state(method, spec, config)
    logits = logits_for_state(state)
    active_logits, positions = constrained_active_logits(state, method, logits, config)
    for row, position in enumerate(positions):
        region = Region(int(state.region_id[position]))
        if region in {Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2}:
            assert torch.isneginf(active_logits[row, list(spec.newline_token_ids)]).all()


def test_opentail_allows_newline_tokens(spec, config):
    state = make_state(Method.V2_OPENTAIL, spec, config)
    resolve_region(state, Region.HARD_SLOT_0)
    resolve_region(state, Region.HARD_SLOT_1)
    active_logits, positions = constrained_active_logits(
        state, Method.V2_OPENTAIL, logits_for_state(state), config
    )
    assert {Region(int(state.region_id[p])) for p in positions} == {Region.OPEN_TAIL}
    assert not torch.isneginf(active_logits[:, list(spec.newline_token_ids)]).any()


def test_hard_slot_32_token_cap_blocks_expand(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    while state.region_length(Region.HARD_SLOT_0) < 32:
        state.expand_at(state.positions_for_region(Region.HARD_SLOT_0)[0])
    active_logits, positions = constrained_active_logits(
        state, Method.V2_HARD, logits_for_state(state), config
    )
    row = positions.index(state.positions_for_region(Region.HARD_SLOT_0)[0])
    assert torch.isneginf(active_logits[row, spec.expand_id])


def test_global_64_token_cap_blocks_expand(spec, config):
    capped = SlotGeneratorConfig(**{**config.__dict__, "max_global_middle_tokens": 15})
    state = make_state(Method.V2_HARD, spec, capped)
    active_logits, _ = constrained_active_logits(
        state, Method.V2_HARD, logits_for_state(state), capped
    )
    assert torch.isneginf(active_logits[:, spec.expand_id]).all()


class AlwaysMaskModel(torch.nn.Module):
    def __init__(self, spec: TokenizerSpec):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.spec = spec

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 16), -9.0, device=input_ids.device)
        logits[..., self.spec.mask_id] = 9.0
        return SimpleNamespace(logits=logits)


def test_forward_cap_with_unresolved_masks_is_protocol_error(tokenizer, spec, config):
    tiny = SlotGeneratorConfig(**{**config.__dict__, "max_total_forwards": 2})
    result = run_slot_generation(
        model=AlwaysMaskModel(spec),
        tokenizer=tokenizer,
        tokenizer_spec=spec,
        method=Method.V2_HARD,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[1, spec.eos_id],
        config=tiny,
        save_trace=True,
    )
    assert result["status"] == "protocol_error"
    assert result["unresolved_mask_count"] > 0
    assert "forward_cap_with_unresolved_masks" in result["protocol_flags"]


def test_prefix_suffix_and_separators_remain_exact_after_dynamic_updates(spec, config):
    state = make_state(Method.V2_HARD, spec, config)
    apply_selected_action(state, state.positions_for_region(Region.HARD_SLOT_0)[1], spec.expand_id, config)
    apply_selected_action(state, state.positions_for_region(Region.HARD_SLOT_0)[0], spec.eos_id, config)
    validate_state(state, Method.V2_HARD)
    assert state.tokens_for_region(Region.CONTEXT_PREFIX) == [spec.bos_id, 0, 1]
    assert state.tokens_for_region(Region.CONTEXT_SUFFIX) == [2, spec.eos_id]
    assert state.tokens_for_region(Region.LOCKED_NEWLINE) == [4, 4, 4]


def test_dynamic_output_extraction_uses_regions_not_prefix_length(spec, config, tokenizer):
    state = make_state(Method.V2_OPENTAIL, spec, config)
    for region, token in ((Region.HARD_SLOT_0, 0), (Region.HARD_SLOT_1, 1)):
        positions = state.positions_for_region(region)
        state.replace_token(positions[0], token)
        for position in reversed(positions[1:]):
            state.delete_at(position)
    tail_positions = state.positions_for_region(Region.OPEN_TAIL)
    for position in reversed(tail_positions[1:]):
        state.delete_at(position)
    state.active_region = Region.OPEN_TAIL
    apply_selected_action(state, state.positions_for_region(Region.OPEN_TAIL)[0], spec.expand_id, config)
    tail_positions = state.positions_for_region(Region.OPEN_TAIL)
    state.replace_token(tail_positions[0], 2)
    state.replace_token(tail_positions[1], 0)
    extracted = extract_completion(state, Method.V2_OPENTAIL, tokenizer)
    assert extracted["region_text"]["HARD_SLOT_0"] == "a"
    assert extracted["region_text"]["HARD_SLOT_1"] == "b"
    assert extracted["region_text"]["OPEN_TAIL"] == "ca"
    assert extracted["completion"] == "a\nb\nca"


class DeterministicTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 16), -10.0, device=input_ids.device)
        logits[..., 0] = 10.0
        return SimpleNamespace(logits=logits)


def test_same_task_method_and_config_are_deterministic(tokenizer, spec, config):
    kwargs = dict(
        model=DeterministicTokenModel(),
        tokenizer=tokenizer,
        tokenizer_spec=spec,
        method=Method.V2_OPENTAIL,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[1, spec.eos_id],
        config=config,
        save_trace=True,
    )
    first = run_slot_generation(**kwargs)
    second = run_slot_generation(**kwargs)
    comparable = [
        "status",
        "completion",
        "completion_token_ids",
        "regions",
        "total_forwards",
        "token_forwards",
        "protocol_flags",
    ]
    assert {key: first[key] for key in comparable} == {
        key: second[key] for key in comparable
    }


def test_source_contains_no_first_line_truncation_or_posthoc_repair():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "repro_scripts/dreamon_slot_generator.py",
        root / "repro_scripts/run_dreamon_progressive_v2.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = [
        ".partition(\"\\n\")",
        ".partition('\\n')",
        "splitlines()[0]",
        "discarded_after_first_line",
        "accept_first_physical_line",
    ]
    assert not any(pattern in source for pattern in forbidden)
