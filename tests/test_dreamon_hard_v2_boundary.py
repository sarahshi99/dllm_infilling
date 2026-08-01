from __future__ import annotations

from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from repro_scripts.dreamon_slot_generator import (
    ExactTransitionCycleDetector,
    Method,
    ProtocolError,
    Region,
    SlotGeneratorConfig,
    TokenizerSpec,
    advance_active_region,
    apply_selected_action,
    build_canvas,
    extract_completion,
    scan_newline_token_metadata,
    state_hash,
    transition_signature,
    validate_state,
    run_slot_generation,
)


class BoundaryTokenizer:
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
        7: "\n    ",
        8: ":\n        ",
        9: "\r\n",
        10: "<bos>",
        11: "<eos>",
        12: "<mask>",
        13: "<expand>",
        14: ":",
        15: "a\nb\nc",
        16: "ab\n",
        17: "x",
        18: "y",
    }

    def __len__(self) -> int:
        return len(self._pieces)

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        encodings = {
            "": [],
            "\n": [4],
            ":": [14],
            "a": [0],
            "ab": [0, 1],
        }
        if text not in encodings:
            raise AssertionError(f"Unexpected fake encode input: {text!r}")
        return encodings[text]


@pytest.fixture()
def tokenizer():
    return BoundaryTokenizer()


@pytest.fixture()
def spec(tokenizer):
    newline_map, _ = scan_newline_token_metadata(tokenizer)
    return TokenizerSpec(
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id,
        expand_id=13,
        newline_token_ids=frozenset(newline_map),
        newline_token_map=newline_map,
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


def make_state(spec, config):
    return build_canvas(
        method=Method.V2_HARD_V2_BOUNDARY,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[2, spec.eos_id],
        tokenizer_spec=spec,
        config=config,
        device=torch.device("cpu"),
    )


def locked_snapshot(state):
    return {
        "prefix": state.tokens_for_region(Region.CONTEXT_PREFIX),
        "suffix": state.tokens_for_region(Region.CONTEXT_SUFFIX),
        "separators": state.tokens_for_region(Region.LOCKED_NEWLINE),
        "future1": state.tokens_for_region(Region.HARD_SLOT_1),
        "future2": state.tokens_for_region(Region.HARD_SLOT_2),
    }


def test_newline_metadata_normalizes_and_retokenizes_once(tokenizer):
    newline_map, metadata = scan_newline_token_metadata(tokenizer)
    assert set(newline_map) == {4, 5, 6, 7, 8, 9, 15, 16}
    assert newline_map[6].left_text == ":"
    assert newline_map[6].left_token_ids == (14,)
    assert newline_map[8].right_text == "        "
    assert newline_map[9].normalized_text == "\n"
    assert newline_map[9].newline_count == 1
    assert newline_map[15].newline_count == 2
    assert newline_map[16].left_token_ids == (0, 1)
    assert metadata["newline_token_mapping_count"] == 8
    assert len(metadata["newline_token_mapping_sha256"]) == 64
    assert metadata["preprocessing_seconds"] >= 0


@pytest.mark.parametrize(
    ("proposal", "expected_ids", "expected_right"),
    [
        (6, [14], ""),
        (4, [], ""),
        (7, [], "    "),
        (8, [14], "        "),
    ],
)
def test_line_boundary_keeps_left_and_discards_internal_right(
    proposal, expected_ids, expected_right, tokenizer, spec, config
):
    state = make_state(spec, config)
    before = locked_snapshot(state)
    position = state.positions_for_region(Region.HARD_SLOT_0)[0]
    result = apply_selected_action(
        state, position, proposal, config, tokenizer=tokenizer
    )
    assert result.action == "line_boundary"
    assert state.tokens_for_region(Region.HARD_SLOT_0) == expected_ids
    assert result.details["right_text"] == expected_right
    assert result.details["discarded_internal_right_text"] == expected_right
    assert locked_snapshot(state) == before
    validate_state(state, Method.V2_HARD_V2_BOUNDARY)


def test_crlf_counts_as_one_boundary(tokenizer):
    newline_map, _ = scan_newline_token_metadata(tokenizer)
    assert newline_map[9].decoded_text == "\r\n"
    assert newline_map[9].normalized_text == "\n"
    assert newline_map[9].newline_count == 1


def test_multiple_newlines_use_only_first_boundary(tokenizer, spec, config):
    state = make_state(spec, config)
    position = state.positions_for_region(Region.HARD_SLOT_0)[0]
    result = apply_selected_action(
        state, position, 15, config, tokenizer=tokenizer
    )
    assert state.tokens_for_region(Region.HARD_SLOT_0) == [0]
    assert result.details["left_text"] == "a"
    assert result.details["right_text"] == "b\nc"
    assert result.details["newline_count"] == 2


def test_boundary_discards_right_masks(tokenizer, spec, config):
    state = make_state(spec, config)
    position = state.positions_for_region(Region.HARD_SLOT_0)[1]
    result = apply_selected_action(
        state, position, 6, config, tokenizer=tokenizer
    )
    assert state.tokens_for_region(Region.HARD_SLOT_0) == [spec.mask_id, 14]
    assert result.details["discarded_masks_after_boundary"] == 2
    assert result.details["discarded_resolved_token_ids_after_boundary"] == []


def test_boundary_discards_right_resolved_tokens_as_contiguous_segments(
    tokenizer, spec, config
):
    state = make_state(spec, config)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    state.replace_token(positions[2], 0)
    state.replace_token(positions[3], 1)
    result = apply_selected_action(
        state, positions[1], 6, config, tokenizer=tokenizer
    )
    assert state.tokens_for_region(Region.HARD_SLOT_0) == [spec.mask_id, 14]
    assert result.details["discarded_resolved_token_ids_after_boundary"] == [0, 1]
    assert result.details["discarded_resolved_segments_after_boundary"] == [
        {"token_ids": [0, 1], "text": "ab"}
    ]
    assert result.details["discarded_resolved_tokens_after_boundary"] == 2


def test_boundary_with_left_mask_keeps_current_slot_active(tokenizer, spec, config):
    state = make_state(spec, config)
    position = state.positions_for_region(Region.HARD_SLOT_0)[1]
    result = apply_selected_action(
        state, position, 6, config, tokenizer=tokenizer
    )
    assert result.details["boundary_event_with_left_masks_remaining"] is True
    assert advance_active_region(state, Method.V2_HARD_V2_BOUNDARY) == Region.HARD_SLOT_0


@pytest.mark.parametrize(
    ("proposal", "expected_count"), [(4, 0), (6, 1), (16, 2)]
)
def test_boundary_left_retokenizes_to_zero_one_or_multiple_tokens(
    proposal, expected_count, tokenizer, spec, config
):
    state = make_state(spec, config)
    result = apply_selected_action(
        state,
        state.positions_for_region(Region.HARD_SLOT_0)[-1],
        proposal,
        config,
        tokenizer=tokenizer,
    )
    assert result.details["boundary_retokenized_left_token_count"] == expected_count


def test_boundary_retokenization_checks_slot_cap_atomically(tokenizer, spec, config):
    state = make_state(spec, config)
    while state.region_length(Region.HARD_SLOT_0) < 32:
        state.expand_at(state.positions_for_region(Region.HARD_SLOT_0)[0])
    before_hash = state_hash(state)
    with pytest.raises(ProtocolError, match="boundary_slot_cap_exceeded"):
        apply_selected_action(
            state,
            state.positions_for_region(Region.HARD_SLOT_0)[-1],
            16,
            config,
            tokenizer=tokenizer,
        )
    assert state_hash(state) == before_hash


def test_boundary_retokenization_checks_global_cap_atomically(tokenizer, spec, config):
    capped = SlotGeneratorConfig(**{**config.__dict__, "max_global_middle_tokens": 15})
    state = make_state(spec, capped)
    before_hash = state_hash(state)
    with pytest.raises(ProtocolError, match="boundary_global_cap_exceeded"):
        apply_selected_action(
            state,
            state.positions_for_region(Region.HARD_SLOT_0)[-1],
            16,
            capped,
            tokenizer=tokenizer,
        )
    assert state_hash(state) == before_hash


def test_boundary_retokenization_checks_context_cap_atomically(tokenizer, spec, config):
    state = make_state(spec, config)
    exact = SlotGeneratorConfig(**{**config.__dict__, "max_context_tokens": state.real_length})
    before_hash = state_hash(state)
    with pytest.raises(ProtocolError, match="boundary_context_cap_exceeded"):
        apply_selected_action(
            state,
            state.positions_for_region(Region.HARD_SLOT_0)[-1],
            16,
            exact,
            tokenizer=tokenizer,
        )
    assert state_hash(state) == before_hash


def test_boundary_preserves_prefix_suffix_separator_and_future(tokenizer, spec, config):
    state = make_state(spec, config)
    before = locked_snapshot(state)
    apply_selected_action(
        state,
        state.positions_for_region(Region.HARD_SLOT_0)[1],
        6,
        config,
        tokenizer=tokenizer,
    )
    assert locked_snapshot(state) == before


def test_region_local_eos_deletes_selected_and_right_unresolved_only(
    tokenizer, spec, config
):
    state = make_state(spec, config)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    state.replace_token(positions[0], 0)
    state.replace_token(positions[2], 1)
    before = locked_snapshot(state)
    result = apply_selected_action(
        state, positions[1], spec.eos_id, config, tokenizer=tokenizer
    )
    assert result.action == "region_local_eos"
    assert state.tokens_for_region(Region.HARD_SLOT_0) == [0, 1]
    assert result.details["deleted_masks"] == 2
    assert result.details["deleted_positions"] == 2
    assert locked_snapshot(state) == before


def test_region_local_eos_does_not_delete_right_resolved_token(tokenizer, spec, config):
    state = make_state(spec, config)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    state.replace_token(positions[2], 17)
    apply_selected_action(
        state, positions[1], spec.eos_id, config, tokenizer=tokenizer
    )
    assert 17 in state.tokens_for_region(Region.HARD_SLOT_0)


def test_region_local_eos_never_crosses_separator_or_future(tokenizer, spec, config):
    state = make_state(spec, config)
    before = locked_snapshot(state)
    apply_selected_action(
        state,
        state.positions_for_region(Region.HARD_SLOT_0)[0],
        spec.eos_id,
        config,
        tokenizer=tokenizer,
    )
    assert locked_snapshot(state) == before
    assert not state.unresolved_positions(Region.HARD_SLOT_0)


@pytest.mark.parametrize(("a", "b"), [(31, 32), (8, 9)])
def test_repeated_transition_is_exact_cycle(a, b):
    detector = ExactTransitionCycleDetector()
    first = transition_signature("state-a", 3, Region.HARD_SLOT_0, 13, "expand", "state-b")
    back = transition_signature("state-b", 3, Region.HARD_SLOT_0, 11, "region_local_eos", "state-a")
    assert detector.observe(first, 1, {"HARD_SLOT_0": a}) is None
    assert detector.observe(back, 2, {"HARD_SLOT_0": b}) is None
    cycle = detector.observe(first, 3, {"HARD_SLOT_0": a})
    assert cycle is not None
    assert cycle["cycle_length"] == 2
    assert cycle["first_forward_index"] == 1
    assert cycle["repeat_forward_index"] == 3


def test_same_length_different_token_state_is_not_cycle(spec, config):
    first = make_state(spec, config)
    second = make_state(spec, config)
    first.replace_token(first.positions_for_region(Region.HARD_SLOT_0)[0], 0)
    second.replace_token(second.positions_for_region(Region.HARD_SLOT_0)[0], 1)
    assert first.region_length(Region.HARD_SLOT_0) == second.region_length(Region.HARD_SLOT_0)
    assert state_hash(first) != state_hash(second)


def test_new_state_correction_path_does_not_trigger_cycle():
    detector = ExactTransitionCycleDetector()
    signatures = [
        transition_signature(f"s{i}", i, Region.HARD_SLOT_0, i, "normal", f"s{i+1}")
        for i in range(4)
    ]
    assert all(
        detector.observe(signature, index, {"HARD_SLOT_0": 4}) is None
        for index, signature in enumerate(signatures, start=1)
    )


def test_completion_contains_no_mask_expand_or_pad(tokenizer, spec, config):
    state = make_state(spec, config)
    for region in (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2):
        positions = state.positions_for_region(region)
        state.replace_token(positions[0], 0)
        while state.unresolved_positions(region):
            apply_selected_action(
                state,
                state.unresolved_positions(region)[0],
                spec.eos_id,
                config,
                tokenizer=tokenizer,
            )
        advance_active_region(state, Method.V2_HARD_V2_BOUNDARY)
    extracted = extract_completion(state, Method.V2_HARD_V2_BOUNDARY, tokenizer)
    assert spec.mask_id not in extracted["completion_token_ids"]
    assert spec.expand_id not in extracted["completion_token_ids"]
    assert spec.pad_id not in extracted["completion_token_ids"]


def test_v2_generation_source_cannot_access_reference_or_tests():
    root = Path(__file__).resolve().parents[1]
    source = (root / "repro_scripts/run_dreamon_progressive_v2.py").read_text(
        encoding="utf-8"
    )
    generation = source.split("def run_generation", 1)[1].split(
        "def score_completion", 1
    )[0]
    forbidden = [
        "canonical_solution",
        "reference_middle",
        "reference_lines",
        "check_correctness",
        "test_result",
    ]
    assert not any(item in generation for item in forbidden)


def test_scoring_builds_prompt_completion_suffix_program():
    root = Path(__file__).resolve().parents[1]
    source = (root / "repro_scripts/run_dreamon_progressive_v2.py").read_text(
        encoding="utf-8"
    )
    scoring = source.split("def score_completion", 1)[1].split(
        "def load_scoring_problems", 1
    )[0]
    assert 'problem["prompt"] + completion + problem["suffix"]' in scoring


def test_new_hard_path_has_no_blanket_newline_logit_ban():
    root = Path(__file__).resolve().parents[1]
    source = (root / "repro_scripts/dreamon_slot_generator.py").read_text(
        encoding="utf-8"
    )
    constrained = source.split("def constrained_active_logits", 1)[1].split(
        "def select_update", 1
    )[0]
    assert "newline_token_ids" not in constrained
    assert "newline_token_map" not in constrained


class AlwaysMaskModel(torch.nn.Module):
    def __init__(self, mask_id: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.mask_id = mask_id

    def forward(self, input_ids, **_kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 19), -10.0, device=input_ids.device)
        logits[..., self.mask_id] = 10.0
        return SimpleNamespace(logits=logits)


class AlwaysNormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, **_kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 19), -10.0, device=input_ids.device)
        logits[..., 0] = 10.0
        return SimpleNamespace(logits=logits)


def test_generation_stops_on_second_identical_transition(tokenizer, spec, config):
    result = run_slot_generation(
        model=AlwaysMaskModel(spec.mask_id),
        tokenizer=tokenizer,
        tokenizer_spec=spec,
        method=Method.V2_HARD_V2_BOUNDARY,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[1, spec.eos_id],
        config=config,
        save_trace=True,
        task_id="cycle-task",
    )
    assert result["status"] == "protocol_error"
    assert result["protocol_flags"] == ["exact_deterministic_cycle"]
    assert result["total_forwards"] == 2
    assert result["exact_cycle_events"][0]["cycle_length"] == 1


def test_boundary_v2_generation_is_deterministic(tokenizer, spec, config):
    kwargs = dict(
        model=AlwaysNormalModel(),
        tokenizer=tokenizer,
        tokenizer_spec=spec,
        method=Method.V2_HARD_V2_BOUNDARY,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[1, spec.eos_id],
        config=config,
        save_trace=True,
        task_id="deterministic-task",
    )
    first = run_slot_generation(**kwargs)
    second = run_slot_generation(**kwargs)
    fields = [
        "status",
        "completion",
        "completion_token_ids",
        "regions",
        "total_forwards",
        "token_forwards",
        "protocol_flags",
        "newline_boundary_event_details",
        "region_local_eos_event_details",
        "exact_cycle_events",
    ]
    assert {field: first[field] for field in fields} == {
        field: second[field] for field in fields
    }
