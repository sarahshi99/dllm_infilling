from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from repro_scripts.dreamon_slot_generator import (
    BudgetedTransitionCycleDetector,
    Method,
    ProtocolError,
    Region,
    SlotGeneratorConfig,
    TokenizerSpec,
    advance_active_region,
    apply_selected_action,
    build_canvas,
    canvas_only_state_hash,
    constrained_active_logits,
    full_state_hash,
    run_slot_generation,
    scan_newline_token_metadata,
    select_update,
    transition_signature,
    validate_state,
)


class BudgetTokenizer:
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
        5: ":\n",
        6: "x",
        7: "y",
        8: "z",
        9: "q",
        10: "<bos>",
        11: "<eos>",
        12: "<mask>",
        13: "<expand>",
        14: ":",
        15: "s",
    }

    def __len__(self) -> int:
        return len(self._pieces)

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        encodings = {"": [], "\n": [4], ":": [14]}
        if text not in encodings:
            raise AssertionError(f"Unexpected fake encode input: {text!r}")
        return encodings[text]


@pytest.fixture()
def tokenizer():
    return BudgetTokenizer()


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
        max_context_tokens=96,
        max_total_forwards=256,
        max_global_middle_tokens=64,
        max_hard_slot_tokens=32,
        temperature=0.0,
        top_p=0.9,
        initial_expand_budget=64,
    )


def make_state(spec, config):
    return build_canvas(
        method=Method.V3_HARD_BUDGETED,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[2, spec.eos_id],
        tokenizer_spec=spec,
        config=config,
        device=torch.device("cpu"),
    )


def first_mask(state, region=Region.HARD_SLOT_0):
    return state.unresolved_positions(region)[0]


def test_initial_budget_is_exactly_64(spec, config):
    state = make_state(spec, config)
    assert state.initial_expand_budget == 64
    assert state.remaining_expand_budget == 64
    assert state.successful_expand_count == 0


def test_successful_expand_consumes_one_budget(spec, config):
    state = make_state(spec, config)
    result = apply_selected_action(state, first_mask(state), spec.expand_id, config)
    assert result.action == "expand"
    assert state.remaining_expand_budget == 63
    assert state.successful_expand_count == 1
    validate_state(state, Method.V3_HARD_BUDGETED)


@pytest.mark.parametrize("proposal", [0, 5, 11])
def test_non_expand_actions_do_not_change_budget(proposal, tokenizer, spec, config):
    state = make_state(spec, config)
    before = state.remaining_expand_budget
    apply_selected_action(
        state, first_mask(state), proposal, config, tokenizer=tokenizer
    )
    assert state.remaining_expand_budget == before


def test_deleting_previously_expanded_positions_does_not_refund_budget(spec, config):
    state = make_state(spec, config)
    position = first_mask(state)
    apply_selected_action(state, position, spec.expand_id, config)
    apply_selected_action(state, position, spec.eos_id, config)
    assert state.remaining_expand_budget == 63
    assert state.successful_expand_count == 1


def test_cap_rejected_expand_does_not_consume_budget(spec, config):
    capped = SlotGeneratorConfig(
        initial_masks_per_region=4,
        max_context_tokens=96,
        max_total_forwards=256,
        max_global_middle_tokens=15,
        max_hard_slot_tokens=32,
        initial_expand_budget=64,
    )
    state = make_state(spec, capped)
    before_ids = state.input_ids.clone()
    with pytest.raises(ProtocolError, match="expand_selected_at_cap"):
        apply_selected_action(state, first_mask(state), spec.expand_id, capped)
    assert torch.equal(state.input_ids, before_ids)
    assert state.remaining_expand_budget == 64


def test_unconstrained_expand_but_constrained_normal_does_not_consume(
    spec, config
):
    state = make_state(spec, config)
    state.remaining_expand_budget = 0
    state.successful_expand_count = 64
    logits = torch.full((state.real_length, 16), -20.0)
    logits[:, spec.expand_id] = 20.0
    logits[:, 0] = 19.0
    selected = select_update(state, Method.V3_HARD_BUDGETED, logits, config)
    assert selected.unconstrained_proposal_token_id == spec.expand_id
    assert selected.proposal_token_id == 0
    apply_selected_action(state, selected.position, selected.proposal_token_id, config)
    assert state.remaining_expand_budget == 0


def test_budget_accumulates_across_slots_without_reset(spec, config):
    state = make_state(spec, config)
    apply_selected_action(state, first_mask(state), spec.expand_id, config)
    for position in state.unresolved_positions(Region.HARD_SLOT_0):
        state.replace_token(position, 0)
    advance_active_region(state, Method.V3_HARD_BUDGETED)
    assert state.active_region == Region.HARD_SLOT_1
    assert state.remaining_expand_budget == 63
    apply_selected_action(state, first_mask(state, Region.HARD_SLOT_1), spec.expand_id, config)
    assert state.remaining_expand_budget == 62


def test_sixty_fourth_successful_expand_reaches_zero(spec, config):
    state = make_state(spec, config)
    state.remaining_expand_budget = 1
    state.successful_expand_count = 63
    apply_selected_action(state, first_mask(state), spec.expand_id, config)
    assert state.remaining_expand_budget == 0
    assert state.successful_expand_count == 64
    validate_state(state, Method.V3_HARD_BUDGETED)


def test_budget_zero_masks_expand_before_entropy_and_selection(spec, config):
    state = make_state(spec, config)
    state.remaining_expand_budget = 0
    state.successful_expand_count = 64
    logits = torch.zeros((state.real_length, 16))
    constrained, positions = constrained_active_logits(
        state, Method.V3_HARD_BUDGETED, logits, config
    )
    assert positions
    assert torch.isneginf(constrained[:, spec.expand_id]).all()


def test_same_canvas_different_budget_has_different_full_hash(spec, config):
    state = make_state(spec, config)
    canvas_hash = canvas_only_state_hash(state)
    full_before = full_state_hash(state)
    state.remaining_expand_budget = 63
    state.successful_expand_count = 1
    assert canvas_only_state_hash(state) == canvas_hash
    assert full_state_hash(state) != full_before


def _signature(pre, post, budgeted=False):
    return transition_signature(
        pre, 3, Region.HARD_SLOT_2, 13, "expand", post
    )


def test_canvas_repeat_with_lower_budget_is_draining_not_exact_cycle():
    detector = BudgetedTransitionCycleDetector()
    first = detector.observe(
        full_signature=_signature("full-a-64", "full-b-63"),
        canvas_signature=_signature("canvas-a", "canvas-b"),
        pre_remaining_budget=64,
        post_remaining_budget=63,
        forward_index=8,
        region_lengths={"HARD_SLOT_2": 9},
    )
    second = detector.observe(
        full_signature=_signature("full-a-63", "full-b-62"),
        canvas_signature=_signature("canvas-a", "canvas-b"),
        pre_remaining_budget=63,
        post_remaining_budget=62,
        forward_index=10,
        region_lengths={"HARD_SLOT_2": 9},
    )
    assert first is None
    assert second["classification"] == "budget_draining_loop"
    assert second["terminal"] is False
    assert second["first_pre_remaining_budget"] == 64
    assert second["repeat_pre_remaining_budget"] == 63


def test_same_full_transition_at_same_budget_is_exact_cycle():
    detector = BudgetedTransitionCycleDetector()
    kwargs = dict(
        full_signature=_signature("full-a", "full-b"),
        canvas_signature=_signature("canvas-a", "canvas-b"),
        pre_remaining_budget=0,
        post_remaining_budget=0,
        region_lengths={"HARD_SLOT_2": 8},
    )
    assert detector.observe(forward_index=8, **kwargs) is None
    repeated = detector.observe(forward_index=10, **kwargs)
    assert repeated["classification"] == "exact_deterministic_cycle"
    assert repeated["terminal"] is True


@pytest.mark.parametrize("low,high", [(8, 9), (31, 32)])
def test_birth_death_canvas_loop_drains_budget_before_exact_cycle(low, high):
    detector = BudgetedTransitionCycleDetector()
    for index, budget in enumerate((64, 63, 62), start=1):
        event = detector.observe(
            full_signature=_signature(
                f"full-{low}-{budget}", f"full-{high}-{budget - 1}"
            ),
            canvas_signature=_signature(f"canvas-{low}", f"canvas-{high}"),
            pre_remaining_budget=budget,
            post_remaining_budget=budget - 1,
            forward_index=index * 2 - 1,
            region_lengths={"HARD_SLOT_2": high},
        )
        if index == 1:
            assert event is None
        else:
            assert event["classification"] == "budget_draining_loop"
            assert event["terminal"] is False


def test_budget_conservation_invariant_rejects_corruption(spec, config):
    state = make_state(spec, config)
    state.remaining_expand_budget = 60
    with pytest.raises(ProtocolError, match="expand_budget_conservation_failed"):
        validate_state(state, Method.V3_HARD_BUDGETED)


class AlwaysNormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **_kwargs):
        logits = torch.full((*input_ids.shape, 16), -20.0, device=input_ids.device)
        logits[..., 0] = 20.0
        return SimpleNamespace(logits=logits)


def _observable_trace(result):
    return [
        (
            step["selected_position"],
            step["selected_region"],
            step["proposal_token_id"],
            step["action"],
        )
        for step in result["step_trace"]
    ]


def test_a_matches_v2_when_budget_never_binds(tokenizer, spec, config):
    model = AlwaysNormalModel()
    v2 = run_slot_generation(
        model,
        tokenizer,
        spec,
        Method.V2_HARD_V2_BOUNDARY,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        SlotGeneratorConfig(max_context_tokens=96),
        save_trace=True,
        task_id="isolation",
    )
    v3 = run_slot_generation(
        model,
        tokenizer,
        spec,
        Method.V3_HARD_BUDGETED,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        config,
        save_trace=True,
        task_id="isolation",
    )
    assert v3["expand_budget_consumed"] == 0
    assert v3["completion"] == v2["completion"]
    assert _observable_trace(v3) == _observable_trace(v2)


def test_a_keeps_allow_empty_behavior(tokenizer, spec, config):
    state = make_state(spec, config)
    apply_selected_action(
        state, first_mask(state), spec.eos_id, config, tokenizer=tokenizer
    )
    assert state.region_length(Region.HARD_SLOT_0) == 0
