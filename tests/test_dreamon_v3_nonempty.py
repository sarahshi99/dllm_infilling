from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from repro_scripts.dreamon_slot_generator import (
    Method,
    ProtocolError,
    Region,
    SelectedUpdate,
    SlotGeneratorConfig,
    TokenizerSpec,
    action_completes_blank_hard_slot,
    apply_selected_action,
    build_canvas,
    clone_canvas_state,
    run_slot_generation,
    scan_newline_token_metadata,
    select_nonempty_guarded_update,
)


class GuardTokenizer:
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 11
    mask_token_id = 12

    _pieces = {
        0: "a",
        1: "b",
        2: "c",
        3: " ",
        4: "\t",
        5: "\n",
        6: ":\n",
        7: "x",
        8: "y",
        9: "z",
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
        encodings = {"": [], "\n": [5], ":": [14]}
        if text not in encodings:
            raise AssertionError(f"Unexpected fake encode input: {text!r}")
        return encodings[text]


@pytest.fixture()
def tokenizer():
    return GuardTokenizer()


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
        literal_newline_ids=(5,),
    )


@pytest.fixture()
def config():
    return SlotGeneratorConfig(
        max_context_tokens=96,
        initial_expand_budget=64,
        nonempty_guard=True,
    )


def make_state(spec, config):
    return build_canvas(
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        spec,
        config,
        torch.device("cpu"),
    )


def keep_only_first_mask(state, config, tokenizer):
    positions = state.unresolved_positions(Region.HARD_SLOT_0)
    apply_selected_action(
        state, positions[1], state.tokenizer_spec.eos_id, config, tokenizer=tokenizer
    )
    assert len(state.unresolved_positions(Region.HARD_SLOT_0)) == 1


def selected(state, position, token_id):
    return SelectedUpdate(
        position=position,
        region=Region(int(state.region_id[position])),
        proposal_token_id=token_id,
        unconstrained_proposal_token_id=token_id,
        confidence=0.0,
    )


def simulate(state, choice, config, tokenizer):
    candidate = clone_canvas_state(state)
    result = apply_selected_action(
        candidate,
        choice.position,
        choice.proposal_token_id,
        config,
        tokenizer=tokenizer,
    )
    return candidate, result


@pytest.mark.parametrize("proposal", [5, 11])
def test_pure_newline_or_local_eos_that_completes_empty_slot_is_rejected(
    proposal, tokenizer, spec, config
):
    state = make_state(spec, config)
    choice = selected(state, state.unresolved_positions(Region.HARD_SLOT_0)[0], proposal)
    candidate, _ = simulate(state, choice, config, tokenizer)
    assert action_completes_blank_hard_slot(candidate, choice.region, tokenizer)


def test_last_whitespace_mask_that_completes_blank_slot_is_rejected(
    tokenizer, spec, config
):
    state = make_state(spec, config)
    positions = state.unresolved_positions(Region.HARD_SLOT_0)
    for position in positions[:-1]:
        state.replace_token(position, 3)
    choice = selected(state, positions[-1], 4)
    candidate, _ = simulate(state, choice, config, tokenizer)
    assert action_completes_blank_hard_slot(candidate, choice.region, tokenizer)


def test_boundary_with_left_unresolved_mask_is_allowed(tokenizer, spec, config):
    state = make_state(spec, config)
    position = state.unresolved_positions(Region.HARD_SLOT_0)[2]
    choice = selected(state, position, 5)
    candidate, _ = simulate(state, choice, config, tokenizer)
    assert not action_completes_blank_hard_slot(candidate, choice.region, tokenizer)


@pytest.mark.parametrize("proposal", [5, 11])
def test_nonempty_resolved_content_allows_boundary_or_eos(
    proposal, tokenizer, spec, config
):
    state = make_state(spec, config)
    positions = state.unresolved_positions(Region.HARD_SLOT_0)
    state.replace_token(positions[0], 0)
    choice = selected(state, positions[1], proposal)
    candidate, _ = simulate(state, choice, config, tokenizer)
    assert not action_completes_blank_hard_slot(candidate, choice.region, tokenizer)


def test_guard_simulation_does_not_mutate_real_canvas_or_budget(
    tokenizer, spec, config
):
    state = make_state(spec, config)
    before_ids = state.input_ids.clone()
    before_regions = state.region_id.clone()
    before_budget = state.remaining_expand_budget
    choice = selected(state, state.unresolved_positions(Region.HARD_SLOT_0)[0], 5)
    simulate(state, choice, config, tokenizer)
    assert torch.equal(state.input_ids, before_ids)
    assert torch.equal(state.region_id, before_regions)
    assert state.remaining_expand_budget == before_budget


def test_guard_reselects_from_same_logits_without_extra_forward(
    tokenizer, spec, config
):
    state = make_state(spec, config)
    keep_only_first_mask(state, config, tokenizer)
    logits = torch.full((state.real_length, 16), -30.0)
    logits[:, 5] = 30.0
    logits[:, 0] = 29.0
    choice, rejections = select_nonempty_guarded_update(
        state,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        logits,
        config,
        tokenizer,
        task_id="guard",
        forward_index=1,
    )
    assert rejections
    assert choice.proposal_token_id == 0
    assert state.remaining_expand_budget == 64


def test_no_valid_candidate_is_explicit_terminal_failure(tokenizer, spec, config):
    state = make_state(spec, config)
    keep_only_first_mask(state, config, tokenizer)
    logits = torch.full((state.real_length, 16), float("-inf"))
    logits[:, 5] = 3.0
    logits[:, 11] = 2.0
    logits[:, 3] = 1.0
    logits[:, 4] = 0.0
    with pytest.raises(ProtocolError, match="nonempty_guard_no_valid_action"):
        select_nonempty_guarded_update(
            state,
            Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
            logits,
            config,
            tokenizer,
            task_id="guard",
            forward_index=1,
        )


class CountingNormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        self.calls += 1
        logits = torch.full((*input_ids.shape, 16), -20.0, device=input_ids.device)
        logits[..., 0] = 20.0
        return SimpleNamespace(logits=logits)


def observable_trace(result):
    return [
        (
            step["selected_position"],
            step["selected_region"],
            step["proposal_token_id"],
            step["action"],
        )
        for step in result["step_trace"]
    ]


def test_completed_b_row_has_three_nonempty_slots_and_no_extra_forward(
    tokenizer, spec, config
):
    model = CountingNormalModel()
    result = run_slot_generation(
        model,
        tokenizer,
        spec,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        config,
        save_trace=True,
        task_id="nonempty",
    )
    assert result["status"] == "completed"
    assert result["final_nonempty_invariant_passed"] is True
    assert not any(result["blank_region_flags"].values())
    assert model.calls == result["total_forwards"]


def test_b_without_guard_activation_matches_a(tokenizer, spec):
    model_a = CountingNormalModel()
    model_b = CountingNormalModel()
    a = run_slot_generation(
        model_a,
        tokenizer,
        spec,
        Method.V3_HARD_BUDGETED,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        SlotGeneratorConfig(max_context_tokens=96, initial_expand_budget=64),
        save_trace=True,
        task_id="isolation",
    )
    b = run_slot_generation(
        model_b,
        tokenizer,
        spec,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        SlotGeneratorConfig(
            max_context_tokens=96,
            initial_expand_budget=64,
            nonempty_guard=True,
        ),
        save_trace=True,
        task_id="isolation",
    )
    assert b["nonempty_guard_rejection_count"] == 0
    assert b["completion"] == a["completion"]
    assert observable_trace(b) == observable_trace(a)


def test_a_still_allows_empty_slot(tokenizer, spec):
    config = SlotGeneratorConfig(max_context_tokens=96, initial_expand_budget=64)
    state = build_canvas(
        Method.V3_HARD_BUDGETED,
        [spec.bos_id],
        [spec.eos_id],
        spec,
        config,
        torch.device("cpu"),
    )
    position = state.unresolved_positions(Region.HARD_SLOT_0)[0]
    apply_selected_action(state, position, spec.eos_id, config, tokenizer=tokenizer)
    assert state.region_length(Region.HARD_SLOT_0) == 0
