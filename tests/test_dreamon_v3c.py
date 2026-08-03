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
    apply_pure_newline_intervention,
    apply_selected_action,
    build_canvas,
    canvas_checkpoint_payload,
    canvas_only_state_hash,
    classify_action_token,
    collect_future_slot_diagnostics,
    extract_completion,
    full_state_hash,
    is_strict_pure_newline_blank_trigger,
    restore_canvas_checkpoint,
    run_slot_generation,
    scan_newline_token_metadata,
    select_update,
    select_update_with_pending_veto,
    validate_state,
)


class V3CTokenizer:
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 11
    mask_token_id = 12

    _pieces = {
        0: "x",
        1: "y",
        2: "z",
        3: "    ",
        4: "\n",
        5: "\r\n",
        6: "    \n",
        7: "\n    ",
        8: "x\n",
        9: ":",
        10: "<bos>",
        11: "<eos>",
        12: "<mask>",
        13: "<expand>",
        14: "q",
        15: "s",
    }

    def __len__(self) -> int:
        return len(self._pieces)

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        encodings = {"": [], "\n": [4], "    ": [3], "x": [0], ":": [9]}
        if text not in encodings:
            raise AssertionError(f"Unexpected fake encode input: {text!r}")
        return encodings[text]


@pytest.fixture()
def tokenizer():
    return V3CTokenizer()


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


def config(**overrides):
    values = {
        "max_context_tokens": 96,
        "max_global_middle_tokens": 64,
        "max_hard_slot_tokens": 32,
        "max_total_forwards": 256,
        "initial_expand_budget": 64,
        "pure_newline_guard_budget_per_slot": 1,
        "pure_newline_guard_global_budget": 3,
    }
    values.update(overrides)
    return SlotGeneratorConfig(**values)


def make_state(method, spec, cfg=None):
    return build_canvas(
        method,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        spec,
        cfg or config(),
        torch.device("cpu"),
    )


def choice(state, token_id, offset=0):
    position = state.unresolved_positions(state.active_region)[offset]
    return SelectedUpdate(
        position=position,
        region=state.active_region,
        proposal_token_id=token_id,
        unconstrained_proposal_token_id=token_id,
        confidence=0.0,
    )


@pytest.mark.parametrize("token_id", [4, 5])
def test_exact_pure_newline_at_left_edge_that_would_blank_triggers(
    token_id, tokenizer, spec
):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    assert is_strict_pure_newline_blank_trigger(
        state, choice(state, token_id), config(), tokenizer
    )


@pytest.mark.parametrize("token_id", [6, 7, 8, 11])
def test_whitespace_mixed_newline_and_eos_do_not_trigger(token_id, tokenizer, spec):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    assert not is_strict_pure_newline_blank_trigger(
        state, choice(state, token_id), config(), tokenizer
    )


def test_non_left_edge_or_nonblank_boundary_does_not_trigger(tokenizer, spec):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    assert not is_strict_pure_newline_blank_trigger(
        state, choice(state, 4, offset=1), config(), tokenizer
    )


@pytest.mark.parametrize(
    "method",
    [
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
    ],
)
def test_per_slot_and_global_guard_budget_are_consumed_once(method, tokenizer, spec):
    state = make_state(method, spec)
    selected = choice(state, 4)
    apply_pure_newline_intervention(state, selected, config(), tokenizer)
    assert state.pure_newline_guard_remaining[Region.HARD_SLOT_0.name] == 0
    assert state.pure_newline_guard_global_remaining == 2
    assert not is_strict_pure_newline_blank_trigger(state, selected, config(), tokenizer)


def test_guard_budget_is_global_across_all_three_slots(tokenizer, spec):
    state = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec)
    for region in (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2):
        assert state.active_region == region
        selected = choice(state, 4)
        apply_pure_newline_intervention(state, selected, config(), tokenizer)
        for position in state.unresolved_positions(region):
            state.replace_token(position, 0)
        from repro_scripts.dreamon_slot_generator import advance_active_region

        advance_active_region(state, state.method)
    assert state.pure_newline_guard_global_remaining == 0
    assert all(value == 0 for value in state.pure_newline_guard_remaining.values())


def test_c0_veto_changes_only_guard_state_and_sets_exact_pending_signature(
    tokenizer, spec
):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    selected = choice(state, 4)
    before_canvas = canvas_only_state_hash(state)
    before_ids = state.input_ids.clone()
    before_regions = state.region_id.clone()
    result = apply_pure_newline_intervention(state, selected, config(), tokenizer)
    assert result.action == "pure_newline_veto"
    assert canvas_only_state_hash(state) == before_canvas
    assert torch.equal(state.input_ids, before_ids)
    assert torch.equal(state.region_id, before_regions)
    assert state.pending_pure_newline_veto == {
        "pre_canvas_hash": before_canvas,
        "selected_position": selected.position,
        "proposal_token_id": 4,
        "action": "line_boundary",
    }


def test_c0_pending_ban_masks_only_exact_position_and_token(tokenizer, spec):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    selected = choice(state, 4)
    apply_pure_newline_intervention(state, selected, config(), tokenizer)
    logits = torch.full((state.real_length, 16), -20.0)
    active = state.unresolved_positions(state.active_region)
    logits[active[0], 4] = 20.0
    logits[active[0], 0] = 19.0
    logits[active[1], 4] = 20.0
    logits[active[1], 0] = -20.0
    reselected, used_pending, fallback = select_update_with_pending_veto(
        state, state.method, logits, config()
    )
    assert used_pending is True
    assert fallback is False
    assert reselected.position == active[1]
    assert reselected.proposal_token_id == 4
    assert state.pending_pure_newline_veto is None


def test_c0_no_finite_alternative_clears_pending_and_returns_a_fallback(
    tokenizer, spec
):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    original = choice(state, 4)
    apply_pure_newline_intervention(state, original, config(), tokenizer)
    logits = torch.full((state.real_length, 16), float("-inf"))
    logits[original.position, 4] = 1.0
    reselected, used_pending, fallback = select_update_with_pending_veto(
        state, state.method, logits, config()
    )
    assert used_pending is True
    assert fallback is True
    assert reselected.position == original.position
    assert reselected.proposal_token_id == original.proposal_token_id
    assert state.pending_pure_newline_veto is None


def test_c_inserts_one_locked_newline_and_preserves_canvas_content(tokenizer, spec):
    state = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec)
    positions = state.positions_for_region(Region.HARD_SLOT_0)
    state.replace_token(positions[2], 0)
    selected = choice(state, 4)
    before_slot = state.tokens_for_region(Region.HARD_SLOT_0)
    before_future = state.tokens_for_region(Region.HARD_SLOT_1)
    before_prefix = state.tokens_for_region(Region.CONTEXT_PREFIX)
    before_suffix = state.tokens_for_region(Region.CONTEXT_SUFFIX)
    before_fixed = state.tokens_for_region(Region.LOCKED_NEWLINE)
    before_budget = state.remaining_expand_budget
    result = apply_pure_newline_intervention(state, selected, config(), tokenizer)
    assert result.action == "insert_locked_blank_newline"
    assert state.tokens_for_region(Region.INSERTED_BLANK_NEWLINE) == [4]
    assert state.tokens_for_region(Region.HARD_SLOT_0) == before_slot
    assert state.tokens_for_region(Region.HARD_SLOT_1) == before_future
    assert state.tokens_for_region(Region.CONTEXT_PREFIX) == before_prefix
    assert state.tokens_for_region(Region.CONTEXT_SUFFIX) == before_suffix
    assert state.tokens_for_region(Region.LOCKED_NEWLINE) == before_fixed
    assert state.active_region == Region.HARD_SLOT_0
    assert state.remaining_expand_budget == before_budget
    validate_state(state, state.method)


def test_inserted_newline_is_not_selectable_or_deletable_and_enters_completion(
    tokenizer, spec
):
    state = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec)
    apply_pure_newline_intervention(state, choice(state, 4), config(), tokenizer)
    inserted = state.positions_for_region(Region.INSERTED_BLANK_NEWLINE)[0]
    with pytest.raises(ProtocolError, match="selected_region_is_not_generation_eligible"):
        apply_selected_action(state, inserted, spec.eos_id, config(), tokenizer=tokenizer)
    for region in (Region.HARD_SLOT_0, Region.HARD_SLOT_1, Region.HARD_SLOT_2):
        for position in state.unresolved_positions(region):
            state.replace_token(position, 0)
        from repro_scripts.dreamon_slot_generator import advance_active_region

        advance_active_region(state, state.method)
    completion = extract_completion(state, state.method, tokenizer)
    assert completion["completion"].startswith("\n")
    assert completion["completion_physical_line_count"] == 4


def test_insert_counts_global_and_context_but_not_slot_or_expand_budget(tokenizer, spec):
    state = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec)
    before = (
        state.real_length,
        state.global_middle_length(),
        state.region_length(Region.HARD_SLOT_0),
        state.remaining_expand_budget,
    )
    apply_pure_newline_intervention(state, choice(state, 4), config(), tokenizer)
    after = (
        state.real_length,
        state.global_middle_length(),
        state.region_length(Region.HARD_SLOT_0),
        state.remaining_expand_budget,
    )
    assert after == (before[0] + 1, before[1] + 1, before[2], before[3])


@pytest.mark.parametrize("cap_kind", ["global", "context"])
def test_insert_cap_fallback_is_atomic_and_uses_a_boundary(cap_kind, tokenizer, spec):
    initial_middle = 15
    cfg = config(
        max_global_middle_tokens=initial_middle if cap_kind == "global" else 64,
        max_context_tokens=19 if cap_kind == "context" else 96,
    )
    state = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec, cfg)
    selected = choice(state, 4)
    fixed_before = state.tokens_for_region(Region.LOCKED_NEWLINE)
    result = apply_pure_newline_intervention(state, selected, cfg, tokenizer)
    assert result.action == "line_boundary"
    assert result.details["blankline_insert_cap_fallback"] is True
    assert state.region_length(Region.HARD_SLOT_0) == 0
    assert state.tokens_for_region(Region.INSERTED_BLANK_NEWLINE) == []
    assert state.tokens_for_region(Region.LOCKED_NEWLINE) == fixed_before


def test_checkpoint_round_trip_preserves_guard_insertions_and_pending(tokenizer, spec):
    c = make_state(Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE, spec)
    apply_pure_newline_intervention(c, choice(c, 4), config(), tokenizer)
    c_payload = canvas_checkpoint_payload(c)
    c_restored = restore_canvas_checkpoint(c_payload, spec, torch.device("cpu"))
    assert canvas_checkpoint_payload(c_restored) == c_payload

    c0 = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    apply_pure_newline_intervention(c0, choice(c0, 4), config(), tokenizer)
    c0_payload = canvas_checkpoint_payload(c0)
    c0_restored = restore_canvas_checkpoint(c0_payload, spec, torch.device("cpu"))
    assert canvas_checkpoint_payload(c0_restored) == c0_payload


def test_full_hash_includes_guard_state_while_canvas_hash_does_not(tokenizer, spec):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    canvas_before = canvas_only_state_hash(state)
    full_before = full_state_hash(state)
    apply_pure_newline_intervention(state, choice(state, 4), config(), tokenizer)
    assert canvas_only_state_hash(state) == canvas_before
    assert full_state_hash(state) != full_before


def test_action_classification_distinguishes_required_classes(tokenizer, spec):
    assert classify_action_token(0, spec) == "normal"
    assert classify_action_token(spec.expand_id, spec) == "expand"
    assert classify_action_token(spec.eos_id, spec) == "EOS"
    assert classify_action_token(4, spec) == "pure_newline"
    assert classify_action_token(7, spec) == "other_newline"
    assert classify_action_token(spec.mask_id, spec) == "special/sentinel"


def test_future_slot_diagnostic_is_read_only_and_reports_normal_advantage(
    tokenizer, spec
):
    state = make_state(Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO, spec)
    selected = choice(state, 4)
    logits = torch.full((state.real_length, 16), -10.0)
    logits[selected.position, 4] = 4.0
    logits[selected.position, 0] = 3.0
    future = state.unresolved_positions(Region.HARD_SLOT_1)[0]
    logits[future, 0] = 12.0
    before_hash = full_state_hash(state)
    before_selection = select_update(state, state.method, logits, config())
    diagnostic = collect_future_slot_diagnostics(
        state, state.method, logits, config(), tokenizer, selected
    )
    after_selection = select_update(state, state.method, logits, config())
    assert full_state_hash(state) == before_hash
    assert before_selection == after_selection
    assert diagnostic["future_has_lower_entropy_normal_candidate"] is True
    assert diagnostic["future_has_higher_top1_probability_normal_candidate"] is True
    assert diagnostic["regions"][Region.HARD_SLOT_1.name]["eligible"] is False
    assert diagnostic["regions"][Region.HARD_SLOT_0.name]["eligible"] is True


class SequenceModel(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.tokens = list(tokens)
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        token = self.tokens[min(self.calls, len(self.tokens) - 1)]
        self.calls += 1
        logits = torch.full((*input_ids.shape, 16), -20.0, device=input_ids.device)
        if token in {4, 5}:
            logits[..., 0] = 0.0
            logits[..., 1] = 0.0
            logits[:, 1, :] = -20.0
            logits[:, 1, token] = 20.0
        else:
            logits[..., token] = 20.0
        return SimpleNamespace(logits=logits)


def observable(result):
    return [
        (
            step["selected_position"],
            step["selected_region"],
            step["proposal_token_id"],
            step["action"],
        )
        for step in result["step_trace"]
    ]


def test_c0_retry_forward_counts_compute_and_reselects(tokenizer, spec):
    model = SequenceModel([4, 0])
    result = run_slot_generation(
        model,
        tokenizer,
        spec,
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        config(),
        save_trace=True,
        task_id="c0",
    )
    assert result["status"] == "completed"
    assert result["pure_newline_veto_triggers"] == 1
    assert result["pure_newline_veto_extra_forwards"] == 1
    assert result["total_forwards"] == model.calls
    assert result["step_trace"][0]["action"] == "pure_newline_veto"
    assert result["step_trace"][1]["action"] == "normal"


def test_c_insertion_reconditions_and_outputs_extra_physical_line(tokenizer, spec):
    model = SequenceModel([4, 0])
    result = run_slot_generation(
        model,
        tokenizer,
        spec,
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
        [spec.bos_id, 0],
        [2, spec.eos_id],
        config(),
        save_trace=True,
        task_id="c",
    )
    assert result["status"] == "completed"
    assert result["blankline_insert_triggers"] == 1
    assert result["blankline_inserted_count"] == 1
    assert result["completion"].startswith("\n")
    assert result["completion_physical_line_count"] == 4
    assert all(
        token not in {spec.mask_id, spec.expand_id, spec.pad_id}
        for token in result["completion_token_ids"]
    )


def test_no_trigger_rows_match_a_actions_completion_and_forwards(tokenizer, spec):
    a_model = SequenceModel([0])
    c0_model = SequenceModel([0])
    c_model = SequenceModel([0])
    common = dict(
        tokenizer=tokenizer,
        tokenizer_spec=spec,
        prefix_ids=[spec.bos_id, 0],
        suffix_ids=[2, spec.eos_id],
        save_trace=True,
        task_id="isolation",
    )
    a = run_slot_generation(
        model=a_model,
        method=Method.V3_HARD_BUDGETED,
        config=SlotGeneratorConfig(max_context_tokens=96, initial_expand_budget=64),
        **common,
    )
    c0 = run_slot_generation(
        model=c0_model,
        method=Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        config=config(),
        **common,
    )
    c = run_slot_generation(
        model=c_model,
        method=Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
        config=config(),
        **common,
    )
    for candidate in (c0, c):
        assert candidate["pure_newline_guard_trigger_count"] == 0
        assert candidate["completion"] == a["completion"]
        assert observable(candidate) == observable(a)
        assert candidate["total_forwards"] == a["total_forwards"]
        assert candidate["token_forwards"] == a["token_forwards"]
