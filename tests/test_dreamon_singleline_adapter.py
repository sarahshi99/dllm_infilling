from __future__ import annotations

import unittest
from functools import wraps

from experiments.dreamon_singleline_adapter import (
    INITIAL_LENGTHS,
    OfficialHFTokenizerWrapper,
    arm_name,
    audit_rows,
    candidate_key,
    call_with_generation_profile,
    derive_row_seed,
    expected_keys,
    generation_target_code,
    movement_counts,
    paired_seed_key,
    protocol_config,
)


class DreamOnSingleLineAdapterTest(unittest.TestCase):
    def test_official_source_protocol_grid_is_frozen(self) -> None:
        self.assertEqual(INITIAL_LENGTHS, (4, 8, 16, 32, 64))
        config = protocol_config(4)
        self.assertEqual(config["min_gen_len"], 4)
        self.assertEqual(config["max_gen_len"], 64)
        self.assertEqual(config["steps"], 256)
        self.assertEqual(config["temperature"], 0.2)
        self.assertEqual(config["top_p"], 0.9)
        self.assertEqual(config["alg"], "entropy")
        self.assertTrue(config["mask_expansion"])

    def test_arm_and_candidate_keys_are_length_specific(self) -> None:
        self.assertEqual(arm_name(32), "dreamon_dynamic_min32_max64")
        manifest = [{"source_row_id": 3}, {"source_row_id": 7}]
        self.assertEqual(
            expected_keys(manifest, arm_name(32)),
            {
                "dreamon_singleline_source_row=3|arm=dreamon_dynamic_min32_max64",
                "dreamon_singleline_source_row=7|arm=dreamon_dynamic_min32_max64",
            },
        )

    def test_dreamcoder_fixed_profile_reuses_decoder_with_fixed_maximum(self) -> None:
        self.assertEqual(
            arm_name(32, "dreamcoder_fixed"), "dreamcoder_fixed32_dreamon_decoder"
        )
        config = protocol_config(32, "dreamcoder_fixed")
        self.assertEqual(config["min_gen_len"], 32)
        self.assertEqual(config["max_gen_len"], 32)
        self.assertEqual(config["steps"], 256)
        self.assertEqual(config["temperature"], 0.2)
        self.assertEqual(config["top_p"], 0.9)
        self.assertEqual(config["alg"], "entropy")
        self.assertTrue(config["delete_eos_token"])

    def test_fixed_control_uses_distinct_raw_key_but_paired_dynamic_seed_key(self) -> None:
        fixed_arm = arm_name(16, "dreamcoder_fixed")
        fixed_key = candidate_key("dreamcoder_fixed", 7, fixed_arm)
        self.assertEqual(
            fixed_key,
            "dreamcoder_singleline_source_row=7|arm=dreamcoder_fixed16_dreamon_decoder",
        )
        self.assertEqual(
            paired_seed_key(7, 16),
            "dreamon_singleline_source_row=7|arm=dreamon_dynamic_min16_max64",
        )

    def test_multiline_profile_changes_only_candidate_namespace_contract(self) -> None:
        arm = arm_name(8)
        self.assertEqual(
            candidate_key("dreamon", 11, arm, "multiline"),
            "dreamon_multiline_source_row=11|arm=dreamon_dynamic_min8_max64",
        )
        self.assertEqual(
            paired_seed_key(11, 8, "multiline"),
            "dreamon_multiline_source_row=11|arm=dreamon_dynamic_min8_max64",
        )
        manifest = [{"source_row_id": 11}, {"source_row_id": 12}]
        self.assertEqual(
            expected_keys(manifest, arm, "dreamon", "multiline"),
            {
                "dreamon_multiline_source_row=11|arm=dreamon_dynamic_min8_max64",
                "dreamon_multiline_source_row=12|arm=dreamon_dynamic_min8_max64",
            },
        )

    def test_row_seed_is_deterministic_and_arm_specific(self) -> None:
        first = derive_row_seed(42, "key-a")
        self.assertEqual(first, derive_row_seed(42, "key-a"))
        self.assertNotEqual(first, derive_row_seed(42, "key-b"))

    def test_official_tokenizer_wrapper_matches_source_semantics(self) -> None:
        class FakeTokenizer:
            bos_token_id = 1
            eos_token_id = 2
            mask_token_id = 3

            def encode(self, value: str) -> list[int]:
                return [len(value)]

            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(map(str, tokens))

        wrapper = OfficialHFTokenizerWrapper(FakeTokenizer())
        self.assertEqual(wrapper.expand_id, 151667)
        self.assertEqual(wrapper.encode("abc", add_bos=True, add_eos=True), [1, 3, 2])
        self.assertEqual(wrapper.decode([4, 5]), "4,5")

    def test_profile_captures_official_source_local_counters(self) -> None:
        def fake_generation() -> str:
            expand_budget = 64
            num_generation_tokens = 4
            expand_budget -= 3
            num_generation_tokens += 3
            num_generation_tokens -= 1
            return "ok"

        result, captured = call_with_generation_profile(fake_generation, fake_generation.__code__)
        self.assertEqual(result, "ok")
        self.assertEqual(captured["expand_budget"], 61)
        self.assertEqual(captured["num_generation_tokens"], 6)
        self.assertEqual(movement_counts(initial_length=4, initial_expand_budget=64, captured=captured), (3, 1))

    def test_generation_target_code_unwraps_decorators(self) -> None:
        def source_generation() -> None:
            return None

        @wraps(source_generation)
        def decorated() -> None:
            return source_generation()

        self.assertIs(generation_target_code(decorated), source_generation.__code__)

    def test_audit_requires_exact_keys_forward_partition_and_length_accounting(self) -> None:
        row = {
            "candidate_key": "a",
            "arm": arm_name(4),
            "metrics": {
                "initial_selected_length": 4,
                "final_generated_length": 6,
                "expansion_moves": 3,
                "contraction_moves": 1,
                "search_forward_calls": 0,
                "decode_forward_calls": 12,
                "total_forward_calls": 12,
            },
        }
        self.assertTrue(audit_rows([row], {"a"}, arm_name(4))["passed"])
        row["metrics"]["total_forward_calls"] = 11
        self.assertFalse(audit_rows([row], {"a"}, arm_name(4))["passed"])


if __name__ == "__main__":
    unittest.main()
