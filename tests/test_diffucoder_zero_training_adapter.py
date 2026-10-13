"""Agreement against this exact DiffuCoder release, including EOS probabilities."""
import importlib.util
import sys
import unittest
from types import SimpleNamespace

import torch

from experiments.diffucoder_zero_training_adapter import (
    MODEL_PATH, decode_fixed_transfer, native_fixed_reference,
)
from experiments.dreamon_singleline_order_parallel import GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER


class Tokenizer:
    mask_token_id = 7
    pad_token_id = eos_token_id = 6
    all_special_ids = [5, 6, 7]
    eos_token = pad_token = '<eos>'

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(str(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)


@unittest.skipUnless((MODEL_PATH / 'generation_utils.py').exists(), 'exact DiffuCoder generation cache required')
class DiffuCoderNativeAgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('diffucoder_native_transfer_cpu', MODEL_PATH / 'generation_utils.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.native = module

    def model(self):
        native = self.native

        class Model(torch.nn.Module):
            _sample = native.DreamGenerationMixin._sample

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.))
                self.device = torch.device('cpu')

            def forward(self, x, *args):
                # EOS is deliberately probable: an accidental DreamCoder EOS
                # penalty would change this independent reference comparison.
                columns = torch.tensor([1.2, 1.5, 2.0, 1.0, 1.7, -50., 1.9, -50.])
                logits = columns.expand(x.shape[0], x.shape[1], -1).clone()
                logits[..., 0] += torch.arange(x.shape[1]) % 7 * .03 + x.float() * .025
                return SimpleNamespace(logits=logits)

        return Model()

    def kwargs(self, policy, k):
        return dict(model=self.model(), tokenizer=Tokenizer(), prefix_ids=[5, 1],
                    suffix_ids=[2, 6], requested_k=k, policy=policy, row_seed=4242)

    def test_two_orders_two_k_match_actual_native_release(self):
        for policy in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER):
            for k in (1, 2):
                with self.subTest(policy=policy, k=k):
                    kwargs = self.kwargs(policy, k)
                    actual = decode_fixed_transfer(**kwargs, sample_tokens_fn=self.native.sample_tokens)
                    native = native_fixed_reference(**kwargs)
                    self.assertEqual(actual['completion_token_ids'], native['completion_token_ids'])
                    self.assertEqual(actual['completion'], native['completion'])
                    self.assertEqual(actual['forward_count'], 64 // k)
                    self.assertGreater(actual['completion_token_ids'].count(6), 0)
                    self.assertEqual(actual['expand_count'] + actual['delete_count'], 0)

    def test_native_eos_penalty_is_not_silently_added(self):
        kwargs = self.kwargs(LEFT_TO_RIGHT_FRONTIER, 1)
        with self.assertRaisesRegex(ValueError, 'no EOS penalty'):
            decode_fixed_transfer(**kwargs, sample_tokens_fn=self.native.sample_tokens, eos_penalty=3.)
        with self.assertRaisesRegex(ValueError, 'no EOS penalty'):
            native_fixed_reference(**kwargs, eos_penalty=3.)

    def test_zero_and_nonzero_heads_use_same_adapter_path(self):
        class Head(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
                self.calls = 0

            def forward(self, ids):
                self.calls += 1
                result = torch.zeros((len(ids), 8))
                result[:, 1] = self.value
                return result

        for policy in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER):
            for value in (0., .01):
                head = Head(value)
                result = decode_fixed_transfer(**self.kwargs(policy, 2), sample_tokens_fn=self.native.sample_tokens, markov_head=head)
                self.assertEqual(result['markov_enabled_count'], head.calls)
                if policy == LEFT_TO_RIGHT_FRONTIER:
                    self.assertGreater(head.calls, 0)
                self.assertEqual(len(result['completion_token_ids']), 64)
                self.assertEqual(result['expand_count'] + result['delete_count'], 0)


if __name__ == '__main__':
    unittest.main()
