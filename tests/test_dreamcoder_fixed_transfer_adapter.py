"""CPU agreement with the actual cached official generation implementation."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.dreamcoder_fixed_transfer_adapter import decode_fixed_transfer, native_fixed_reference
from experiments.dreamon_singleline_order_parallel import GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER


NATIVE = Path('/home/shx/.cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Base-7B/snapshots/2346ccd3be517d0d314152b988a3b9bafa7d6d63/generation_utils.py')


class Tokenizer:
    mask_token_id = 7
    pad_token_id = eos_token_id = 6
    all_special_ids = [5, 6, 7]
    eos_token = pad_token = '<eos>'

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(str(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)


@unittest.skipUnless(NATIVE.exists(), 'audited native generation cache required')
class NativeAgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('dreamcoder_native_cpu_test', NATIVE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.native = module

    def make_model(self):
        native = self.native

        class Model(torch.nn.Module):
            _sample = native.DreamGenerationMixin._sample

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
                self.device = torch.device('cpu')

            def forward(self, x, *args):
                columns = torch.tensor([1.2, 1.5, 2.0, 1.0, 1.7, -50., -50., -50.])
                # Content-sensitive probabilities exercise shifts and RNG path.
                pos = torch.arange(x.shape[1]).view(1, -1, 1)
                content = x.float().view(1, -1, 1)
                logits = columns.expand(x.shape[0], x.shape[1], -1).clone()
                logits[..., 0] += (pos[..., 0] % 7) * .03 + content[..., 0] * .025
                return SimpleNamespace(logits=logits)

        return Model()

    def kwargs(self, policy, k=1):
        return dict(model=self.make_model(), tokenizer=Tokenizer(), prefix_ids=[5, 1],
                    suffix_ids=[2, 6], requested_k=k, policy=policy, row_seed=4242)

    def test_both_orders_match_explicitly_patched_native_k1_and_k2(self):
        for policy in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER):
            for k in (1, 2, 4):
                args = self.kwargs(policy, k)
                actual = decode_fixed_transfer(**args, sample_tokens_fn=self.native.sample_tokens)
                reference = native_fixed_reference(**args)
                self.assertEqual(actual['completion_token_ids'], reference['completion_token_ids'])
                self.assertEqual(actual['completion'], reference['completion'])
                self.assertEqual(actual['forward_count'], 64 // k)
                self.assertEqual(actual['final_dynamic_length'], 64)

    def test_zero_head_and_nonzero_head_use_same_sampling_path(self):
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
            args = self.kwargs(policy, 2)
            zero, trained = Head(0), Head(.01)
            z = decode_fixed_transfer(**args, sample_tokens_fn=self.native.sample_tokens, markov_head=zero)
            t = decode_fixed_transfer(**args, sample_tokens_fn=self.native.sample_tokens, markov_head=trained)
            if policy == LEFT_TO_RIGHT_FRONTIER:
                self.assertGreater(z['markov_enabled_count'], 0)
            self.assertEqual(z['markov_enabled_count'], zero.calls)
            self.assertEqual(t['markov_enabled_count'], trained.calls)
            for result in (z, t):
                self.assertEqual(result['expand_count'] + result['delete_count'], 0)
                self.assertEqual(len(result['completion_token_ids']), 64)


    def test_k4_native_rounds_match_and_head_eos_never_stops_round(self):
        from tests.test_dreamon_markov_k2 import deterministic_sample
        args=self.kwargs(LEFT_TO_RIGHT_FRONTIER,4)
        nohead=decode_fixed_transfer(**args,sample_tokens_fn=self.native.sample_tokens,capture_actions=True)
        reference=native_fixed_reference(**args,capture_actions=True)
        canvas=[7]*64;snapshots=[]
        for step in nohead['action_trace']:
            for position,token in step['submitted']:canvas[position-2]=token
            snapshots.append(list(canvas))
        self.assertEqual(snapshots,reference['native_canvas_history'])
        class EOSHead(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,ids):
                self.calls+=1;out=torch.zeros((len(ids),8));out[:,6]=1000;return out
        head=EOSHead()
        result=decode_fixed_transfer(**args,sample_tokens_fn=deterministic_sample,markov_head=head,capture_actions=True)
        self.assertEqual(result['forward_count'],16)
        self.assertEqual(head.calls,result['markov_enabled_count'])
        self.assertEqual(head.calls,32)
        self.assertTrue(all(t['actual_submitted_count']==4 for t in result['action_trace']))
        self.assertEqual(result['structural_stop_count'],0)
        self.assertEqual(result['delete_count']+result['expand_count'],0)


if __name__ == '__main__':
    unittest.main()
