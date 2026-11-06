from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
import torch

from experiments.context_foundation.adapter import (
    Engine, FrozenHeadAdapter, aligned_hidden, distribution, history_mask,
    inverse_cdf, sample_key, uniform,
)
from experiments.context_foundation.common import Store
from experiments.dreamon_markov_head_training import MarkovHead


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.model = SimpleNamespace(norm=torch.nn.Identity())
        self.seen = []
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        self.lm_head.weight.data.zero_()
        self.lm_head.weight.data[:4] = torch.eye(4)

    def forward(self, x, attention, positions, **_):
        self.seen.append(x.clone())
        h = torch.zeros((*x.shape, 4), device=x.device)
        # Bidirectional ordinary-prefix influence, and a positional alignment signal.
        h[..., 0] = (x < 4).sum() * .05
        h[..., 1] = torch.arange(x.shape[1]) * .01
        h = self.model.norm(h)
        logits = self.lm_head(h)
        logits[..., 4:] = -100
        return SimpleNamespace(logits=logits)


def engine():
    model = Toy()
    tokenizer = SimpleNamespace(mask_id=4, eos_id=5, expand_id=6, decode=lambda t, **_: str(t))
    head = MarkovHead(vocab_size=7, rank=256, structural_token_ids=[4,5,6])
    e = Engine(model, tokenizer, FrozenHeadAdapter(head), [4,5,6])
    e.zero = FrozenHeadAdapter(head, True)
    return e


class AdapterTest(unittest.TestCase):
    def test_finite_vocab_inverse_cdf_and_ties(self):
        logits = torch.tensor([0.,0.,0.,0.])
        p = distribution(logits, temperature=1, top_p=.5).exp()
        self.assertTrue(torch.equal(p > 0, torch.tensor([True,True,True,False])))
        full = distribution(logits, temperature=1, top_p=None)
        self.assertEqual([int(inverse_cdf(full, u)) for u in [.01,.26,.51,.76]], [0,1,2,3])
        self.assertTrue(torch.isneginf(distribution(torch.tensor([100.,0.,-100.]))[1:]).all())

    def test_key_reproducible_independent_of_other_cases(self):
        key = sample_key('a', 3, 7)
        u = uniform(key)
        for i in range(100):
            uniform(sample_key('other', i, 1))
        self.assertEqual(u, uniform(key))
        self.assertNotEqual(u, uniform(sample_key('a',4,7)))

    def test_alignment_including_boundary(self):
        h = torch.arange(20).reshape(1,5,4)
        self.assertTrue(torch.equal(aligned_hidden(h, torch.tensor([0,1,4])), h[:,[0,0,3]]))

    def test_masks_no_future_other_chain_or_provisional(self):
        p = torch.tensor([[2,3,7,8]])
        c = torch.tensor([[0,0,1,1]])
        s = torch.ones_like(p, dtype=torch.bool)
        t = torch.tensor([[1,2,1,0]])
        hm = history_mask(p,c,s,t,torch.tensor([[1,3]]))
        self.assertEqual(hm.tolist(), [[[True,False,False,False],[False,False,True,False]]])

    def test_zero_nohead_every_token_and_canvas(self):
        e = engine()
        for order in ['global','frontier']:
            a = e.decode([0],[1], 'case', 'nohead', order=order, detailed=True)
            b = e.decode([0],[1], 'case', 'Z4', order=order, detailed=True)
            self.assertEqual(a['completion_token_ids'], b['completion_token_ids'])
            self.assertEqual([r['x_after'] for r in a['action_trace']], [r['x_after'] for r in b['action_trace']])
            self.assertEqual(a['head_calls'], 0)
            self.assertGreater(b['head_calls'], 0)

    @torch.inference_mode()
    def test_teacher_copy_and_no_future_leak(self):
        e = engine()
        collected = []
        def observe(kind, s):
            if kind == 'query' and int(s['history_mask'].sum()) == 2 and not collected:
                collected.append({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in s.items()})
        e.decode([0],[1], 'case', 'nohead', order='frontier', observer=observe)
        s = collected[0]
        before = s['x0'].clone()
        e.fresh(s)
        seen = e.model.seen[-1]
        changed = (before != seen)[0].nonzero().flatten().tolist()
        self.assertEqual(changed, s['position_ids'][0,:2].tolist())
        self.assertTrue(torch.equal(before, s['x0']))
        s['token_ids'][0,3] = 3
        a = e.fresh(s)
        s['token_ids'][0,3] = 0
        b = e.fresh(s)
        self.assertTrue(torch.equal(a,b))

    def test_structure_barrier(self):
        e = engine()
        old_forward = e.model.forward
        def forward(*args, **kwargs):
            out = old_forward(*args, **kwargs)
            out.logits[...,5] = 100
            return out
        e.model.forward = forward
        a = e.decode([0],[1], 'case', 'Z4', order='frontier', detailed=True)
        self.assertEqual(len(a['action_trace'][0]['events']), 1)
        self.assertEqual(a['head_calls'], 0)
        self.assertEqual(a['delete_count'], 64)

    def test_corrected_structure_refresh_never_commits_guess(self):
        e=engine()
        e.head.head.embedding.weight.data.fill_(1)
        e.head.head.output.weight.data[:4].fill_(-1)
        old=e.model.forward
        def forward(*args,**kwargs):
            out=old(*args,**kwargs)
            out.logits[...,5]=-80
            return out
        e.model.forward=forward
        a=e.decode([0],[1],'case','H4-old',order='frontier',detailed=True)
        first=a['action_trace'][0]
        self.assertEqual(first['barrier']['reason'],'corrected_structure_refresh_only')
        self.assertEqual(first['actual_submitted_count'],1)
        self.assertEqual(first['delete'],[])
        self.assertEqual(first['events'][-1]['committed'],False)

    @torch.inference_mode()
    def test_teacher_never_writes_other_chain_commits(self):
        e=engine()
        collected=[]
        def observer(kind,s):
            if kind=='query' and int(s['history_mask'].sum())==3 and not collected:
                collected.append({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in s.items()})
        e.decode([0],[1],'case','nohead',order='frontier',observer=observer)
        s=collected[0]
        s['position_ids'][0]=torch.tensor([1,2,8,9])
        s['chain_ids'][0]=torch.tensor([0,0,1,1])
        s['history_mask'].zero_()
        s['history_mask'][0,0,2]=True
        e.fresh(s)
        changed=(e.model.seen[-1]!=s['x0'])[0].nonzero().flatten().tolist()
        self.assertEqual(changed,[8])

    def test_restart_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'ledger.sqlite'
            a = Store(path)
            a.put('case', 'x', {'seed':42})
            a.db.close()
            b = Store(path)
            b.put('case', 'x', {'seed':42})
            self.assertEqual(len(b.rows('case')), 1)


if __name__ == '__main__':
    unittest.main()
