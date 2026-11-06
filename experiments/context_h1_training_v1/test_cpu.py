"""Targeted formal-run checks. No validation-improvement gate."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.context_heads_v1.test_cpu import fixture,cfg
from experiments.context_heads_v1.contract import HeadBatch,validate
from experiments.context_heads_v1.loss import teacher_kl
from experiments.context_heads_v1.training_v1.run import save_checkpoint
from experiments.context_foundation.adapter import aligned_hidden
from tests.context_foundation.test_adapter import engine
from .bank import cells
from .train import initialize,accumulated_update,rng_state,restore_rng


class FormalTests(unittest.TestCase):
    def test_zero_then_nonzero_finite_signal(self):
        c=cfg(seed=20261107);head=initialize(c);b=fixture()
        self.assertTrue(torch.equal(head(b).delta_logits,torch.zeros(2,4,11)))
        self.assertGreater(float(head.embedding.weight.abs().sum()),0)
        self.assertGreater(float(head.hidden_proj.weight.abs().sum()),0)
        self.assertTrue(torch.equal(head.gamma[-1].weight,torch.zeros_like(head.gamma[-1].weight)))
        teacher=torch.randn_like(b.old_logits)
        opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=0)
        norms=[]
        for step in range(3):
            opt.zero_grad();loss=teacher_kl(b.old_logits,head(b).delta_logits,teacher,b.history_mask.any(-1));loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
            norms.append(float(head.output.weight.grad.abs().sum()))
            opt.step()
        self.assertTrue(all(n>0 for n in norms))
        self.assertGreater(float(head.hidden_proj.weight.grad.abs().sum()),0)
        self.assertGreater(float(head.gamma[-1].weight.grad.abs().sum()),0)

    def test_batch_step_future_chain_barrier(self):
        head=initialize(cfg(seed=20261107));head.output.weight.data.normal_(std=.01)
        b=fixture(positions=[2,3,8,9],chains=[0,0,1,1])
        full=head(b).delta_logits
        for q in range(4):
            one=b.queries(b.query_indices[:,q:q+1],b.history_mask[:,q:q+1],b.old_logits[:,q:q+1])
            torch.testing.assert_close(head.step(one).delta_logits,full[:,q:q+1],atol=1e-7,rtol=1e-5)
        q=b.queries(b.query_indices[:,1:2],b.history_mask[:,1:2],b.old_logits[:,1:2])
        changed=replace(q,token_ids=q.token_ids.clone())
        changed.token_ids[:,1:]=9
        self.assertTrue(torch.equal(head(q).delta_logits,head(changed).delta_logits))
        cleared=head.reset(q,chain_id=0)
        self.assertFalse(cleared.history_mask.any())
        self.assertTrue(torch.equal(head(cleared).delta_logits,torch.zeros(2,1,11)))

    def test_real_k2_and_alignment(self):
        e=engine();queries=[]
        e.decode([0],[1],'actualK2','Z4',order='frontier',requested_k=2,observer=lambda kind,s:queries.append(int(s['history_mask'].sum())) if kind=='query' else None)
        self.assertIn(1,queries);self.assertLessEqual(max(queries),1)
        h=torch.arange(28).reshape(1,7,4)
        self.assertTrue(torch.equal(aligned_hidden(h,torch.tensor([0,1,6])),h[:,[0,0,5]]))
        e.hook.remove()

    def test_quota_32_each_split(self):
        c=cells()
        self.assertEqual(len(c),64)
        self.assertEqual(sum(n for k,n in c.items() if k.startswith('train|')),200000)
        self.assertEqual(sum(n for k,n in c.items() if k.startswith('validation|')),20000)

    def test_accumulation_and_optimizer_resume(self):
        b=fixture();b=b.queries(b.query_indices[:,3:4],b.history_mask[:,3:4],b.old_logits[:,3:4])
        teacher=torch.randn(2,1,11)
        class ToyBank:
            def batch(self,ix,device):
                return HeadBatch(**{k:v[ix] if v is not None else None for k,v in vars(b).items()}),teacher[ix]
        bank=ToyBank();a=initialize(cfg());z=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=0)
        oz=torch.optim.AdamW(z.parameters(),lr=3e-4,weight_decay=0)
        accumulated_update(a,oa,bank,[0,1],'cpu',2)
        accumulated_update(z,oz,bank,[0,1],'cpu',1)
        for x,y in zip(a.parameters(),z.parameters()): torch.testing.assert_close(x,y,atol=2e-7,rtol=1e-5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'last.pt';save_checkpoint(p,dict(head=a.state_dict(),optimizer=oa.state_dict(),rng=rng_state(),epoch=2,cursor=128))
            ck=torch.load(p,weights_only=False);r=initialize(cfg());r.load_state_dict(ck['head'])
            op=torch.optim.AdamW(r.parameters(),lr=3e-4,weight_decay=0);op.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
            accumulated_update(a,oa,bank,[0,1],'cpu',2);accumulated_update(r,op,bank,[0,1],'cpu',2)
            for x,y in zip(a.parameters(),r.parameters()): self.assertTrue(torch.equal(x,y))


if __name__=='__main__': unittest.main()
