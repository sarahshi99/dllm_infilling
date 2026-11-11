"""H3 formal learned-attention checks; CPU only, no backbone or data collection."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from experiments.context_heads_v1.test_cpu import fixture,cfg
from experiments.context_heads_v1.contract import HeadBatch
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter
from experiments.context_h1_training_v1.train import initialize,accumulated_update,rng_state,restore_rng
from experiments.context_heads_v1.training_v1.run import save_checkpoint
from experiments.context_h2_training_v1 import test_cpu as h2_tests
from .dependency import ReadOnlyStore


class H3Tests(unittest.TestCase):
    def head(self): return initialize(cfg(seed=20261107),'H3-short-attention')

    def test_zero_learning_accumulation_optimizer_resume(self):
        h=self.head();b=fixture();b=b.queries(b.query_indices[:,3:4],b.history_mask[:,3:4],b.old_logits[:,3:4])
        self.assertTrue(torch.equal(h(b).delta_logits,torch.zeros_like(b.old_logits)))
        self.assertGreater(int(h.q_proj.weight.count_nonzero()),0)
        teacher=torch.randn_like(b.old_logits)
        class Bank:
            def batch(self,ix,device): return HeadBatch(**{k:v[ix] if v is not None else None for k,v in vars(b).items()}),teacher[ix]
        bank=Bank();a=copy.deepcopy(h)
        op=torch.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=0)
        oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=0)
        loss,norm=accumulated_update(h,op,bank,[0,1],'cpu',1)
        accumulated_update(a,oa,bank,[0,1],'cpu',2)
        self.assertGreater(norm,0);self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for x,y in zip(h.parameters(),a.parameters()): torch.testing.assert_close(x,y,atol=2e-7,rtol=1e-5)
        accumulated_update(h,op,bank,[0,1],'cpu',1)
        self.assertGreater(float(h.q_proj.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in h.parameters()))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'last.pt'
            save_checkpoint(path,dict(head=h.state_dict(),optimizer=op.state_dict(),rng=rng_state(),epoch=2,cursor=128))
            ck=torch.load(path,weights_only=False);a=self.head();a.load_state_dict(ck['head']);oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=0);oa.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
            self.assertEqual((ck['epoch'],ck['cursor']),(2,128))
            accumulated_update(h,op,bank,[0,1],'cpu',1);accumulated_update(a,oa,bank,[0,1],'cpu',1)
            for x,y in zip(h.parameters(),a.parameters()): self.assertTrue(torch.equal(x,y))

    def test_learned_attention_all_visible_records(self):
        h=self.head();h.output.weight.data.normal_(std=.01);b=fixture()
        self.assertEqual(h.config.attention_mode,'learned-attention')
        out=h(b);w=out.attention_weights
        visible=b.history_mask[:,:,None,:].expand_as(w)
        self.assertEqual(int(w.masked_select(~visible).count_nonzero()),0)
        self.assertTrue(torch.equal(w[:,1,:,0],torch.ones(2,4)))
        torch.testing.assert_close(w[:,1:].sum(-1),torch.ones(2,3,4),atol=1e-6,rtol=1e-6)
        self.assertTrue(bool((w[:,3,:,:3]>0).all()))
        for j in range(4):
            one=b.queries(b.query_indices[:,j:j+1],b.history_mask[:,j:j+1],b.old_logits[:,j:j+1])
            torch.testing.assert_close(h.step(one).delta_logits,out.delta_logits[:,j:j+1],atol=1e-7,rtol=1e-5)
        # Test with nonzero output so independence cannot pass vacuously.
        q=b.queries(b.query_indices[:,3:4],b.history_mask[:,3:4],b.old_logits[:,3:4])
        changed=replace(q,token_ids=q.token_ids.clone());changed.token_ids[:,0]=(changed.token_ids[:,0]+1)%10
        self.assertFalse(torch.equal(h(q).delta_logits,h(changed).delta_logits))

    test_future_other_chain_padding_and_case_reset=h2_tests.H2Tests.test_future_other_chain_padding_and_case_reset
    test_bf16_live_adapter_uses_same_fp32_definition=h2_tests.H2Tests.test_bf16_live_adapter_uses_same_fp32_definition

    def test_attention_resource_counter(self):
        from .pipeline import CountedAdapter
        h=self.head();adapter=CountedAdapter(h);b=fixture()
        adapter(**vars(b))
        self.assertEqual(adapter.attention_calls,1)
        self.assertEqual(adapter.attention_rows,8)
        self.assertEqual(adapter.active_records,int(b.history_mask.sum()))
        self.assertEqual(adapter.calls,8)
        adapter.handle.remove()

    def test_public_store_cannot_write(self):
        from experiments.context_foundation.common import Store
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bank.sqlite';writer=Store(p);writer.put('state','id',{'state_id':'id'});writer.db.close()
            reader=ReadOnlyStore(p);self.assertEqual(reader.count('state'),1)
            with self.assertRaises(RuntimeError): reader.put('state','id',{})
            with self.assertRaises(Exception): reader.db.execute('delete from rows')
            reader.db.close()


if __name__=='__main__': unittest.main()
