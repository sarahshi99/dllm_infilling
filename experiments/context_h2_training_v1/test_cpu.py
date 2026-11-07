"""Meaningful H2 checks in addition to A/B and the shared formal trainer suite."""
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
from .dependency import ReadOnlyStore


class H2Tests(unittest.TestCase):
    def head(self): return initialize(cfg(seed=20261107),'H2-GRU')

    def test_zero_output_learning_and_full_optimizer_resume(self):
        h=self.head();b=fixture();b=b.queries(b.query_indices[:,3:4],b.history_mask[:,3:4],b.old_logits[:,3:4])
        self.assertTrue(torch.equal(h(b).delta_logits,torch.zeros_like(b.old_logits)))
        self.assertTrue(h.gru.weight_hh.count_nonzero());self.assertEqual(h.config.memory_mode,'full-memory')
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
        self.assertGreater(float(h.gru.weight_hh.grad.abs().sum()),0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'last.pt'
            save_checkpoint(path,dict(head=h.state_dict(),optimizer=op.state_dict(),rng=rng_state(),epoch=2,cursor=128))
            ck=torch.load(path,weights_only=False);a=self.head();a.load_state_dict(ck['head']);oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=0);oa.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
            self.assertEqual((ck['epoch'],ck['cursor']),(2,128))
            accumulated_update(h,op,bank,[0,1],'cpu',1);accumulated_update(a,oa,bank,[0,1],'cpu',1)
            for x,y in zip(h.parameters(),a.parameters()): self.assertTrue(torch.equal(x,y))

    def test_one_two_three_history_matches_explicit_zero_state_gru(self):
        h=self.head();h.output.weight.data.normal_(std=.01);b=fixture()
        for n in (1,2,3):
            q=b.queries(b.query_indices[:,n:n+1],b.history_mask[:,n:n+1],b.old_logits[:,n:n+1])
            state=torch.zeros(2,256)
            for j in range(n):
                rel=torch.tensor([(n-j)/1024,(n-j)/4]).repeat(2,1)
                rec=h.record(torch.cat([h.embedding(b.token_ids[:,j]),h.hidden_proj(b.old_hidden[:,j]),rel],-1))
                state=h.gru(rec,state)
            fixed=torch.tensor([n/1024,n/4,1]).repeat(2,1)
            z=h.readout(torch.cat([state,h.hidden_proj(b.old_hidden[:,n]),fixed],-1))
            torch.testing.assert_close(h(q).latent[:,0],z,atol=1e-7,rtol=1e-5)
            full=h(b).delta_logits[:,n:n+1]
            torch.testing.assert_close(h.step(q).delta_logits,full,atol=1e-7,rtol=1e-5)

    def test_future_other_chain_padding_and_case_reset(self):
        h=self.head();h.output.weight.data.normal_(std=.01)
        b=fixture(positions=[0,1,3,4],chains=[0,0,1,1]);q=b.queries(b.query_indices[:,1:2],b.history_mask[:,1:2],b.old_logits[:,1:2])
        before=h(q).delta_logits
        change=replace(q,token_ids=q.token_ids.clone(),old_hidden=q.old_hidden.clone())
        change.token_ids[:,1:]=9;change.old_hidden[:,2:]=1000
        self.assertTrue(torch.equal(before,h(change).delta_logits))
        for _ in range(2): h(fixture());h.reset(q,chain_id=0);h.reset(q)
        self.assertTrue(torch.equal(before,h(q).delta_logits))
        reset=h.reset(q);self.assertTrue(torch.equal(h(reset).delta_logits,torch.zeros_like(before)))
        p=replace(q,selected_mask=q.selected_mask.clone(),old_hidden=q.old_hidden.clone())
        p.selected_mask[:,2:]=False;p.old_hidden[:,2:]=float('nan')
        self.assertTrue(torch.equal(before,h(p).delta_logits))

    def test_bf16_live_adapter_uses_same_fp32_definition(self):
        h=self.head();h.output.weight.data.normal_(std=.01);b=fixture()
        b=replace(b,old_hidden=b.old_hidden.bfloat16())
        wrapper=KeywordHeadAdapter(h)
        expected=h(replace(b,old_hidden=b.old_hidden.float())).delta_logits
        self.assertTrue(torch.equal(wrapper(**vars(b))['delta_logits'],expected))

    def test_public_store_cannot_write(self):
        from experiments.context_foundation.common import Store
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bank.sqlite';writer=Store(p);writer.put('state','id',{'state_id':'id'});writer.db.close()
            reader=ReadOnlyStore(p);self.assertEqual(reader.count('state'),1)
            with self.assertRaises(RuntimeError): reader.put('state','id',{})
            with self.assertRaises(Exception): reader.db.execute('delete from rows')
            reader.db.close()


if __name__=='__main__': unittest.main()
