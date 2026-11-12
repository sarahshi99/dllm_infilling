from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sqlite3
import tempfile
import unittest
import torch

from experiments.context_foundation.adapter import Engine as Reference, FrozenHeadAdapter
from experiments.dreamon_markov_head_training import MarkovHead
from experiments.threshold_eval.adapter import Engine
from experiments.threshold_eval.common import Ledger, configs


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(()))
        self.model=SimpleNamespace(norm=torch.nn.Identity())
        self.special_token=None

    def forward(self,x,*args,**kwargs):
        h=self.model.norm(torch.zeros((*x.shape,4)))
        logits=torch.full((*x.shape,7),-100.)
        logits[...,:2]=0.
        if self.special_token is not None:
            logits[...,self.special_token]=100.
        return SimpleNamespace(logits=logits)


def engine(cls=Engine,with_head=True,**kwargs):
    model=Toy()
    tok=SimpleNamespace(mask_id=4,eos_id=5,expand_id=6,decode=lambda ids,**kw:str(ids))
    h=MarkovHead(vocab_size=7,rank=256,structural_token_ids=[4,5,6])
    return cls(model,tok,FrozenHeadAdapter(h) if with_head else None,[4,5,6],**kwargs)


class ThresholdTests(unittest.TestCase):
    def test_matrix(self):
        c=configs()
        self.assertEqual(len(c),34)
        self.assertEqual(len({x['id'] for x in c}),34)
        self.assertEqual(sum(x['method']=='nohead' for x in c),18)

    def test_off_matches_legacy_tokens_canvas_structure(self):
        for order in ('global','left'):
            for k in (1,4):
                for method,arm in [('nohead','K1' if k==1 else 'nohead'),('trained','H4-old')]:
                    if k==1 and method=='trained':continue
                    a,b=engine(),engine(Reference)
                    v=a.decode([0],[1],'case',method,order,k,detailed=True)
                    w=b.decode([0],[1],'case',arm,order,detailed=True)
                    for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count'):
                        self.assertEqual(v[field],w[field])
                    for t,u in zip(v['action_trace'],w['action_trace']):
                        for f in ('selected','submitted','expand','delete','x_after','barrier'):
                            self.assertEqual(t[f],u[f])

    def test_zero_interface_matches_nohead_with_threshold(self):
        e=engine()
        for k in (2,4):
            for t in (None,.80,.95):
                a=e.decode([0],[1],'case','nohead','left',k,t,detailed=True)
                b=e.decode([0],[1],'case','zero','left',k,t,detailed=True)
                self.assertEqual(a['completion_token_ids'],b['completion_token_ids'])
                self.assertEqual([r['x_after'] for r in a['action_trace']],[r['x_after'] for r in b['action_trace']])
                self.assertEqual(a['head_calls'],0)
                self.assertGreater(b['head_calls'],0)

    def test_k2_k4_and_nohead_has_no_head_instance(self):
        for k in (2,4):
            e=engine(with_head=False)
            r=e.decode([0],[1],'case','nohead','left',k)
            self.assertEqual(max(len(t['selected']) for t in r['action_trace']),k)
            self.assertEqual(r['head_calls'],0)

    def test_reject_no_resample_no_memory_stop_same_chain(self):
        e=engine()
        seen=[]
        def observe(kind,s):
            if kind=='query':seen.append((s['round_index'],int(s['query_indices'][0,0]),int(s['history_mask'].sum())))
        r=e.decode([0],[1],'case','trained','left',4,.80,observer=observe,detailed=True)
        t=r['action_trace'][0]
        self.assertEqual(len(t['events']),2)
        self.assertTrue(t['events'][1]['rejected'])
        self.assertEqual(t['events'][1]['candidate_probability'],.5)
        self.assertEqual(len(t['submitted']),1)
        self.assertEqual(r['head_calls'],63) # final one-mask round has no extra candidate
        self.assertEqual(r['statistics']['extra_ordinary_accepted'],0)
        self.assertEqual(sum(row.count(4) for row in [t['x_after']]),63)

    def test_gap_chain_anchor_not_exempt(self):
        e=engine()
        from experiments.threshold_eval.adapter import select_positions as original
        def choose(positions,conf,requested_k,policy):
            if len(positions)>=6:return [positions[i] for i in (0,1,4,5)]
            return original(positions,conf,requested_k=requested_k,policy=policy)
        with patch('experiments.threshold_eval.adapter.select_positions',side_effect=choose):
            r=e.decode([0],[1],'case','trained','global',4,.80,detailed=True)
        events=r['action_trace'][0]['events']
        self.assertEqual([e['slot'] for e in events],[0,1,2])
        self.assertEqual(events[2]['visible_slots'],[])
        self.assertTrue(events[2]['rejected'])
        self.assertEqual(len(r['action_trace'][0]['submitted']),1)

    def test_sampled_probability_includes_special_denominator_and_boundary(self):
        e=engine(with_head=False)
        # p(actual=0)=.8, legal special mask remains in denominator with p=.2.
        old=e.model.forward
        def forward(*args,**kw):
            out=old(*args,**kw)
            out.logits.fill_(-100.)
            out.logits[...,0]=torch.log(torch.tensor(.8))
            out.logits[...,4]=torch.log(torch.tensor(.2))
            return out
        e.model.forward=forward
        r=e.decode([0],[1],'case','nohead','left',4,.80)
        self.assertGreater(r['statistics']['extra_ordinary_accepted'],0)
        self.assertEqual(r['statistics'].get('extra_ordinary_rejected',0),0)
        r=e.decode([0],[1],'case','nohead','left',4,.90)
        self.assertEqual(r['statistics'].get('extra_ordinary_accepted',0),0)
        p=r['action_trace'][0]['events'][1]['candidate_probability']
        self.assertAlmostEqual(p,.8)
        # Force draw of a low-probability ordinary token; top1=.95 must not pass it.
        def low_forward(*args,**kw):
            out=old(*args,**kw);out.logits.fill_(-100.)
            out.logits[...,0]=torch.log(torch.tensor(.95));out.logits[...,1]=torch.log(torch.tensor(.05))
            return out
        e.model.forward=low_forward
        with patch('experiments.threshold_eval.adapter.inverse_cdf',return_value=torch.tensor(1)):
            r=e.decode([0],[1],'case','nohead','left',4,.80)
        self.assertAlmostEqual(r['action_trace'][0]['events'][1]['candidate_probability'],.05)
        self.assertTrue(r['action_trace'][0]['events'][1]['rejected'])

    def test_structure_once_delete_provenance_and_corrected_barrier(self):
        e=engine()
        e.model.special_token=5
        r=e.decode([0],[1],'case','trained','left',4,.95,detailed=True)
        self.assertEqual(r['delete_count'],64)
        self.assertEqual(r['normal_token_count'],0)
        self.assertEqual(len(r['action_trace'][0]['events']),1)
        self.assertEqual(r['head_calls'],0)
        # Existing EOS in prefix makes broadcast global; deletion only old mask coordinates.
        r=e.decode([5],[1],'case','trained','left',4,.95,detailed=True)
        self.assertEqual(r['delete_count'],64)
        self.assertEqual(r['action_trace'][0]['x_after'][0],5)
        e=engine()
        e.head.head.embedding.weight.data.fill_(1)
        e.head.head.output.weight.data[:2].fill_(-1)
        r=e.decode([0],[1],'case','trained','left',4,.95)
        t=r['action_trace'][0]
        self.assertEqual(t['barrier']['reason'],'corrected_structure_refresh_only')
        self.assertEqual(len(t['submitted']),1)
        self.assertFalse(t['events'][-1]['committed'])
        self.assertEqual(t['delete'],[])

    def test_resume_unique_transaction(self):
        with tempfile.TemporaryDirectory() as td,patch('experiments.threshold_eval.common.SHARED',Path(td)):
            db=Ledger('nohead');db.put('quality','a','case',{'version':'x'});db.db.close()
            db=Ledger('nohead')
            self.assertTrue(db.has('quality','a','case'))
            self.assertEqual(db.count('quality'),1)
            with self.assertRaises(sqlite3.IntegrityError):db.put('quality','a','case',{'version':'x'})
            self.assertEqual(db.count('quality'),1)



class DeploymentPathTests(unittest.TestCase):
    def test_trace_disabled_preserves_tokens_actions_and_counters(self):
        for method in ('nohead','trained'):
            for threshold in (None,.90):
                e=engine()
                a=e.decode([0],[1],'case',method,'left',4,threshold,record_trace=True)
                b=e.decode([0],[1],'case',method,'left',4,threshold,record_trace=False)
                for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count','statistics','normal_submissions_histogram','actual_submitted_count'):
                    self.assertEqual(a[field],b[field],field)
                self.assertEqual(b['action_trace'],[])

    def test_expand_after_delete_coordinate_fix_matches_legacy(self):
        from experiments.threshold_eval.adapter import select_positions as select
        def setup(cls):
            e=engine(cls);count=[0];old=e.model.forward
            def forward(*args,**kw):
                o=old(*args,**kw);count[0]+=1
                if count[0]==1:
                    o.logits[...,5]=100 # select last mask; deletes just that one
                elif count[0]==2:
                    o.logits[...,6]=100 # one legal expand consumes new spare capacity
                return o
            e.model.forward=forward
            return e
        calls=[0]
        def choose(positions,confidence,requested_k,policy):
            calls[0]+=1
            if calls[0]==1:return [positions[-1]]
            return select(positions,confidence,requested_k=requested_k,policy=policy)
        e=setup(Engine)
        with patch('experiments.threshold_eval.adapter.select_positions',side_effect=choose):
            a=e.decode([0],[1],'case','nohead','global',4,detailed=True)
        calls[0]=0;e=setup(Reference)
        with patch('experiments.context_foundation.adapter.select_positions',side_effect=choose):
            b=e.decode([0],[1],'case','nohead','global',detailed=True)
        self.assertEqual(a['expand_count'],1)
        self.assertEqual(a['delete_count'],1)
        self.assertEqual(a['action_trace'][1]['dynamic_length'],64)
        self.assertEqual([t['x_after'] for t in a['action_trace']],[t['x_after'] for t in b['action_trace']])

if __name__=='__main__':unittest.main()
