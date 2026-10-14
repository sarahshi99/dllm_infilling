import unittest
from unittest.mock import patch
import torch
from tests.test_dreamon_markov_k2 import DreamOnMarkovK2Test, FixedModel, deterministic_sample
from experiments.dreamon_singleline_order_parallel import decode_with_policy, GLOBAL_CONFIDENCE

class ChainHead(torch.nn.Module):
    def __init__(self, zero=False, structural=False):
        super().__init__(); self.inputs=[]; self.zero=zero; self.structural=structural
    def forward(self, ids):
        self.inputs.extend(ids.tolist()); out=torch.zeros((len(ids),8))
        if not self.zero:
            out[:,5]=-100
            out[:,2 if self.structural else (6 if int(ids[0])==5 else 7)]=100
        return out

class K4Test(unittest.TestCase):
    decode=DreamOnMarkovK2Test.decode
    def run4(self, **kw):
        return self.decode(requested_k=4,min_gen_len=4,max_gen_len=4,diagnostics=False,capture_actions=True,**kw)
    def test_third_and_fourth_depend_on_final_predecessor(self):
        head=ChainHead(); r=self.run4(markov_head=head,markov_lambda=1)
        self.assertEqual(head.inputs,[5,6,7]); self.assertEqual(r['completion_token_ids'],[5,6,7,7])
        self.assertEqual(r['forward_count'],1); self.assertEqual(r['markov_enabled_count'],3)
    def test_nonadjacent_chains_do_not_cross_gap(self):
        def sample(logits,**kw):
            c,p=deterministic_sample(logits,**kw)
            return torch.tensor([.8,.9,.1,.7,.6][:len(c)]),p
        head=ChainHead(); r=self.decode(requested_k=4,min_gen_len=5,max_gen_len=5,steps=1,policy=GLOBAL_CONFIDENCE,markov_head=head,markov_lambda=1,diagnostics=False,capture_actions=True,sample_tokens_fn=sample)
        self.assertEqual(head.inputs,[5,5]); self.assertEqual(r['completion_token_ids'],[5,6,3,5,6])
    def test_corrected_structure_stops_entire_round_without_execution(self):
        head=ChainHead(structural=True); r=self.run4(markov_head=head,markov_lambda=1,steps=1)
        self.assertEqual(r['completion_token_ids'],[5,3,3,3]); self.assertEqual(r['delete_count'],0)
        self.assertEqual(r['corrected_structural_stop_count'],1)
    def test_original_structure_commits_prefix_then_stops(self):
        def sample(logits,**kw):
            c,p=deterministic_sample(logits,**kw); p[:]=torch.tensor([5,2,5,5][:len(p)]); return c,p
        r=self.run4(sample_tokens_fn=sample,steps=1)
        self.assertEqual(r['normal_token_count'],1); self.assertEqual(r['structural_stop_count'],1)
    def test_zero_really_calls_and_rng_isolation(self):
        def sample(logits,**kw):
            torch.rand(3); return deterministic_sample(logits,**kw)
        torch.manual_seed(99); self.run4(sample_tokens_fn=sample); state=torch.random.get_rng_state()
        head=ChainHead(zero=True); torch.manual_seed(99)
        r=self.run4(markov_head=head,markov_lambda=1,sample_tokens_fn=sample)
        self.assertEqual(len(head.inputs),3); self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
    def test_tail_less_than_k(self):
        head=ChainHead(); r=self.decode(requested_k=4,markov_head=head,markov_lambda=1,diagnostics=False)
        self.assertEqual(r['markov_enabled_count'],1); self.assertEqual(r['forward_count'],1)
    def test_expand_budget_never_negative_native_multiple_actions(self):
        # Force one initial delete to create capacity, then multiple expands.
        calls=[0]
        def sample(logits,**kw):
            calls[0]+=1; c,p=deterministic_sample(logits,**kw)
            p[:]=3
            if calls[0]==1:p[-1]=2
            elif calls[0]==2:p[:]=4
            else:p[:]=5
            return c,p
        r=self.run4(policy=GLOBAL_CONFIDENCE,k2_action_mode='legacy',sample_tokens_fn=sample,expand_budget=1)
        self.assertEqual(r['expand_count'],1)
        self.assertGreaterEqual(r['expand_budget_remaining'],0)
        self.assertLessEqual(r['final_dynamic_length'],4)

class K4MaskAndAliasTest(unittest.TestCase):
    decode=DreamOnMarkovK2Test.decode
    def test_corrected_mask_cuts_dependency_and_continues_round(self):
        class MaskHead(ChainHead):
            def forward(self,ids):
                out=super().forward(ids)
                if len(self.inputs)==1:out[:]=0;out[:,3]=100
                return out
        head=MaskHead()
        r=self.decode(requested_k=4,min_gen_len=4,max_gen_len=4,steps=1,markov_head=head,markov_lambda=1,diagnostics=False)
        self.assertEqual(r['completion_token_ids'],[5,3,5,6])
        self.assertEqual(head.inputs,[5,5]);self.assertEqual(r['structural_stop_count'],0)
    def test_frontier_nohead_native_and_chain_round_equivalence(self):
        for normal in [2,5]:
            values=[]
            for mode in ['legacy','sequential_pair']:
                values.append(self.decode(requested_k=4,min_gen_len=7,max_gen_len=7,model=FixedModel(normal_id=normal),k2_action_mode=mode,diagnostics=False,capture_actions=True))
            a,b=values
            self.assertEqual(a['completion_token_ids'],b['completion_token_ids'])
            self.assertEqual(a['forward_count'],b['forward_count'])
            for left,right in zip(a['action_trace'],b['action_trace']):
                for field in ['selected_raw_absolute','submitted_absolute','normal','expand','delete']:
                    self.assertEqual(left[field],right[field])

class PersistenceTest(unittest.TestCase):
    def test_generated_output_survives_eval_interruption_and_failure_is_complete(self):
        import tempfile,sqlite3,json
        from pathlib import Path
        from experiments.run_k4_multiline_stage import open_store,records
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rows.sqlite';db=open_store(path)
            row=dict(policy='global_confidence',arm='D',task_id='t',output='fixed',row_seed=42)
            with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,NULL)',('global_confidence','D','t',json.dumps(row)))
            db.close();db=open_store(path)
            self.assertEqual(records(db)[0]['output'],'fixed')
            self.assertNotIn('passed',records(db)[0])
            with db:db.execute('UPDATE cases SET evaluated=?',(json.dumps(dict(passed=False,failure_reason='timed out')),))
            db.close();db=open_store(path)
            self.assertFalse(records(db)[0]['passed'])
            self.assertEqual(db.execute('SELECT count(*) FROM cases WHERE evaluated IS NOT NULL').fetchone()[0],1)
            with self.assertRaises(sqlite3.IntegrityError):
                with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,NULL)',('global_confidence','D','t',json.dumps(row)))
            db.close()

class BarrierBudgetOrderTest(unittest.TestCase):
    decode=DreamOnMarkovK2Test.decode
    def test_late_chain_cannot_consume_first_chain_expansion_budget(self):
        # Create two vacancies by deleting the final two positions, then select
        # two expand drafts with later position ranked first by confidence.
        calls=[0]
        def sample(logits,**kw):
            calls[0]+=1;c,p=deterministic_sample(logits,**kw)
            if calls[0]==1:
                p[:]=3;p[-2:]=2
                return torch.arange(len(p),dtype=torch.float),p
            if calls[0]==2:
                p[:]=4
                return torch.arange(len(p),dtype=torch.float),p
            p[:]=5;return c,p
        r=self.decode(requested_k=4,min_gen_len=4,max_gen_len=4,expand_budget=1,policy=GLOBAL_CONFIDENCE,
            steps=2,sample_tokens_fn=sample,diagnostics=False,capture_actions=True)
        # First barrier deletion at position2 broadcasts the tail. At round2,
        # the first spatial expand executes, despite its lower global rank.
        self.assertEqual(r['expand_count'],1)
        self.assertEqual(r['action_trace'][1]['barrier'][0],0)
        self.assertEqual(r['expand_budget_remaining'],0)
