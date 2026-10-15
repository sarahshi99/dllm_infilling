"""CPU structural coordinate regression tests, with explicit canvas fixtures."""
import unittest
import torch
from tests.test_dreamon_markov_k2 import DreamOnMarkovK2Test,deterministic_sample,FixedModel

class CoordinateDecoderTest(unittest.TestCase):
    decode=DreamOnMarkovK2Test.decode
    def mixed(self,k=2,policy='confidence_global',mode='legacy',tokens=None,capacity=None):
        tokens=tokens or [4]+[2]*(k-1)
        def sample(logits,**kwargs):
            c,p=deterministic_sample(logits,**kwargs)
            p[:]=torch.tensor(tokens[:len(p)])
            return torch.arange(len(p),dtype=torch.float),p
        return self.decode(prefix_ids=[1]*11,suffix_ids=[2],min_gen_len=k,max_gen_len=k+(capacity or k),
                           requested_k=k,max_tokens=64,steps=1,policy=policy,k2_action_mode=mode,sample_tokens_fn=sample,
                           diagnostics=False,capture_actions=True)
    def test_k2_expand_then_eos_moves_delete_eligibility(self):
        r=self.mixed();self.assertEqual(r['completion_token_ids'],[3,3])
        self.assertEqual((r['expand_count'],r['delete_count'],r['final_dynamic_length']),(1,1,2))
    def test_k4_expand_then_broadcast_eos(self):
        r=self.mixed(k=4);self.assertEqual(r['completion_token_ids'],[3,3])
        self.assertEqual((r['expand_count'],r['delete_count']),(1,3))
    def test_nonadjacent_k2_fallback_expand_and_eos(self):
        def sample(logits,**kwargs):
            c,p=deterministic_sample(logits,**kwargs);p[:]=torch.tensor([4,5,2]);return torch.tensor([.9,.1,.8]),p
        r=self.decode(prefix_ids=[1]*11,suffix_ids=[2],min_gen_len=3,max_gen_len=4,requested_k=2,
            steps=1,policy='confidence_global',sample_tokens_fn=sample,diagnostics=False,capture_actions=True)
        self.assertEqual(r['completion_token_ids'],[3,3,3]);self.assertEqual(r['delete_count'],1)
    def test_deleted_position_left_of_expand(self):
        # EOS broadcast also overwrites the later expand: original broadcast wins.
        r=self.mixed(tokens=[2,4]);self.assertEqual(r['completion_token_ids'],[])
        self.assertEqual((r['expand_count'],r['delete_count']),(0,2))
    def test_multiple_expands_shift_delete_twice(self):
        r=self.mixed(k=4,tokens=[4,4,2,5]);self.assertEqual(r['completion_token_ids'],[3,3,3,3])
        self.assertEqual((r['expand_count'],r['delete_count']),(2,2))
    def test_capacity_clipping_still_deletes_original_eligible_eos(self):
        r=self.mixed(k=4,tokens=[4,4,2,5],capacity=1)
        self.assertEqual(r['expand_count'],1);self.assertEqual(r['delete_count'],2)
        self.assertEqual(r['final_dynamic_length'],3)
    def test_structural_barriers_leave_later_drafts_uncommitted(self):
        for policy in ['confidence_global','left_to_right_frontier']:
            r=self.mixed(k=4,policy=policy,mode='sequential_pair')
            self.assertEqual((r['expand_count'],r['delete_count']),(1,0))
            self.assertEqual(r['completion_token_ids'],[3]*5)

class EligibilityBoundsTest(unittest.TestCase):
    def test_insertion_right_of_delete_keeps_left_coordinate(self):
        from experiments.dreamon_singleline_order_parallel import _expand_delete_eligibility
        old=torch.tensor([[False,True,False,False,False,False]])
        actual=_expand_delete_eligibility(old,4,6)
        self.assertEqual(actual.tolist(),[[False,True,False,False,False,False]])
    def test_truncate_eligibility_with_canvas_and_never_mark_new_masks(self):
        from experiments.dreamon_singleline_order_parallel import _expand_delete_eligibility
        old=torch.tensor([[False,False,False,True,False,True]])
        actual=_expand_delete_eligibility(old,2,6)
        self.assertEqual(actual.tolist(),[[False,False,False,False,True,False]])
    def test_no_mixed_actions_matches_actual_baseline_and_rng(self):
        import subprocess,types,sys
        name='coordinate_v1_decoder_cpu'
        module=types.ModuleType(name);sys.modules[name]=module
        exec(compile(subprocess.check_output(['git','show','dbdd0bd:experiments/dreamon_singleline_order_parallel.py'],text=True),'<baseline decoder>','exec'),module.__dict__)
        from tests.test_dreamon_markov_k2 import FakeTokenizer
        from experiments.dreamon_singleline_order_parallel import decode_with_policy
        for policy in ['confidence_global','left_to_right_frontier']:
            for k,mode in [(1,'legacy'),(2,'sequential_pair'),(4,'sequential_pair'),(4,'legacy')]:
                for token in [2,5]:
                    results=[];states=[]
                    for decoder in [module.decode_with_policy,decode_with_policy]:
                        torch.manual_seed(987)
                        results.append(decoder(model=FixedModel(normal_id=token),tokenizer=FakeTokenizer(),sample_tokens_fn=deterministic_sample,
                            prefix_ids=[1],suffix_ids=[6,2],min_gen_len=7,max_gen_len=7,max_tokens=64,steps=8,expand_budget=64,
                            temperature=.2,top_p=.9,top_k=None,requested_k=k,policy=policy,device='cpu',k2_action_mode=mode,diagnostics=False,capture_actions=True))
                        states.append(torch.random.get_rng_state())
                    for field in ['completion_token_ids','forward_count','expand_count','delete_count','normal_token_count','final_dynamic_length']:
                        self.assertEqual(results[0][field],results[1][field])
                    self.assertTrue(torch.equal(*states))
    def test_published_minimal_fixtures_have_expected_delete_coordinates(self):
        r=CoordinateDecoderTest().mixed(k=4)
        self.assertEqual(r['action_trace'][0]['expanded_absolute'],[11])
        self.assertEqual(r['action_trace'][0]['deleted_absolute'],[13,14,15])

class PortableAnalysisTest(unittest.TestCase):
    def test_recovery_uses_original_nohead_and_zero_different_denominators(self):
        from analysis.analyze_structure_coordinate_fix import recovery
        def group(n):return {f'SingleLineInfilling/HumanEval/{i}/L0':{'passed':i<n} for i in range(10)}
        self.assertAlmostEqual(recovery(group(7),group(2),group(9))['line_recovery_pct'],500/7)
        self.assertAlmostEqual(recovery(group(7),group(4),group(9))['line_recovery_pct'],60)
        self.assertIsNone(recovery(group(7),group(9),group(9))['line_recovery_pct'])
    def test_cluster_bootstrap_preserves_task_group_weighting(self):
        from analysis.analyze_structure_coordinate_fix import bootstrap_pair,span
        keys=['MultiLineInfilling/HumanEval/0/L0_L0','MultiLineInfilling/HumanEval/0/L0_L1','MultiLineInfilling/HumanEval/0/L0_L2','MultiLineInfilling/HumanEval/1/L0_L0']
        left={k:{'passed':i<3} for i,k in enumerate(keys)};right={k:{'passed':False} for k in keys}
        stats=bootstrap_pair(left,right)
        self.assertEqual(stats['line_delta_pp'],75);self.assertEqual(stats['base_task_equal_delta_pp'],50)
        self.assertEqual(stats['cluster_count'],2);self.assertEqual(span('MultiLineInfilling/HumanEval/0/L1_L8'),8)
    def test_replay_distinguishes_stale_and_transformed_delete_positions(self):
        from analysis.dreamon_structure_replay import Canvas,apply_round
        state=Canvas([1]*11+[3,3]+[2,3],2,64,11,1,15)
        trace={'submitted_absolute':[[11,4],[12,2]],'selected_raw_absolute':[[11,4],[12,2]],'barrier':None}
        args=dict(stage='A',arm='A',policy='global_confidence',mask_id=3,eos_id=2,expand_id=4)
        old,oa=apply_round(state,trace,**args,fixed=False);new,na=apply_round(state,trace,**args,fixed=True)
        self.assertEqual(old.x[11:11+old.length],[3,3,2]);self.assertEqual(new.x[11:11+new.length],[3,3])
        self.assertEqual(oa['delete_indices'],[]);self.assertEqual(na['delete_indices'],[13])
