import copy
import unittest
from tests.threshold_eval.test_threshold import engine
from experiments.threshold_trace_fork.engine import Engine, clone_state, integrity


class ReplayTests(unittest.TestCase):
    def test_original_and_restored_trajectory(self):
        for method in ('nohead','trained'):
            for token in (None,5):
                old=engine(); old.model.special_token=token
                ref=old.decode([0],[1],'case',method,'global',4,.95,detailed=True)
                e=engine(Engine); e.model.special_token=token
                saved=[]
                def capture(s):
                    if not saved and (s['step']>=2 or token==5): saved.append(clone_state(s))
                got=e.decode([0],[1],'case',method,'global',4,.95,capture=capture)
                for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count','statistics','normal_submissions_histogram','stop_reason','sampling_key_sha256'):
                    self.assertEqual(ref[field],got[field],field)
                for a,b in zip(ref['action_trace'],got['action_trace']):
                    for field in ('selected','submitted','barrier','expand','delete','x0','x_after'):
                        self.assertEqual(a[field],b[field],field)
                integrity(got)
                restored=e.decode(snapshot=saved[0])
                self.assertEqual(got,restored)

    def test_snapshot_immutable_and_current_event_seed(self):
        e=engine(Engine); saved=[]
        def capture(s):
            if not saved and s['step']==1: saved.append(clone_state(s))
        e.decode([0],[1],'case','trained','global',4,.95,capture=capture)
        s=saved[0]; before=copy.deepcopy(s)
        e.decode(snapshot=s,continuation_seed=43)
        self.assertTrue((s['x']==before['x']).all())
        self.assertTrue((s['tokens']==before['tokens']).all())

    def test_rejected_chain_set_matches_original(self):
        # Original-space traversal test above exercises same-chain skip on every round.
        e=engine(Engine)
        got=e.decode([0],[1],'case','nohead','global',4,.95)
        self.assertGreater(got['statistics']['chain_skipped'],0)
        integrity(got)

if __name__=='__main__': unittest.main()

class InterventionTests(unittest.TestCase):
    def test_nonmonotonic_order_preserves_spatial_rejection_chains(self):
        from unittest.mock import patch
        e=engine(Engine);snap=[]
        from experiments.threshold_trace_fork.engine import select_positions as original
        def choose(positions,confidence,requested_k,policy):
            if len(positions)>=6:return [positions[i] for i in ((0,1,4,5) if requested_k==4 else (4,5))]
            return original(positions,confidence,requested_k=requested_k,policy=policy)
        with patch('experiments.threshold_trace_fork.engine.select_positions',side_effect=choose):
            factual=e.decode([0],[1],'case','nohead','global',4,.95,capture=lambda s:snap.append(clone_state(s)) if not snap else None)
        s=snap[0]
        altered=e.decode(snapshot=s,intervention=dict(kind='order',a2_first=True,a2_exempt=False))
        t=altered['action_trace'][0]
        self.assertEqual(t['processing_order'],[5,1,2,6])
        self.assertEqual(t['chains'],[0,0,1,1])
        self.assertEqual(t['events'][-1]['reason'],'chain_skipped')
        self.assertEqual(t['exempt_position'],1)
        self.assertTrue(t['exempt_reached'])

    def test_one_rejection_nonpropagation_rechecks_later_candidate(self):
        e=engine(Engine);saved=[]
        def capture(s):
            if not saved and s['step']==0 and s['slot']==1:saved.append(clone_state(s))
        factual=e.decode([0],[1],'case','nohead','global',4,.95,capture=capture)
        s=saved[0];r=e.decode(snapshot=s,intervention=dict(kind='no_propagation'))
        events=r['action_trace'][0]['events']
        self.assertTrue(events[1]['rejected']);self.assertTrue(events[2]['rejected'])
        self.assertEqual(events[3]['reason'],'chain_skipped')
        self.assertEqual(factual['completion_token_ids'],r['completion_token_ids'])

    def test_resume_never_renews_256_round_budget(self):
        e=engine(Engine);e.model.special_token=4;saved=[]
        r=e.decode([0],[1],'case','nohead','global',4,.95,capture=lambda s:saved.append(clone_state(s)) if s['step']==255 and not saved else None)
        self.assertEqual(r['forward_count'],256)
        self.assertEqual(e.decode(snapshot=saved[0]),r)
        self.assertEqual(r['stop_reason'],'step_budget')

    def test_head_four_arms_factual_and_unchanged_candidate(self):
        from experiments.threshold_trace_fork.forks import distributions
        for method,factual_arm in [('nohead','PP'),('trained','QQ')]:
            e=engine(Engine);saved=[]
            r=e.decode([0],[1],'case',method,'global',4,.95,capture=lambda s:saved.append(clone_state(s)) if s['slot']==1 and not saved else None)
            s=saved[0];calls=e.call_count();p,q,yp,yq=distributions(e,s)
            self.assertEqual(e.call_count(),calls)
            for arm in ('PP','PQ','QP','QQ'):
                cs,ss=arm.lower()
                got=e.decode(snapshot=s,intervention=dict(kind='head',candidate=yp if cs=='p' else yq,score_logits=p if ss=='p' else q,candidate_source=cs,score_source=ss),continuation_seed=42)
                if arm==factual_arm:self.assertEqual(got,r)

    def test_midround_snapshot_restore(self):
        for method in ('nohead','trained'):
            e=engine(Engine);saved=[]
            r=e.decode([0],[1],'case',method,'global',4,.95,capture=lambda s:saved.append(clone_state(s)) if s['slot']==1 and not saved else None)
            self.assertTrue(saved[0]['committed'])
            self.assertEqual(e.decode(snapshot=saved[0]),r)
