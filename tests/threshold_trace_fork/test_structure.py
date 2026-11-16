import unittest
from tests.threshold_eval.test_threshold import engine
from experiments.threshold_trace_fork.engine import Engine,clone_state


class StructureInterventions(unittest.TestCase):
    def setup_snapshot(self):
        e=engine(Engine);saved=[];old=e.model.forward
        def forward(x,*args,**kwargs):
            out=old(x,*args,**kwargs)
            out.logits.fill_(-100.)
            out.logits[...,0]=0.
            out.logits[...,5]=100.
            # The first spatial candidate commits an ordinary token before EOS.
            out.logits[:,0,0]=200.
            return out
        e.model.forward=forward
        r=e.decode([0],[1],'case','nohead','global',4,.95,capture=lambda s:saved.append(clone_state(s)) if s['slot']==1 and not saved else None)
        return e,r,saved[0]

    def test_defer_retains_pending_ordinary_prefix_and_rejudges_next_round(self):
        e,factual,s=self.setup_snapshot()
        r=e.decode(snapshot=s,intervention=dict(kind='structure',action='defer'))
        t=r['action_trace'][0]
        self.assertEqual(t['round_end'],'structure_deferred')
        self.assertEqual(t['normal_submissions'],1)
        self.assertEqual(t['delete'],[])
        self.assertGreater(r['forward_count'],factual['forward_count'])
        self.assertGreater(r['delete_count'],0)

    def test_veto_resampling_scores_with_original_denominator(self):
        e,_,s=self.setup_snapshot()
        r=e.decode(snapshot=s,intervention=dict(kind='structure',action='veto'))
        ev=r['action_trace'][0]['events'][1]
        self.assertEqual(ev['raw'],5)
        self.assertEqual(ev['resampled_base'],0)
        self.assertTrue(ev['rejected'])
        self.assertLess(ev['candidate_probability'],.01)
        self.assertEqual(r['action_trace'][0]['delete'],[])

if __name__=='__main__':unittest.main()
