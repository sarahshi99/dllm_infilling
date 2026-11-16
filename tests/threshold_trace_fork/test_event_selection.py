import unittest
from experiments.threshold_trace_fork.forks import first_action_difference


class EventSelectionTests(unittest.TestCase):
    def test_rejected_proposal_change_is_not_an_executed_action_change(self):
        def event(slot,token,rejected):
            return dict(slot=slot,position=slot+10,chain=slot,raw=token,final=token,
                        processed=True,rejected=rejected,committed=not rejected,
                        reason='threshold_rejected' if rejected else 'ordinary_accepted')
        a=dict(action_trace=[dict(round=0,x0=[0],selected=[10,11],events=[event(0,3,True),event(1,5,True)])])
        b=dict(action_trace=[dict(round=0,x0=[0],selected=[10,11],events=[event(0,4,True),event(1,5,False)])])
        t,x,y=first_action_difference(a,b)
        self.assertEqual(x['slot'],1)
        self.assertEqual(y['slot'],1)

if __name__=='__main__':unittest.main()
