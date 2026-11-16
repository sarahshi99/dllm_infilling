import unittest
import torch
from experiments.threshold_trace_fork.teacher_numeric import distances


class TeacherNumerics(unittest.TestCase):
    def test_full_distribution_kl_survives_probability_underflow(self):
        p=torch.tensor([0.,-120.]).log_softmax(-1)
        q=torch.tensor([0.,-100.]).log_softmax(-1)
        f=torch.tensor([-1.,0.]).log_softmax(-1)
        self.assertEqual(float(p.exp()[1]),0.)
        result=distances(f,p,q)
        self.assertGreater(result['kl_fresh_p'],80.)
        self.assertLess(result['kl_fresh_q'],result['kl_fresh_p'])

    def test_zero_mass_illegal_actions_do_not_create_nan(self):
        f=torch.tensor([0.,-torch.inf]).log_softmax(-1)
        result=distances(f,f,f)
        self.assertEqual(result,dict(kl_fresh_p=0.,kl_fresh_q=0.,tv_fresh_p=0.,tv_fresh_q=0.))

if __name__=='__main__':unittest.main()
