import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from experiments.context_foundation import evidence
from experiments.context_foundation.adapter import FrozenHeadAdapter, Engine
from experiments.context_foundation.common import Store, Budget, BudgetExceeded
from experiments.dreamon_markov_head_training import MarkovHead
from test_adapter import Toy
from types import SimpleNamespace


class EvidenceTest(unittest.TestCase):
    def test_deploy_infinite_kl_is_not_epsilon_hidden(self):
        a=torch.tensor([0.,-20.,-30.])
        b=torch.tensor([-20.,0.,-30.])
        m=evidence.pair_metrics(a,b,{2})
        self.assertEqual(m['kl_deploy'],'infinite')
        self.assertGreater(m['tv_deploy'],.99)
        self.assertTrue(np.isfinite(m['kl_T1']))

    def test_special_zero_delta_can_change_special_probability(self):
        a=torch.tensor([0.,0.,0.])
        b=torch.tensor([3.,0.,0.])
        m=evidence.pair_metrics(a,b,{2})
        self.assertNotEqual(m['special_teacher_T1'],m['special_estimate_T1'])

    @torch.inference_mode()
    def test_lossless_state_dcd_log_authority_and_counterfactual(self):
        model=Toy()
        model.lm_head=torch.nn.Linear(4,24,bias=False)
        model.lm_head.weight.data.copy_(torch.arange(96).reshape(24,4)*.001)
        tokenizer=SimpleNamespace(mask_id=4,eos_id=5,expand_id=6,decode=lambda ids,**_:str(ids))
        head=MarkovHead(vocab_size=24,rank=256,structural_token_ids=[4,5,6])
        e=Engine(model,tokenizer,FrozenHeadAdapter(head),[4,5,6])
        captured=[]
        def observe(kind,s):
            if kind=='query' and int(s['history_mask'].sum()) in (1,2) and len(captured)<2:
                captured.append({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in s.items()})
        e.decode([0],[1],'evidence','nohead',order='frontier',observer=observe)
        self.assertEqual(len(captured),2)
        with tempfile.TemporaryDirectory() as td, patch.object(evidence,'CACHE',Path(td)):
            metas=[]
            for s in captured:
                meta=dict(state_id=evidence.state_id(s),group='g',split='train',order='frontier',span='single',
                    history_length=int(s['history_mask'].sum()),arm='Z4')
                meta=evidence.save_state(s,meta)
                # Encoding itself must be replayable, not only tensor values.
                self.assertEqual(evidence.save_state(s,meta)['raw_sha256'],meta['raw_sha256'])
                relabelled=dict(s,order='different-order',arm='different-head',round_index=s['round_index']+10)
                self.assertEqual(evidence.state_id(s),evidence.state_id(relabelled))
                restored=evidence.load_state(meta,'cpu')
                self.assertTrue(torch.equal(restored['old_hidden'],s['old_hidden']))
                metas.append(evidence.measure_state(e,meta))
            dcd=evidence.dcd_one(e,metas[0])
            with np.load(Path(td)/dcd['raw_path'],allow_pickle=False) as z:
                self.assertEqual(z['fresh_cond2'].shape,(16,16))
                self.assertTrue(np.isfinite(z['fresh_logcond2']).all())
                self.assertFalse(any(np.asarray(z[k]).dtype==object for k in z.files))
                self.assertTrue(np.all(z['fresh_candidate_mass']<=1.000001))
                self.assertFalse(set(z['candidate_ids_1']) & {4,5,6})
            cf=evidence.counterfactual(e,metas[1])
            self.assertEqual(cf['status'],'complete')
            self.assertNotEqual(cf['original_token'],cf['alternative_token'])
            self.assertEqual(cf['head_branch_tv_T1'],0)

    def test_budget_never_refills_and_stage_cap(self):
        import experiments.context_foundation.common as common
        with tempfile.TemporaryDirectory() as td, patch.object(common,'OUT',Path(td)):
            common.atomic_json(Path(td)/'budget.local.json',dict(used_sec=14400,stage_sec={'mechanism':14400},resident=False))
            b=Budget()
            b.switch('mechanism')
            with self.assertRaises(BudgetExceeded):
                b.check()
            b.switch('generation')
            b.check()
            b.close()
            c=Budget()
            self.assertGreaterEqual(c.used,14400)
            c.close()

    def test_group_bootstrap_and_nonpositive_recovery(self):
        root=Path(__file__).resolve().parents[2]
        spec=importlib.util.spec_from_file_location('context_analysis',root/'analysis/analyze_context_foundation.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows=[dict(task_id='a',base_task_id='1',arm=a,passed=p) for a,p in [('K1',True),('Z4',True),('H4-old',False)]]
        result=module.learning_recovery(rows)
        self.assertIsNone(result['value'])
        delta=module.paired_quality(rows,'H4-old','Z4')
        self.assertEqual(delta['harm'],1)
        self.assertEqual(delta['mean'],-100)


if __name__=='__main__':
    unittest.main()
