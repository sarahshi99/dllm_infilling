import unittest
import numpy as np
from experiments.threshold_eval.analyze import grouped


class AnalysisTests(unittest.TestCase):
    def test_base_task_cluster_weights_not_row_independence(self):
        keys=[];values=[]
        for g in range(164):
            for line in range(1 if g else 10):
                keys.append(f'SingleLineInfilling/HumanEval/{g}/L{line}')
                values.append(100 if g==0 else 0)
        w=np.random.default_rng(20261104).multinomial(164,np.full(164,1/164),size=10000)
        s=grouped(values,keys,w)
        self.assertAlmostEqual(s['line_mean'],1000/173)
        self.assertAlmostEqual(s['task_mean'],100/164)
        self.assertEqual(s['task_ci95'][0],0)
        self.assertGreater(s['task_ci95'][1],s['task_mean'])
        zero=grouped([0]*len(keys),keys,w)
        self.assertEqual(zero['task_ci95'],[0,0])

class ExportTests(unittest.TestCase):
    def test_partial_export_keeps_all_negative_groups_and_separate_timing(self):
        import csv,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.threshold_eval.common import configs
        from experiments.threshold_eval.analyze import analyze
        quality=[];keys=['SingleLineInfilling/HumanEval/0/L0','SingleLineInfilling/HumanEval/1/L0']
        for c in configs():
            for i,key in enumerate(keys):
                quality.append(dict(config=c['id'],task_id=key,base_task_id=str(i),method=c['method'],execution_commit='v',
                    passed=bool(c['k']==1 or i==0),completion_token_ids=[0],forward_count=1,normal_token_count=1,
                    head_calls=int(c['method']=='trained'),expand_count=0,delete_count=0,final_remaining_masks=0,
                    statistics=dict(extra_ordinary_attempted=0,extra_ordinary_accepted=0,extra_ordinary_rejected=0),
                    normal_submissions_histogram={'1':1},generation_sec=.1,peak_memory_allocated_bytes=1,peak_memory_reserved_bytes=1))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'READY.json').write_text(json.dumps({'execution_commit':'v'}));(p/'commands.txt').write_text('test\n')
            sp=dict(configs=configs(),cases=keys,timing_cases=keys)
            with patch('experiments.threshold_eval.analyze.OUT',p),patch('experiments.threshold_eval.analyze.SHARED',p),patch('experiments.threshold_eval.common.SHARED',p),patch('experiments.threshold_eval.analyze.spec',return_value=sp),patch('experiments.threshold_eval.analyze.load_results',return_value=(quality,[])),patch('experiments.threshold_eval.analyze.charts'),patch('experiments.threshold_eval.analyze.update_checkpoint'):
                analyze(partial=True)
            with (p/'summary.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),34)
            with (p/'timing.csv').open() as f:timed=list(csv.DictReader(f))
            self.assertEqual(len(timed),34)
            self.assertTrue(all(r['timing_n']=='0' for r in rows))
            self.assertFalse(json.loads((p/'integrity_audit.json').read_text())['complete'])
            self.assertTrue(any(float(r['vs_k1_task_delta_pp'])<0 for r in rows))

class CheckpointTests(unittest.TestCase):
    def test_updates_task_block_and_preserves_history(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.threshold_eval.analyze import update_checkpoint
        from experiments.threshold_eval.common import TASK
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'docs/paper_agent').mkdir(parents=True)
            for name in ('pause_checkpoint.current.md','paper_agent_dashboard.zh.md'):
                (root/'docs/paper_agent'/name).write_text('# Historical checkpoint\nuser history\n')
            a=dict(quality_actual=0,scored=0,timing_actual=0,expected_execution_version='x')
            with patch('experiments.threshold_eval.analyze.REPO',root),patch('experiments.threshold_eval.analyze.OUT',root):
                update_checkpoint(a,'initial');update_checkpoint(a,'running');update_checkpoint(a,'completed')
            text=(root/'docs/paper_agent/pause_checkpoint.current.md').read_text()
            self.assertEqual(text.count(f'<!-- {TASK} -->'),1)
            self.assertIn('状态：completed',text)
            self.assertNotIn('状态：initial',text)
            self.assertIn('# Historical checkpoint\nuser history\n',text)
