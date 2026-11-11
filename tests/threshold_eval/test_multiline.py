import unittest
import numpy as np
from experiments.threshold_eval import common
from experiments.threshold_eval.analyze import grouped


class MultilineTests(unittest.TestCase):
    def test_original_endpoint_span(self):
        self.assertEqual(common.case_metadata('MultiLineInfilling/HumanEval/3/L2_L9'),
                         dict(base_task_id='3',hole_span_lines=8,span_bucket='8+',crossline=True))
        self.assertEqual(common.case_metadata('SingleLineInfilling/HumanEval/3/L2')['hole_span_lines'],1)

    def test_bootstrap_only_present_groups(self):
        keys=['MultiLineInfilling/HumanEval/3/L0_L1']*2+['MultiLineInfilling/HumanEval/160/L0_L1']
        result=grouped([100,100,0],keys,None)
        self.assertAlmostEqual(result['line_mean'],200/3)
        self.assertEqual(result['task_mean'],50)
        self.assertEqual(result['line_ci95'],[0,100])

    def test_multiline_population_and_fixed_selection(self):
        from experiments.k4_batch_common import population
        rows=population('multi-line')
        audit=common.population_counts(rows)
        self.assertEqual(audit['cases'],5815)
        self.assertEqual(audit['crossline_cases'],4782)
        self.assertEqual(audit['span_layers']['8+'],dict(cases=1401,base_tasks=60))
        chosen=common.select_timing(rows)
        self.assertEqual(len(chosen),164)
        self.assertEqual(sum(common.case_metadata(k)['crossline'] for k in chosen),127)

class ResumeTests(unittest.TestCase):
    def test_role_lock_prevents_second_model(self):
        import fcntl,json,os,subprocess,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'READY.json').write_text('{}')
            with (root/'trained.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                run=subprocess.run(['bash',str(common.REPO/'scripts/start_threshold_worker.sh'),'--role','trained'],
                    env=dict(os.environ,THRESHOLD_SHARED=td),text=True,capture_output=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('already running',run.stdout)

    def test_report_only_commit_does_not_invalidate_execution(self):
        from unittest.mock import patch
        with patch('experiments.threshold_eval.common.subprocess.run') as run:
            common.verify_execution('execution')
            args=run.call_args.args[0]
            self.assertIn('execution',args)
            self.assertIn('experiments/threshold_eval/runner.py',args)
            self.assertFalse(any('analysis_outputs' in x for x in args))

class PopulationExportTests(unittest.TestCase):
    def test_crossline_summary_uses_matching_timing_denominator(self):
        from experiments.threshold_eval.analyze import population_summary
        keys=['MultiLineInfilling/HumanEval/3/L0_L1','MultiLineInfilling/HumanEval/160/L0_L7']
        summary,timed,pairs,nearest,hist=population_summary([],[],keys,keys,'crossline')
        self.assertEqual(len(summary),34)
        self.assertTrue(all(r['expected']==2 and r['base_tasks']==2 and r['timing_expected']==6 for r in summary))
        self.assertTrue(all(r['line_pass_pct'] is None and not r['exploratory_case_engineering_flag'] for r in summary))

    def test_generation_paths_include_input_and_sampling_dependencies(self):
        self.assertIn('experiments/k4_batch_common.py',common.GENERATION_PATHS)
        self.assertIn('experiments/context_foundation/adapter.py',common.GENERATION_PATHS)

class PipelineTests(unittest.TestCase):
    def test_score_then_timing_only_after_both_models_release(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.threshold_eval import runner
        class Ledger:
            def __init__(self,role):self.db=self
            def count(self,kind):return 1
            def close(self):pass
        sp=dict(configs=[dict(method='nohead'),dict(method='trained')],cases=['case'],expected_quality=2)
        with tempfile.TemporaryDirectory() as td,patch.object(runner,'SHARED',Path(td)),patch.object(runner,'spec',return_value=sp),patch.object(runner,'Ledger',Ledger),patch.object(runner,'role_running',return_value=False),patch.object(runner.subprocess,'call',return_value=0) as call,patch.object(runner.subprocess,'run') as run:
            self.assertEqual(runner.pipeline(),0)
            commands=[c.args[0] for c in call.call_args_list]
            self.assertIn('--score-only',commands[0])
            self.assertIn('timing',commands[1])
            self.assertIn('--publish',commands[2])
            run.assert_not_called()
