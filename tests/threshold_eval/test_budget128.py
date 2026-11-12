import json
import os
import subprocess
import sys
import unittest

from test_threshold import engine
from experiments.context_foundation.adapter import Engine as Reference


class Budget128Tests(unittest.TestCase):
    def test_expansion_can_exceed_64_and_stops_at_128(self):
        e = engine(max_gen_len=128)
        e.model.special_token = e.tokenizer.expand_id
        value = e.decode([0], [1], 'case', 'nohead', 'left', 4, detailed=True)
        self.assertEqual(value['expand_count'], 64)
        self.assertEqual(len(value['completion_token_ids']), 128)
        self.assertEqual(value['final_remaining_masks'], 0)
        self.assertEqual(value['head_calls'], 0)
        self.assertLessEqual(max(t['dynamic_length'] for t in value['action_trace']), 128)

    def test_128_off_matches_reference_tokens_canvas_and_structure(self):
        for order in ('global', 'left'):
            a, b = engine(max_gen_len=128), engine(Reference, max_gen_len=128)
            a.model.special_token = b.model.special_token = 6
            v = a.decode([0], [1], 'case', 'nohead', order, 4, detailed=True)
            w = b.decode([0], [1], 'case', 'nohead', order, detailed=True)
            self.assertEqual(v['completion_token_ids'], w['completion_token_ids'])
            self.assertEqual(v['expand_count'], 64)
            self.assertEqual([t['x_after'] for t in v['action_trace']],
                             [t['x_after'] for t in w['action_trace']])

    def test_64_default_remains_capped_and_matches_explicit_64(self):
        a, b = engine(), engine(max_gen_len=64)
        a.model.special_token = b.model.special_token = 6
        v = a.decode([0], [1], 'case', 'nohead', 'left', 4, detailed=True)
        w = b.decode([0], [1], 'case', 'nohead', 'left', 4, detailed=True)
        self.assertEqual(v, w)
        self.assertEqual(v['expand_count'], 0)
        self.assertEqual(len(v['completion_token_ids']), 64)

    def test_64_matches_unmodified_parent_reference(self):
        source = subprocess.check_output(['git', 'show',
            '7557627f847960eb8500eea8570cdd3572ac0851:experiments/context_foundation/adapter.py'], text=True)
        namespace = {'__name__': 'parent_reference'}
        exec(compile(source, 'parent_reference.py', 'exec'), namespace)
        for order in ('global', 'left'):
            for method, arm in [('nohead', 'nohead'), ('trained', 'H4-old')]:
                a, b = engine(), engine(namespace['Engine'])
                v = a.decode([0], [1], 'case', method, order, 4, detailed=True)
                w = b.decode([0], [1], 'case', arm, order, detailed=True)
                for field in ('completion_token_ids', 'forward_count', 'head_calls',
                              'normal_token_count', 'expand_count', 'delete_count'):
                    self.assertEqual(v[field], w[field])
                self.assertEqual([t['x_after'] for t in v['action_trace']],
                                 [t['x_after'] for t in w['action_trace']])

    def test_task_selection_isolates_128_storage(self):
        code = ('import json; from experiments.threshold_eval.common import '
                'TASK,SHARED,DATASET,MAX_GEN_LEN; '
                'print(json.dumps([TASK,str(SHARED),DATASET,MAX_GEN_LEN]))')
        task = 'dreamon_multiline_threshold_128_20261112_v1'
        r = subprocess.run([sys.executable, '-c', code],
                           env=dict(os.environ, THRESHOLD_TASK_ID=task),
                           capture_output=True, text=True, check=True)
        value = json.loads(r.stdout)
        self.assertEqual(value, [task, '/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1', 'multi-line', 128])

    def test_audit_rejects_64_rows_in_128_task(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.threshold_eval import analyze
        from experiments.threshold_eval.common import configs
        c = configs()[0]
        key = 'MultiLineInfilling/HumanEval/0/L0_L1'
        r = dict(config=c['id'], task_id=key, method='nohead', execution_commit='v', passed=True,
                 completion_token_ids=[0], forward_count=1, head_calls=0, normal_token_count=1,
                 normal_submissions_histogram={'1': 1}, statistics=dict(extra_ordinary_attempted=0,
                 extra_ordinary_accepted=0, extra_ordinary_rejected=0), max_gen_len=64)
        sp = dict(configs=[c], cases=[key], timing_cases=[key], expected_quality=1,
                  expected_timing=3, protocol=dict(max_gen_len=128))
        with tempfile.TemporaryDirectory() as td, patch.object(analyze, 'OUT', Path(td)), \
             patch.object(analyze, 'SHARED', Path(td)), patch.object(analyze, 'spec', return_value=sp):
            (Path(td)/'READY.json').write_text(json.dumps(dict(execution_commit='v')))
            timed = [dict(r, repetition=i, generation_sec=.1) for i in range(3)]
            result = analyze.audit([r], timed)
            self.assertFalse(result['complete'])
            self.assertIn([c['id'], key, 'generation_budget'], result['invalid'])


if __name__ == '__main__':
    unittest.main()
