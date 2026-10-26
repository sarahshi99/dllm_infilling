import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.context_foundation.common import Store,atomic_json


class PipelineTest(unittest.TestCase):
    def test_empty_partial_export_is_auditable_and_does_not_invent_rows(self):
        root=Path(__file__).resolve().parents[2]
        spec=importlib.util.spec_from_file_location('context_pipeline_analysis',root/'analysis/analyze_context_foundation.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            (out/'cpu_tests.txt').write_text('OK\n')
            atomic_json(out/'source_audit.json',dict(status='passed'))
            store=Store(out/'test.sqlite')
            with patch.object(module,'OUT',out),patch.object(module,'CACHE',out/'cache'):
                matrix,table,comparisons,cf,chains=module.export(store)
                module.report(matrix,table,comparisons,cf,chains)
                self.assertEqual(matrix['generation']['missing'],4132)
                self.assertEqual(matrix['A1']['actual'],0)
                self.assertEqual(matrix['status'],'blocked_technical_or_resource')
                self.assertEqual(cf['paired_states'],0)
                self.assertTrue((out/'report.zh.md').exists())
                self.assertIsNone(table[0]['pass_pct'])
                self.assertIn('N/A',(out/'report.zh.md').read_text())
            store.db.close()


if __name__=='__main__':
    unittest.main()
