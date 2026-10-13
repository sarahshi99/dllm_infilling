import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.run_zero_training_transfer import open_store, save_case, read_cases, charge, ARMS

class ControllerTests(unittest.TestCase):
    def test_identity_deduplicates_and_distinguishes_target_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            db=open_store(Path(directory)/'rows.sqlite')
            row=dict(target='dreamcoder',policy='global_confidence',arm='C',task_id='p/0/1/2',seed=42)
            save_case(db,row)
            with self.assertRaises(sqlite3.IntegrityError): save_case(db,row)
            save_case(db,{**row,'target':'dream'})
            save_case(db,{**row,'seed':43})
            self.assertEqual(len(read_cases(db)),3)
            db.close()
    def test_global_budget_accumulates_sessions_without_double_charging_heartbeats(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'budget.json'
            path.write_text(json.dumps(dict(limit_gpu_job_sec=100,used_gpu_job_sec=0,sessions=[])))
            with patch('time.perf_counter',return_value=30):
                self.assertEqual(charge(path,'a',10,'dream'),80)
            with patch('time.perf_counter',return_value=40):
                self.assertEqual(charge(path,'a',10,'dream'),70)
            with patch('time.perf_counter',return_value=70):
                self.assertEqual(charge(path,'b',20,'diffucoder'),20)
            self.assertEqual(json.loads(path.read_text())['used_gpu_job_sec'],80)
    def test_matrix_has_eight_unique_configurations(self):
        self.assertEqual(len(ARMS),8)
        self.assertEqual(len(set(ARMS)),8)
        self.assertEqual({a for p,a in ARMS},set('ABCD'))

if __name__=='__main__': unittest.main()
