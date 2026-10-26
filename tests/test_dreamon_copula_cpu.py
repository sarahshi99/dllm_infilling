"""Substantive CPU tests; no torch, model, CUDA, generation or external data."""
import json
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import numpy as np

from analysis.dreamon_copula_cpu import (
    InputError, analyze_package, atomic_json, cluster_bootstrap, conditional_kl,
    ipf, kl, lognormalize, sha256, verify_package,
)
from scripts import consume_dreamon_copula_cpu as consumer


def log(x):
    with np.errstate(divide='ignore'):
        return np.log(np.asarray(x, dtype=np.float64))


def fixture(root, n=4):
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    d = root / 'inputs'
    d.mkdir()
    rng = np.random.default_rng(22)
    a = {'state_id': np.array([f's{i}' for i in range(n)]),
         'candidate_ids_1': np.tile(np.arange(16), (n, 1)),
         'candidate_ids_2': np.tile(np.arange(16, 32), (n, 1))}
    for prob, lp, shape in [('old_p1', 'old_logp1', (n, 16)),
                            ('old_p2', 'old_logp2', (n, 16)),
                            ('head_cond2', 'head_logcond2', (n, 16, 16)),
                            ('fresh_cond2', 'fresh_logcond2', (n, 16, 16))]:
        x = rng.uniform(.2, 1, shape)
        x = .98 * x / x.sum(axis=-1, keepdims=True)
        a[prob], a[lp] = x, np.log(x)
    for j in (1, 2):
        a[f'old_candidate_mass{j}'] = a[f'old_p{j}'].sum(-1)
        a[f'old_special_mass{j}'] = np.full(n, .001)
    for prefix in ('head', 'fresh'):
        a[prefix + '_candidate_mass'] = a[prefix + '_cond2'].sum(-1)
        a[prefix + '_special_mass'] = np.full((n, 16), .001)
    np.savez(d / 'states.npz', **a)
    metadata = [dict(state_id=f's{i}', group=f'g{i//2}', split='train',
                     source_state_id=f'parent{i//2}', order='global' if (i//2) % 2 == 0 else 'frontier',
                     span='single' if i % 2 else 'crossline') for i in range(n)]
    (d / 'meta.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metadata))
    buckets = {f'{o}/{s}': 64-sum(x['order']==o and x['span']==s for x in metadata)
               for o in ('global', 'frontier') for s in ('single', 'crossline')}
    manifest = dict(contract_id='dreamon-context-v1-20261015', schema_version=1,
                    status='partial', source_commit='a'*40,
                    producer_branch='codex/dreamon-context-foundation-20261015-v1',
                    count=n, expected=256, coverage_gap=256-n, missing_buckets=buckets,
                    budget_stop={'stopped': True, 'reason': 'toy budget'}, T=1,
                    probability_authority='full-vocabulary log_softmax before extraction',
                    model={'revision': 'b'*40}, model_shards_sha256={'model': 'c'*64},
                    head={'sha256': 'd'*64}, adapter_sha256='e'*64,
                    data=[dict(path='inputs/'+p.name, sha256=sha256(p), rows=n)
                          for p in (d/'states.npz', d/'meta.jsonl')])
    atomic_json(d/'manifest.json', manifest)
    ready = {k: manifest[k] for k in ('contract_id', 'schema_version', 'status', 'source_commit', 'producer_branch')}
    ready.update(manifest_absolute_path=str(d/'manifest.json'), manifest_sha256=sha256(d/'manifest.json'),
                 created_at='2026-10-26T00:00:00Z')
    atomic_json(root/'ready.json', ready)
    return root/'ready.json'


def refresh_manifest(ready_path, manifest):
    ready = json.loads(ready_path.read_text())
    path = Path(ready['manifest_absolute_path'])
    atomic_json(path, manifest)
    ready['manifest_sha256'] = sha256(path)
    atomic_json(ready_path, ready)


class ProjectionTests(unittest.TestCase):
    def test_already_satisfied_unchanged(self):
        r = np.array([[.1, .2], [.3, .4]])
        result = ipf(log(r), log(r.sum(1)), log(r.sum(0)))
        self.assertEqual(result['status'], 'converged')
        self.assertEqual(result['iterations'], 0)
        np.testing.assert_allclose(np.exp(result['logq']), r, atol=1e-14)

    def test_projection_and_independent_product(self):
        for r in (np.array([[.01, .49], [.25, .25]]), np.outer([.1, .9], [.8, .2])):
            result = ipf(log(r), log([.3, .7]), log([.6, .4]))
            self.assertEqual(result['status'], 'converged')
            q = np.exp(result['logq'])
            np.testing.assert_allclose(q.sum(1), [.3, .7], atol=1e-6)
            np.testing.assert_allclose(q.sum(0), [.6, .4], atol=1e-6)
        np.testing.assert_allclose(q, np.outer([.3, .7], [.6, .4]), atol=1e-6)

    def test_teacher_margins_kl_pythagoras(self):
        rng = np.random.default_rng(18)
        for _ in range(30):
            r = lognormalize(log(rng.uniform(.01, 1, (16, 16))))
            t = lognormalize(log(rng.uniform(.01, 1, (16, 16))))
            tp = np.exp(t)
            result = ipf(r, log(tp.sum(1)), log(tp.sum(0)))
            self.assertEqual(result['status'], 'converged')
            self.assertLessEqual(kl(t, result['logq']), kl(t, r)+1e-5)
            self.assertAlmostEqual(kl(t, r)-kl(t, result['logq']), kl(result['logq'], r), delta=1e-5)

    def test_wrong_marginals_hurt_concrete_counterexample(self):
        t = log([[.4, .1], [.1, .4]])
        q = ipf(t, log([.9, .1]), log([.9, .1]))
        self.assertGreater(kl(t, q['logq']), .5)
        self.assertEqual(kl(t, t), 0)

    def test_block_vs_row_truncation_counterexample(self):
        p = np.array([.5, .5])
        cond = np.array([[.45, .45], [.05, .05]])
        whole = np.exp(lognormalize(log(p[:, None])+log(cond)))
        wrong = p[:, None]*cond/cond.sum(1, keepdims=True)
        np.testing.assert_allclose(whole.sum(1), [.9, .1])
        np.testing.assert_allclose(wrong.sum(1), [.5, .5])

    def test_feasible_and_hall_infeasible_zero_support(self):
        feasible = ipf(log(np.eye(2)), log([.3, .7]), log([.3, .7]))
        self.assertEqual(feasible['status'], 'converged')
        self.assertTrue(np.isneginf(feasible['logq'][0, 1]))
        impossible = ipf(log(np.eye(2)), log([.3, .7]), log([.6, .4]))
        self.assertEqual(impossible['status'], 'support_infeasible')
        # Every row/column has an edge, but the Hall mass constraint fails.
        impossible = ipf(log([[1, 0, 0], [1, 0, 0], [0, 1, 1]]),
                         log([.4, .4, .2]), log([.2, .4, .4]))
        self.assertEqual(impossible['status'], 'support_infeasible')

    def test_zero_target_and_extreme_log_support(self):
        q = ipf(log([[.2, .3], [.3, .2]]), log([0, 1]), log([.4, .6]))
        self.assertEqual(q['status'], 'converged')
        self.assertTrue(np.isneginf(q['logq'][0]).all())
        q = ipf(np.array([[-1200., -1201.], [-1202., -1200.]]), log([.3, .7]), log([.4, .6]))
        self.assertEqual(q['status'], 'converged')
        self.assertTrue(np.isfinite(q['logq']).all())

    def test_chain_rule_and_limit(self):
        t, r = log([[.1, .2], [.3, .4]]), log([[.4, .1], [.1, .4]])
        row_kl = kl(log(np.exp(t).sum(1)), log(np.exp(r).sum(1)))
        self.assertAlmostEqual(kl(t, r), row_kl+conditional_kl(t, r), places=12)
        q = ipf(r, log([.95, .05]), log([.3, .7]), max_iter=0)
        self.assertEqual(q['status'], 'max_iterations')


class PackageTests(unittest.TestCase):
    def test_complete_256_float32_package_all_states_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = fixture(root, n=256)
            p = root/'inputs/states.npz'
            with np.load(p, allow_pickle=False) as z:
                a = {key: z[key].astype(np.float32) if z[key].dtype.kind == 'f' else z[key] for key in z.files}
            np.savez(p, **a)
            manifest = json.loads((root/'inputs/manifest.json').read_text())
            manifest['status'] = 'complete'
            manifest['data'][0]['sha256'] = sha256(p)
            refresh_manifest(ready, manifest)
            value = json.loads(ready.read_text())
            value['status'] = 'complete'
            atomic_json(ready, value)
            package = verify_package(ready)
            summary = analyze_package(package, root/'out', bootstrap_reps=30)
            self.assertEqual(summary['actual_n'], 256)
            self.assertEqual(summary['high_coverage']['n'], 256)
            self.assertEqual(set(summary['counts'].values()), {64})
            self.assertEqual(summary['audit']['Q_failures'], 0)
            self.assertEqual(summary['audit']['oracle_violations'], 0)
            self.assertEqual(len((root/'out/per_state.jsonl').read_text().splitlines()), 256)
            for name in ('main_table.csv', 'stratified_table.csv', 'high_coverage_table.csv', 'probability_audit.json'):
                self.assertTrue((root/'out'/name).is_file())

    def test_hash_manifest_and_every_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = fixture(root)
            verify_package(ready)
            p = root/'inputs/states.npz'
            with p.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(InputError, 'hash'):
                verify_package(ready)
        with tempfile.TemporaryDirectory() as tmp:
            ready = fixture(Path(tmp))
            p = Path(tmp)/'inputs/manifest.json'
            p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(InputError, 'hash'):
                verify_package(ready)

    def test_partial_requires_budget_and_missing_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = fixture(Path(tmp))
            manifest = json.loads((Path(tmp)/'inputs/manifest.json').read_text())
            del manifest['budget_stop']
            refresh_manifest(ready, manifest)
            with self.assertRaisesRegex(InputError, 'partial'):
                verify_package(ready)

    def test_missing_logs_and_probability_mismatch_block(self):
        for mutation in ('missing', 'mismatch', 'underflow'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ready = fixture(root)
                p = root/'inputs/states.npz'
                with np.load(p, allow_pickle=False) as z:
                    a = {k: z[k] for k in z.files}
                if mutation == 'missing':
                    del a['head_logcond2']
                elif mutation == 'mismatch':
                    a['head_cond2'][0, 0, 0] = .2
                else:
                    # Genuine FP32 probability underflow; log stays authoritative.
                    a['head_cond2'] = a['head_cond2'].astype(np.float32)
                    a['head_cond2'][0, 0, 0] = 0
                    a['head_logcond2'][0, 0, 0] = -200
                    a['head_candidate_mass'] = a['head_cond2'].sum(-1)
                np.savez(p, **a)
                manifest = json.loads((root/'inputs/manifest.json').read_text())
                manifest['data'][0]['sha256'] = sha256(p)
                refresh_manifest(ready, manifest)
                if mutation == 'underflow':
                    package = verify_package(ready)
                    self.assertEqual(package['numeric_audit']['head_cond2']['probability_underflow'], 1)
                else:
                    with self.assertRaises(InputError):
                        verify_package(ready)

    def test_resume_idempotence_preserves_all_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = fixture(root)
            package = verify_package(ready)
            out = root/'output'
            with self.assertRaisesRegex(InterruptedError, 'test interruption'):
                analyze_package(package, out, bootstrap_reps=30, interrupt_after=2)
            first = (out/'per_state.jsonl').read_text()
            with (out/'per_state.jsonl').open('ab') as f:
                f.write(b'{"interrupted_append":')
            summary = analyze_package(package, out, bootstrap_reps=30)
            self.assertTrue((out/'per_state.jsonl').read_text().startswith(first))
            self.assertEqual(summary['actual_n'], 4)
            self.assertEqual(summary['high_coverage']['status'], '高覆盖证据不足')
            self.assertEqual(len((out/'per_state.jsonl').read_text().splitlines()), 4)
            before = sha256(out/'per_state.jsonl')
            analyze_package(package, out, bootstrap_reps=30)
            self.assertEqual(before, sha256(out/'per_state.jsonl'))
            for row in map(json.loads, (out/'per_state.jsonl').read_text().splitlines()):
                self.assertIn('source_state_id', row)
                self.assertIn('delta_kl_Q_R', row)
                self.assertLessEqual(abs(row['chain_rule_residual']), 1e-10)

    def test_probability_regimes_cannot_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = fixture(Path(tmp))
            manifest = json.loads((Path(tmp)/'inputs/manifest.json').read_text())
            manifest.update(T=.2, top_p=.9)
            refresh_manifest(ready, manifest)
            with self.assertRaisesRegex(InputError, 'probability regime'):
                verify_package(ready)

    def test_resume_refuses_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = fixture(root)
            package = verify_package(ready)
            analyze_package(package, root/'out', bootstrap_reps=10)
            package['ready']['manifest_sha256'] = 'f'*64
            with self.assertRaisesRegex(InputError, 'resume input/code/config changed'):
                analyze_package(package, root/'out', bootstrap_reps=10)

    def test_bootstrap_resamples_groups_not_states(self):
        rows = [dict(group='a', delta=1.)]*20+[dict(group='b', delta=-1.)]*20
        x = cluster_bootstrap(rows, 'delta', reps=1000)
        self.assertEqual(x['groups'], 2)
        self.assertEqual(x['ci95'], [-1., 1.])
        self.assertEqual(x['mean'], 0.)

    def test_nonfinite_states_do_not_become_complete_case_main_mean(self):
        x = cluster_bootstrap([dict(group='a', delta=1.), dict(group='b', delta='Infinity')], 'delta', reps=10)
        self.assertEqual(x['n'], 2)
        self.assertEqual(x['unavailable'], 1)
        self.assertIsNone(x['mean'])


class ConsumerTests(unittest.TestCase):
    def test_restart_deadline_is_not_reset_or_extended(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = consumer.wait_window(root, 86400, now=100)
            second = consumer.wait_window(root, 86400, now=10000)
            self.assertEqual(first['deadline_epoch'], second['deadline_epoch'])
            third = consumer.wait_window(root, 86400, new_window=True, now=10000)
            self.assertEqual(third['deadline_epoch'], 96400)
            self.assertEqual(len((root/'wait_history.local.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(ValueError):
                consumer.wait_window(root, 86401)

    def test_wait_for_ready_not_tensor_existence_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'states.npz').write_bytes(b'not ready')
            self.assertEqual(consumer.inspect_producer(root)[0], 'waiting')
            atomic_json(root/'A_done.json', dict(contract_id=consumer.CONTRACT, status='failed'))
            self.assertEqual(consumer.inspect_producer(root)[0], 'failed')
            atomic_json(root/'dcd_ready.json', dict(status='partial'))
            self.assertEqual(consumer.inspect_producer(root)[0], 'ready')

    def test_timeout_records_blocked_archives_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(consumer, 'OUT', root), mock.patch.object(consumer, 'DOC', root/'docs'), \
                 mock.patch.object(consumer, 'relay_dir', return_value=root/'relay'), \
                 mock.patch.object(consumer, 'git', return_value=consumer.BRANCH), \
                 mock.patch.object(consumer, 'archive') as archive, \
                 mock.patch.object(consumer.sys, 'argv', ['consumer', '--wait-seconds', '0']), redirect_stdout(io.StringIO()):
                self.assertEqual(consumer.main(), 2)
                self.assertEqual(archive.call_args.args[0], 'blocked')
                record = json.loads((root/'blocked.json').read_text())
                self.assertEqual(record['status'], 'blocked')
                self.assertIn('--new-wait-window', record['restore_command'])

    def test_incompatible_ready_exits_without_analyzing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'relay').mkdir()
            atomic_json(root/'relay/dcd_ready.json', dict(contract_id='wrong', status='complete'))
            with mock.patch.object(consumer, 'OUT', root), mock.patch.object(consumer, 'DOC', root/'docs'), \
                 mock.patch.object(consumer, 'relay_dir', return_value=root/'relay'), \
                 mock.patch.object(consumer, 'git', return_value=consumer.BRANCH), \
                 mock.patch.object(consumer, 'archive') as archive, \
                 mock.patch.object(consumer, 'analyze_package') as analyze, \
                 mock.patch.object(consumer.sys, 'argv', ['consumer', '--wait-seconds', '0']), redirect_stdout(io.StringIO()):
                self.assertEqual(consumer.main(), 2)
                analyze.assert_not_called()
                self.assertEqual(archive.call_args.args[0], 'blocked')

    def test_archive_push_verification_and_failed_push_preserves_commit(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'work'
                root.mkdir()
                subprocess.run(['git', 'init', '-q', '-b', consumer.BRANCH, str(root)], check=True)
                for name, value in [('user.email', 'cpu-test@example.invalid'), ('user.name', 'CPU test')]:
                    subprocess.run(['git', '-C', str(root), 'config', name, value], check=True)
                (root/'initial').write_text('test')
                subprocess.run(['git', '-C', str(root), 'add', 'initial'], check=True)
                subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'test initial'], check=True)
                remote = Path(tmp)/'remote.git'
                if not fail:
                    subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True)
                subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', str(remote)], check=True)
                for rel in ('analysis/dreamon_copula_cpu.py', 'scripts/consume_dreamon_copula_cpu.py',
                            'tests/test_dreamon_copula_cpu.py'):
                    (root/rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(consumer.REPO/rel, root/rel)
                out = root/'analysis_outputs/dreamon_copula_diagnostic_20261015_v1'
                out.mkdir(parents=True)
                atomic_json(out/'summary.json', dict(actual_n=4))
                doc = root/'docs/paper_agent/copula_diagnostic_20261015_v1'
                doc.mkdir(parents=True)
                (doc/'checkpoint.zh.md').write_text('toy archive')
                relay = Path(tmp)/'relay'
                relay.mkdir()
                with mock.patch.object(consumer, 'REPO', root), mock.patch.object(consumer, 'OUT', out), \
                     mock.patch.object(consumer, 'DOC', doc), redirect_stdout(io.StringIO()):
                    done = consumer.archive('partial', relay, dict(manifest_sha256='f'*64))
                    self.assertEqual(consumer.existing_done(relay)['manifest_sha256'], done['manifest_sha256'])
                self.assertEqual(done['publication']['local_commit'],
                                 subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
                self.assertEqual(done['publication']['status'], 'local_commit_saved_remote_unverified' if fail else 'pushed_verified')
                if not fail:
                    self.assertEqual(done['publication']['local_commit'], done['publication']['remote_head'])


if __name__ == '__main__':
    unittest.main()
