"""CPU-only audit of the completed finite manifest and exported evidence."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.context_foundation.common import atomic_json, utc
from experiments.threshold_trace_fork.engine import integrity
from experiments.threshold_trace_fork.run import Ledger, OUT, LOCAL, IDS, FOCUS, FIELDS


def exported(name):
    with gzip.open(OUT / name, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    db = Ledger()
    expected = dict(smoke=114, replay=5168, fork=2920, multiseed=54,
                    teacher=64, teacher_logspace_validation=64,
                    original_full_validation=5168, capture_cost=91, shadow_cost=80)
    counts = {kind: db.count(kind) for kind in expected}
    assert counts == expected, counts
    assert all(r['original_matches'] and r['restore_matches'] for r in db.rows('smoke'))
    rows = {kind: db.rows(kind) for kind in ('replay', 'fork', 'multiseed')}
    for kind, name in [('replay', 'replay_per_case.jsonl.gz'),
                       ('fork', 'fork_per_case.jsonl.gz'),
                       ('multiseed', 'from_scratch_multiseed.jsonl.gz')]:
        assert exported(name) == rows[kind], name
    replay = {(r['config'], r['task_id']): r for r in rows['replay']}
    assert len(replay) == 5168
    assert Counter(r['config'] for r in rows['replay']) == Counter(
        {**{cid: 1033 for cid in IDS}, 'global_k2_nohead_090': 3})
    for cid in IDS:
        assert len({r['base_task_id'] for r in rows['replay'] if r['config'] == cid}) == 164
    scratch_keys = {(r['task_id'], r['config'], r['seed']) for r in rows['multiseed']}
    assert scratch_keys == {(case, cid, seed) for case in FOCUS
                            for cid in IDS + ['global_k2_nohead_090']
                            for seed in (43, 44, 45)}
    scores = {(case, text): json.loads(payload) for case, text, payload in
              db.db.execute('SELECT case_id,output,payload FROM scores')}
    stop_reasons, incomplete = {}, {}
    for kind, rr in rows.items():
        stop_reasons[kind] = dict(Counter(r['stop_reason'] for r in rr))
        incomplete[kind] = sum(r['final_remaining_masks'] > 0 for r in rr)
        for r in rr:
            integrity(r)
            assert r['outer_rounds'] <= 256 and 0 <= r['expansion_budget'] <= 64
            assert r['score_identity'] == [r['task_id'], r['output']]
            s = scores[(r['task_id'], r['output'])]
            assert s['passed'] == r['passed']
            assert s['official_evaluation'] == r['official_evaluation']
            assert s['failure_reason'] == r['failure_reason']
            if kind != 'fork':
                file = LOCAL / (r['config'] + '__' + r['task_id'].replace('/', '_')
                                + '__' + str(r['seed']) + '.json.gz')
                assert file.exists(), file
    groups = defaultdict(list)
    for r in rows['fork']:
        groups[(r['sample_group'], r['kind'], r['state'])].append(r)
        assert r['actual_forward_count'] == (0 if r['reused_execution'] else r['remaining_forward_count'])
        assert r['remaining_forward_count'] >= 0
        assert (LOCAL / ('fork__' + hashlib.sha256(r['actual_execution'].encode()).hexdigest()[:24]
                         + '.json.gz')).exists()
        assert Path(r['snapshot']).exists()
        factual = (r['intervention'] == 'original' or
                   r['intervention'] == 'PP' and 'nohead' in r['config'] or
                   r['intervention'] == 'QQ' and 'trained' in r['config'])
        if r['continuation_seed'] == 42 and factual:
            ref = replay[(r['config'], r['task_id'])]
            for field in FIELDS + ['output']:
                assert r[field] == ref[field], (r['state'], field)
    for (group, kind, state), rr in groups.items():
        labels = ('PP', 'PQ', 'QP', 'QQ') if kind == 'head' else (
            ('original', 'no_propagation') if kind == 'no_propagation' else
            ('original', 'a2_order', 'a2_exempt', 'a2_both') if kind == 'order' else
            ('original', 'defer', 'veto'))
        assert {(r['intervention'], r['continuation_seed']) for r in rr} == {
            (label, seed) for label in labels for seed in range(42, 50)}
        assert len(rr) == len(labels) * 8
    panel = json.loads((OUT / 'state_panel.json').read_text())
    special = json.loads((OUT / 'special_state_manifest.json').read_text())
    assert len(panel['selected']) == 64 and len(special['head']) == 16 and len(special['decoder']) == 17
    assert len(groups) == 97
    checks = json.loads((OUT / 'common_state_integrity.json').read_text())
    assert len(checks) == 11 and all(r['complete_pre_action_state_equal'] for r in checks)
    assert len({r['snapshot'] for r in rows['fork']}) <= 106
    assert len(list(LOCAL.glob('snapshot__*.pt'))) == 106
    for x in json.loads((OUT / 'replay_integrity.json').read_text()).values():
        assert x['completed'] == x['expected'] and not x['historical_mismatches'] and not x['score_mismatches']
    orig = json.loads((OUT / 'original_full_validation.json').read_text())
    assert orig['status'] == 'passed' and orig['completed'] == 5168
    assert len(exported('original_validation_per_case.jsonl.gz')) == 5168
    numeric = json.loads((OUT / 'teacher_numeric_integrity.json').read_text())
    assert numeric['states'] == 64 and numeric['new_states'] == 0
    assert numeric['extra_teacher_forwards'] == 64 and not numeric['used_in_branch_decisions']
    assert len(exported('teacher_logspace_per_state.jsonl.gz')) == 64
    mapping = exported('raw_trace_index.jsonl.gz')
    assert len(mapping) == 2920
    assert Counter(r['actual_execution'] for r in mapping) == Counter(r['actual_execution'] for r in rows['fork'])
    cost = json.loads((OUT / 'complete_forward_accounting.json').read_text())
    assert cost['original_validation_rows'] == 5168
    assert cost['branch_actual_forwards'] == sum(r['actual_forward_count'] for r in rows['fork'])
    assert cost['extra_original_validation_forwards'] == cost['replay_forwards']
    historical = ['experiments/threshold_eval', 'experiments/context_foundation',
                  'experiments/dreamon_singleline_order_parallel.py',
                  'experiments/dreamon_singleline_adapter.py', 'experiments/k4_batch_common.py']
    assert not subprocess.check_output(['git', 'diff', '285f8f1462481d7c73e1e6659822f45fe7b59a5e',
                                        '--', *historical], text=True)
    env = json.loads((LOCAL / 'environment.json').read_text())
    atomic_json(OUT / 'environment_summary.json', env)
    summary = dict(at=utc(), status='passed', counts=counts,
        logical_generated_rows=sum(len(r) for r in rows.values()), logical_branch_groups=len(groups),
        unique_snapshots=106, common_state_checks=11, exported_evidence_matches_ledger=True,
        factual_seed42_compact_fields_match=True, all_conservation_checks_passed=True,
        raw_trace_and_snapshot_mapping_complete=True, unchanged_historical_source=True,
        incomplete_cases=incomplete, stop_reasons=stop_reasons,
        quality_counts={kind: dict(passed=sum(r['passed'] for r in rr),
                       failed=sum(not r['passed'] for r in rr),
                       timed_out=sum(r['failure_reason'] == 'timed out' for r in rr))
                       for kind, rr in rows.items()},
        score_executions=len(scores), score_reuse_logical_rows=sum(r['score_reused'] for rr in rows.values() for r in rr),
        phase_versions=json.loads((OUT / 'phase_versions.json').read_text()),
        original_validation_version=orig['execution_version'], teacher_numeric_validation=numeric,
        unresolved_required_experiment_items=[], not_applicable=special['not_applicable'],
        interpretation_limits=['candidate factor not activated in selected states; interaction unidentified',
            'three focal cases and conditional seeds are not independent tasks',
            'repeated development population, not independent test data',
            'no deployment controller or logged-runtime speedup claim'])
    atomic_json(OUT / 'delivery_verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
