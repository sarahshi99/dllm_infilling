#!/usr/bin/env python3
"""CPU-only delivery audit; preserves all frozen inference and original results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collections
import argparse
import json
import subprocess
import tempfile

import numpy as np

from experiments.context_foundation.common import (
    BASE,BRANCH,CACHE,CONTRACT,OUT,PYTHON,REPO,atomic_json,common_dir,
    head_commit,manifest_entries,read_jsonl,sha,utc,write_jsonl,
)

AUDIT=OUT/'continuation_audit_20261027'
RESULT_COMMIT='f521bd3f91719c40a1fcfda7c39f977fdd48b5be'


def verify_manifest(path):
    manifest=json.loads(path.read_text())
    assert manifest['contract_id']==CONTRACT and manifest['schema_version']==1
    for entry in manifest['data']:
        assert not Path(entry['path']).is_absolute()
        assert sha(REPO/entry['path'])==entry['sha256'],entry['path']
    return len(manifest['data'])


def counterfactual_support(output):
    parents=list(read_jsonl(OUT/'blind_pair_parents.jsonl.gz'))
    attempts={r['state_id']:r for r in read_jsonl(OUT/'counterfactual_pairs.jsonl.gz')}
    special=json.loads((OUT/'protocol_schema.json').read_text())['special_token_ids']
    assert len(parents)==len(attempts)==512
    cells=collections.defaultdict(collections.Counter)
    records=[]
    for meta in parents:
        path=CACHE/meta['raw_path']
        assert sha(path)==meta['raw_sha256']
        with np.load(path,allow_pickle=False) as z:
            visible=np.flatnonzero(z['history_mask'][0,0])
            assert len(visible)==meta['history_length'] and len(visible)>=2
            counts=[]
            for slot in visible[:-1]:
                lp=z['branch_logp'][0,slot]
                ordinary=np.isfinite(lp)
                ordinary[special]=False
                original=int(z['token_ids'][0,slot])
                assert ordinary[original]
                counts.append(int(ordinary.sum()))
            supported=counts[0]>1
            result=attempts[meta['state_id']]
            assert (result['status']=='complete')==supported
            if supported:
                assert result['intervention_slot']==int(visible[0])
                assert result['latest_position']==int(z['position_ids'][0,visible[-1]])
                assert result['latest_token']==int(z['token_ids'][0,visible[-1]])
                assert result['original_token']!=result['alternative_token']
                assert result['alternative_token'] not in special
                lp=z['branch_logp'][0,visible[0]]
                assert np.isfinite(lp[result['alternative_token']])
                assert abs(float(lp[result['alternative_token']])-result['alternative_branch_logp'])<1e-6
            else:
                assert result['status']=='no_alternative_ordinary_support'
            cell=f"{meta['order']}|{meta['history_length']}|{meta['span']}"
            cells[cell]['parents']+=1
            cells[cell]['earliest_slot_feasible']+=supported
            cells[cell]['earliest_slot_singleton']+=not supported
            cells[cell]['any_earlier_slot_feasible']+=any(n>1 for n in counts)
            records.append(dict(state_id=meta['state_id'],group=meta['group'],split=meta['split'],
                order=meta['order'],history_length=meta['history_length'],span=meta['span'],
                earlier_slot_ordinary_support_sizes=counts,original_status=result['status'],
                any_earlier_slot_feasible=any(n>1 for n in counts)))
    assert sum(v['earliest_slot_feasible'] for v in cells.values())==22
    assert sum(v['any_earlier_slot_feasible'] for v in cells.values())==32
    write_jsonl(output/'counterfactual_parent_support.jsonl.gz',records)
    return dict(fixed_parents=512,complete=22,earliest_slot_no_alternative=490,
        any_earlier_slot_feasible=32,not_run_other_slot_interventions=10,
        cells={k:dict(v) for k,v in sorted(cells.items())},
        distinction='22 is feasible under the sealed earliest-token rule; 32 is a CPU support audit only, not 32 computed intervention pairs',
        gap_reason='normal deployment T=.2/top-p=.9 ordinary support is singleton at frozen earliest intervention slot',
        decision='no temperature/top-p changes, parent replacement, copied pairs or new GPU work; retain partial')


def quality_integrity():
    rows=list(read_jsonl(OUT/'per_case.jsonl.gz'))
    generated={(r['arm'],r['task_id']):r for r in read_jsonl(OUT/'generated_rows.jsonl.gz')}
    assert len(rows)==len(generated)==4132
    assert len({(r['arm'],r['task_id']) for r in rows})==4132
    by_arm={a:{} for a in ['K1','Z4','H4-old','F4-chain']}
    for r in rows:
        assert r['seed']==42 and type(r['passed']) is bool
        original=generated[r['arm'],r['task_id']]
        for field in ['output','completion_token_ids','sampling_key_sha256','forward_count','head_calls']:
            assert original[field]==r[field]
        assert r['actual_submitted_count']==sum(t['actual_submitted_count'] for t in r['action_trace'])
        by_arm[r['arm']][r['task_id']]=r
    population=set(by_arm['K1'])
    for arm,index in by_arm.items():
        assert len(index)==1033 and set(index)==population
        assert len({r['base_task_id'] for r in index.values()})==164
        assert sum(r['detailed_trace'] for r in index.values())==32
    results={a:dict(scored=len(rs),passed=sum(r['passed'] for r in rs.values())) for a,rs in by_arm.items()}
    for comparison in json.loads((OUT/'paired_comparisons.json').read_text())['comparisons']:
        a,b=by_arm[comparison['a']],by_arm[comparison['b']]
        wins=sum(a[k]['passed'] and not b[k]['passed'] for k in population)
        losses=sum(not a[k]['passed'] and b[k]['passed'] for k in population)
        assert wins==comparison['help'] and losses==comparison['harm']
    harms=list(read_jsonl(OUT/'fresh_harms_vs_old_head.jsonl.gz'))
    actual={k for k in population if by_arm['H4-old'][k]['passed'] and not by_arm['F4-chain'][k]['passed']}
    assert {r['task_id'] for r in harms}==actual and len(harms)==8
    return dict(status='passed',arms=results,logical_rows=4132,missing=0,duplicates=0,
        paired_counts_recomputed=True,generated_and_scored_outputs_equal=True,detailed_cases_per_arm=32,
        fresh_harm_failure_reasons=dict(collections.Counter(r['failure_reason'] for r in harms)))


def dcd_integrity():
    rows=list(read_jsonl(OUT/'dcd_metadata.jsonl'))
    assert len(rows)==256 and len({r['state_id'] for r in rows})==256
    with np.load(OUT/'dcd_states.npz',allow_pickle=False) as z:
        assert all(z[k].dtype!=object for k in z.files)
        assert list(z['state_id'])==[r['state_id'] for r in rows]
        for label in ['head','fresh']:
            p,lp=z[label+'_cond2'],z[label+'_logcond2']
            assert p.shape==lp.shape==(256,16,16) and np.isfinite(lp).all()
            assert np.allclose(p,np.exp(lp),rtol=1e-6,atol=1e-8)
            assert np.allclose(p.sum(-1),z[label+'_candidate_mass'],rtol=1e-5,atol=1e-6)
            assert (z[label+'_candidate_mass']+z[label+'_special_mass']<=1+1e-5).all()
        for slot in [1,2]:
            assert z['candidate_ids_'+str(slot)].shape==(256,16)
            assert np.isfinite(z['old_logp'+str(slot)]).all()
            assert np.allclose(z['old_p'+str(slot)],np.exp(z['old_logp'+str(slot)]),rtol=1e-6,atol=1e-8)
    return dict(status='passed',rows=256,object_arrays=False,log_fields_finite=True,
        probability_log_agreement=True,candidate_and_special_mass_consistent=True,
        cells=dict(collections.Counter(f"{r['order']}|{r['span']}|{r['split']}" for r in rows)))


def run_audit(output):
    output.mkdir(exist_ok=True)
    print('CPU audit: original portable manifests',flush=True)
    manifests={name:verify_manifest(OUT/name) for name in ['protocol_manifest.json','dcd_manifest.json','A_manifest.json']}
    print('CPU audit: fixed 512 intervention parents',flush=True)
    support=counterfactual_support(output)
    quality=quality_integrity()
    dcd=dcd_integrity()
    states=list(read_jsonl(OUT/'state_metrics.jsonl.gz'))
    assert len(states)==len({r['state_id'] for r in states})==3072
    assert collections.Counter(r['split'] for r in states)==dict(train=2304,validation=768)
    assert collections.Counter(r['arm'] for r in states)=={'Z4':1536,'H4-old':1536}
    assert all(int(row['max_per_group'])<=8 for row in __import__('csv').DictReader((OUT/'A1_coverage.csv').open()))
    raw=json.loads((OUT/'tensor_manifest.json').read_text())['data']
    for i,entry in enumerate(raw):
        assert not Path(entry['path']).is_absolute()
        assert sha(CACHE/entry['path'])==entry['sha256'],entry['path']
        if i%1024==0:
            print(f'CPU raw shard hash audit {i}/{len(raw)}',flush=True)
    with (output/'cpu_tests.txt').open('w') as f:
        test=subprocess.run([PYTHON,'-m','unittest','discover','-s','tests/context_foundation','-v'],cwd=REPO,
            stdout=f,stderr=subprocess.STDOUT)
    assert test.returncode==0
    budget=json.loads((OUT/'budget.local.json').read_text())
    assert budget['resident'] is False and budget['used_sec']<43200
    result=dict(contract_id=CONTRACT,schema_version=1,status='verified_partial',at=utc(),
        original_result_commit=RESULT_COMMIT,source_commit=head_commit(),manifest_files_verified=manifests,
        raw_shards_verified=len(raw),A1=dict(actual=3072,train=2304,validation=768,zero_origin=1536,old_head_origin=1536),
        A2=support,DCD=dcd,generation=quality,gpu_budget_used_sec=budget['used_sec'],
        mechanism_budget_used_sec=budget['stage_sec']['mechanism'],additional_gpu_sec=0,additional_model_calls=0,
        original_inference_code_and_artifacts_unchanged=True)
    atomic_json(output/'audit.json',result)
    print(json.dumps(dict(status=result['status'],A2=support['complete'],raw_shards=len(raw),additional_gpu_sec=0)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true',help='verify in a temporary directory, leaving published hashes unchanged')
    args=parser.parse_args()
    if args.verify_only:
        with tempfile.TemporaryDirectory(prefix='dreamon-context-delivery-audit-') as temporary:
            run_audit(Path(temporary))
    else:
        if (AUDIT/'delivery_manifest.json').exists():
            raise SystemExit('Audit is already published/prepared. Use --verify-only; do not overwrite manifest-bound evidence.')
        run_audit(AUDIT)


if __name__=='__main__':
    main()
