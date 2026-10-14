#!/usr/bin/env python3
"""CPU preparation of portable historical controls and full population records."""
from pathlib import Path
import collections,gzip,hashlib,json,subprocess,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.k4_batch_common import ROOT,DIAG,EVALUATOR,population,read_jsonl,write_gzip
from experiments.dreamon_singleline_adapter import derive_row_seed
from experiments.lrdllm_dreamcoder_adapter import atomic_write_json

ROOT.mkdir(parents=True,exist_ok=True)
oldroot=REPO.parent/'dreamon-singleline-order-parallel-markov-v1'
old=oldroot/'analysis_outputs/dreamon_singleline_order_parallelism_markov_diagnostic_20260823_v2'
diag=json.loads(DIAG.read_text()); cfg=json.loads((old/'run_config.json').read_text())
for key in ['model_snapshot','source_root','evaluator_root','official_protocol','seed','evaluator']:
    assert cfg[key]==diag['original_config'][key], 'historical protocol mismatch '+key
rows=population('single-line');ids={r['task_id'] for r in rows}
assert json.loads((old/'manifest.snapshot.json').read_text())['task_ids_common_to_all_variants']==1033
assert not subprocess.check_output(['git','-C',str(EVALUATOR),'status','--porcelain','--','data','human_eval_infilling/data.py'],text=True).strip()
source=read_jsonl(old/'per_case_results.jsonl')
traces=collections.defaultdict(list)
for t in read_jsonl(old/'per_step_trace.jsonl'):
    if t['variant'] in ['C4','L4']:traces[t['variant'],t['task_id']].append(t)
sha={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in ['run_config.json','per_case_results.jsonl','per_step_trace.jsonl']}
reusable=[];rejected=[]
for original in source:
    if original['variant'] not in ['C4','L4']:continue
    assert original['task_id'] in ids and original['row_seed']==derive_row_seed(42,original['task_id']) and original['seed']==42
    steps=sorted(traces[original['variant'],original['task_id']],key=lambda t:t['step_index'])
    assert len(steps)==original['forward_count'] and [t['step_index'] for t in steps]==list(range(len(steps)))
    before=64; budget=64;safe=True;normalized=[]
    for t in steps:
        expand=t['executed_expand_count'];delete=t['executed_delete_count']
        if expand>min(budget,64-before):safe=False
        budget-=expand
        assert t['canvas_length']==before+expand-delete
        before=t['canvas_length']
        top={v['position']:v for v in t['confidence_top4']}
        positions=t['selected_positions'];barrier=t['earliest_structural_action_position']
        direct_structure=sum(top[pos]['token_type'] in ['expand','delete'] for pos in positions if pos in top) if original['variant']=='C4' else int(barrier is not None)
        normalized.append(dict(step=t['step_index'],selected_raw_relative=[[pos,top[pos]['token_id'] if pos in top else None] for pos in positions],
            chain_edges=[],selected_count=t['selected_candidate_count'],actual_submitted_count=t['actual_committed_normal_tokens']+direct_structure,
            normal=t['actual_committed_normal_tokens'],expand=expand,delete=delete,
            barrier=barrier,corrected_stop=False,stop_reason='historical_original_structure' if barrier is not None else 'historical_selected_committed',
            expand_budget_remaining=budget,legacy_trace=t))
    policy='global_confidence' if original['variant']=='C4' else 'left_to_right_frontier'
    if not safe:
        rejected.append(dict(policy=policy,task_id=original['task_id'],reason='historical parallel expand exceeded current shared budget/capacity repair'))
        continue
    reusable.append(dict(stage='A',policy=policy,arm='A',task_id=original['task_id'],base_task_id=original['task_id'].split('/')[2],seed=42,row_seed=original['row_seed'],
       code_commit='historical 5d5d5f2 branch; original generation source archived below',output=original['output'],passed=original['passed'],failure_reason=original['failure_reason'],
       completion_token_ids=original['completion_token_ids'],output_chars=len(original['output']),output_tokens=len(original['completion_token_ids']),
       forward_count=original['forward_count'],base_forward_count=original['forward_count'],outer_round_count=original['forward_count'],recompute_forward_count=0,recompute_enabled_count=0,
       normal_token_count=original['normal_token_count'],expand_count=original['expand_count'],delete_count=original['delete_count'],
       markov_enabled_count=0,structural_stop_count=sum(t['barrier'] is not None for t in normalized),corrected_structural_stop_count=0,
       final_dynamic_length=original['final_dynamic_length'],final_remaining_masks=original['final_remaining_masks'],
       generation_sec=None,evaluator_sec=None,head_projection_cuda_sec=0,head_serial_cuda_sec=0,action_trace=normalized,
       provenance='reused_historical_no_head_K4',reuse_source_record=original,reuse_source_sha256=sha['per_case_results.jsonl'],
       timing_limitation='historical wall_sec includes evaluator and lacks CUDA timing boundary; excluded from new generation timing means',
       trace_limitation='old trace preserves selection/counts and top4 drafts; frontier selected draft outside top4 is null; original final IDs present, no reconstructed causal trace'))
for p in ['C4','L4']:
    subset=[r for r in source if r['variant']==p]
    assert len(subset)==1033 and {r['task_id'] for r in subset}==ids
write_gzip(ROOT/'historical_a4_candidates.jsonl.gz',reusable)
atomic_write_json(ROOT/'reuse_audit.json',dict(source_branch='codex/dreamon-singleline-order-parallel-markov-diagnostic-v1',source_commit='5d5d5f2',
    source_files_sha256=sha,eligible=len(reusable),rejected=rejected,
    counts_by_policy=dict(collections.Counter(r['policy'] for r in reusable)),
    protocol_checked=['same pinned input file and normal loader fields','same model/source revision','same bf16 sampling/length/seed42/derive_row_seed per row','same no-head legacy K4 semantics; shared expansion repair checked from every historical round'],
    activation='16 fixed stable-ID GPU current A4 outputs/actions must match historical results before reuse; if any unexplained mismatch, disable reuse for that policy and regenerate it',
    key_sets={p:sorted(r['task_id'] for r in reusable if r['policy']==p) for p in ['global_confidence','left_to_right_frontier']},
    head_k4_completed=False,server_inventory='no active shx model/experiment GPU process at inspection; known diagnosis/transfer SQLite and old C4/L4 reviewed; no matching existing trained-head K4 or exact full multiline protocol evidence found',
    multiline_k1_reuse=False,multiline_reason='old 3415/5815 or old5079 controller outputs do not supply an exact full current per-row protocol; regenerate full K1'))
# Explicit data use record, without changing or inspecting sealed controller labels.
for ds in ['single-line','multi-line']:write_gzip(ROOT/(ds+'_contexts.jsonl.gz'),population(ds))
atomic_write_json(ROOT/'population_use_20261014.json',dict(date='2026-10-14',singleline=1033,multiline=5815,base_task_ids=list(range(164)),
    loader='pinned evaluator read_problems; all fields/IDs equal cached loubnabnl/humaneval_infilling test@ad46002f24b153968a3d0949e6fa9576780530ba',
    permission='user explicitly authorized full Markov multiline development population, including old controller-excluded 16 base groups',
    old_test_lock='not read or modified',scope='shared 164 base tasks; not independent unseen test'))
# Portable anchor evidence, checked per row against the exact current loader/seed.
for name,path in [('dreamon',REPO.parent/'dreamon-markov-k2-diagnosis-v1/analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/evidence/per_case.jsonl.gz'),
                  ('dreamcoder_base',REPO/'analysis_outputs/zero_training_transfer_20261013/dreamcoder_base/per_case.jsonl.gz')]:
    anchor=read_jsonl(path)
    for r in anchor:assert r['task_id'] in ids and r['row_seed']==derive_row_seed(42,r['task_id']) and r['seed']==42
    for p in {r['policy'] for r in anchor}:
        for a in {r['arm'] for r in anchor}:
            assert {r['task_id'] for r in anchor if (r['policy'],r['arm'])==(p,a)}==ids
    write_gzip(ROOT/(name+'_old_anchors.jsonl.gz'),[r for r in anchor if r['arm']=='A' or (r['arm']=='B' and (name=='dreamcoder_base' or r['expand_count']==0))])
print('Historical A4 eligible:',len(reusable),'shared-budget affected:',len(rejected),flush=True)

atomic_write_json(ROOT/'anchor_manifest.json',dict(dreamon_source_commit='c843847',base_source_commit='bf62fdf',dreamon_k1='all1033 per policy; single-action expansion cannot exceed one-slot capacity or remaining budget, all other K1 behavior unchanged',dreamon_k2='only rows with original expand_count=0, because original K2 has no matching round trajectories to exclude changed expansion boundary; pair tables disclose actual subset N',base='all1033 K1/K2 per policy; native selection/sampling/cleanup/clock unchanged, existing CPU native tests retained',inputs='same pinned loader fields and exact stable IDs checked; every row derive_row_seed42 checked; same model revision and bf16/head mapping/protocol',timing='old anchor timing unchanged; reused rows do not count as new independent evidence'))
