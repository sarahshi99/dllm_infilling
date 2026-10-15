#!/usr/bin/env python3
"""One GPU worker, same sampler/evaluator, only frozen affected keys regenerate."""
from pathlib import Path
import copy,fcntl,json,os,subprocess,sys,time,traceback
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
from experiments.structure_coordinate_common import OUT,OLD,DIAG,CODE_PATHS,influence,old_rows,digest,read_jsonl,write_gzip
from experiments.run_k4_multiline_stage import open_store,records,anonymous_gpu,official_evaluate,MeasuredHead,utc
from experiments.lrdllm_dreamcoder_adapter import atomic_write_json,append_jsonl
from experiments.dreamon_singleline_adapter import CountingModel,OfficialHFTokenizerWrapper,derive_row_seed,ensure_modeling_rope_utils_available
from experiments.dreamon_singleline_order_parallel import decode_with_policy,GLOBAL_CONFIDENCE,LEFT_TO_RIGHT_FRONTIER
from experiments.run_dreamon_singleline_order_parallel import load_generator_module
from analysis.dreamon_structure_replay import initial_canvas,replay_old,apply_round


def check_unaffected(old,new,encoded,tokenizer):
    for field in ['completion_token_ids','base_forward_count','expand_count','delete_count','normal_token_count','final_dynamic_length','markov_enabled_count']:
        assert new[field]==old[field], 'unaffected GPU redecode differs: '+field
    prefix,suffix=encoded['prefix_ids'],encoded['suffix_ids']
    start=initial_canvas(prefix,suffix,tokenizer.mask_id)
    final=replay_old(old,start,mask_id=tokenizer.mask_id,eos_id=tokenizer.eos_id,expand_id=tokenizer.expand_id)
    resolved={t:v for t,v in zip(final.x[len(prefix):len(prefix)+final.length],old['completion_token_ids']) if isinstance(t,str)}
    left=right=start
    checks=[]
    for before,after in zip(old['action_trace'],new['action_trace']):
        kwargs=dict(stage=old['stage'],arm=old['arm'],policy=old['policy'],mask_id=tokenizer.mask_id,eos_id=tokenizer.eos_id,expand_id=tokenizer.expand_id)
        left,la=apply_round(left,before,**kwargs,fixed=False,resolved=resolved)
        right,ra=apply_round(right,after,**kwargs,fixed=True)
        assert left.x==right.x and left.length==right.length, 'unaffected round canvas differs'
        checks.append(dict(step=after['step'],submitted=ra['submitted'],delete_indices=ra['delete_indices'],canvas_length=right.length,full_canvas_equal=True))
    return checks


@torch.inference_mode()
def run():
    manifest=influence();old=old_rows()
    pending=[r for r in manifest['rows'] if r['regenerate']]
    expected={(r['policy'],r['arm'],r['task_id']) for r in pending}
    spec=json.loads((OUT/'run_manifest.json').read_text())
    for path,blob in spec['code_versions'].items():
        assert subprocess.check_output(['git','hash-object',path],cwd=REPO,text=True).strip()==blob,'source changed after formal version: '+path
    assert digest(OUT/'influence_manifest.json')==spec['influence_sha256']
    assert digest(OUT/'encoded_inputs.jsonl.gz')==spec['encoded_inputs_sha256']
    for stage,sha in manifest['source_files_sha256'].items():assert digest(OLD/stage/'per_case.jsonl.gz')==sha
    db=open_store(OUT/'per_case.sqlite')
    lock=open('/tmp/dllm_infilling_h200_diagnosis.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    started=utc(); spec.setdefault('first_started_utc',started); spec.update(status='waiting_gpu_capacity',worker_pid=os.getpid(),started_utc=started)
    atomic_write_json(OUT/'run_manifest.json',spec)
    try:
        while True:
            snapshot=anonymous_gpu();append_jsonl(OUT/'gpu_background.jsonl',snapshot)
            if snapshot['total_mib']-snapshot['used_mib']>=32768:break
            print('Waiting for >=32GiB free; no model loaded; other jobs untouched',flush=True);time.sleep(30)
        ensure_modeling_rope_utils_available()
        from transformers import AutoModel,AutoTokenizer
        from experiments.train_dreamon_markov_head import make_head
        original=json.loads(DIAG.read_text());cfg=original['original_config']
        assert Path(original['head_registry_entry']['path']).stat().st_size==original['head_registry_entry']['size']
        assert Path(cfg['model_snapshot']).name==spec['model']['revision']
        assert subprocess.check_output(['git','-C',cfg['source_root'],'rev-parse','HEAD'],text=True).strip()==spec['source_revision']
        assert original['head_registry_entry']['sha256']==spec['head']['sha256']
        assert cfg['official_protocol']==spec['official_protocol']
        tokenizer=OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(cfg['model_snapshot'],trust_remote_code=True,local_files_only=True))
        model=CountingModel(AutoModel.from_pretrained(cfg['model_snapshot'],trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval()).eval().requires_grad_(False)
        trained=make_head(model.model.config,'cpu');checkpoint=torch.load(original['head_registry_entry']['path'],map_location='cpu',weights_only=False)
        trained.load_state_dict(checkpoint['head'],strict=True);del checkpoint
        zero=copy.deepcopy(trained)
        with torch.no_grad():zero.output.weight.zero_()
        heads={a:MeasuredHead(h.to('cuda:0').eval().requires_grad_(False)) for a,h in [('C',zero),('D',trained)]}
        assert next(model.parameters()).dtype==torch.bfloat16 and next(trained.parameters()).dtype==torch.float32
        sample=load_generator_module(Path(cfg['source_root'])).sample_tokens
        contexts={r['task_id']:r for ds in ['single-line','multi-line'] for r in read_jsonl(OLD/(ds+'_contexts.jsonl.gz'))}
        encoded={(r['stage'],r['task_id']):r for r in read_jsonl(OUT/'encoded_inputs.jsonl.gz')}
        index={r['stable_id']:r for r in manifest['rows']}
        def decode(proof):
            stage,p,a,task=proof['stage'],proof['policy'],proof['arm'],proof['task_id'];seed=derive_row_seed(42,task)
            assert seed==proof['row_seed']
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            head=heads.get(a) if not (stage=='A' and a in 'AB') else None
            if stage=='C' and a=='B':head=None
            if head:head.reset()
            inp=encoded[stage,task];protocol=spec['official_protocol']
            torch.cuda.synchronize();t=time.perf_counter();before=model.forward_calls;begin=utc()
            result=decode_with_policy(model=model,tokenizer=tokenizer,sample_tokens_fn=sample,prefix_ids=inp['prefix_ids'],suffix_ids=inp['suffix_ids'],
                **{k:protocol[k] for k in ['min_gen_len','max_gen_len','max_tokens','steps','temperature','top_p','top_k']},
                expand_budget=protocol['max_gen_len'],requested_k=4 if stage=='A' else 1 if a=='A' else 2,
                policy=GLOBAL_CONFIDENCE if p=='global_confidence' else LEFT_TO_RIGHT_FRONTIER,device='cuda:0',
                markov_head=head,markov_lambda=1 if head else 0,k2_action_mode='legacy' if a=='A' else 'sequential_pair',
                markov_sampling_seed=seed,diagnostics=False,capture_actions=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            assert model.forward_calls-before==result['forward_count']
            assert result['markov_enabled_count']==(len(head.events) if head else 0)==sum(len(tr['chain_edges']) for tr in result['action_trace'])
            assert result['expand_budget_remaining']>=0
            result.update(generation_sec=elapsed,generation_start_utc=begin,generation_end_utc=utc(),head_projection_cuda_sec=head.elapsed() if head else 0,head_serial_cuda_sec=result.pop('_head_serial_cuda_sec',0))
            for field in ['step_trace','markov_transitions','global_neighbor_observations','replay_transitions','replay_exclusion_counts']:result.pop(field,None)
            return result
        spec.update(status='technical_smoke');atomic_write_json(OUT/'run_manifest.json',spec)
        smoke=json.loads((OUT/'technical_smoke_manifest.json').read_text());checks=[]
        controls_path=OUT/'unaffected_gpu_checks.json'
        if controls_path.exists():checks=json.loads(controls_path.read_text())['checks']
        checked={r['stable_id'] for r in checks}
        for key in smoke['unaffected_controls']:
            if key in checked:continue
            proof=index[key]
            if proof['stage']=='B':
                checks.append(dict(stable_id=key,method='all6198 CPU full canvas replay plus unchanged native Base module and native reference regression; no unnecessary Base GPU model loaded'))
            else:
                new=decode(proof)
                rounds=check_unaffected(old[key],new,encoded[proof['stage'],proof['task_id']],tokenizer)
                checks.append(dict(stable_id=key,input=encoded[proof['stage'],proof['task_id']],old_final_token_ids=old[key]['completion_token_ids'],new_final_token_ids=new['completion_token_ids'],round_assertions=rounds,action_trace=new['action_trace']))
            atomic_write_json(controls_path,dict(status='passed',checks=checks))
        order=smoke['affected_smoke']+[r['stable_id'] for r in sorted(pending,key=lambda r:r['stable_id']) if r['stable_id'] not in set(smoke['affected_smoke'])]
        completed={(p,a,t) for p,a,t in db.execute('SELECT policy,arm,task_id FROM cases WHERE evaluated IS NOT NULL')}
        assert completed<=expected
        for key in order:
            proof=index[key];p,a,task=proof['policy'],proof['arm'],proof['task_id'];dbkey=(p,a,task)
            if dbkey in completed:continue
            prior=db.execute('SELECT generated FROM cases WHERE policy=? AND arm=? AND task_id=?',dbkey).fetchone()
            if prior:generated=json.loads(prior[0])
            else:
                result=decode(proof);completion=result.pop('completion')
                generated=dict(stage=proof['stage'],policy=p,arm=a,task_id=task,base_task_id=task.split('/')[2],seed=42,row_seed=proof['row_seed'],
                    code_commit=spec['fix_commit'],provenance='coordinate_v2_new',v2_provenance='regenerated',influence_id=key,
                    output=completion,output_chars=len(completion),output_tokens=len(result['completion_token_ids']),**result)
                with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,NULL)',(*dbkey,json.dumps(generated,ensure_ascii=False)))
            evaluation=official_evaluate(contexts[task],generated['output'])
            assert evaluation['evaluator_interpreter']==spec['official_evaluator_python'],'scoring Python changed'
            with db:db.execute('UPDATE cases SET evaluated=? WHERE policy=? AND arm=? AND task_id=?',(json.dumps(evaluation,ensure_ascii=False),*dbkey))
            completed.add(dbkey)
            atomic_write_json(OUT/'progress.json',dict(status='running',completed=len(completed),expected=len(expected),last_stable_id=key,time_utc=utc()))
            append_jsonl(OUT/'gpu_background.jsonl',anonymous_gpu())
            print(f'{len(completed)}/{len(expected)} {key}',flush=True)
            if all((index[k]['policy'],index[k]['arm'],index[k]['task_id']) in completed for k in smoke['affected_smoke']):
                spec['status']='running_affected_full';atomic_write_json(OUT/'run_manifest.json',spec)
        assert completed==expected
        spec.update(status='completed',completed=len(completed));return 0
    except Exception as exc:
        spec.update(status='technical_failure',error=str(exc))
        append_jsonl(OUT/'failure_retry_log.jsonl',dict(time_utc=utc(),error=str(exc),traceback=traceback.format_exc(),action='retained any completed generation; resume exact seed, never turn infrastructure failure into model score'))
        raise
    finally:
        write_gzip(OUT/'new_rows.jsonl.gz',records(db));db.close();lock.close()
        spec['ended_utc']=utc();atomic_write_json(OUT/'run_manifest.json',spec)
        log=OUT/'failure_retry_log.jsonl'
        atomic_write_json(OUT/'failure_retry_coverage.json',dict(start_utc=spec['first_started_utc'],end_utc=utc(),events=sum(1 for _ in log.open()) if log.exists() else 0,empty=not log.exists() or log.stat().st_size==0,scope='all accumulated worker sessions under this fix manifest; generation/evaluation infrastructure exceptions; ordinary official failures are scores'))

if __name__=='__main__':run()
