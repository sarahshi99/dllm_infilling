#!/usr/bin/env python3
"""One serial GPU worker; transactional generation/evaluation, no quality gate."""
from pathlib import Path
import argparse,copy,fcntl,gzip,json,os,shlex,sqlite3,subprocess,sys,tempfile,time,traceback
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from experiments.k4_batch_common import ROOT,DIAG,EVALUATOR,PYTHON,STAGES,CODE_PATHS,population,read_jsonl,write_gzip
from experiments.dreamon_singleline_adapter import CountingModel,OfficialHFTokenizerWrapper,derive_row_seed,ensure_modeling_rope_utils_available
from experiments.dreamon_singleline_order_parallel import decode_with_policy,GLOBAL_CONFIDENCE,LEFT_TO_RIGHT_FRONTIER
from experiments.dreamcoder_fixed_transfer_adapter import decode_fixed_transfer,native_fixed_reference
from experiments.run_dreamon_singleline_order_parallel import load_generator_module
from experiments.run_dreamon_markov_k2_diagnosis import stable_smoke
from experiments.lrdllm_dreamcoder_adapter import atomic_write_json,append_jsonl,load_pinned_evaluator,selection_view
POLICIES={'global_confidence':GLOBAL_CONFIDENCE,'left_to_right_frontier':LEFT_TO_RIGHT_FRONTIER}


def open_store(path):
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS cases(policy TEXT,arm TEXT,task_id TEXT,generated TEXT NOT NULL,evaluated TEXT,PRIMARY KEY(policy,arm,task_id))')
    db.commit();return db


def records(db):
    return [dict(json.loads(g),**(json.loads(e) if e else {})) for g,e in db.execute('SELECT generated,evaluated FROM cases ORDER BY policy,arm,task_id')]


def anonymous_gpu():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert len(fields)==5 and 'H200' in fields[0], 'requires current single H200'
    return dict(time_utc=utc(),model=fields[0].strip(),used_mib=float(fields[1]),total_mib=float(fields[2]),utilization_pct=float(fields[3]),temperature_c=float(fields[4]))


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def official_evaluate(problem,output):
    # Same interpreter/environment as generation; bwrap restricts mounts/network,
    # while official check_correctness, concatenation and 3-second clock stay intact.
    _,info=load_pinned_evaluator(EVALUATOR)
    from experiments.lrdllm_dreamcoder_adapter import enable_pinned_evaluator_source
    runtime=PYTHON.resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='k4-official-') as temp:
        p=Path(temp)
        (p/'payload.json').write_text(json.dumps(dict(problem=problem,output=output)))
        (p/'execution.py').write_text(enable_pinned_evaluator_source((EVALUATOR/'human_eval_infilling/execution.py').read_text()))
        (p/'worker.py').write_text((REPO/'experiments/k4_official_eval_worker.py').read_text())
        cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
          '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
          '--ro-bind',str(runtime),str(runtime),'--ro-bind',str(p),'/input','--proc','/proc','--dev','/dev',
          '--tmpfs','/tmp','--chdir','/tmp','--setenv','OMP_NUM_THREADS','1',
          '--setenv','OPENBLAS_NUM_THREADS','1',str(runtime/'bin/python'),'-I','/input/worker.py']
        started=time.perf_counter(); start_utc=utc(); load=os.getloadavg()
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=12)
        if run.returncode:
            raise RuntimeError('official evaluator infrastructure exit '+str(run.returncode)+': '+run.stderr[-1500:])
        data=json.loads(run.stdout); result=data['official_result']
        if result['task_id']!=problem['task_id'] or type(result['passed']) is not bool:
            raise RuntimeError('official result schema/identity failure')
        return dict(passed=result['passed'],failure_reason=result['result'],official_evaluation=result,
                    evaluator_interpreter=data['interpreter'],evaluator_sec=time.perf_counter()-started,
                    evaluation_start_utc=start_utc,evaluation_end_utc=utc(),cpu_load=load)


class MeasuredHead(torch.nn.Module):
    """CUDA events buffered, read only after the row's existing sync boundary."""
    def __init__(self,head):
        super().__init__();self.head=head;self.events=[]
    def forward(self,ids):
        before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
        before.record(); value=self.head(ids);after.record();self.events.append((before,after))
        return value
    def reset(self):self.events=[]
    def elapsed(self):return sum(a.elapsed_time(b) for a,b in self.events)/1000


def prepare(stage):
    spec=STAGES[stage]; out=ROOT/stage;out.mkdir(parents=True,exist_ok=True)
    original=json.loads(DIAG.read_text()); config=original['original_config']
    rows=population(spec['dataset'])
    # Compare every loader field and key with the actual cached official HF test,
    # never consult the old controller/test_lock or sealed labels.
    from datasets import Dataset
    hf_config='HumanEval-SingleLineInfilling' if spec['dataset']=='single-line' else 'HumanEval-MultiLineInfilling'
    arrow=next((Path('/home/shx/.cache/huggingface/datasets/loubnabnl___humaneval_infilling')/hf_config).glob('1.0.0/ad460*/humaneval_infilling-test.arrow'))
    assert {r['task_id']:r for r in Dataset.from_file(str(arrow))}=={r['task_id']:r for r in rows}
    _, evaluator=load_pinned_evaluator(EVALUATOR);assert evaluator==config['evaluator']
    assert subprocess.check_output(['git','-C',config['source_root'],'rev-parse','HEAD'],text=True).strip()==original['source_revision']
    registry=json.loads((REPO/'analysis_outputs/zero_training_transfer_20261013/controller/target_registry.json').read_text())
    model_entry=registry['dreamcoder_base'] if stage=='B' else dict(id='Dream-org/DreamOn-v0-7B',revision=original['model_revision'],local_path=config['model_snapshot'])
    assert Path(model_entry['local_path']).name==model_entry['revision']
    head_entry=original['head_registry_entry'];assert Path(head_entry['path']).stat().st_size==head_entry['size']
    assert (original['head_rank'],original['head_lambda'],original['seed'])==(256,1.0,42)
    protocol=config['official_protocol']
    assert (protocol['min_gen_len'],protocol['max_gen_len'],protocol['steps'],protocol['temperature'],protocol['top_p'])==(64,64,256,.2,.9)
    keys=[(p,a) for p in spec['policies'] for a in spec['arms']]
    manifest=dict(stage=stage,status='prepared',dataset=hf_config,dataset_repository='loubnabnl/humaneval_infilling',
        dataset_revision='ad46002f24b153968a3d0949e6fa9576780530ba',dataset_split='test',normal_loader='pinned human_eval_infilling.data.read_problems',
        population_count=len(rows),base_task_count=164,population_scope='full authorized development population; single/multiline share 164 base tasks; no claim of independent unseen test; old test_lock untouched',
        official_protocol=protocol,base_parameters=json.loads((REPO/'analysis_outputs/zero_training_transfer_20261013/dreamcoder_base/run_manifest.json').read_text())['parameters'] if stage=='B' else None,
        model={k:v for k,v in model_entry.items() if k in ['id','revision','weight_kind']},source_revision=original['source_revision'],
        evaluator=evaluator,evaluator_timeout_sec=3.0,evaluator_isolation='bwrap private PID/filesystem/network, same Python runtime and official check_correctness; sequential CPU tests',
        head={k:head_entry[k] for k in ['head','phase','label','sha256','size']},rank=256,head_lambda=1,model_dtype='bfloat16',head_dtype='float32',frozen=True,
        seed=42,row_seed='derive_row_seed(42, task_id)',edge_seed='(row_seed + 1000003*step + 10007*left_absolute + right_absolute) % (2**63-1)',rng='fork_rng around each local head/sample; base draft global stream isolated',
        arms=[dict(policy=p,arm=a,k=1 if stage=='C' and a=='A' else spec['k']) for p,a in keys],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        code_versions={p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=REPO,text=True).strip() for p in CODE_PATHS},
        evidence_blobs={p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=REPO,text=True).strip() for p in ['analysis_outputs/k4_multiline_20261014_v1/'+spec['dataset']+'_contexts.jsonl.gz','analysis_outputs/k4_multiline_20261014_v1/historical_a4_candidates.jsonl.gz']},
        smoke_ids=[r['task_id'] for r in stable_smoke(rows)],smoke_selection='16 evenly spaced sorted stable IDs; fixed before outcomes',
        expand_budget_source='max_gen_len=64; unchanged source semantics, dynamic length changes with deletion/expansion, not a no-expansion promise',
        shared_budget_fix='parallel original expands clipped in released descending-coordinate execution order to remaining expansion budget and canvas capacity, shared by all controls; barrier paths reserve only one expansion slot; excess becomes mask_noop; K1/K2 unchanged when bound not exceeded',
        timing='CUDA synchronize before/after generation, excludes model loading, official execution, row writes; includes discrete trace buffering and CUDA event recording. Shared H200 observation only; no extra timing reruns.',
        trace_coverage='all newly generated rows/rounds: selected drafts/final submitted IDs, edges/edge seeds, actions/stop reasons; no full logits/hidden states; old reuse coverage in provenance',
        head_mapping='direct shared token IDs, persistent output mask strict loaded; Base complete tokenizer audit: ../zero_training_transfer_20261013/a_dreamcoder/tokenizer_audit.json',
        head_timing='head_projection_cuda_sec covers head forward only; head_serial_cuda_sec covers head plus local sampling; counters are actual calls, not nominal K',
        base_schedule_confound='steps64 unchanged for K1/K2/K4; K4 fills ~16 forwards at earlier timestep/EOS penalty progress than K2~32; not corrected by filled fraction',
        reviewer_gate_disabled=True,command=shlex.join([sys.executable,*sys.argv]))
    mpath=out/'run_manifest.json'
    if mpath.exists():
        old=json.loads(mpath.read_text())
        for k in ['arms','code_versions','official_protocol','base_parameters','model','head','smoke_ids','dataset_revision','evidence_blobs']:
            assert old[k]==manifest[k], 'resume mismatch '+k
        manifest=old
    atomic_write_json(mpath,manifest)
    contexts=ROOT/(spec['dataset']+'_contexts.jsonl.gz')
    if contexts.exists():assert {r['task_id']:r for r in read_jsonl(contexts)}=={r['task_id']:r for r in rows}
    else:write_gzip(contexts,rows)
    return manifest,rows,original,model_entry


@torch.inference_mode()
def run(stage,smoke_only=False):
    manifest,rows,original,model_entry=prepare(stage);spec=STAGES[stage];out=ROOT/stage
    keys=[(p,a) for p in spec['policies'] for a in spec['arms']]
    db=open_store(out/'per_case.sqlite')
    lock=open('/tmp/dllm_infilling_h200_diagnosis.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    while True:
        gpu=anonymous_gpu();append_jsonl(out/'gpu_background.jsonl',gpu)
        if gpu['total_mib']-gpu['used_mib']>=32768:break
        manifest.update(status='waiting_gpu_capacity');atomic_write_json(out/'run_manifest.json',manifest)
        print('Waiting for >=32GiB free; no model loaded',flush=True);time.sleep(30)
    try:
        ensure_modeling_rope_utils_available()
        from transformers import AutoModel,AutoTokenizer,AutoConfig
        from experiments.train_dreamon_markov_head import make_head
        model_path=model_entry['local_path'];config=original['original_config']
        hf_tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        model=CountingModel(AutoModel.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval()).eval().requires_grad_(False)
        trained=make_head(AutoConfig.from_pretrained(config['model_snapshot'],trust_remote_code=True,local_files_only=True),'cpu')
        checkpoint=torch.load(original['head_registry_entry']['path'],map_location='cpu',weights_only=False)
        trained.load_state_dict(checkpoint['head'],strict=True);del checkpoint
        zero=copy.deepcopy(trained)
        with torch.no_grad():zero.output.weight.zero_()
        heads={a:MeasuredHead(h.to('cuda:0').eval().requires_grad_(False)) for a,h in [('C',zero),('D',trained)]}
        tokenizer=hf_tokenizer if stage=='B' else OfficialHFTokenizerWrapper(hf_tokenizer)
        assert model.model.config.vocab_size==trained.vocab_size, 'head vocabulary mapping differs'
        assert next(trained.parameters()).dtype==torch.float32 and next(model.parameters()).dtype==torch.bfloat16
        sampler=type(model.model)._sample.__globals__['sample_tokens'] if stage=='B' else load_generator_module(Path(config['source_root'])).sample_tokens
        manifest.update(status='technical_smoke',actual_device=torch.cuda.get_device_name(0),python=sys.version,torch_version=torch.__version__,cuda_version=torch.version.cuda)
        atomic_write_json(out/'run_manifest.json',manifest)
        def decode(row,p,a,reference=False):
            safe=selection_view(row);row_seed=derive_row_seed(42,row['task_id'])
            torch.manual_seed(row_seed);torch.cuda.manual_seed_all(row_seed)
            head=heads.get(a)
            if head:head.reset()
            before=model.forward_calls;torch.cuda.synchronize();start=time.perf_counter();start_utc=utc()
            if stage=='B':
                kw=dict(model=model,tokenizer=tokenizer,sample_tokens_fn=sampler,prefix_ids=[tokenizer.bos_token_id]+tokenizer.encode(safe['prefix'],add_special_tokens=False),suffix_ids=tokenizer.encode(safe['suffix'],add_special_tokens=False)+[tokenizer.eos_token_id],requested_k=4,policy=POLICIES[p],row_seed=row_seed,capture_actions=True,**{k:v for k,v in manifest['base_parameters'].items() if k!='canvas_length'})
                decoded=native_fixed_reference(**kw) if reference else decode_fixed_transfer(**kw,markov_head=head)
            else:
                protocol=manifest['official_protocol']; k=1 if stage=='C' and a=='A' else spec['k']
                decoded=decode_with_policy(model=model,tokenizer=tokenizer,sample_tokens_fn=sampler,
                   prefix_ids=tokenizer.encode(safe['prefix'],add_bos=True,add_eos=False),suffix_ids=tokenizer.encode(safe['suffix'],add_bos=False,add_eos=True),
                   **{k:protocol[k] for k in ['min_gen_len','max_gen_len','max_tokens','steps','temperature','top_p','top_k']},
                   expand_budget=protocol['max_gen_len'],requested_k=k,policy=POLICIES[p],device='cuda:0',
                   markov_head=head,markov_lambda=1 if head else 0,k2_action_mode='legacy' if a=='A' else 'sequential_pair',
                   markov_sampling_seed=row_seed,diagnostics=False,capture_actions=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if not reference:
                assert model.forward_calls-before==decoded['forward_count']
                assert decoded['markov_enabled_count']==(len(head.events) if head else 0)
                assert decoded['markov_enabled_count']==sum(len(t['chain_edges']) for t in decoded['action_trace'])
                assert len(decoded['action_trace'])==decoded['base_forward_count']
                assert all(0<=t['actual_submitted_count']<=spec['k'] for t in decoded['action_trace'])
                assert decoded.get('expand_budget_remaining',0)>=0
            for field in ['step_trace','markov_transitions','global_neighbor_observations','replay_transitions','replay_exclusion_counts']:
                decoded.pop(field,None)
            decoded.update(generation_sec=elapsed,generation_start_utc=start_utc,generation_end_utc=utc(),head_projection_cuda_sec=head.elapsed() if head else 0,
                           head_serial_cuda_sec=decoded.pop('_head_serial_cuda_sec',0))
            return decoded
        reuse_candidates=read_jsonl(ROOT/'historical_a4_candidates.jsonl.gz') if stage=='A' else []
        reuse_index={(r['policy'],r['task_id']):r for r in reuse_candidates}
        reuse_disabled=set()
        smoke=out/'smoke_checks.json'
        if not smoke.exists() or json.loads(smoke.read_text())['status']!='passed':
            checks=[]
            frontier_alias_verified=[]
            for index,row in enumerate(stable_smoke(rows)):
                print(f'{stage} technical smoke {index+1}/16',flush=True)
                row_results={}
                for p,a in keys:
                    decoded=decode(row,p,a)
                    row_results[p,a]=decoded
                    if stage=='B' and a=='A':
                        reference=decode(row,p,a,reference=True)
                        assert reference['completion_token_ids']==decoded['completion_token_ids'], 'native final disagreement'
                        canvas=[int(tokenizer.mask_token_id)]*64
                        snapshots=[];offset=len([tokenizer.bos_token_id]+tokenizer.encode(row['prompt'],add_special_tokens=False))
                        for trace in decoded['action_trace']:
                            for pos,token in trace['submitted']:canvas[pos-offset]=token
                            snapshots.append(list(canvas))
                        assert snapshots==reference['native_canvas_history'],'native per-round disagreement'
                    if stage=='A' and a=='A' and (p,row['task_id']) in reuse_index:
                        historical=reuse_index[p,row['task_id']]
                        fields=['completion_token_ids','forward_count','normal_token_count','expand_count','delete_count','final_dynamic_length']
                        if any(decoded[k]!=historical[k] for k in fields):reuse_disabled.add(p)
                        else:
                            old_actions=historical['action_trace'];new_actions=decoded['action_trace']
                            if any((old['normal'],old['expand'],old['delete'])!=(new['normal'],new['expand'],new['delete']) for old,new in zip(old_actions,new_actions)):
                                reuse_disabled.add(p)
                    evaluated=official_evaluate(row,decoded['completion'])
                    # False/timeout/syntax are valid; smoke has no quality threshold.
                    checks.append(dict(task_id=row['task_id'],policy=p,arm=a,head_calls=decoded['markov_enabled_count'],nfe=decoded['forward_count'],reference_per_round=stage=='B' and a=='A',generation_sec=decoded['generation_sec'],evaluation=evaluated['official_evaluation']))
                if stage=='A':
                    left=row_results['left_to_right_frontier','A'];right=row_results['left_to_right_frontier','B']
                    assert all(left[k]==right[k] for k in ['completion_token_ids','forward_count','normal_token_count','expand_count','delete_count']), 'frontier A/B equivalence failure'
                    assert len(left['action_trace'])==len(right['action_trace'])
                    for l,r in zip(left['action_trace'],right['action_trace']):
                        assert all(l[k]==r[k] for k in ['selected_raw_absolute','submitted_absolute','normal','expand','delete']), 'frontier per-round alias failure'
                    frontier_alias_verified.append(row['task_id'])
                atomic_write_json(out/'smoke_progress.json',dict(completed=len(checks),expected=16*len(keys)))
            atomic_write_json(smoke,dict(status='passed',checks=checks,quality_gate=False,reuse_disabled_policies=sorted(reuse_disabled),frontier_alias_verified=frontier_alias_verified,technical_generation_calls=len(checks),native_reference_calls=32 if stage=='B' else 0,native_reference_steps=64 if stage=='B' else None,native_reference_trailing_noops='typically48 after K4 fills64; output history compares actual changed rounds' if stage=='B' else None))
        if smoke_only:
            manifest.update(status='technical_smoke_passed');return 0
        if stage=='A':
            reuse_disabled=set(json.loads(smoke.read_text()).get('reuse_disabled_policies',[]))
            for reused in reuse_candidates:
                if reused['policy'] in reuse_disabled:continue
                key=(reused['policy'],reused['arm'],reused['task_id'])
                if db.execute('SELECT 1 FROM cases WHERE policy=? AND arm=? AND task_id=?',key).fetchone():continue
                evaluated={k:reused[k] for k in ['passed','failure_reason']}
                evaluated['official_evaluation_recorded_fields_only']=evaluated.copy()
                with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(*key,json.dumps(reused,ensure_ascii=False),json.dumps(evaluated,ensure_ascii=False)))
        expected={(p,a,r['task_id']) for p,a in keys for r in rows}
        prior=records(db);assert {(r['policy'],r['arm'],r['task_id']) for r in prior}<=expected
        assert all(r['row_seed']==derive_row_seed(42,r['task_id']) for r in prior)
        completed={(p,a,t) for p,a,t in db.execute('SELECT policy,arm,task_id FROM cases WHERE evaluated IS NOT NULL')}
        manifest.update(status='running_full');atomic_write_json(out/'run_manifest.json',manifest)
        for index,row in enumerate(rows):
            if all((p,a,row['task_id']) in completed for p,a in keys):continue
            append_jsonl(out/'gpu_background.jsonl',anonymous_gpu())
            offset=index%len(keys)
            ordered=keys[offset:]+keys[:offset]
            if stage=='A':ordered.sort(key=lambda key: 0 if key==('left_to_right_frontier','A') else 1 if key==('left_to_right_frontier','B') else 2)
            for p,a in ordered:
                key=(p,a,row['task_id'])
                if key in completed:continue
                if stage=='A' and (p,a)==('left_to_right_frontier','B'):
                    source=db.execute('SELECT generated,evaluated FROM cases WHERE policy=? AND arm=? AND task_id=?',(p,'A',row['task_id'])).fetchone()
                    assert source is not None and source[1] is not None
                    alias=json.loads(source[0]);alias.update(arm='B',provenance='alias_frontier_B_to_A',alias_of={'policy':p,'arm':'A','task_id':row['task_id']},
                        alias_proof='same selected contiguous positions and left-to-right original structural barrier; no head/no resampling; CPU and 16 GPU per-round submitted IDs equivalent')
                    with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(*key,json.dumps(alias,ensure_ascii=False),source[1]))
                    completed.add(key);continue
                prior_row=db.execute('SELECT generated FROM cases WHERE policy=? AND arm=? AND task_id=?',key).fetchone()
                if prior_row:
                    generated=json.loads(prior_row[0])
                else:
                    decoded=decode(row,p,a);completion=decoded.pop('completion')
                    generated=dict(stage=stage,policy=p,arm=a,task_id=row['task_id'],base_task_id=row['task_id'].split('/')[2],seed=42,row_seed=derive_row_seed(42,row['task_id']),
                        code_commit=manifest['code_commit'],output=completion,output_chars=len(completion),output_tokens=len(decoded['completion_token_ids']),provenance='new',**decoded)
                    with db:db.execute('INSERT INTO cases VALUES(?,?,?,?,NULL)',(*key,json.dumps(generated,ensure_ascii=False)))
                evaluated=official_evaluate(row,generated['output'])
                with db:db.execute('UPDATE cases SET evaluated=? WHERE policy=? AND arm=? AND task_id=?',(json.dumps(evaluated,ensure_ascii=False),*key))
                completed.add(key)
                atomic_write_json(out/'progress.json',dict(status='running',completed=len(completed),expected=len(expected),row_index=index,time_utc=utc()))
            if index%25==0:print(f'{stage} {len(completed)}/{len(expected)}',flush=True)
        assert completed==expected
        manifest.update(status='completed');return 0
    except Exception as exc:
        manifest.update(status='technical_failure',error=str(exc))
        append_jsonl(out/'failure_journal.jsonl',dict(time_utc=utc(),reason=str(exc),traceback=traceback.format_exc(),policy='infrastructure/technical only; retain generated payload, resume same row seed; no model result substituted'))
        print(traceback.format_exc(),flush=True);return 2
    finally:
        atomic_write_json(out/'run_manifest.json',manifest)
        write_gzip(out/'per_case.jsonl.gz',records(db));db.close();lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=STAGES,required=True);parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();sys.exit(run(args.stage,args.smoke_only))
