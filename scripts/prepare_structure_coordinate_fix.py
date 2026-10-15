#!/usr/bin/env python3
"""Freeze an outcome-blind influence manifest before any new generation."""
from pathlib import Path
import collections,gzip,hashlib,json,subprocess,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.k4_batch_common import ROOT as OLD,DIAG,read_jsonl,write_gzip
from experiments.lrdllm_dreamcoder_adapter import atomic_write_json
from experiments.dreamon_singleline_adapter import OfficialHFTokenizerWrapper,derive_row_seed
from analysis.dreamon_structure_replay import compare_row
from transformers import AutoTokenizer
OUT=REPO/'analysis_outputs/structure_coordinate_fix_20261015_v2'
OUT.mkdir(exist_ok=True,parents=True)
assert not (OUT/'influence_manifest.json').exists(),'frozen manifest exists; verify/reuse, never regenerate over it'
config=json.loads(DIAG.read_text())['original_config']
tok=OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(config['model_snapshot'],trust_remote_code=True,local_files_only=True))
base_path=json.loads((REPO/'analysis_outputs/zero_training_transfer_20261013/controller/target_registry.json').read_text())['dreamcoder_base']['local_path']
base=AutoTokenizer.from_pretrained(base_path,trust_remote_code=True,local_files_only=True)
contexts={ds:{r['task_id']:r for r in read_jsonl(OLD/(ds+'_contexts.jsonl.gz'))} for ds in ['single-line','multi-line']}
encodings={}
for stage,ds in [('A','single-line'),('B','single-line'),('C','multi-line')]:
 encodings[stage]={}
 for task,row in contexts[ds].items():
  if stage=='B':prefix=[base.bos_token_id]+base.encode(row['prompt'],add_special_tokens=False);suffix=base.encode(row['suffix'],add_special_tokens=False)+[base.eos_token_id]
  else:prefix=tok.encode(row['prompt'],add_bos=True,add_eos=False);suffix=tok.encode(row['suffix'],add_bos=False,add_eos=True)
  encodings[stage][task]=(prefix,suffix)
manifest=[];source_files={};union=collections.defaultdict(set);unknown=[]
for stage in 'ABC':
 source=OLD/stage/'per_case.jsonl.gz';source_files[stage]=hashlib.sha256(source.read_bytes()).hexdigest()
 count=collections.Counter()
 for row in read_jsonl(source):
  assert row['seed']==42 and row['row_seed']==derive_row_seed(42,row['task_id'])
  key=f"{stage}::{row['policy']}::{row['arm']}::{row['task_id']}"
  try:
   proof=compare_row(row,*encodings[stage][row['task_id']],mask_id=base.mask_token_id if stage=='B' else tok.mask_id,eos_id=base.eos_token_id if stage=='B' else tok.eos_id,expand_id=-1 if stage=='B' else tok.expand_id)
  except (AssertionError,KeyError,IndexError,TypeError) as exc:
   proof=dict(status='unknown',first_step=None,reason=type(exc).__name__+': '+str(exc))
   unknown.append(dict(key=key,reason=proof['reason']))
  eligible=(stage=='A' and row['policy']=='global_confidence' and row['arm']=='A') or (stage=='C' and row['arm'] in 'BCD')
  if proof['status']!='proved_equivalent':
   assert eligible, 'unreconstructed non-interference path needs CPU reconstruction: '+key+' '+proof['reason']
   union[stage].add(row['task_id'])
  count[proof['status']]+=1
  manifest.append(dict(stable_id=key,stage=stage,policy=row['policy'],arm=row['arm'],task_id=row['task_id'],row_seed=row['row_seed'],
                       source_sha256=source_files[stage],source_provenance=row['provenance'],regenerate=proof['status']!='proved_equivalent',**proof))
 print(stage,dict(count),flush=True)
assert len(manifest)==37722 and len({r['stable_id'] for r in manifest})==37722
pending=[r for r in manifest if r['regenerate']]
assert len(pending)<=18478
smoke=[]
for stage in ['A','C']:
 group=sorted([r for r in pending if r['stage']==stage],key=lambda r:r['stable_id'])
 # Fixed indices from outcome-blind affected-ID order; no passed field read.
 for i in sorted({0,len(group)//3,2*len(group)//3,len(group)-1}) if group else []:smoke.append(group[i]['stable_id'])
controls=[]
for stage in ['A','B','C']:
 group=sorted([r for r in manifest if r['stage']==stage and not r['regenerate']],key=lambda r:r['stable_id'])
 for i in [0,len(group)//2,len(group)-1]:controls.append(group[i]['stable_id'])
value=dict(version='v2 coordinate only',baseline_commit='dbdd0bd5cbbf21a049a989213136aa2650844aea',source_files_sha256=source_files,
           selection='outcome-blind complete discrete replay, no passed field used; first divergence ends counterfactual replay; unknown regenerates',
           rows=manifest,regenerate_count=len(pending),unknown_count=len(unknown),paired_union={k:sorted(v) for k,v in union.items()},
           smoke_ids=smoke,unaffected_control_ids=controls,smoke_rule='first, 1/3, 2/3, last affected stable configuration ID per stage; controls first/middle/last equivalent per stage',
           limitation='only actions before first state divergence are same-state model evidence; no corrected pass rate inferred from stale subsequent actions',
           bootstrap=dict(replicates=10000,seed=20261013,cluster='base task',weights=['line','base_task_equal']))
atomic_write_json(OUT/'influence_manifest.json',value)
atomic_write_json(OUT/'influence_summary.json',dict(counts=dict(collections.Counter(r['status'] for r in manifest)),regenerate_count=len(pending),by_configuration=dict(collections.Counter((r['stage']+'/'+r['policy']+'/'+r['arm']) for r in pending)),unknown=unknown,paired_union_sizes={k:len(v) for k,v in union.items()},smoke_ids=smoke))
# Cache CPU encoded inputs and contexts once, no model weights and no sampling.
write_gzip(OUT/'encoded_inputs.jsonl.gz',[dict(stage=s,task_id=t,prefix_ids=p,suffix_ids=q) for s,items in encodings.items() for t,(p,q) in items.items()])
print('FROZEN',len(pending),'rows to regenerate;',len(unknown),'unknown',flush=True)
