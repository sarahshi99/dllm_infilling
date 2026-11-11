"""H2's committed-prefix generation path with H1's fixed trained checkpoint."""
import gc
import time
import traceback
import torch
from experiments.context_foundation.common import REPO,read_jsonl,utc,head_commit
from experiments.context_h1_training_v1 import pipeline as shared
from experiments.context_h1_training_v1.train import initialize
from experiments.context_h1_training_v1.bank import config
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter
from .pipeline_state import RUN,SOURCE,progress

def generate(store):
    contexts=list(read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz'))
    if len(contexts)!=1033 or len({r['task_id'] for r in contexts})!=1033 or len({r['task_id'].split('/')[2] for r in contexts})!=164:
        raise ValueError('1033 line / 164 source population mismatch')
    pending=[r for r in contexts if not store.get('generated','H1::'+r['task_id']) and not store.get('generation_failure','H1::'+r['task_id'])]
    if not pending: return
    engine=None;candidate=None;adapter=None
    try:
        engine=shared.load_engine(progress_fn=progress,memory_reserve_gib=4)
        ck=torch.load(SOURCE/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        assert ck['model']=='H1-context-prev' and ck['epoch']==5
        candidate=initialize(config(),'H1-context-prev');candidate.load_state_dict(ck['head']);del ck
        candidate=candidate.to('cuda:0').eval().requires_grad_(False)
        adapter=KeywordHeadAdapter(candidate,legacy_zero_head=engine.head.head)
        engine.head=adapter
        for r in pending:
            key='H1::'+r['task_id']
            # Do not choose among repeated stochastic attempts after interruption.
            if store.get('generation_started',key) and not store.get('oom_deferred',key):
                store.put('generation_failure',key,dict(arm='H1',task_id=r['task_id'],reason='interrupted in-flight attempt; no automatic regeneration'))
                continue
            prefix=engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
            suffix=engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
            store.put('generation_started',key,dict(at=utc(),best=str(SOURCE/'checkpoints/best.pt')))
            torch.cuda.synchronize();ts=time.monotonic()
            try:
                result=engine.decode(prefix,suffix,r['task_id'],'H4-old',order='global')
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                store.put('oom_deferred',key,dict(at=utc(),reason='batch1 memory; resume same checkpoint after memory available'))
                raise
            except InterruptedError: raise
            except Exception as exc:
                store.put('generation_failure',key,dict(arm='H1',task_id=r['task_id'],reason=str(exc),traceback=traceback.format_exc()))
                raise
            result.update(arm='H1',task_id=r['task_id'],base_task_id=r['task_id'].split('/')[2],seed=42,order='global',
                generation_shared_gpu_sec=time.monotonic()-ts,source_commit=head_commit())
            store.put('generated',key,result)
            progress('generation',completed=store.count('generated'),expected=1033,task_id=r['task_id'],last_shared_sec=result['generation_shared_gpu_sec'])
    finally:
        if engine is not None: shared.release(engine)
        del engine,candidate,adapter
        gc.collect();torch.cuda.empty_cache()

