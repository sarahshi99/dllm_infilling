"""Five-epoch resumable H1 KL training; shared schedule for subsequent heads."""
import collections
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.context_heads_v1.heads import ContextHead
from experiments.context_heads_v1.loss import teacher_kl
from experiments.context_heads_v1.training_v1.run import save_checkpoint
from experiments.context_foundation.adapter import distribution,inverse_cdf,uniform
from .bank import RUN,SEED,Bank,config,cell,progress,atomic_json


def initialize(cfg, head_name='H1-context-prev'):
    head=ContextHead(head_name,cfg).float()
    # Existing H1 gamma=1+tanh(g) starts at 1. All other internal layers keep
    # normal random initialization. Only residual vocabulary output is zeroed.
    torch.nn.init.zeros_(head.output.weight)
    return head


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])


def accumulated_update(head,opt,bank,selected,device,micro):
    opt.zero_grad(set_to_none=True)
    total=0.
    for start in range(0,len(selected),micro):
        ix=selected[start:start+micro]
        batch,teacher=bank.batch(ix,device)
        delta=head(batch).delta_logits
        loss=teacher_kl(batch.old_logits,delta,teacher,batch.history_mask.any(-1))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite KL')
        (loss*(len(ix)/len(selected))).backward()
        total+=float(loss.detach())*len(ix)/len(selected)
        del batch,teacher,delta,loss
    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return total,float(norm)


@torch.inference_mode()
def evaluate(head,bank,store,epoch,device,micro,*,run=RUN,progress_fn=progress):
    RUN=run;progress=progress_fn
    head.eval()
    indices=bank.indices('validation')
    kind=f'validation_{epoch}'
    done={r['state_id'] for r in store.rows(kind)}
    missing=[int(i) for i in indices if bank.rows[int(i)]['state_id'] not in done]
    for start in range(0,len(missing),micro):
        ix=missing[start:start+micro]
        batch,teacher=bank.batch(ix,device)
        old=batch.old_logits.float()
        scores=old+head(batch).delta_logits
        lp,lq=teacher.float().log_softmax(-1),scores.log_softmax(-1)
        p,q=lp.exp(),lq.exp()
        kl=(p*(lp-lq)).sum(-1).flatten()
        tv=(p-q).abs().sum(-1).flatten()/2
        pd,qd,od=distribution(teacher),distribution(scores),distribution(old)
        deploy_tv=(pd.exp()-qd.exp()).abs().sum(-1).flatten()/2
        us=[uniform(bank.rows[i]['sampling_key']) for i in ix]
        u=torch.tensor(us,device=device,dtype=torch.float64)[:,None]
        teacher_tokens=inverse_cdf(pd,u).flatten()
        new_tokens=inverse_cdf(qd,u).flatten()
        old_tokens=inverse_cdf(od,u).flatten()
        vals=torch.stack([kl,tv,deploy_tv,(new_tokens!=old_tokens).float(),
            (new_tokens!=teacher_tokens).float(),(old_tokens!=teacher_tokens).float()],-1).cpu().tolist()
        with store.db:
            for i,v in zip(ix,vals):
                if not all(math.isfinite(x) for x in v): raise FloatingPointError('validation nonfinite')
                row=bank.rows[i]
                r=dict(state_id=row['state_id'],group=row['group'],cell=cell(row),stage=row['stage'],
                    kl_T1=v[0],tv_T1=v[1],tv_deploy=v[2],sample_change_old=v[3],sample_mismatch_teacher=v[4],old_mismatch_teacher=v[5])
                store.db.execute('INSERT INTO rows VALUES(?,?,?)',(kind,r['state_id'],json.dumps(r)))
        if start%1024==0:
            progress('validation',epoch=epoch,completed=len(done)+min(start+micro,len(missing)),target=len(indices))
    rows=store.rows(kind)
    if len(rows)!=len(indices): raise ValueError('incomplete validation')
    keys=('kl_T1','tv_T1','tv_deploy','sample_change_old','sample_mismatch_teacher','old_mismatch_teacher')
    buckets=collections.defaultdict(list)
    groups=collections.defaultdict(list)
    for r in rows:
        buckets['all'].append(r);buckets[r['cell']].append(r);buckets['stage='+r['stage']].append(r)
        groups[r['group']].append(r)
    def mean(rs): return {k:float(np.mean([r[k] for r in rs])) for k in keys}
    result=dict(epoch=epoch,by_bucket={k:dict(count=len(rs),**mean(rs)) for k,rs in buckets.items()},
        source_groups=len(groups),source_equal={k:float(np.mean([mean(rs)[k] for rs in groups.values()])) for k in keys})
    atomic_json(RUN/f'validation_epoch_{epoch}.json',result)
    return result


def train(store,*,run=RUN,head_name='H1-context-prev',bank=None,progress_fn=progress):
    RUN=run;progress=progress_fn
    if (RUN/'training_summary.json').exists(): return
    cfg=config();bank=bank if bank is not None else Bank(store,cfg);device='cuda:0'
    plan=dict(head=head_name,seed=SEED,config=vars(cfg),initialization=('W_out=0; gamma=1; remaining internal parameters random' if head_name=='H1-context-prev' else 'W_out=0; GRU h0=0; remaining internal parameters random'),
        optimizer='AdamW',lr=3e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_norm=1,
        effective_batch=128,epochs=5,head_dtype='FP32',loss='full-vocab T1 mean KL(fresh||old+delta)',
        train_states=len(bank.indices('train')),validation_states=len(bank.indices('validation')),
        best='strictly lower full validation mean KL; ties earlier')
    # An ordinary persisted schedule prevents accidental resume with changed lr,
    # data population or initialization. No new protocol hash/gate is introduced.
    canonical=json.loads(json.dumps(plan))
    if (RUN/'plan.json').exists() and json.loads((RUN/'plan.json').read_text())!=canonical:
        raise ValueError('resume training schedule/data/config changed')
    atomic_json(RUN/'plan.json',canonical)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    free,total=torch.cuda.mem_get_info()
    micro=32 if free>12*2**30 else 16 if free>6*2**30 else 4
    if (RUN/'microbatch.json').exists(): micro=min(micro,json.loads((RUN/'microbatch.json').read_text())['micro'])
    reductions=0
    while True:
        head=opt=None
        try:
            head=initialize(cfg,head_name).to(device)
            opt=torch.optim.AdamW(head.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            path=RUN/'checkpoints/last.pt'
            state=dict(epoch=0,cursor=0,step=0,stage='validation',best_kl=float('inf'),best_epoch=0,curve=[],train_loss_sum=0.,train_seen=0)
            if path.exists():
                ck=torch.load(path,map_location='cpu',weights_only=False)
                head.load_state_dict(ck['head']);opt.load_state_dict(ck['optimizer']);state=ck['state'];restore_rng(ck['rng'])
                del ck
            def save():
                save_checkpoint(path,dict(model=head_name,config=vars(cfg),head=head.state_dict(),optimizer=opt.state_dict(),state=state,rng=rng_state(),micro=micro))
            if not path.exists(): save()
            while state['epoch']<=5:
                epoch=state['epoch']
                if state['stage']=='validation':
                    result=evaluate(head,bank,store,epoch,device,micro,run=run,progress_fn=progress_fn)
                    kl=result['by_bucket']['all']['kl_T1']
                    if kl<state['best_kl']:
                        state['best_kl']=kl;state['best_epoch']=epoch
                        save_checkpoint(RUN/'checkpoints/best.pt',dict(model=head_name,config=vars(cfg),head=head.state_dict(),epoch=epoch,step=state['step'],validation=result))
                    if not any(r['epoch']==epoch for r in state['curve']):
                        state['curve'].append(dict(epoch=epoch,step=state['step'],validation_kl=kl,
                            train_kl=state['train_loss_sum']/state['train_seen'] if state['train_seen'] else None))
                    atomic_json(RUN/'curve.json',state['curve'])
                    if epoch==5:
                        state['stage']='complete';save()
                        curve=state['curve']
                        improving=curve[-2]['validation_kl']-curve[-1]['validation_kl']>max(1e-5,abs(curve[-2]['validation_kl'])*.001)
                        atomic_json(RUN/'training_summary.json',dict(status='complete',epochs=5,steps=state['step'],best_epoch=state['best_epoch'],
                            best_is_untrained_initial=state['best_epoch']==0,last_epoch=5,curve=curve,
                            not_fully_converged_at_epoch5=improving,extra_epochs_started=False,
                            best=str(RUN/'checkpoints/best.pt'),last=str(path),micro=micro,effective_batch=128,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                        return
                    state.update(epoch=epoch+1,cursor=0,stage='train',train_loss_sum=0.,train_seen=0)
                    save()
                epoch=state['epoch']
                order=np.random.default_rng(SEED+epoch).permutation(bank.indices('train'))
                head.train()
                while state['cursor']<len(order):
                    ix=order[state['cursor']:state['cursor']+128]
                    ts=time.monotonic()
                    loss,norm=accumulated_update(head,opt,bank,ix,device,micro)
                    state['cursor']+=len(ix);state['step']+=1
                    state['train_loss_sum']+=loss*len(ix);state['train_seen']+=len(ix)
                    if state['step']%32==0: save()
                    if state['step']%16==0:
                        progress('training',epoch=epoch,cursor=state['cursor'],target=len(order),step=state['step'],loss=loss,grad_norm=norm,micro=micro,step_shared_sec=time.monotonic()-ts)
                state['stage']='validation';save()
        except torch.cuda.OutOfMemoryError:
            # A partial optimizer step must never become a checkpoint. Reload the
            # last durable optimizer boundary; at most31 completed steps replay.
            reductions+=1
            atomic_json(RUN/f'oom_training_{reductions}.json',dict(micro=micro,free_bytes=torch.cuda.mem_get_info()[0]))
            if micro==1 or reductions>5: raise
            micro=max(1,micro//2)
            atomic_json(RUN/'microbatch.json',dict(micro=micro))
            progress('training_oom_resume',micro=micro,reductions=reductions)
        finally:
            del head,opt
            gc.collect();torch.cuda.empty_cache()
