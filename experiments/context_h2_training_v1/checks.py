"""H2-only real public-state checks; no DreamOn is loaded."""
import gc
import time
import torch
from experiments.context_foundation.common import atomic_json
from experiments.context_h1_training_v1.train import initialize, accumulated_update
from experiments.context_heads_v1.contract import HeadBatch
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter
from .dependency import RUN,progress


def technical(bank):
    device='cuda:0'
    ix=[next(i for i,r in enumerate(bank.rows) if r['history_length']==n) for n in (1,2,3)]
    b,t=bank.batch(ix,device)
    h=initialize(bank.config,'H2-GRU').to(device)
    assert h.config.memory_mode=='full-memory'
    assert torch.equal(h(b).delta_logits,torch.zeros_like(b.old_logits))
    assert torch.equal(b.old_logits+h(b).delta_logits,b.old_logits)
    assert h.embedding.weight.count_nonzero() and h.gru.weight_hh.count_nonzero()
    opt=torch.optim.AdamW(h.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    losses=[];norms=[];start=time.monotonic()
    for _ in range(3):
        loss,norm=accumulated_update(h,opt,bank,ix,device,1)
        assert norm>0
        losses.append(loss);norms.append(norm)
    with torch.no_grad():
        full=h(b).delta_logits
        wrapper=KeywordHeadAdapter(h)
        for j in range(3):
            one=HeadBatch(**{k:v[j:j+1] if v is not None else None for k,v in vars(b).items()})
            torch.testing.assert_close(h.step(one).delta_logits,full[j:j+1],atol=2e-6,rtol=2e-5)
        # Exactly the BF16 values emitted by DreamOn, promoted as cached training
        # inputs are. This used to fail on the live generation path with FP32 heads.
        values=vars(b).copy();values['old_hidden']=b.old_hidden.bfloat16()
        expected=HeadBatch(**dict(values,old_hidden=values['old_hidden'].float()))
        torch.testing.assert_close(wrapper(**values)['delta_logits'],h(expected).delta_logits,atol=0,rtol=0)
    timings=[]
    selected=bank.indices('train')[:128]
    for _ in range(2):
        torch.cuda.synchronize();ts=time.monotonic()
        accumulated_update(h,opt,bank,selected,device,32)
        torch.cuda.synchronize();timings.append(time.monotonic()-ts)
    atomic_json(RUN/'pilot_measurement.json',dict(effective_batch=128,micro=32,step_shared_seconds=timings,
        projected_7815_train_steps_hours=timings[-1]*7815/3600,includes_shard_io=True,excludes_full_validation_and_generation=True,
        caveat='two disposable-head updates only; shared GPU load and shard access vary'))
    atomic_json(RUN/'technical_checks.json',dict(status='passed',head='H2-GRU',history_lengths=[1,2,3],
        zero_residual_exact=True,internal_random_initialization=True,finite_losses=losses,finite_nonzero_gradient_norms=norms,
        batch_step_consistent=True,bf16_live_adapter_matches_fp32_training=True,
        aligned_hidden='reuse H1 exact real-model audit; same cached target-aligned old_hidden, no additional shift',
        cpu_causality_padding_reset_checks='see cpu_checks.txt',smoke_optimizer_discarded=True,shared_gpu_sec=time.monotonic()-start))
    progress('technical_checks_complete',losses=losses,gradient_norms=norms)
    del h,opt,b,t,full,wrapper,expected,values
    gc.collect();torch.cuda.empty_cache()
