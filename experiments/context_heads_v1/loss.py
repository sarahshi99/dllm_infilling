"""Future common T=1 teacher KL only. No collection, CE, DCD, or training loop."""
import torch
from torch import Tensor


def teacher_kl(old_logits: Tensor, delta_logits: Tensor, teacher_logits: Tensor,
               query_mask: Tensor) -> Tensor:
    """Full-vocabulary KL(teacher || softmax(old+delta)), averaged over queries.

    Teacher: frozen DreamOn(X0 + visible committed prefix of THIS chain only).
    old_logits/teacher_logits [B,Q,V] are detached; gradients flow through delta.
    Full support and T=1 are mandatory. T=.2/top-p=.9 metrics belong elsewhere.
    """
    if old_logits.shape != delta_logits.shape or teacher_logits.shape != old_logits.shape:
        raise ValueError("all logits must have identical [B,Q,V] full support")
    if query_mask.shape != old_logits.shape[:-1] or query_mask.dtype != torch.bool:
        raise ValueError("query_mask must be bool[B,Q]")
    # Select first: ignored padding is allowed to contain NaN.
    student = (old_logits.detach()[query_mask].float() + delta_logits[query_mask].float())
    if not student.shape[0]:
        return delta_logits[query_mask].sum() * 0
    teacher = teacher_logits.detach()[query_mask].float().log_softmax(-1)
    return (teacher.exp() * (teacher - student.log_softmax(-1))).sum(-1).mean()
