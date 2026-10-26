"""Example for A: replace this boundary with A's already aligned tensor producer.

Nothing imports or edits A's runner. The real sampler, structure scheduler,
canvas mutation, position alignment, and ordinary/structural support stay in A.
"""
import torch

from .contract import HeadBatch


def corrected_scores(head, aligned_tensors, *, zero_override=False):
    """[B,Q,V] old_logits + lambda(=1)*delta; differentiable for future KL.

    aligned_tensors contains exactly HeadBatch's fields, including FULL selected
    set old_hidden[B,K,H], coordinates/chains/valid masks and explicit history.
    Q can be 1 (step) or a teacher-forced query batch. No hidden-position shift.
    Teacher batches must have only legal committed prefixes; no provisional.
    """
    batch = HeadBatch(**aligned_tensors)
    if batch.old_logits is None:
        raise ValueError("A must supply aligned target old_logits for score addition")
    out = head(batch, zero_override=zero_override)
    return batch.old_logits.detach() + out.delta_logits


def cpu_mock_example():
    from .heads import ContextHead
    from .contract import HeadConfig
    from .test_cpu import fixture

    head = ContextHead("H1-context-prev", HeadConfig(11, 7, (10,)))
    tensors = vars(fixture(b=1))
    with torch.no_grad():
        scores = corrected_scores(head, tensors)
        assert scores.shape == (1, 4, 11)
        assert torch.equal(scores[:, 0], tensors["old_logits"][:, 0])
        assert torch.equal(corrected_scores(head, tensors, zero_override=True), tensors["old_logits"])
    return scores.shape


if __name__ == "__main__":
    print(cpu_mock_example())
