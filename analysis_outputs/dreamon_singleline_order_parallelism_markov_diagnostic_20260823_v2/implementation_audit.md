# Implementation audit

Valid implementation: commit `742b6b0` plus analysis commit pending. It reuses the released DreamOn logits shift, entropy confidence, prompt construction, sampling, expansion budget, and evaluator path.

- C1/C2/C4: select `min(K, active_masks)` global confidence ranks from the aligned active-mask logits; structural processing retains released `pad_eos_to_right`, expansion, and broadcast delete behavior.
- L1/L2/L4: choose only the leftmost contiguous unresolved masks. Normal proposals before the earliest selected structural proposal are committed; later old-coordinate proposals are discarded. The chosen structural action still executes the released EOS broadcast-delete behavior, then forces the next forward to use fresh state.
- v1 is invalidated because L delete actions were incorrectly converted to one-point deletion, omitting released EOS broadcast behavior. v2 corrects this and is the only reportable population.
- `confidence_top4` and leftmost contiguous positions are traced per v2 step. L1 computes full-vocabulary stale/fresh TV online for offsets 1/2/3, retaining only scalar records. Reference-token and fresh global-rank promotion fields were not collected; the report explicitly treats them as unavailable.
