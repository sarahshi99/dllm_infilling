# Paper Claim Rewrite

## Final Central Claim

Unknown-length diffusion language model infilling exhibits a measurable gap between canvas adequacy and rescue adequacy. Oracle-sufficient canvas can recover many failures across H200 SingleLine, Dream-Coder, and official second-regime data, but deployable length/canvas selection remains harm-prone and rescue/non-canvas limitations dominate substantial hard-tail, RandomSpan, and extreme-length cases.

## Three Main Contributions

1. A diagnostic decomposition of unknown-length infilling failures into canvas inadequacy, rescue/non-canvas limitations, and selection harm.
2. A sealed-test-preserving evidence base spanning H200 SingleLine attribution, Dream-Coder second-backbone audit, and official MultiLine/RandomSpan full allowed stress evidence.
3. A negative but actionable controller route closure showing that oracle action-bank headroom does not yet translate into a safe deployable controller under validation harm gates.

## Abstract Draft

Diffusion language models can infill code when the missing span length is known, but practical infilling requires deciding how much canvas to allocate without seeing the reference. We present a diagnostic study of unknown-length code infilling that separates canvas inadequacy from rescue inadequacy. On H200 SingleLine HumanEval infilling, oracle-sufficient canvas recovers many missed long-span failures, while already-triggered long failures remain rescue-limited under longer-trajectory and trace-remasking actions. A second-backbone Dream-Coder audit confirms substantial oracle-canvas recoverability but shows that the missed/triggered boundary is model-dependent. On the full allowed official MultiLine/RandomSpan infilling population, control fixed64 passes 2019/6707 cases, deployable cal-lite passes 1464/6707, and oracle-sufficient canvas passes 3180/6707, strengthening the mixed diagnostic claim while exposing deployable harm and rescue/non-canvas limitations. We close the current controller route as negative evidence: validation-visible selectors cannot safely harvest oracle headroom under sealed-test discipline. The result is not a new positive controller, but a claim-bounded map of where canvas expansion helps, where it harms, and where future rescue mechanisms are needed.

## Intro Contribution Bullets

- We quantify the canvas-rescue split for unknown-length DLLM code infilling and show that oracle canvas is a strong diagnostic tool but not a deployable method.
- We provide official second-regime stress evidence over 6707 non-frozen MultiLine/RandomSpan rows, showing both large oracle recoverability and substantial rescue/harm boundaries.
- We audit generalization on Dream-Coder and find recoverability transfers but qualitative strata are model-dependent.
- We preserve frozen-test discipline and report Controller V1/V2/V3 as route-closure evidence rather than overstating a weak deployable selector.

## Forbidden Claims

- No positive deployable controller claim.
- No frozen-test performance claim.
- No model-agnostic claim that Dream-Coder replicates the LLaDA missed-vs-triggered split.
- No official LR-DLLM reproduction claim.
- No synthetic easy-regime benchmark claim.
- No claim that hard-tail full104 is an unbiased benchmark aggregate.
