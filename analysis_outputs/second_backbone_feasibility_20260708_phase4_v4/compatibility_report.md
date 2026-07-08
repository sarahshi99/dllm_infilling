# Second-Backbone Feasibility Audit

Verdict: `recommended_backbone_available`.

The most feasible second backbone is `Dream-org/Dream-Coder-v0-Base-7B`: it is cached locally, has a dedicated official-canvas infilling runner, and has existing full SingleLine evidence. It requires tokenizer/canvas adaptation, but no new backbone weights or evaluator changes.

Dream-Coder does not expose the LLaDA trace-remasking E/F/G action family. For Phase 4, the diagnostic action set is primary/cal-lite, current best simple length policy, and oracle-sufficient canvas.
