# M3 Birth-Death Canvas Diffusion V0 Experiment Brief

M3 is an independent candidate method. It is neither an M1/M2 extension nor a selected paper method.

Each task begins with four inference-visible particles on canvas `16/32/64/128`. The uniform control runs all four through `64` denoising forwards (`256` total). The birth-death method has the same four-particle, `256`-forward budget; after rounds 15, 31, and 47 it may kill the weakest visible particle and birth a child on the best visible particle's canvas. A child is initialized only from the parent’s current decoded candidate state, never from reference or evaluator labels, and has a local schedule that finishes by the global final round.

Particle rank/death/birth/selection reads only candidate token state, final confidence, syntax compatibility of prefix+candidate+suffix, and prefix/suffix obligations/contradictions. It cannot read tests, verifier outcomes, reference/canonical code, oracle length, task ID/group/source ID, split label, or passed labels. Test code is constructed only for post-selection evaluation. Every raw method output is isolated, resumable, deduplicated, and audited for missing/duplicate/error plus exact `256` forwards per task.

The route is 12-case technical smoke then automatic 148-task RandomSpanLight full with no performance gate. Analysis compares uniform fixed-grid and birth-death at equal compute using equal-weight task-macro accuracy as primary, descriptive span-micro accuracy, paired wins/losses, selected canvases, birth/death event counts, and wall time. Frozen test remains sealed with `test_evaluation_count=0`.
