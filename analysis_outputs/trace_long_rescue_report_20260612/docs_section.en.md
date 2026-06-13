## LLaDA-Base Full Trace Long-Rescue Diagnostics

Full trace collection completed for the previous local method and the current `midcons` method. These diagnostics use trace/decode dynamics for triggers and labels only for offline Gate A/B accounting; they are not a new SOTA claim.

| Run | Output | Rows | Trace rows | Pass rate |
| --- | --- | --- | --- | --- |
| previous local method trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552 | 1033 | 35257 | 769/1033 = 74.44% |
| current midcons trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846 | 1033 | 35768 | 795/1033 = 76.96% |

Offline route analysis:

| Trace source | Route | Triggers | Failed-long | Short | Current-pass risk | Gate A | Gate B | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous local method | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |
| current midcons | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |
