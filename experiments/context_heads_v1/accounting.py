"""Meta parameter counts and analytic dense-forward MACs, never latency claims."""
import torch

from .contract import HeadConfig
from .heads import ContextHead, MODEL_NAMES


def accounting(config: HeadConfig, *, batch=1, selected=4, queries=3):
    r, h, v, k, q, b = (config.rank, config.hidden_size, config.vocab_size,
                        selected, queries, batch)
    result = []
    for name in MODEL_NAMES:
        with torch.device("meta"):
            head = ContextHead(name, config)
        groups = {}
        for key, parameter in head.named_parameters():
            group = key.split(".")[0]
            groups[group] = groups.get(group, 0) + parameter.numel()
        tables = groups.get("embedding", 0) + groups["output"]
        if name == "H0-static":
            core = 0
        elif name == "Hctx-old-context-only":
            core = q*h*r + q*((r+3)*r + r*r)
        elif name == "Hgated-author-family":
            core = q*(h+r)*r
        elif name == "H1-context-prev":
            core = 2*q*h*r + q*((2*r+2)*r + r*r)
        elif name == "H2-GRU":
            core = (k+q)*h*r + q*k*(2*r+2)*r + q*k*6*r*r + q*((2*r+3)*r + r*r)
        elif name == "H3-short-attention":
            core = ((k+q)*h*r + q*k*(2*r+2)*r + q*r*r + 2*q*k*r*r
                    + 2*q*k*r + q*((2*r+3)*r + r*r))
        else:
            core = (q+k+1)*h*r + q*3*r + (q+k)*3*r*r + q*k*r + q*2*r*r
        result.append(dict(model=name, parameter_groups=groups,
                           embedding_parameters=groups.get("embedding", 0),
                           output_parameters=groups["output"], vocabulary_table_parameters=tables,
                           extra_parameters=sum(groups.values())-tables,
                           total_parameters=sum(groups.values()),
                           fp32_parameter_bytes=4*sum(groups.values()),
                           output_mask_buffer_bytes=4*v,
                           core_dense_MACs=b*core, full_vocab_projection_MACs=b*q*r*v,
                           total_dense_MACs=b*(core+q*r*v)))
    inventory = {
        "old_hidden_input": [b, k, h], "old_logits_input": [b, q, v],
        "latent_output": [b, q, r], "delta_logits_output": [b, q, v],
        "adapter_student_logits": [b, q, v],
        "mock_softmax_probabilities": [b, q, v], "mock_cdf": [b, q, v],
        "H2_H3_record_concat": [b, q, k, 2*r+2],
        "H2_H3_record_activations": [b, q, k, r],
        "H2_recurrent_states_per_unroll": [b, q, k, r],
        "H2_GRU_three_gate_values_per_unroll": [b, q, k, 3*r],
        "H3_keys_and_values_each": [b, q, 4, k, r//4],
        "H3_scores_and_weights_each": [b, q, 4, k],
        "H4_mixer_coefficients": [b, q, k, r],
        "H4_record_activations": [b, k, r],
    }
    activations = {}
    for key, shape in inventory.items():
        n = 1
        for d in shape:
            n *= d
        activations[key] = dict(shape=shape, fp32_elements=n, fp32_bytes=4*n)
    # Actual reference execution: no cached old projections; anchors excluded.
    sequential = {x["model"]: accounting_core_for_calls(config, x["model"], k, q, 1)
                  for x in result}
    jacobi = {x["model"]: accounting_core_for_calls(config, x["model"], k, q, 3)
              for x in result if x["model"] in ("H3-short-attention", "H4-causal-mixer")}
    return dict(config=vars(config), B=b, K=k, Q=q, models=result, activation_inventory=activations,
                sequential_K4=sequential, jacobi3_K4=jacobi,
                scope="Dense forward matmul MACs, 1 MAC=one multiply-accumulate; not measured timing or peak memory. "
                      "Bias adds, nonlinearities, mask/index ops, scalar distance/gate products, backward and allocator overhead excluded. "
                      "Full-vocab projections included even for zero override. "
                      "Inventory lists logical tensors (some coexist/copy); autograd saves additional intermediates. "
                      "The reference H2 recomputes prefix and GRU over all K masked slots for each query. "
                      "Mock sampling is softmax+CDF O(V) per projected row; A top-p sorting may be O(V log V).")


def accounting_core_for_calls(config, name, k, q, rounds):
    """Same formulas as above, factored to describe B=1 executors without recursion."""
    r, h, v = config.rank, config.hidden_size, config.vocab_size
    calls = q if rounds == 1 else rounds
    per_q = 1 if rounds == 1 else q
    if name == "H0-static":
        core = 0
    elif name == "Hctx-old-context-only":
        core = per_q*(h*r+(r+3)*r+r*r)
    elif name == "Hgated-author-family":
        core = per_q*(h+r)*r
    elif name == "H1-context-prev":
        core = per_q*(2*h*r+(2*r+2)*r+r*r)
    elif name == "H2-GRU":
        core = (k+per_q)*h*r+per_q*k*((2*r+2)*r+6*r*r)+per_q*((2*r+3)*r+r*r)
    elif name == "H3-short-attention":
        core = (k+per_q)*h*r+per_q*k*((2*r+2)*r+2*r*r+2*r)+per_q*(r*r+(2*r+3)*r+r*r)
    else:
        core = (per_q+k+1)*h*r+per_q*3*r+(per_q+k)*3*r*r+per_q*k*r+per_q*2*r*r
    rows = calls*per_q
    return dict(projection_calls=calls, positions_per_call=[per_q]*calls,
                projected_positions=rows, full_vocab_projection_MACs=rows*r*v,
                core_dense_MACs=calls*core, sampling_rows=rows,
                sampling_vocab_elements=rows*v,
                mock_sampling_complexity="O(rows*V), softmax then inverse CDF; no measured ms")
