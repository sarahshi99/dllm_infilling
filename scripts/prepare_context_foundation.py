#!/usr/bin/env python3
"""CPU-only source verification and blind trajectory manifest. No test labels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collections
import json
import subprocess

from experiments.context_foundation.common import *


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from experiments.dreamon_singleline_adapter import ensure_modeling_rope_utils_available
    from experiments.dreamon_markov_head_training import _normalize_record, _canonical_json, _sha256_text
    from analysis.prepare_dreamon_markov_training_data import extract_code_block
    ensure_modeling_rope_utils_available()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol_manifest.json').exists():
        for entry in json.loads((OUT/'protocol_manifest.json').read_text())['data']:
            assert sha(REPO/entry['path']) == entry['sha256']
        print('Frozen protocol already verified; source preparation reused.')
        return
    spec = json.loads((REPO/'analysis_outputs/structure_coordinate_fix_20261015_v2/run_manifest.json').read_text())
    assert spec['fix_commit'] == '560e98aa84667fbaad7ee0caae99e76fdca01275'
    assert spec['head']['sha256'] == sha(HEAD)
    for name in ['prepared_data_reaudit.json', 'actual_bank_membership_audit_v2.json']:
        assert json.loads((OLD_DATA/name).read_text())['status'] == 'passed'
    summary = json.loads((OLD_DATA/'data_manifest_summary.json').read_text())
    split_path = OLD_DATA/'split_manifest.jsonl.zst'
    assert sha(split_path) == summary['split_manifest_sha256']
    with subprocess.Popen(['zstd', '-dc', str(split_path)], stdout=subprocess.PIPE, text=True) as proc:
        members = {r['record_id']: r for r in map(json.loads, proc.stdout)}
        assert proc.wait() == 0
    excluded = json.loads((OLD_DATA/'humaneval_exclusion_audit.json').read_text())
    excluded_ids = {r for g in excluded['groups'] for r in g['member_record_ids']}
    excluded_groups = {g['problem_group_id'] for g in excluded['groups']}
    assert len(excluded_groups) == 39 and len(excluded_ids) == 559
    group_splits, counts, seen, source_ids = {}, collections.Counter(), set(), set()
    eligible = []
    # External payload is never used; only identity/split counts are checked.
    for r in read_jsonl(PREPARED):
        rid, group, split = r['record_id'], r['problem_group_id'], r['split']
        assert rid not in seen and r['source_seq_id'] not in source_ids
        assert rid not in excluded_ids and group not in excluded_groups
        assert rid in members
        for field in ['problem_group_id', 'source_seq_id', 'split']:
            assert r[field] == members[rid][field]
        assert group_splits.setdefault(group, split) == split
        seen.add(rid)
        source_ids.add(r['source_seq_id'])
        counts[split] += 1
        if split in ('train', 'validation'):
            eligible.append(r)
    assert counts == summary['split_counts'] and len(seen) == 106491
    # Choose sources without outcomes/distribution differences, at most one record/group.
    chosen = []
    for split, target in [('train',768), ('validation',256)]:
        groups = set()
        for r in sorted((r for r in eligible if r['split'] == split), key=lambda r: stable(['source', r['record_id']])):
            if r['problem_group_id'] in groups:
                continue
            groups.add(r['problem_group_id'])
            chosen.append(r)
            if len(groups) == target:
                break
    raw_root = Path('/home/shx/.cache/huggingface/datasets/OpenCoder-LLM___opc-sft-stage2/educational_instruct/0.0.0/7d28f40d579edd7c24402d17d0c7639f991e6f8d')
    info = json.loads((raw_root/'dataset_info.json').read_text())
    assert any('@7d28f40d579edd7c24402d17d0c7639f991e6f8d/' in k for k in info['download_checksums'])
    source = Dataset.from_file(str(raw_root/'opc-sft-stage2-train.arrow'))
    assert len(source) == 118278
    wanted = {r['source_seq_id'] for r in chosen}
    raw = {r['seq_id']:r for r in source if r['seq_id'] in wanted}
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True, local_files_only=True)
    records, gaps = [], []
    for r in chosen:
        row = _normalize_record(raw[r['source_seq_id']])
        rid = _sha256_text(_canonical_json({k:row[k] for k in ['instruction','code','entry_point','testcase']}))
        assert rid == r['record_id']
        response_prefix, code, response_suffix = extract_code_block(row['output'], row['code'])
        lines = code.strip('\n').splitlines(keepends=True)
        for span, width in [('single',1), ('crossline',2)]:
            holes = [(start,start+width) for start in range(1,len(lines)-width)
                     if len(''.join(lines[start:start+width]).split()) >= 2]
            if not holes:
                gaps.append(dict(record_id=rid, span=span, reason='no_legal_source_hole'))
                continue
            start,end = min(holes, key=lambda h: stable(['hole',rid,span,h]))
            prefix = tok.apply_chat_template([{'role':'user','content':row['instruction']}], add_generation_prompt=True, tokenize=True)
            prefix += tok.encode(response_prefix+''.join(lines[:start]), add_special_tokens=False)
            suffix = tok.encode(''.join(lines[end:])+response_suffix+tok.eos_token, add_special_tokens=False)
            if len(prefix)+64+len(suffix) > 1024:
                gaps.append(dict(record_id=rid, span=span, reason='frozen_source_context_over_1024'))
                continue
            for order in ['global','frontier']:
                for arm in ['Z4','H4-old']:
                    trajectory_id = stable(['trajectory',rid,span,order,arm])
                    records.append(dict(trajectory_id=trajectory_id, case=rid+':'+span, record_id=rid,
                        source_seq_id=r['source_seq_id'], group=r['problem_group_id'], split=r['split'],
                        span=span, hole_start_line=start, hole_end_line=end, hole_line_count=width,
                        source_code_sha256=stable(code), prefix_ids=prefix, suffix_ids=suffix, order=order, arm=arm))
    records.sort(key=lambda r: stable(['trajectory_order',r['trajectory_id']]))
    write_jsonl(OUT/'blind_trajectories.jsonl.gz', records)
    atomic_json(OUT/'source_audit.json',dict(status='passed', contract_id=CONTRACT, schema_version=1,
        source=dict(dataset_id=summary['dataset_id'], config=summary['dataset_config'], revision=summary['dataset_revision'],
                    arrow_sha256=sha(raw_root/'opc-sft-stage2-train.arrow'), prepared_sha256=sha(PREPARED)),
        counts=dict(counts), unique_records=len(seen), cross_split_groups=0, excluded_groups=39,
        excluded_records=559, excluded_intersection=0, prepared_split_manifest_sha256=sha(split_path),
        selected_sources=len(chosen), blind_trajectories=len(records), source_hole_gaps=gaps,
        audit_reuse='Original full retokenization/decontamination audit; all prepared memberships rechecked; chosen source record hashes rederived. External test payload unused.',
        prior_audit_sha256={n:sha(OLD_DATA/n) for n in ['prepared_data_reaudit.json','actual_bank_membership_audit_v2.json','humaneval_exclusion_audit.json']}))
    special = sorted(set(tok.all_special_ids) | {151667} | set(range(len(tok),152064)))
    fields = dict(old_hidden='float[B,K,H]: entire selected set, target-aligned H0',
        position_ids='int64[B,K]: original X0 coordinates', chain_ids='int64[B,K]: contiguous chain IDs',
        selected_mask='bool[B,K]: valid slots', query_indices='int64[B,Q]: target slots',
        old_logits='float[B,Q,V]: old scores; needed only by adapter addition/sampling',
        token_ids='int64[B,K]: recorded or guessed tokens', token_state='int[B,K]: 0 unknown, 1 committed, 2 provisional',
        history_mask='bool[B,Q,K]: explicit same-chain earlier valid visibility; never inferred from padding/token values',
        latent='float[B,Q,256]', delta_logits='optional float[B,Q,V]')
    atomic_json(OUT/'protocol_schema.json',dict(contract_id=CONTRACT,schema_version=1, tensor_contract=fields,
        tensor_contract_text=[
            'old_hidden[B,K,H]：整个选中集合的目标对齐H0；position_ids[B,K]：X0原坐标；chain_ids[B,K]：链编号；selected_mask[B,K]：有效槽位。',
            'query_indices[B,Q]：要修正的目标在选中集合中的槽位；old_logits[B,Q,V]：这些目标的旧分数，仅adapter相加/采样时必需。',
            'token_ids[B,K]：记录或猜测；token_state[B,K]：0未知、1committed已确定、2provisional仅迭代猜测。',
            'history_mask[B,Q,K]：该query可读的同链更早有效槽位；不能靠padding/token值推断可见性。',
            '输出 latent[B,Q,256] 与可选 delta_logits[B,Q,V]；每个query严格遵守history_mask。锚点不修正。V/H/特殊token集合在配置中固定。A落盘保留完整selected-set旧h/原坐标/链/有效mask，不只存目标和已发生词。',
            'teacher状态库只含committed合法前缀。provisional只供H3/H4的Jacobi执行器，不能被标记为已经提交；本批A的真实生成不使用新Jacobi执行器。'],
        batch=1, max_k=4, rank=256, H=3584, V=152064, special_token_ids=special,
        anchors='no correction call', zero_override='strict zero delta through actual interface',
        teacher_bank='only legal committed prefixes; provisional only for future Jacobi execution, unused by A'))
    tokenizer_files = ['tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json','tokenization_dream.py']
    atomic_json(OUT/'frozen_protocol.json',dict(contract_id=CONTRACT, schema_version=1, baseline_commit=BASE,
        producer_branch=BRANCH, official_protocol=spec['official_protocol'], model=spec['model'],
        model_index_sha256=sha(SNAPSHOT/'model.safetensors.index.json'),
        model_runtime_sha256={n:sha(SNAPSHOT/n) for n in ['config.json','modeling_dream.py','configuration_dream.py']},
        model_shards={p.name:sha(p) for p in sorted(SNAPSHOT.glob('*.safetensors'))},
        tokenizer_sha256={n:sha(SNAPSHOT/n) for n in tokenizer_files}, head=spec['head'],
        model_dtype='bfloat16', head_dtype='float32', head_lambda=1, seed=42,
        differences_from_history=['common-u keyed inverse-CDF sampler replaces historical categorical RNG',
            'FP32 score addition, filtering, softmax and entropy; FP64 CDF; stable nucleus ties by ascending token ID',
            'global confidence ties by ascending original coordinate; entropy formula retains original epsilon 1e-10',
            'all tokenizer special and unassigned vocabulary IDs excluded from ordinary head/teacher memory'],
        sampler=dict(key='[contract_id,42,case,main_round,original_position,purpose]', purpose='token',
            uniform='SHA256 first52bits / 2^52 + 2^-53', inverse_cdf='ascending token ID; first CDF >= u; FP64 CDF normalized by final sum',
            nucleus='stable descending FP32 score, token ID tie; include first crossing .9, masked support exactly zero',
            trajectory_caveat='pairing device only; coordinates need not retain identical semantics after trajectories diverge; next round creates new keys after structure'),
        teacher='chain-local fresh from X0, only actual ordinary committed prefix of this chain; fixed attention/time/selection; not full-canvas refresh, K1, answer or upper bound',
        limits=dict(total_gpu_job_sec=43200, A1_A2_sec=14400, A1_candidate_collection_sec=7200, A3_sec=7200,
                    gpu_wait_sec=86400, gpu_wait_scope='cumulative across resumes; not charged as resident GPU time'),
        plan='smoke -> A1/A2 -> A3 and atomic DCD publication -> 4132 global rows -> GPU process exit -> CPU scoring/analysis/archive/commit/push',
        sampling='no outcome/teacher/head-distance selection; stable hash of encountered eligible states; at most8 per group/cell; per split and source branch quotas',
        metrics='full vocabulary FP32 online KL(fresh||q), TV at T1; deployment TV/support/infinite KL explicitly; special masses included',
        head_training=False, sealed_test_opened=False, new_jacobi_execution=False))
    write_jsonl(OUT/'source_hole_gaps.jsonl',gaps)
    protocol_paths = [OUT/n for n in ['blind_trajectories.jsonl.gz','source_audit.json','source_hole_gaps.jsonl','protocol_schema.json','frozen_protocol.json']]
    atomic_json(OUT/'protocol_manifest.json',dict(contract_id=CONTRACT,schema_version=1,status='frozen',
        source_commit=head_commit(), producer_branch=BRANCH, data=manifest_entries(protocol_paths)))
    print(f'Frozen {len(records)} blind trajectories; source membership verified; protocol ready for commit.')


if __name__ == '__main__':
    main()
