"""CPU-only paired supplement for H1's completed, separately archived repair."""
import json
from pathlib import Path
from analysis.analyze_context_foundation import paired_quality
from experiments.context_foundation.common import read_jsonl,atomic_json
from .dependency import RUN,OUT,PUBLIC


def main():
    repair=RUN.parent/'context_h1_eval_repair_20261111_v1'
    assert json.loads((RUN/'exit.json').read_text())['status']=='complete'
    assert json.loads((repair/'exit.json').read_text())['status']=='complete'
    ours=json.loads((RUN/'plan.json').read_text());theirs=json.loads((PUBLIC/'plan.json').read_text())
    keys=('seed','config','optimizer','lr','betas','eps','weight_decay','clip_norm','effective_batch','epochs','head_dtype','loss','train_states','validation_states','best')
    assert all(ours[k]==theirs[k] for k in keys)
    assert all(json.loads((p/'control_reuse.json').read_text())['reused'] for p in (RUN,repair))
    h3=[r for r in read_jsonl(RUN/'per_case.jsonl.gz') if r['arm']=='H3']
    h1=[r for r in read_jsonl(repair/'per_case.jsonl.gz') if r['arm']=='H1']
    assert len(h3)==len(h1)==1033
    assert len({r['task_id'] for r in h3})==1033
    assert {r['task_id'] for r in h3}=={r['task_id'] for r in h1}
    assert len({r['base_task_id'] for r in h3})==164
    result=paired_quality(h3+h1,'H3','H1')
    assert result['net']==sum(r['passed'] for r in h3)-sum(r['passed'] for r in h1)
    atomic_json(OUT/'paired_h1_supplement.json',dict(status='complete',h1_source=str(repair/'per_case.jsonl.gz'),
        protocol_fields_checked=list(keys),baseline_reuse_audits_match=True,result=result,
        original_h1_failure_archive_preserved=True,training_or_generation_started=False))
    summary=json.loads((OUT/'generation_comparison.json').read_text())
    table=summary['table']+[dict(arm='H1',passed=sum(r['passed'] for r in h1),N=1033)]
    order=['Z4','H4-old','H1','H2','H3','K1']
    h2=RUN.parent/'context_h2_training_20261107_v1'
    peer=[r for r in read_jsonl(h2/'per_case.jsonl.gz') if r['arm']=='H2']
    table.append(dict(arm='H2',passed=sum(r['passed'] for r in peer),N=len(peer)))
    rows={r['arm']:r for r in table}
    text=['# H1/H2/H3完整结果补充','',
        '同协议SingleLine/global/K4/1033行/164基础题；seed42，T=.2，top_p=.9，batch1。K1为逐步模型重算对照。H1使用已完成的dtype修复评测，不重训。','',
        '|方法|通过/N|通过率|','|---|---:|---:|']
    for arm in order:
        row=rows[arm];text.append(f"|{arm}|{row['passed']}/{row['N']}|{100*row['passed']/row['N']:.2f}%|")
    text+=['',f"H3−H1：救回{result['help']}，改错{result['harm']}，净增{result['net']}；百分点95%区间{result['ci95']}。"]
    for row in summary['comparisons']+[summary['h2']['result']]:
        text.append(f"H3−{row['b']}：救回{row['help']}，改错{row['harm']}，净增{row['net']}；百分点95%区间{row['ci95']}。")
    text+=['','区间使用164基础题分组的10000次配对bootstrap；上述通过率为行加权指标。H3相对零头有收益；相对旧头、H1和H2是否有确定优势，以对应区间为准。',
        'H3完整训练5轮/7815步，best和last均为第5轮，验证KL从0.2901485降至0.1210964。第5轮仍改善，标记尚未充分收敛，未自动追加训练。',
        '纯学习恢复比例39/129=30.23%。没有同数据Hctx和必要消融，不宣称多词信息有效或选择性注意力优于简单平均。',
        '原generation_comparison.json里的H1 pending来自旧失败归档；本补充使用已完成的独立H1补评测。原归档保留，准确配对见paired_h1_supplement.json。']
    (OUT/'completed_comparison.zh.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
