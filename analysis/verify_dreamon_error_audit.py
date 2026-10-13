"""Fresh verification of new joins, labels, coverage, and portable HTML data."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_dreamon_markov_errors import OUT, bundle, read_jsonl, write_json, utc


def verify():
    source,problems,joined=bundle()
    compact=read_jsonl(OUT/'per_case_compact.jsonl')
    assert compact==source
    assert len(compact)==10330
    assert all(type(r['passed']) is bool for r in compact)
    assert all(n==1033 for n in Counter((r['policy'],r['arm']) for r in compact).values())
    for policy in {r['policy'] for r in compact}:
        for arm in 'ABCDE':
            assert {r['task_id'] for r in compact if r['policy']==policy and r['arm']==arm}==set(problems)
    assert all(r['base_task_id']==r['task_id'].split('/')[2] and r['line_id']==r['task_id'].split('/')[3] for r in compact)
    queue=read_jsonl(OUT/'review_queue.jsonl')
    reviews=read_jsonl(OUT/'semantic_reviews.jsonl')
    assert len(queue)==len(reviews)==326
    assert sum(c['status']=='reviewed' for c in read_jsonl(OUT/'all_cases.jsonl'))==326
    assert all(q['status']=='reviewed' for q in queue)
    assert {q['stable_id'] for q in queue}=={p+'::'+t for (p,t),a in joined.items() if any(r['passed'] is False for r in a.values())}
    labels=read_jsonl(OUT/'failure_labels.jsonl')
    assert len(labels)==1220
    assert len({(f['stable_id'],f['arm']) for f in labels})==1220
    assert all(f['first_generation_divergence']=='unknown' and f['review_status']=='reviewed' for f in labels)
    assert not any(f['observable_label']=='证据不足' for f in labels)
    for f in labels:
        r=joined[f['policy'],f['task_id']][f['arm']];p=problems[f['task_id']]
        assert r['passed'] is False
        assert f['output']==r['output'] and f['official_failure_reason']==r['failure_reason']
        assert f['candidate_program']==p['prompt']+r['output']+p['suffix']+'\n'+p['test']+'\n'+f"check({p['entry_point']})"
        assert f['explanation_zh'].strip()
    for rv,q in zip(sorted(reviews,key=lambda x:x['index']),queue):
        assert set(''.join(rv['notes']))==set(q['failed_arms'])
    joint=list(csv.DictReader((OUT/'joint_outcomes.csv').open()))
    assert len(joint)==80
    for policy in {q['policy'] for q in queue}:
        for axes,n in (('ABCDE',32),('ABD',8)):
            cells=[r for r in joint if r['policy']==policy and r['axes']==axes]
            assert len(cells)==n and sum(int(r['count']) for r in cells)==1033
            ids=[x for r in cells for x in json.loads(r['stable_ids'])]
            assert len(ids)==len(set(ids))==1033
            for cell in cells:
                assert len({x.split('/')[-2] for x in json.loads(cell['stable_ids'])})==int(cell['base_tasks'])
    pairs=list(csv.DictReader((OUT/'paired_recheck.csv').open()))
    assert len(pairs)==10
    for r in pairs:
        cells=json.loads(r['cells_ids']);assert sum(map(len,cells.values()))==1033
        new,old=r['comparison'].split('-')
        subset={t:a for (p,t),a in joined.items() if p==r['policy']}
        direct_help={r['policy']+'::'+t for t,a in subset.items() if not a[old]['passed'] and a[new]['passed']}
        reverse_harm={r['policy']+'::'+t for t,a in subset.items() if a[new]['passed'] and not a[old]['passed']}
        assert direct_help==reverse_harm==set(json.loads(r['help_ids']))
        assert int(r['harm'])==sum(a[old]['passed'] and not a[new]['passed'] for a in subset.values())
        assert r['difference'] in ('0','')
    replay=read_jsonl(OUT/'official_test_replay.jsonl')
    assert len(replay)==1220
    timeouts=[r for r in replay if r['original_failure_reason']=='timed out']
    assert len(timeouts)==13 and all(not r.get('reused_identical_program') and 'instrumented' not in r for r in timeouts)
    assert all(r['official']['audit_failure_reason']=='timed out' and r['official']['child_exitcode']==0 for r in timeouts)
    assert all(r['instrumentation_agrees'] is True for r in replay if 'instrumented' in r)
    assert all('infrastructure_error' not in r.get('official',{}) and 'infrastructure_error' not in r.get('instrumented',{}) for r in replay)
    historical_differences=[dict(policy=r['policy'],arm=r['arm'],task_id=r['task_id'],
        original=r['original_failure_reason'],audit=r['official']['audit_failure_reason'])
        for r in replay if 'official' in r and r['original_failure_reason']!=r['official']['audit_failure_reason']]
    document=(OUT/'casebook.html').read_text()
    embedded=re.search(r'<script type="application/json" id="audit-data">(.*?)</script>',document,re.S).group(1)
    assert '<' not in embedded
    data=json.loads(embedded)
    assert len(data['cases'])==2066 and sum(c['status']=='reviewed' for c in data['cases'])==326
    assert sum(len(c['arms']) for c in data['cases'])==10330
    assert len(data['contexts'])==1033
    for c in data['cases']:
        assert all(c['arms'][a]==joined[c['policy'],c['task_id']][a] for a in 'ABCDE')
    page_script=re.findall(r'<script>(.*?)</script>',document,re.S)[0]
    assert 'innerHTML' not in page_script and 'eval(' not in page_script
    write_json(OUT/'verification.json',dict(status='passed',verified_utc=utc(),compact_exact=True,joins_and_labels=True,
        joint_cells=80,paired_tables=10,symmetric_pairs=True,failures=1220,reviewed=326,
        timeout_replays_once=13,instrumentation_disagreements=0,historical_replay_differences=historical_differences,html_embedded_cases=2066,html_output_records=10330,
        browser='no browser tool/binary or playwright/puppeteer installed; Node DOM interaction verification separately recorded',
        static_html_safety='all data escaped in application/json; DOM textContent; no eval/innerHTML/CDN'))
    print('Fresh data verification passed: 10330 original rows, 326 reviews, 1220 failures, 80 joint cells, 10 symmetric pair tables.')


if __name__=='__main__':verify()
