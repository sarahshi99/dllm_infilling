"""Frozen-head finite K2 I-projection diagnostic. NumPy/stdlib only; never loads a model."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np

CONTRACT = 'dreamon-context-v1-20261015'
PRODUCER = 'codex/dreamon-context-foundation-20261015-v1'
SEED = 20261015
TOL = 1e-6
MAX_ITER = 1000
BUCKETS = tuple(f'{o}/{s}' for o in ('global', 'frontier') for s in ('single', 'crossline'))


class InputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InputError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def portable(value):
    """Strict JSON: nonfinite metrics are explicit strings, never silent NaN/drop."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [portable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ('Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN')
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(portable(value), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def lse(x, axis=None):
    # logaddexp handles all-minus-infinity rows without epsilon or inf-inf.
    return np.logaddexp.reduce(np.asarray(x, dtype=np.float64), axis=axis)


def lognormalize(x):
    x = np.asarray(x, dtype=np.float64)
    z = lse(x.reshape(-1))
    require(np.isfinite(z), 'empty or invalid probability support')
    return x-z


def kl(logt, logm):
    logt, logm = np.asarray(logt), np.asarray(logm)
    mask = np.isfinite(logt)
    if np.any(~np.isfinite(logm[mask])):
        return float('inf')
    return float(np.sum(np.exp(logt[mask])*(logt[mask]-logm[mask])))


def conditional_kl(logt, logm):
    tr, mr = lse(logt, 1), lse(logm, 1)
    total = 0.
    for i in range(len(tr)):
        if not np.isfinite(tr[i]):
            continue
        if not np.isfinite(mr[i]):
            return float('inf')
        value = kl(logt[i]-tr[i], logm[i]-mr[i])
        if not np.isfinite(value):
            return value
        total += float(np.exp(tr[i]))*value
    return total


def support_feasible(support, r, c):
    """Small bipartite transport max-flow: detects Hall failures, not just empty rows.

    Used only for structural zeros; positive matrices are automatically feasible.
    FP64 capacity resolution is audited separately from mathematical log support.
    """
    nr, nc = support.shape
    size, source, sink = nr+nc+2, nr+nc, nr+nc+1
    residual = np.zeros((size, size), dtype=np.float64)
    residual[source, :nr] = r
    residual[:nr, nr:nr+nc] = support.astype(float)
    residual[nr:nr+nc, sink] = c
    flow = 0.
    while True:
        parent = np.full(size, -1)
        parent[source] = source
        queue = [source]
        for u in queue:
            for v in np.flatnonzero(residual[u] > 0):
                if parent[v] == -1:
                    parent[v] = u
                    queue.append(int(v))
            if parent[sink] != -1:
                break
        if parent[sink] == -1:
            return flow >= 1.-1e-12
        v, capacity = sink, 1.
        while v != source:
            u = parent[v]
            capacity = min(capacity, residual[u, v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= capacity
            residual[v, u] += capacity
            v = u
        flow += capacity


def ipf(logr, log_r, log_c, tol=TOL, max_iter=MAX_ITER):
    """Argmin KL(Q||R) under row/column constraints; no smoothing."""
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    q = lognormalize(logr).copy()
    lr, lc = lognormalize(log_r), lognormalize(log_c)
    require(q.shape == (len(lr), len(lc)), 'IPF target dimension mismatch')
    active_r, active_c = np.isfinite(lr), np.isfinite(lc)
    r, c = np.exp(lr), np.exp(lc)
    q[~active_r, :] = -np.inf
    q[:, ~active_c] = -np.inf
    support = np.isfinite(q)

    def result(status, iterations, error):
        return dict(logq=q, status=status, iterations=iterations, max_margin_error=float(error),
                    wall_seconds=time.perf_counter()-start_wall, cpu_seconds=time.process_time()-start_cpu,
                    structural_zeros=int(np.isneginf(logr).sum()),
                    target_fp64_underflow=int(((r == 0) & active_r).sum()+((c == 0) & active_c).sum()),
                    output_fp64_underflow=int(((np.exp(q) == 0) & np.isfinite(q)).sum()))

    if np.any(active_r & ~support.any(1)) or np.any(active_c & ~support.any(0)):
        return result('support_infeasible', 0, float('inf'))
    if not support[np.ix_(active_r, active_c)].all() and not support_feasible(support, r, c):
        return result('support_infeasible', 0, float('inf'))
    for iteration in range(max_iter+1):
        rows, cols = lse(q, 1), lse(q, 0)
        error = max(np.max(np.abs(np.exp(rows)-r)), np.max(np.abs(np.exp(cols)-c)))
        if error <= tol:
            return result('converged', iteration, error)
        if iteration == max_iter:
            return result('max_iterations', iteration, error)
        q[active_r] += (lr[active_r]-rows[active_r])[:, None]
        cols = lse(q, 0)
        q[:, active_c] += (lc[active_c]-cols[active_c])[None, :]
    raise AssertionError('unreachable')


def validate_identity(value, label):
    require(value.get('contract_id') == CONTRACT and value.get('schema_version') == 1,
            label+': incompatible contract/schema')
    require(value.get('producer_branch') == PRODUCER, label+': unexpected producer branch')
    require(bool(re.fullmatch('[0-9a-f]{40}', str(value.get('source_commit', '')))), label+': missing source commit')


def validate_numeric(arrays, manifest):
    n = len(arrays['state_id'])
    audit = {}
    for prob, lp, shape in [('old_p1', 'old_logp1', (n, 16)), ('old_p2', 'old_logp2', (n, 16)),
                            ('head_cond2', 'head_logcond2', (n, 16, 16)),
                            ('fresh_cond2', 'fresh_logcond2', (n, 16, 16))]:
        require(prob in arrays and arrays[prob].shape == shape, 'missing/wrong probability shape: '+prob)
        raw = arrays[prob]
        require(raw.dtype.kind == 'f', 'probability must be floating point: '+prob)
        p = np.asarray(raw, dtype=np.float64)
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'invalid probabilities: '+prob)
        if lp not in arrays:
            require(manifest.get('stable_fp64_full_support') is True and raw.dtype == np.float64 and (p > 0).all(),
                    'numerical blockage: missing authoritative log field '+lp)
            arrays[lp] = np.log(p)
            authority = 'explicitly attested stable FP64 positive probabilities'
        else:
            authority = 'producer full-vocabulary log_softmax'
        logs = np.asarray(arrays[lp], dtype=np.float64)
        require(logs.shape == shape and not np.isnan(logs).any() and not np.isposinf(logs).any()
                and (logs <= 1e-6).all(), 'invalid log probabilities: '+lp)
        nonzero = p > 0
        require(np.isfinite(logs[nonzero]).all(), 'probability/log support mismatch: '+lp)
        # Positive subnormals have large relative rounding error. Permit only the
        # absolute rounding uncertainty implied by their original dtype.
        smallest = float(np.nextafter(np.array(0, dtype=raw.dtype), np.array(1, dtype=raw.dtype)))
        expected = np.exp(logs)
        require(np.all(np.abs(p[nonzero]-expected[nonzero]) <= 2e-5*expected[nonzero]+smallest),
                'probability/log mismatch: '+lp)
        underflow = (p == 0) & np.isfinite(logs)
        require(np.all(logs[underflow] <= math.log(smallest/2) + 1e-4) if smallest/2 > 0
                else np.all(logs[underflow] <= math.log(smallest)-math.log(2)+1e-4),
                'unexplained probability zero: '+lp)
        mass = np.exp(lse(logs, -1))
        require((mass <= 1+5e-6).all(), 'candidate mass exceeds full vocabulary: '+lp)
        arrays[lp] = logs
        audit[prob] = dict(authority=authority, probability_underflow=int(underflow.sum()),
                           structural_zeros=int(np.isneginf(logs).sum()),
                           extracted_fp64_underflow=int(((expected == 0) & np.isfinite(logs)).sum()))
    for j in (1, 2):
        ids = arrays.get(f'candidate_ids_{j}')
        require(ids is not None and ids.shape == (n, 16) and ids.dtype.kind in 'iu' and (ids >= 0).all(),
                'invalid candidate IDs')
        require(all(len(set(row.tolist())) == 16 for row in ids), 'duplicate candidate IDs')
    for prefix in ('old', 'head', 'fresh'):
        suffixes = ('1', '2') if prefix == 'old' else ('',)
        for suffix in suffixes:
            candidate = prefix+'_candidate_mass'+suffix
            special = prefix+'_special_mass'+suffix
            expected_shape = (n,) if prefix == 'old' else (n, 16)
            lp = 'old_logp'+suffix if prefix == 'old' else prefix+'_logcond2'
            mass = np.exp(lse(arrays[lp], -1))
            for key in (candidate, special):
                require(key in arrays and arrays[key].shape == expected_shape and np.isfinite(arrays[key]).all()
                        and (arrays[key] >= 0).all() and (arrays[key] <= 1+5e-6).all(), 'missing/invalid mass audit: '+key)
            require(np.allclose(arrays[candidate], mass, rtol=2e-5, atol=1e-7), 'candidate mass/log mismatch: '+candidate)
            require((mass+arrays[special] <= 1+5e-6).all(), 'special and ordinary mass inconsistent: '+special)
    return audit


def verify_package(ready_path):
    """Read only atomically published packages; verify bytes before decoding NPZ."""
    ready_path = Path(ready_path)
    ready = json.loads(ready_path.read_text())
    validate_identity(ready, 'ready')
    require(ready.get('status') in ('complete', 'partial'), 'ready status is not analyzable')
    require(isinstance(ready.get('created_at'), str) and bool(ready['created_at']), 'ready missing created_at')
    path = Path(ready.get('manifest_absolute_path', ''))
    require(path.is_absolute() and path.is_file(), 'manifest absolute path missing')
    require(sha256(path) == ready.get('manifest_sha256'), 'manifest hash mismatch')
    manifest = json.loads(path.read_text())
    validate_identity(manifest, 'manifest')
    require(manifest['status'] == ready['status'], 'manifest/ready status mismatch')
    require(manifest.get('T') == 1 and manifest.get('top_p', 1) in (1, None)
            and 'log_softmax' in manifest.get('probability_authority', ''),
            'incompatible probability regime: require temperature 1 full-support log_softmax; no top-p mixing')
    require(manifest.get('expected') == 256 and isinstance(manifest.get('count'), int)
            and 0 <= manifest['count'] <= 256, 'invalid declared count/expected')
    for label, digest in [('head', manifest.get('head', {}).get('sha256')),
                          ('sampler adapter', manifest.get('adapter_sha256'))]:
        require(bool(re.fullmatch('[0-9a-f]{64}', str(digest))), 'missing SHA256: '+label)
    shards = manifest.get('model_shards_sha256', {})
    require(shards and all(re.fullmatch('[0-9a-f]{64}', str(x)) for x in shards.values()), 'missing model shard SHA256')
    require(manifest.get('model', {}).get('revision'), 'missing model revision')
    repo = Path(subprocess.check_output(['git', '-C', str(path.parent), 'rev-parse', '--show-toplevel'], text=True).strip())
    entries = manifest.get('data', [])
    require(entries, 'manifest has no shards')
    resolved, seen = [], set()
    for entry in entries:
        rel = Path(entry['path'])
        p = (repo/rel).resolve()
        require(not rel.is_absolute() and repo in p.parents and p not in seen, 'invalid/duplicate manifest shard path')
        require(p.is_file() and sha256(p) == entry.get('sha256'), 'shard hash mismatch: '+str(rel))
        seen.add(p)
        resolved.append((p, entry))
    npzs, metadata = [], []
    for p, entry in resolved:
        if p.suffix == '.npz':
            with np.load(p, allow_pickle=False) as z:
                a = {key: z[key].copy() for key in z.files}
            require('state_id' in a and a['state_id'].dtype.kind in 'US' and a['state_id'].ndim == 1,
                    'NPZ needs non-pickle string state_id')
            require(all(x.ndim >= 1 and len(x) == len(a['state_id']) and x.dtype.kind != 'O' for x in a.values()),
                    'invalid NPZ state dimension/object dtype')
            require(entry.get('rows') == len(a['state_id']), 'NPZ rows mismatch')
            npzs.append(a)
        elif p.name.endswith('.jsonl'):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            require(entry.get('rows') == len(rows), 'metadata rows mismatch')
            metadata.extend(rows)
        else:
            raise InputError('unsupported declared shard format: '+p.name)
    require(npzs and all(set(z) == set(npzs[0]) for z in npzs), 'missing/inconsistent NPZ shards')
    arrays = {key: np.concatenate([z[key] for z in npzs]) for key in npzs[0]}
    arrays['state_id'] = arrays['state_id'].astype('U')
    ids = [str(x) for x in arrays['state_id']]
    require(len(ids) == manifest['count'] and len(ids) == len(set(ids)), 'actual N/state IDs mismatch')
    require(len(metadata) == len(ids), 'metadata count mismatch')
    index = {m.get('state_id'): m for m in metadata}
    require(len(index) == len(metadata) and set(index) == set(ids), 'metadata/state ID mismatch')
    metadata = [index[key] for key in ids]
    parent_groups, group_splits = {}, {}
    for m in metadata:
        require(all(m.get(key) is not None and str(m[key]) for key in ('group', 'split', 'source_state_id')),
                'missing source group/split/parent identity')
        require(m.get('order') in ('global', 'frontier') and m.get('span') in ('single', 'crossline')
                and m['split'] in ('train', 'validation'), 'unexpected source stratum')
        parent = str(m['source_state_id'])
        identity = (str(m['group']), m['split'])
        require(parent not in parent_groups or parent_groups[parent] == identity, 'same parent crosses source groups/splits')
        parent_groups[parent] = identity
        group = str(m['group'])
        require(group not in group_splits or group_splits[group] == m['split'], 'same source group crosses train/validation')
        group_splits[group] = m['split']
        for key, expected in [('model_revision', manifest['model']['revision']),
                              ('head_sha256', manifest['head']['sha256']),
                              ('adapter_sha256', manifest['adapter_sha256'])]:
            require(key not in m or m[key] == expected, 'metadata provenance mismatch: '+key)
    counts = {key: sum(m['order']+'/'+m['span'] == key for m in metadata) for key in BUCKETS}
    require(all(x <= 64 for x in counts.values()), 'bucket exceeds predeclared 64')
    require(manifest.get('coverage_gap') == 256-len(ids), 'coverage gap disagrees with actual N')
    if manifest['status'] == 'complete':
        require(len(ids) == 256 and set(counts.values()) == {64}, 'complete package is not 256/four balanced buckets')
    else:
        missing = {key: 64-counts[key] for key in BUCKETS}
        require(manifest.get('missing_buckets') == missing, 'partial manifest must explicitly list exact missing_buckets')
        stop = manifest.get('budget_stop')
        require(isinstance(stop, dict) and isinstance(stop.get('stopped'), bool) and bool(stop.get('reason')),
                'partial manifest must explicitly list budget_stop.stopped and reason')
    audit = validate_numeric(arrays, manifest)
    # Catch a producer rewriting files during validation instead of trusting existence.
    require(sha256(path) == ready['manifest_sha256'] and json.loads(ready_path.read_text()) == ready,
            'ready/manifest changed while validating')
    for p, entry in resolved:
        require(sha256(p) == entry['sha256'], 'shard hash changed while validating: '+p.name)
    return dict(ready=ready, manifest=manifest, arrays=arrays, metadata=metadata,
                numeric_audit=audit, counts=counts, ready_path=str(ready_path.resolve()))


def state_diagnostic(arrays, metadata, i):
    wall, cpu = time.perf_counter(), time.process_time()
    lp1, lp2 = arrays['old_logp1'][i], arrays['old_logp2'][i]
    lh, lf = arrays['head_logcond2'][i], arrays['fresh_logcond2'][i]
    lr, lc = lognormalize(lp1), lognormalize(lp2)
    rawr, rawt = lp1[:, None]+lh, lp1[:, None]+lf
    logr, logt = lognormalize(rawr), lognormalize(rawt)
    tr, tc = lse(logt, 1), lse(logt, 0)
    projected, oracle = ipf(logr, lr, lc), ipf(logr, tr, tc)
    row = dict(metadata)
    row['group'] = str(row['group'])
    row['candidate_ids_1'] = arrays['candidate_ids_1'][i].tolist()
    row['candidate_ids_2'] = arrays['candidate_ids_2'][i].tolist()
    row.update(old_mass1=float(np.exp(lse(lp1))), old_mass2=float(np.exp(lse(lp2))),
               R_raw_mass=float(np.exp(lse(rawr.reshape(-1)))), T_raw_mass=float(np.exp(lse(rawt.reshape(-1)))),
               log_R_raw_mass=float(lse(rawr.reshape(-1))), log_T_raw_mass=float(lse(rawt.reshape(-1))))
    for key in ('old_mass1', 'old_mass2', 'R_raw_mass', 'T_raw_mass'):
        row[key+'_tail'] = max(0., 1-row[key])
    row['high_coverage'] = (row['old_mass1'] >= .95 and row['old_mass2'] >= .95
                            and row['R_raw_mass'] >= .90 and row['T_raw_mass'] >= .90)
    for name, lp in [('r', lr), ('c', lc), ('R_row', lse(logr, 1)), ('R_col', lse(logr, 0)),
                     ('T_row', tr), ('T_col', tc)]:
        row[name+'_marginal'] = np.exp(lp).tolist()
    for name, lp in [('R', logr), ('T', logt)]:
        for axis, target, suffix in [(1, lr, 'row'), (0, lc, 'col')]:
            observed = lse(lp, axis)
            row[name+'_'+suffix+'_target_tv'] = float(.5*np.abs(np.exp(observed)-np.exp(target)).sum())
            row[name+'_'+suffix+'_target_maxabs'] = float(np.abs(np.exp(observed)-np.exp(target)).max())
    distributions = {'I': lr[:, None]+lc, 'R': logr, 'Q': projected['logq'], 'Q_T': oracle['logq']}
    for name, lp in distributions.items():
        valid = name not in ('Q', 'Q_T') or (projected if name == 'Q' else oracle)['status'] == 'converged'
        row['kl_'+name] = kl(logt, lp) if valid else None
        row['tv_'+name] = float(.5*np.abs(np.exp(logt)-np.exp(lp)).sum()) if valid else None
        row['conditional_kl_'+name] = conditional_kl(logt, lp) if valid else None
        row['row_kl_'+name] = kl(tr, lse(lp, 1)) if valid else None
        row['col_kl_'+name] = kl(tc, lse(lp, 0)) if valid else None
        row['T_support_missing_in_'+name] = int((np.isfinite(logt) & ~np.isfinite(lp)).sum())
    for name, result in [('Q', projected), ('Q_T', oracle)]:
        row.update({name+'_'+key: value for key, value in result.items() if key != 'logq'})
    for prefix in ('kl', 'tv', 'conditional_kl', 'row_kl'):
        a, b = row[prefix+'_Q'], row[prefix+'_R']
        row['delta_'+prefix+'_Q_R'] = a-b if a is not None and np.isfinite(a) and np.isfinite(b) else None
    d, dr, dc = (row['delta_'+key+'_Q_R'] for key in ('kl', 'row_kl', 'conditional_kl'))
    row['chain_rule_residual'] = d-dr-dc if all(x is not None for x in (d, dr, dc)) else None
    row['first_margin_share_of_kl_gain'] = dr/d if d is not None and d < -1e-12 else None
    row['conditional_kl_improved'] = dc < -1e-8 if dc is not None else None
    row['oracle_theory_applicable'] = bool(np.isfinite(logr).all() and np.isfinite(logt).all())
    row['oracle_kl_nonincrease'] = (row['kl_Q_T'] <= row['kl_R']+1e-5
                                    if row['kl_Q_T'] is not None and np.isfinite(row['kl_R']) else None)
    for prefix, logcond in [('head', lh), ('fresh', lf)]:
        mass = np.exp(lse(logcond, 1))
        row[prefix+'_candidate_mass_by_a'] = mass.tolist()
        row[prefix+'_tail_by_a'] = np.maximum(0., 1-mass).tolist()
        row[prefix+'_special_mass_by_a'] = arrays[prefix+'_special_mass'][i].tolist()
        row[prefix+'_candidate_mass_r_weighted'] = float(np.dot(np.exp(lr), mass))
        row[prefix+'_tail_r_weighted'] = max(0., 1-row[prefix+'_candidate_mass_r_weighted'])
        row[prefix+'_special_mass_r_weighted'] = float(np.dot(np.exp(lr), arrays[prefix+'_special_mass'][i]))
    for prob, lp in [('old_p1', lp1), ('old_p2', lp2), ('head_cond2', lh), ('fresh_cond2', lf)]:
        row[prob+'_input_probability_underflow'] = int(((arrays[prob][i] == 0) & np.isfinite(lp)).sum())
        row[prob+'_input_structural_zeros'] = int(np.isneginf(lp).sum())
    for j in (1, 2):
        row[f'old_special_mass{j}'] = float(arrays[f'old_special_mass{j}'][i])
    row['support_log_zero_R'] = int(np.isneginf(logr).sum())
    row['support_log_zero_T'] = int(np.isneginf(logt).sum())
    row['normalized_R_fp64_underflow'] = int(((np.exp(logr) == 0) & np.isfinite(logr)).sum())
    row['normalized_T_fp64_underflow'] = int(((np.exp(logt) == 0) & np.isfinite(logt)).sum())
    row['cpu_seconds'] = time.process_time()-cpu
    row['wall_seconds'] = time.perf_counter()-wall
    return portable(row)


def finite_number(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def cluster_bootstrap(rows, field, reps=2000):
    """Paired source-group bootstrap, retaining all children of each sampled group."""
    valid = [r for r in rows if finite_number(r.get(field))]
    groups = sorted({r['group'] for r in rows})
    result = dict(n=len(rows), available=len(valid), unavailable=len(rows)-len(valid), groups=len(groups),
                  mean=None, ci95=None, wins=None, losses=None, ties=None)
    if not rows or len(valid) != len(rows):
        return result  # Never turn a favorable complete-case subset into the primary table.
    values = np.asarray([r[field] for r in rows], dtype=float)
    result['mean'] = float(values.mean())
    if field.startswith('delta'):
        result.update(wins=int((values < -1e-8).sum()), losses=int((values > 1e-8).sum()),
                      ties=int((np.abs(values) <= 1e-8).sum()))
    if len(groups) < 2:
        return result
    sums = np.array([sum(r[field] for r in rows if r['group'] == g) for g in groups])
    counts = np.array([sum(r['group'] == g for r in rows) for g in groups])
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(groups), size=(reps, len(groups)))
    means = sums[samples].sum(1)/counts[samples].sum(1)
    result['ci95'] = np.quantile(means, [.025, .975]).tolist()
    return result


def table_summary(rows, reps):
    fields = [f'{metric}_{name}' for metric in ('kl', 'tv', 'conditional_kl') for name in ('I', 'R', 'Q', 'Q_T')]
    fields += ['delta_kl_Q_R', 'delta_tv_Q_R', 'delta_conditional_kl_Q_R', 'delta_row_kl_Q_R',
               'R_row_target_tv', 'T_row_target_tv', 'R_col_target_tv', 'T_col_target_tv']
    return {key: cluster_bootstrap(rows, key, reps) for key in fields}


def distribution_summary(rows, field):
    values = [r[field] for r in rows if finite_number(r.get(field))]
    return dict(n=len(rows), available=len(values), minimum=min(values) if values else None,
                quantiles_p0_p10_p25_p50_p75_p90_p100=np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist() if values else [],
                mean=float(np.mean(values)) if values else None)


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else ['state_id']
    temp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    temp.replace(path)


def build_report(summary):
    n = summary['actual_n']
    main, high = summary['main'], summary['high_coverage']
    def fmt(value):
        return f'{value:.6g}' if finite_number(value) else '不可估计'
    text = [f'# 2026-10-26 第一批 C：冻结头 K2 Copula CPU 诊断', '',
            f"输入状态：{summary['input_status']}；实际 N={n}/256，来源题组={summary['groups']}。所有实际完成状态均保留。",
            f"输入 manifest SHA256：`{summary['input_manifest_sha256']}`。详见 input_reference.json（含原 manifest、ready、分片引用）。", '',
            '本批回答：在同一候选 A×B 内，硬保 DreamOn 旧边缘是否比现有冻结头 joint R 更接近固定顺序 fresh T。',
            '本批没有回答：真实代码正确率、端到端速度、完整 V² 解码质量、新头训练或 H3/K4 融合效果。T 不是正确代码分布。', '',
            '## 主表（KL 单位 nats，状态加权均值）', '',
            '|分布|KL(T‖·)|TV(T,·)|teacher 行边缘加权条件 KL|', '|---|---:|---:|---:|']
    for name in ('I', 'R', 'Q', 'Q_T'):
        text.append('|'+name+'|'+'|'.join(fmt(main[f'{key}_{name}']['mean']) for key in ('kl', 'tv', 'conditional_kl'))+'|')
    text += ['', '完整均值、配对差、来源题组 bootstrap 95% CI、全部分层见 main_table.csv / stratified_table.csv / high_coverage_table.csv。',
             f"CI 使用 {summary['bootstrap']['replicates']} 次来源题组重采样（固定 seed=20261015），同母状态的派生样本整体保留；属于开发探索证据。", '',
             f"高覆盖子集：N={high['n']}、来源题组={high['groups']}；{high['status']}。",
             '阈值固定为 old_p1/p2 候选质量≥.95 且 R_raw/T_raw 块质量均≥.90；这是诊断选择，不是理论充分条件。', '',
             '## 判断与候选下一步', '', summary['verdict'], '',
             f"算法审计：Q失败={summary['audit']['Q_failures']}，Q_T失败={summary['audit']['Q_T_failures']}，"
             f"适用理论条件下oracle违反={summary['audit']['oracle_violations']}；CPU合计={summary['audit']['cpu_seconds']:.6g}秒。",
             '只有 Q_T 使用 T 的块内边缘，因此是不可部署的 teacher-oracle 检查。旧边缘 Q 不自动继承 KL 改善保证。',
             'KL 链式分解 Δjoint=Δrow+Δconditional；主表及分层同时给出这两项。若只有 Δrow 改善，应视为首位截断修正证据。',
             'order/span 分层方向与置信区间完整发布；样本不足或方向不一致时不推荐部署。', '',
             '## 概率与支持边界', '',
             '权威输入是完整词表 log_softmax 直接抽出的 log 字段；float64/log-space 计算。先记录 old_p1×cond2 的块质量，再全块归一化。',
             '逐行先归一化 cond2 会改变 joint 定义，本分析没有这样做。R 与 T 的相同候选不代表相同覆盖。',
             'Q 是 IPF(R;r,c)，Q_T 是 IPF(R;T.row,T.col)，误差上限1e-6、最多1000轮；不加epsilon。',
             '真实零支持、概率下溢、FP64呈现下溢、最大轮数、迭代数、最大边缘误差、特殊质量和所有尾质量见 probability_audit.json / per_state.jsonl。',
             '非有限或失败状态不删：主表会给 unavailable，避免只报成功子集。覆盖不足不能肯定或否定完整词表方法。', '',
             '## 复现与限制', '',
             'NumPy/标准库 CPU 分析，无 GPU、训练、生成或模型加载。模型/头/adapter SHA 和来源 split 均随输入引用归档。',
             '逐状态 CSV/JSONL 可移植；r/c、R/T 截断边缘与所有条件候选质量均保留。',
             '文献固定为 [Discrete Copula Diffusion v2 (2025-03-19)](https://arxiv.org/html/2410.01949v2)，§4.1及附录A；本批不复现其AR推理流程。',
             '不以 Q 胜 I 替代 Q/R 比较，不自动搜索软约束参数，不开启第二批。',
             '`reviewer_gate_disabled`：本地串行审查与fresh CPU验证；没有独立reviewer，实数据结论仍受候选/开发来源限制。', '']
    return '\n'.join(text)


def analyze_package(package, out, bootstrap_reps=2000, interrupt_after=None):
    invocation_wall, invocation_cpu = time.perf_counter(), time.process_time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Input identity plus code version protects resumed rows from mixing algorithms.
    identity = dict(input_manifest_sha256=package['ready']['manifest_sha256'],
                    analysis_sha256=sha256(__file__), seed=SEED, tol=TOL, max_iter=MAX_ITER,
                    bootstrap_reps=bootstrap_reps)
    checkpoint = out/'analysis_checkpoint.json'
    if checkpoint.exists():
        require(json.loads(checkpoint.read_text())['identity'] == identity, 'resume input/code/config changed; use a new C output directory')
    else:
        atomic_json(checkpoint, dict(identity=identity, status='running', completed=0))
    reference = dict(ready=package['ready'], ready_path=package['ready_path'], manifest=package['manifest'],
                     numeric_audit=package['numeric_audit'])
    atomic_json(out/'input_reference.json', reference)
    rows_path = out/'per_state.jsonl'
    rows = []
    if rows_path.exists():
        # Recover an interrupted append by keeping only complete JSON lines.
        data = rows_path.read_bytes()
        lines = data.splitlines(keepends=True)
        valid_bytes = 0
        for line in lines:
            if not line.endswith(b'\n'):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError('corrupt completed per-state row') from exc
            rows.append(row)
            valid_bytes += len(line)
        if valid_bytes != len(data):
            with rows_path.open('r+b') as f:
                f.truncate(valid_bytes)
    metadata = package['metadata']
    require(len(rows) <= len(metadata) and all(r['state_id'] == m['state_id'] for r, m in zip(rows, metadata)),
            'resume completed rows disagree with input state order')
    with rows_path.open('a') as f:
        for i in range(len(rows), len(metadata)):
            row = state_diagnostic(package['arrays'], metadata[i], i)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            f.flush()
            os.fsync(f.fileno())
            rows.append(row)
            atomic_json(checkpoint, dict(identity=identity, status='running', completed=len(rows)))
            if interrupt_after is not None and len(rows) >= interrupt_after:
                raise InterruptedError('test interruption after durable state row')
    write_csv(out/'per_state.csv', rows)
    main = table_summary(rows, bootstrap_reps)
    high = [r for r in rows if r['high_coverage']]
    high_groups = len({r['group'] for r in high})
    enough = len(high) >= 32 and high_groups >= 8
    high_summary = dict(n=len(high), groups=high_groups,
                        status='达到预声明样本下限' if enough else '高覆盖证据不足',
                        table=table_summary(high, bootstrap_reps))
    strata = {}
    for key in ('order', 'span', 'split'):
        for value in sorted({r[key] for r in rows}):
            strata[key+'='+str(value)] = table_summary([r for r in rows if r[key] == value], bootstrap_reps)
    for bucket in BUCKETS:
        strata['order/span='+bucket] = table_summary([r for r in rows if r['order']+'/'+r['span'] == bucket], bootstrap_reps)
    audit = dict(Q_failures=sum(r['Q_status'] != 'converged' for r in rows),
                 Q_T_failures=sum(r['Q_T_status'] != 'converged' for r in rows),
                 oracle_violations=sum(r['oracle_theory_applicable'] and r['oracle_kl_nonincrease'] is False for r in rows),
                 oracle_applicable=sum(r['oracle_theory_applicable'] for r in rows),
                 numeric_input=package['numeric_audit'], cpu_seconds=sum(r['cpu_seconds'] for r in rows),
                 wall_seconds=sum(r['wall_seconds'] for r in rows),
                 convergence={key: distribution_summary(rows, key) for key in
                              ('Q_iterations', 'Q_max_margin_error', 'Q_cpu_seconds', 'Q_T_iterations', 'Q_T_max_margin_error', 'Q_T_cpu_seconds')},
                 coverage={key: distribution_summary(rows, key) for key in
                           ('old_mass1', 'old_mass2', 'R_raw_mass', 'T_raw_mass', 'R_row_target_tv', 'R_col_target_tv',
                            'old_mass1_tail', 'old_mass2_tail', 'R_raw_mass_tail', 'T_raw_mass_tail',
                            'T_row_target_tv', 'T_col_target_tv', 'old_special_mass1', 'old_special_mass2',
                            'head_candidate_mass_r_weighted', 'fresh_candidate_mass_r_weighted',
                            'head_tail_r_weighted', 'fresh_tail_r_weighted',
                            'head_special_mass_r_weighted', 'fresh_special_mass_r_weighted')},
                 support_failure_states=[r['state_id'] for r in rows if r['Q_status'] != 'converged' or r['Q_T_status'] != 'converged'])
    audit['invocation_cpu_seconds_including_aggregation'] = time.process_time()-invocation_cpu
    audit['invocation_wall_seconds_including_aggregation'] = time.perf_counter()-invocation_wall
    mean = main['delta_kl_Q_R']['mean']
    hmean = high_summary['table']['delta_kl_Q_R']['mean']
    if not rows or audit['Q_failures'] or audit['Q_T_failures'] or audit['oracle_violations'] or mean is None:
        verdict = '数值/支持证据受限或没有可分析状态；不作方法有效性判断，先恢复稳定支持与算法检查。'
    elif not enough:
        verdict = '高覆盖证据不足：有限候选诊断不足，需要扩大候选或换估计器；不能肯定或否定全词表方法。'
        verdict += ' 整体Q相对R的ΔKL='+str(mean)+'（负数为改善）；仅作截断块描述。'
    elif mean > 0 and hmean is not None and hmean > 0:
        verdict = 'Q在整体与高覆盖均更差：强保旧边缘在这些状态不合适；这不否定所有依赖建模。软约束等留下一批决定。'
    elif mean < 0 and hmean is not None and hmean < 0:
        consistent = all(v['delta_kl_Q_R']['mean'] is not None and v['delta_kl_Q_R']['mean'] < 0
                         for k, v in strata.items() if k.startswith('order/span='))
        cond = high_summary['table']['delta_conditional_kl_Q_R']['mean']
        tvs = (main['delta_tv_Q_R']['mean'], high_summary['table']['delta_tv_Q_R']['mean'])
        if consistent and cond is not None and cond < 0 and all(x is not None and x < 0 for x in tvs):
            verdict = 'Q在整体与预声明高覆盖的KL/TV均改善，四个order/span方向一致，条件KL亦改善；可建议下一批小范围可部署校准pilot。尚无代码通过率或端到端速度改善证据。'
        else:
            verdict = '整体与高覆盖KL改善，但分层、TV或条件KL未一致支持；首位截断偏移可能解释收益，暂不作可部署pilot的肯定建议。'
    else:
        verdict = '整体/高覆盖方向混合或无改善：不足以推荐强保旧边缘，保留各分层结果；不自动调参或训练。'
    summary = dict(contract_id=CONTRACT, schema_version=1, actual_n=len(rows), expected_n=256,
                   groups=len({r['group'] for r in rows}), input_status=package['manifest']['status'],
                   input_manifest_sha256=package['ready']['manifest_sha256'], counts=package['counts'],
                   main=main, strata=strata, high_coverage=high_summary, audit=audit, verdict=verdict,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, gpu='none',
                                    CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                    platform=platform.platform()), bootstrap=dict(seed=SEED, replicates=bootstrap_reps, unit='source group'))
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'probability_audit.json', audit)
    write_csv(out/'main_table.csv', [dict(metric=k, **v) for k, v in main.items()])
    write_csv(out/'stratified_table.csv', [dict(stratum=s, metric=k, **v) for s, table in strata.items() for k, v in table.items()])
    write_csv(out/'high_coverage_table.csv', [dict(status=high_summary['status'], metric=k, **v) for k, v in high_summary['table'].items()])
    (out/'report.zh.md').write_text(build_report(summary))
    atomic_json(checkpoint, dict(identity=identity, status='analyzed', completed=len(rows)))
    return portable(summary)
