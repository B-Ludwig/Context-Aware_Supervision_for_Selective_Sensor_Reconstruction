"""Pure kernels from original/signature_panel.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np
from .kernels.common import unique_top
from .kernels.prominence import excess

TEMPLATES = tuple((k, a) for k in ('bias', 'drift') for a in (-50., -10., -2., 2., 10., 50.)) +\
            tuple(('gain', a) for a in (-.25, -.10, .10, .25)) + (('stuck', 0.),)

def signature_features(signed, whitener):
    """Keep signs and six consecutive ten-row bins in each scoring window."""
    z = np.asarray(signed, float)
    if z.shape[0:2] != (5, 60) or not np.isfinite(z).all():
        raise ValueError('expected five signed 60-row residual trajectories')
    return (z.reshape(5, 6, 10, z.shape[-1]).mean(axis=2) @ whitener).reshape(-1)

def signature_scores(feature, bank):
    """Global sensor hypotheses, one amplitude per temporal signature; no truth."""
    x = np.asarray(feature, float)
    bases, directions = bank['bases'], bank['directions']
    delta = x[None, :] - bases
    alpha = np.clip(np.sum(delta * directions, axis=1) /
                    np.maximum(np.sum(directions ** 2, axis=1), 1e-20), .1, 4.)
    distance = np.mean((delta - alpha[:, None] * directions) ** 2, axis=1)
    normal = float(np.min(np.mean((bank['normal'] - x) ** 2, axis=1)))
    f = int(bank['feature_count'][0])
    scores = np.array([distance[bank['targets'] == j].min() for j in range(f)])
    winner = unique_top(-scores)
    order = np.sort(scores)
    improvement = float((normal - order[0]) / max(normal, 1e-12))
    margin = float((order[1] - order[0]) / max(normal, 1e-12))
    return {'winner': winner, 'normal_distance': normal, 'fit_distance': float(order[0]),
            'improvement': improvement, 'margin': margin, 'candidate_distances': scores.tolist()}

def envelope_max(after, profile):
    return np.max(excess(after, profile['residual_profile']), axis=-1)

def envelope_veto(result, after, valid, profile):
    result = dict(result)
    result['envelope_veto'] = False
    if result['route'] == 'reconstruction':
        j = result['target_index']
        passed = bool(valid[j] and envelope_max(after, profile)[j] <= profile['residual_threshold'])
        result['envelope_veto'] = not passed
        if not passed:
            result.update(route='abstain', target_index=None, probe_sensor_flag=False)
    return result


def bank_arrays(bank):
    """Materialize a JSON bank without reading model or recording files."""
    return {key: np.asarray(bank[key], dtype=int if key == 'targets' else float)
            for key in ('bases', 'directions', 'targets', 'normal', 'whitener', 'feature_count')}


def fit_bank(records):
    """Fit the 17-template signed dictionary from already computed evidence.

    Dictionary base records have family ``unchanged``. Each corrupted dictionary
    record declares ``base_id``, ``target_index``, ``family`` and ``amplitude``.
    These are development construction records, never prediction inputs.
    """
    records = list(records)
    bases = [r for r in records if r['case'].get('family', 'unchanged') == 'unchanged']
    if not bases:
        raise ValueError('Signature fitting requires unchanged dictionary bases')
    ids = [str(r['case']['case_id']) for r in bases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate dictionary base identity')
    signed = np.stack([np.asarray(r['payload']['signed'], float) for r in bases])
    if signed.ndim != 4 or signed.shape[1:3] != (5, 60) or not np.isfinite(signed).all():
        raise ValueError('Dictionary residuals require finite N x5x60xchannels')
    d = signed.shape[-1]
    if d < 2:
        raise ValueError('The selective comparison requires at least two channels')
    cov = np.atleast_2d(np.cov(signed.reshape(-1, d), rowvar=False))
    cov = .8 * cov + .2 * np.diag(np.diag(cov))
    eigen, vectors = np.linalg.eigh(cov)
    floor = max(float(eigen.max()) * 1e-6, 1e-12)
    whitener = (vectors * (1 / np.sqrt(np.maximum(eigen, floor)))) @ vectors.T
    normal = np.stack([signature_features(z, whitener) for z in signed])
    lookup = {identity: i for i, identity in enumerate(ids)}
    bank_bases, directions, targets, specifications = [], [], [], []
    seen = set()
    for record in records:
        case = record['case']
        family = case.get('family', 'unchanged')
        if family == 'unchanged':
            continue
        base_id = str(case.get('base_id', ''))
        if base_id not in lookup:
            raise ValueError('Dictionary template has no declared unchanged base: ' + base_id)
        j = int(case['target_index'])
        spec = (family, float(case['amplitude']))
        if not 0 <= j < d or spec not in TEMPLATES:
            raise ValueError('Template does not match the fixed 17-template dictionary')
        key = (base_id, j, *spec)
        if key in seen:
            raise ValueError('Duplicate dictionary template')
        seen.add(key)
        anchor = lookup[base_id]
        bank_bases.append(normal[anchor])
        directions.append(signature_features(record['payload']['signed'], whitener) - normal[anchor])
        targets.append(j)
        specifications.append([base_id, j, family, spec[1]])
    expected = {(identity, j, *spec) for identity in ids for j in range(d) for spec in TEMPLATES}
    if seen != expected:
        raise ValueError(f'Incomplete fixed signature dictionary: {len(seen)}/{len(expected)} templates')
    return {'bases': np.asarray(bank_bases).tolist(), 'directions': np.asarray(directions).tolist(),
            'targets': targets, 'normal': normal.tolist(), 'whitener': whitener.tolist(),
            'feature_count': [d], 'specifications': specifications,
            'fit_receipt': {'role': 'dictionary', 'base_ids': ids, 'template_n': len(seen),
                            'covariance_shrinkage': .2, 'amplitude_bounds': [.1, 4.]}}
