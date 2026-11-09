"""Pure kernels from original/final_decision.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np
from .common import unique_top

FEATURES = ('log_normal_distance', 'relative_signature_fit', 'signature_improvement',
            'signature_margin', 'own_residual_share', 'cross_gain',
            'log_repaired_residual_ratio', 'log_change_to_error_ratio',
            'log_observed_variation_ratio', 'residual_rank', 'signature_rank', 'context_supported')

CONFIDENCE_GRID = tuple(np.linspace(0., .95, 20))

def high_quantile(values, q=.95):
    a = np.asarray(values, float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('finite calibration values required')
    return np.quantile(a, q, axis=0, method='higher')

def fit_noise(signed, phases):
    signed = np.asarray(signed, float)
    if signed.ndim != 4 or signed.shape[1:3] != (5, 60):
        raise ValueError('normal signed residuals must be contexts x 5 x 60 x channels')
    flat = signed.reshape(-1, signed.shape[-1])
    center = np.median(flat, axis=0)
    scale = np.maximum(np.sqrt(np.mean((flat-center)**2, axis=0)), 1e-6)
    groups = {}
    for phase in sorted({p for row in phases for p in row if p != 'unavailable'}):
        selected = [signed[i, k] for i, row in enumerate(phases) for k, p in enumerate(row) if p == phase]
        contexts = sum(phase in row for row in phases)
        if contexts < 2 or len(selected) < 3:
            continue
        a = np.concatenate(selected)
        c = np.median(a, axis=0); s = np.sqrt(np.mean((a-c)**2, axis=0))
        groups[phase] = {'center': (.8*c+.2*center).tolist(),
                         'scale': np.maximum(.8*s+.2*scale, 1e-6).tolist(),
                         'contexts': contexts, 'windows': len(selected)}
    profile = {'center': center.tolist(), 'scale': scale.tolist(), 'groups': groups}
    components = np.stack([event_components(z, p, profile, 1.) for z, p in zip(signed, phases)])
    profile['component_scale'] = np.maximum(high_quantile(components), 1e-6).tolist()
    return profile

def standardized(signed, phases, profile):
    z = np.asarray(signed, float)
    if z.ndim != 3 or z.shape[:2] != (5, 60) or len(phases) != 5 or not np.isfinite(z).all():
        raise ValueError('expected five finite 60-row signed residual windows and aligned phases')
    result = []
    supported = []
    for window, phase in zip(z, phases):
        p = profile['groups'].get(phase, profile)
        result.append((window-np.asarray(p['center']))/np.asarray(p['scale']))
        supported.append(phase in profile['groups'])
    return np.asarray(result), supported

def event_components(signed, phases, profile, original_threshold):
    z, _ = standardized(signed, phases, profile)
    residual = (np.asarray(signed)**2).mean(axis=1)
    # Three of five overlapping native windows; no independent-sample claim.
    global_score = float(np.sort(residual.mean(axis=1)/max(original_threshold, 1e-20))[-3])
    channel = float(np.sort((z*z).mean(axis=1), axis=0)[-3].max())
    means = z.mean(axis=1)
    cumulative = float(np.max(np.abs(np.cumsum(means, axis=0))/np.sqrt(np.arange(1, 6))[:, None]))
    return np.array([global_score, channel, cumulative])

def event_scores(signed, phases, profile, original_threshold):
    # Global component is normalized using the same original threshold at fit.
    components = event_components(signed, phases, profile, original_threshold)
    scale = np.asarray(profile['component_scale'])
    return components/np.maximum(scale, 1e-12)

def fit_event_limits(scores, phases):
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] != 3 or len(scores) != len(phases):
        raise ValueError('aligned three-component event calibration required')
    limits = {'joint': float(high_quantile(scores.max(axis=1))),
              'channel': float(high_quantile(scores[:, :2].max(axis=1))), 'groups': {}}
    for phase in sorted(set(phases)):
        a = scores[np.array(phases) == phase]
        if phase != 'unavailable' and len(a) >= 4:
            limits['groups'][phase] = {'joint': float(high_quantile(a.max(axis=1))),
                                       'channel': float(high_quantile(a[:, :2].max(axis=1))), 'n': len(a)}
    return limits

def event_alarm(scores, phase, limits, *, signed=True):
    p = limits['groups'].get(phase, limits)
    key = 'joint' if signed else 'channel'
    return bool(np.max(scores if signed else scores[:2]) > p[key])

def calibration_lookup(profile, phase):
    return profile['groups'].get(phase, profile)

def candidate_features(raw, signed, scores, corrected, after, fractions, valid,
                       phase, calibration, training_std, context_supported):
    """All-input candidate rows; no injected target, family or clean reference."""
    raw = np.asarray(raw, float); corrected = np.asarray(corrected, float)
    after = np.asarray(after, float); valid = np.asarray(valid, bool)
    f = raw.shape[1]
    if raw.shape != (60, f) or corrected.shape != (f, 60, f) or after.shape != (f, f):
        raise ValueError('all-candidate shape mismatch')
    p = calibration_lookup(calibration, phase)
    distances = np.asarray(scores['candidate_distances'], float)
    if distances.shape != (f,) or valid.shape != (f,):
        raise ValueError('candidate evidence order mismatch')
    normal = max(float(scores['normal_distance']), 1e-12)
    before = (np.asarray(signed[-1])**2).mean(axis=0)
    residual_rank = np.argsort(np.argsort(-before, kind='stable'), kind='stable')/max(f-1, 1)
    signature_rank = np.argsort(np.argsort(distances, kind='stable'), kind='stable')/max(f-1, 1)
    change = np.array([np.mean(np.abs(corrected[j, :, j]-raw[:, j])) for j in range(f)])
    error = np.maximum(np.asarray(p['error_mae']), 1e-12)
    repaired_max = np.max(after/np.maximum(np.asarray(calibration['residual_reference']), 1e-12), axis=1)
    envelope = np.maximum(np.asarray(p['repaired_limit']), 1e-12)
    features = np.column_stack([
        np.full(f, np.log1p(normal)), np.clip(distances/normal, 0, 100),
        np.clip((normal-distances)/normal, -100, 1),
        np.array([(np.min(np.delete(distances, j))-distances[j])/normal for j in range(f)]).clip(-100, 100),
        before/max(float(before.sum()), 1e-20), np.asarray(fractions).clip(-100, 1),
        np.log1p(repaired_max/envelope), np.log1p(change/error),
        np.log1p(np.std(raw, axis=0)/np.maximum(training_std, 1e-12)),
        residual_rank, signature_rank, np.full(f, float(context_supported))])
    if features.shape != (f, len(FEATURES)) or not np.isfinite(features).all():
        raise ValueError('nonfinite candidate features')
    return features

def choose_threshold(records, *, max_wrong=.05, max_normal=.05):
    """Development-only selection; labels must never reach decide()."""
    candidates = []
    for threshold in CONFIDENCE_GRID:
        selected = [r for r in records if r['event_alarm'] and r['margin'] > threshold]
        good = sum(r['correct_beneficial'] for r in selected)
        wrong = len(selected)-good
        normals = [r for r in records if r['normal']]
        false_normal = sum(r['normal'] for r in selected)
        acceptable = bool(selected and wrong/len(selected) <= max_wrong
                          and (not normals or false_normal/len(normals) <= max_normal))
        candidates.append({'threshold': float(threshold), 'selected': len(selected), 'good': good,
                           'wrong_or_not_beneficial': wrong, 'normal_referrals': false_normal,
                           'normal_n': len(normals), 'acceptable': acceptable})
    eligible = [r for r in candidates if r['acceptable']]
    chosen = min(eligible, key=lambda r: (-r['good'], r['wrong_or_not_beneficial'], -r['threshold'])) if eligible else None
    return {'threshold': chosen['threshold'] if chosen else 1., 'eligible': chosen is not None,
            'grid': candidates, 'qualification': 'empirical seed23 development selection, not a risk guarantee'}
