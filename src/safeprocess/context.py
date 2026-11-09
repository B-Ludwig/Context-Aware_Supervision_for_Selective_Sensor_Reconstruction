"""Pure kernels from refinement/context_refinement.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import numpy as np

STARTS = (200, 210, 220, 230, 240)

SCALE_FLOOR = 1e-6

RIDGE_ALPHA = 10.0

SCHEMA = 'safeprocess-component-context-ridge-v1'

def component_states(frame, unit):
    """Only original observable actuator/state variables; no sensor values."""
    result = {}
    if unit.startswith('hai_'):
        groups = {}
        for col in frame:
            if '.state_' in col and col.endswith('.active'):
                groups.setdefault(col.split('.state_', 1)[0], []).append(col)
        for component, cols in sorted(groups.items()):
            cols = sorted(cols)
            values = frame[cols].to_numpy(float)
            valid = np.isin(values, [0, 1]).all(axis=1) & (values.sum(axis=1) == 1)
            states = [c.split('.state_', 1)[1].removesuffix('.active') for c in cols]
            result[component] = [states[int(j)] if ok else None
                                 for j, ok in zip(values.argmax(axis=1), valid)]
        if not groups:
            result['__missing_components__'] = [None] * len(frame)
    elif unit in ('four_tank', 'swat'):
        columns = ['CuStepNo ValueY'] if unit == 'four_tank' else [
            'MV301', 'MV302', 'P301', 'P302', 'P401', 'P402']
        for col in columns:
            if col not in frame:
                result[col] = [None] * len(frame)
                continue
            values = frame[col].to_numpy(float)
            result[col] = [str(int(v)) if np.isfinite(v) and v == np.floor(v)
                           else None for v in values]
    else:
        raise ValueError('Unknown original context adapter: ' + unit)
    return result

def component_history(frame, component_columns=None, window_end=None,
                      window_size=60, prefix_start=0):
    """Summarize component-state sequences through inclusive row window_end.

    A mapping component -> sequence is accepted as frame. For a DataFrame,
    component_columns identifies already categorical state columns. Change count
    covers edges wholly inside the scoring window; age uses the entire available
    causal prefix and is left-censored at the available prefix length.
    """
    if isinstance(frame, dict):
        states = frame
        n = len(next(iter(states.values()))) if states else 0
    else:
        states = {c: frame[c].tolist() for c in component_columns or []}
        n = len(frame)
    end = n - 1 if window_end is None else int(window_end)
    begin = end - int(window_size) + 1
    if begin < prefix_start or end >= n or end < 0:
        raise ValueError('Scoring window not contained in available causal prefix')
    out = {'window_end': end, 'window_start': begin, 'components': {},
           'valid': bool(states), 'history_class': 'unavailable',
           'change_count': None, 'age_rows': None}
    prefixes = []
    for name, values in sorted(states.items()):
        prefix = list(values[prefix_start:end + 1])
        valid = all(x is not None and not (isinstance(x, float) and not np.isfinite(x))
                    for x in prefix)
        if not valid:
            out['valid'] = False
            out['components'][name] = {'current_state': None, 'occupancy': {},
                                      'change_count': None, 'age_rows': None,
                                      'age_censored': True}
            continue
        prefix = [str(x) for x in prefix]
        window = prefix[-window_size:]
        changes = [i for i in range(1, len(prefix)) if prefix[i] != prefix[i - 1]]
        out['components'][name] = {
            'current_state': prefix[-1],
            'occupancy': {k: v / window_size for k, v in sorted(Counter(window).items())},
            'change_count': sum(window[i] != window[i - 1] for i in range(1, len(window))),
            'age_rows': len(prefix) - 1 - changes[-1] if changes else len(prefix),
            'age_censored': not bool(changes)}
        prefixes.append(prefix)
    if out['valid']:
        joint = list(zip(*prefixes))
        changes = [i for i in range(1, len(joint)) if joint[i] != joint[i - 1]]
        out.update(history_class='transition_settling' if changes else 'steady',
                   change_count=len(changes),
                   age_rows=len(joint) - 1 - changes[-1] if changes else len(joint))
    return out

def context_features(frame, unit, context_status='available'):
    """Five causal records. Future rows cannot affect an earlier window."""
    states = component_states(frame, unit)
    windows = [component_history(states, window_end=s + 59) for s in STARTS]
    invalid = context_status != 'available' or not all(w['valid'] for w in windows)
    return {'unit': unit, 'windows': windows,
            'history_class': windows[-1]['history_class'],
            'invalid_context': invalid, 'context_status': context_status}

def _windows(records):
    return [w for r in records for w in r['windows']]

def _numeric(w, components):
    values = [float(w.get('change_count') or 0), float(w.get('age_rows') or 0)]
    for comp in components:
        c = w['components'].get(comp, {})
        values.extend([float(c.get('change_count') or 0),
                       float(c.get('age_rows') or 0),
                       float(c.get('age_censored', True))])
    return values

def feature_transform_fit(reference_records, level="full"):
    """Fit only on supplied valid reference windows; portable deterministic map."""
    windows = [w for r in reference_records if not r.get('invalid_context', False)
               for w in r['windows'] if w['valid']]
    if not windows:
        raise ValueError('No valid normal reference metadata for feature transforms')
    components = sorted({c for w in windows for c in w['components']})
    categories = {c: sorted({s for w in windows for s, p in
                            w['components'].get(c, {}).get('occupancy', {}).items() if p > 0})
                  for c in components}
    numeric = np.asarray([_numeric(w, components) for w in windows], float)
    names = ['joint.change_count', 'joint.age_rows']
    for comp in components:
        names.extend([comp + '.change_count', comp + '.age_rows', comp + '.age_censored'])
    numeric_names = list(names)
    names.extend(['history_class.steady', 'history_class.transition_settling'])
    for comp in components:
        names.extend([comp + '.current=' + s for s in categories[comp]])
        names.extend([comp + '.occupancy=' + s for s in categories[comp]])
    if level not in ('intercept', 'shared', 'state', 'full'):
        raise ValueError('History level must be intercept, shared, state or full')
    shared = [0, 1, len(numeric_names), len(numeric_names) + 1]
    selected = (list(range(len(names))) if level == 'full' else
                shared + [i for i, name in enumerate(names) if '.current=' in name]
                if level == 'state' else shared if level == 'shared' else [])
    return {'level': level, 'selected_features': selected,
            'full_feature_names': names,
            'components': components, 'categories': categories,
            'numeric_mean': numeric.mean(axis=0).tolist(),
            'numeric_scale': np.maximum(numeric.std(axis=0), 1.).tolist(),
            'numeric_features': numeric_names, 'feature_names': [names[i] for i in selected] if selected else ['intercept_placeholder'],
            'fit_windows': len(windows), 'fit_source': 'normal_reference_only'}

def feature_transform(record, transform):
    """Return X[5,D] and per-window unknown/invalid flags; never update vocab."""
    rows, flags = [], []
    for w in record['windows']:
        num = (np.asarray(_numeric(w, transform['components'])) -
               np.asarray(transform['numeric_mean'])) / np.asarray(transform['numeric_scale'])
        row = num.tolist() + [float(w['history_class'] == c)
                             for c in ('steady', 'transition_settling')]
        unknown = set(w['components']) != set(transform['components'])
        missing = (not w['valid'] or record.get('context_status', 'available') != 'available' or
                   bool(set(transform['components']) - set(w['components'])))
        for comp in transform['components']:
            c = w['components'].get(comp, {})
            vocab = transform['categories'][comp]
            occupancy = c.get('occupancy', {})
            unknown |= (c.get('current_state') not in vocab or
                        any(s not in vocab and p > 0 for s, p in occupancy.items()))
            row.extend([float(c.get('current_state') == s) for s in vocab])
            row.extend([float(occupancy.get(s, 0)) for s in vocab])
        rows.append(row)
        flags.append({'unknown_state': bool(unknown), 'invalid_context': bool(missing)})
    x = np.asarray(rows, float)
    selected = transform.get('selected_features')
    if selected is not None:
        x = x[:, selected] if selected else np.zeros((len(x), 1))
    return x, flags

def _ridge_fit(x, y):
    # Multi-target Ridge is algebraically a separate alpha10 ridge per sensor.
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver='svd').fit(x, y)
    return {'coef': np.asarray(model.coef_).tolist(),
            'intercept': np.asarray(model.intercept_).tolist(), 'alpha': RIDGE_ALPHA}

def _ridge_predict(model, x):
    return np.asarray(x) @ np.asarray(model['coef']).T + np.asarray(model['intercept'])

def _pooled(a):
    center = np.median(a.reshape(-1, a.shape[-1]), axis=0)
    scale = np.maximum(np.sqrt(np.mean((a - center) ** 2, axis=(0, 1, 2))), SCALE_FLOOR)
    return {'center': center.tolist(), 'scale': scale.tolist(), 'n_reference': len(a)}

def _arrays(signed, metadata, name):
    a = np.asarray(signed, float)
    if a.ndim != 4 or a.shape[1:3] != (5, 60) or len(a) != len(metadata) or not len(a):
        raise ValueError(name + ' requires aligned nonempty [N,5,60,F] residuals/metadata')
    if not np.isfinite(a).all():
        raise ValueError(name + ' contains nonfinite signed residuals')
    if any(len(r['windows']) != 5 for r in metadata):
        raise ValueError(name + ' metadata must contain exactly five causal windows')
    return a

def fit_context_profile(signed_reference, reference_contexts, source_groups, *,
                        calibration_signed, calibration_contexts,
                        calibration_groups=None, level="full"):
    """Normal-only fit, grouped OOF centre errors, separate calibration q95."""
    from sklearn.model_selection import GroupKFold
    a = _arrays(signed_reference, reference_contexts, 'reference')
    b = _arrays(calibration_signed, calibration_contexts, 'calibration')
    if a.shape[-1] != b.shape[-1] or len(source_groups) != len(a):
        raise ValueError('Channel or reference source-period alignment differs')
    groups = np.asarray(source_groups, str)
    if calibration_groups is not None and len(calibration_groups) != len(b):
        raise ValueError('Calibration source-period alignment differs')
    if calibration_groups is not None and set(groups) & set(map(str, calibration_groups)):
        raise ValueError('Normal reference/calibration source periods overlap')
    valid = np.array([not r.get('invalid_context', False) and all(w['valid'] for w in r['windows'])
                      for r in reference_contexts])
    profile = {'schema': SCHEMA, 'level': level, 'pooled': _pooled(a),
               'n_reference': len(a), 'n_valid_reference': int(valid.sum()),
               'n_reference_groups': len(set(groups)), 'alpha': RIDGE_ALPHA,
               'scale_bounds': [.25, 4.], 'folds': [], 'conditional_available': False,
               'fit_role': 'normal_reference', 'labels_used': False,
               'calibration_used_for_fit': False,
               'reference_classes': dict(Counter(r['history_class'] for r in reference_contexts))}
    unique = sorted(set(groups[valid]))
    if valid.any() and len(unique) >= 2:
        refs = [r for r, ok in zip(reference_contexts, valid) if ok]
        av = a[valid]
        gv = groups[valid]
        oof_center = np.empty((len(av), 5, av.shape[-1]), float)
        for fold, (train, test) in enumerate(GroupKFold(n_splits=min(5, len(unique))).split(av, groups=gv)):
            tr = [refs[i] for i in train]
            tf = feature_transform_fit(tr, level=level)
            x = np.concatenate([feature_transform(r, tf)[0] for r in tr])
            model = _ridge_fit(x, av[train].mean(axis=2).reshape(-1, av.shape[-1]))
            base = _pooled(av[train])
            for i in test:
                xx, ff = feature_transform(refs[i], tf)
                pred = _ridge_predict(model, xx)
                for j, f in enumerate(ff):
                    if f['unknown_state'] or f['invalid_context']:
                        pred[j] = base['center']
                oof_center[i] = pred
            train_groups, test_groups = sorted(set(gv[train])), sorted(set(gv[test]))
            assert not set(train_groups) & set(test_groups)
            profile['folds'].append({'fold': fold, 'train_groups': train_groups,
                                     'test_groups': test_groups, 'train_periods': len(train),
                                     'test_periods': len(test),
                                     'transform_fit_windows': tf['fit_windows'],
                                     'transform_sha256': hashlib.sha256(
                                         json.dumps(tf, sort_keys=True).encode()).hexdigest()})
        tf = feature_transform_fit(refs, level=level)
        x = np.concatenate([feature_transform(r, tf)[0] for r in refs])
        center_model = _ridge_fit(x, av.mean(axis=2).reshape(-1, av.shape[-1]))
        oof_rms = np.maximum(np.sqrt(np.mean((av - oof_center[:, :, None, :]) ** 2, axis=2)), SCALE_FLOOR)
        scale_model = _ridge_fit(x, np.log(oof_rms).reshape(-1, av.shape[-1]))
        profile.update(transform=tf, center_model=center_model, scale_model=scale_model,
                       conditional_available=True,
                       scale_target='log_RMS_60_signed_minus_source_period_OOF_center',
                       oof_error_sha256=hashlib.sha256(oof_rms.tobytes()).hexdigest(),
                       oof_error_min=float(oof_rms.min()), oof_error_max=float(oof_rms.max()))
    profile['split_sha256'] = hashlib.sha256(
        json.dumps(profile['folds'], sort_keys=True).encode()).hexdigest()
    scores, classes, supports = [], [], []
    for signed, metadata in zip(b, calibration_contexts):
        result = score_context(signed, metadata, profile, calibrating=True)
        scores.append(result['event_score'])
        classes.append(metadata['history_class'])
        supports.append(result['unknown_state'])
    profile['calibration'] = {'pooled': {'limit': empirical_limit(scores), 'n_calibration': len(scores)},
                              'groups': {}}
    for cls in ('steady', 'transition_settling', 'unavailable'):
        selected = [s for s, c in zip(scores, classes) if c == cls]
        profile['calibration']['groups'][cls] = {
            'n_calibration': len(selected),
            'limit': empirical_limit(selected) if len(selected) >= 4 else None,
            'uses_class_limit': len(selected) >= 4}
    profile.update(calibration_scores=scores, calibration_classes=classes,
                   calibration_unknown_states=int(sum(supports)),
                   n_calibration=len(b), no_normal_outlier_deletion=True)
    return profile

def empirical_limit(scores):
    a = np.asarray(scores, float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Missing finite normal calibration scores')
    return float(np.quantile(a, .95, method='higher'))

def conditional_normalizer(context, profile):
    base = profile['pooled']
    center = np.repeat(np.asarray(base['center'])[None, :], 5, axis=0)
    scale = np.repeat(np.asarray(base['scale'])[None, :], 5, axis=0)
    flags = [{'unknown_state': False, 'invalid_context': context.get('invalid_context', False)
              or not w['valid']} for w in context['windows']]
    if profile['conditional_available']:
        x, flags = feature_transform(context, profile['transform'])
        predicted_center = _ridge_predict(profile['center_model'], x)
        # Clipping in log space avoids overflow while implementing exact bounds.
        lower = np.log(.25 * np.asarray(base['scale']))
        upper = np.log(4. * np.asarray(base['scale']))
        predicted_scale = np.exp(np.clip(_ridge_predict(profile['scale_model'], x), lower, upper))
        for i, f in enumerate(flags):
            if not (f['unknown_state'] or f['invalid_context']):
                center[i], scale[i] = predicted_center[i], predicted_scale[i]
    return center, scale, flags

def event_score(signed, center, scale):
    a = np.asarray(signed, float)
    if a.ndim != 3 or a.shape[:2] != (5, 60) or not np.isfinite(a).all():
        raise ValueError('Five finite60-row signed windows required; no cache-miss inference')
    z = (a - np.asarray(center)[:, None, :]) / np.asarray(scale)[:, None, :]
    persistent = float(np.sort((z * z).mean(axis=1), axis=0)[-3].max())
    accumulated = float(np.max(np.abs(np.cumsum(z.mean(axis=1), axis=0)) /
                               np.sqrt(np.arange(1, 6))[:, None])) ** 2
    return {'event_score': max(persistent, accumulated),
            'persistent_score': persistent, 'accumulated_score': accumulated}

def score_context(signed, context, profile, calibrating=False):
    center, scale, flags = conditional_normalizer(context, profile)
    score = event_score(signed, center, scale)
    invalid = bool(context.get('invalid_context', False) or any(f['invalid_context'] for f in flags))
    unknown = any(f['unknown_state'] for f in flags)
    result = {**score, 'invalid_context': invalid, 'unknown_state': unknown,
              'history_class': context['history_class'],
              'reference_fallback': bool(unknown or invalid or not profile['conditional_available']),
              'unknown_state_windows': [i for i, f in enumerate(flags) if f['unknown_state']],
              'review_required': invalid, 'proposal_eligible': not invalid}
    if calibrating:
        return result
    cls = context['history_class'] if not invalid else 'unavailable'
    cell = profile['calibration']['groups'].get(cls, {'n_calibration': 0, 'uses_class_limit': False})
    fallback = not cell['uses_class_limit'] or invalid
    cal = profile['calibration']['pooled'] if fallback else cell
    nref = profile['reference_classes'].get(cls, 0)
    result.update(numeric_alarm=bool(score['event_score'] > cal['limit']),
                  limit=cal['limit'], calibration_n=cal['n_calibration'],
                  class_calibration_n=cell['n_calibration'], reference_class_n=nref,
                  reference_weight=1. if profile['conditional_available'] and not unknown else 0.,
                  calibration_fallback=fallback,
                  low_support=bool(cal['n_calibration'] < 20 or cell['n_calibration'] < 4 or
                                   nref < 8 or profile['n_reference_groups'] < 5 or unknown))
    return result


context_from_frame = context_features
admission = score_context
HISTORY_LEVELS = ("intercept", "shared", "state", "full")
