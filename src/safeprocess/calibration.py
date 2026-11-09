"""Development-only fitting of portable SafeProcess calibration profiles.

These functions consume already computed numerical evidence. They do not import
Torch, run a model, open recordings or use assessment labels. JSON stores every
learned coefficient/tree and every selected threshold.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from . import context
from .signatures import fit_bank
from .policies import (metadata_binding, phases, observable_features, absolute_features,
                       event_admissions, process_evidence, relative_decision, _finish)
from .kernels import relative, history, prominence, ranker, selection
from .kernels.common import unique_top

RANKER_PARAMETERS = {'max_iter': 100, 'max_leaf_nodes': 7, 'max_depth': 3,
                     'min_samples_leaf': 20, 'l2_regularization': 1., 'learning_rate': .1,
                     'early_stopping': False, 'class_weight': 'balanced',
                     'categorical_features': None, 'random_state': 31017}
ALLOWED_ROLES = frozenset(('dictionary', 'normal_reference', 'normal_calibration',
                          'decision_fit', 'risk_calibration', 'selection_normal_fit'))


def _records(records):
    result = list(records)
    if not result:
        raise ValueError('Calibration records are required')
    for record in result:
        role = record['case'].get('role')
        if role not in ALLOWED_ROLES:
            raise ValueError('Forbidden calibration role: ' + str(role))
        if record.get('payload') is None:
            raise ValueError('A missing numerical payload cannot be silently fitted')
    keys = [(r['case']['role'], str(r['case']['case_id'])) for r in result]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate case within a calibration role')
    return result


def _role(records, role):
    return [r for r in records if r['case']['role'] == role]


def _group(record):
    if record.get('source_group') is None:
        raise ValueError('Calibration requires an explicit source-period group')
    return str(record['source_group'])


def _status(record):
    return record['case'].get('context_status', 'available')


def _contexts(records, metadata):
    return [context.context_features(r['frame'], metadata['context_unit'], _status(r))
            for r in records]


def _disabled(reason):
    return {'threshold': 1., 'eligible': False, 'reason': reason, 'grid': [],
            'process_and_context_guards_during_threshold_fit': False}


def _normal_inputs(records, metadata):
    reference, calibration = _role(records, 'normal_reference'), _role(records, 'normal_calibration')
    if not reference or not calibration:
        raise ValueError('Separate normal_reference and normal_calibration evidence is required')
    for r in [*reference, *calibration]:
        if r['case'].get('family', 'unchanged') != 'unchanged':
            raise ValueError('Faults cannot enter normal-reference/calibration fitting')
    if set(map(_group, reference)) & set(map(_group, calibration)):
        raise ValueError('Normal reference and calibration periods overlap')
    return reference, calibration


def fit_process_profile(reference, calibration, *, metadata):
    """Normal q95 excess, residual-weighted Jacobians and separate limits."""
    residual = np.concatenate([np.asarray(r['payload']['residuals'], float) for r in reference])
    rp = prominence.fit_scale(residual)
    sp = prominence.fit_scale(np.concatenate([prominence.weighted_sensitivity(
        r['payload']['residuals'], r['payload']['influence'], rp) for r in reference]))
    residual_threshold = float(relative.high_quantile([
        prominence.excess(r['payload']['residuals'], rp).max() for r in calibration]))
    sensitivity_threshold = float(relative.high_quantile([
        prominence.excess(prominence.weighted_sensitivity(
            r['payload']['residuals'], r['payload']['influence'], rp), sp).max()
        for r in calibration]))
    gains = [float(np.max(r['payload']['fractions'])) for r in calibration]
    return {'residual_profile': rp, 'sensitivity_profile': sp,
            'residual_threshold': residual_threshold,
            'sensitivity_threshold': sensitivity_threshold,
            'probe_threshold': max(0., float(relative.high_quantile(gains))),
            'normal_probe_best_gains': gains,
            'reference_periods': list(map(_group, reference)),
            'calibration_periods': list(map(_group, calibration)),
            'fit_role': 'normal_reference', 'limit_role': 'normal_calibration',
            'graph_policy': 'connected induced graph on persistent peak owners using declared connections only'}


def fit_dominance_profile(reference):
    """Per-channel normal q99 and max(1.4826 MAD, q99-median, 1e-12)."""
    values = np.concatenate([np.asarray(r['payload']['residuals'], float) for r in reference])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Dominance calibration requires finite nonnegative residuals')
    med = np.quantile(values, .5, axis=0, method='linear')
    mad = np.quantile(np.abs(values - med), .5, axis=0, method='linear')
    q99 = np.quantile(values, .99, axis=0, method='linear')
    scale = np.maximum.reduce([1.4826 * mad, q99 - med, np.full_like(q99, 1e-12)])
    return {'q99': q99.tolist(), 'scale': scale.tolist(), 'median': med.tolist(),
            'mad': mad.tolist(), 'quantile_method': 'linear', 'fit_role': 'normal_reference',
            'normal_window_n': len(values)}


def fit_reconstruction_profile(records, labels):
    """The normal candidate error/envelope reference used by the 12 features."""
    if any(not np.asarray(r['payload']['valid'], bool).all() for r in records):
        return None
    payloads = [r['payload'] for r in records]
    before = np.stack([(np.asarray(p['signed'])[-1] ** 2).mean(axis=0) for p in payloads])
    reference = np.maximum(relative.high_quantile(before), 1e-12)
    errors, repaired = [], []
    for p in payloads:
        raw, corrected = np.asarray(p['raw']), np.asarray(p['corrected'])
        errors.append([np.mean(np.abs(corrected[j, :, j] - raw[-60:, j]))
                       for j in range(raw.shape[1])])
        repaired.append(np.max(np.asarray(p['after']) / reference[None], axis=1))
    errors, repaired = np.asarray(errors), np.asarray(repaired)
    result = {'error_mae': relative.high_quantile(errors, .9).tolist(),
              'repaired_limit': relative.high_quantile(repaired).tolist(),
              'residual_reference': reference.tolist(), 'groups': {}, 'n': len(records)}
    final = [row[-1] for row in labels]
    for phase in sorted(set(final)):
        mask = np.asarray(final) == phase
        if phase != 'unavailable' and mask.sum() >= 3:
            result['groups'][phase] = {'error_mae': relative.high_quantile(errors[mask], .9).tolist(),
                                       'repaired_limit': relative.high_quantile(repaired[mask]).tolist(),
                                       'n': int(mask.sum())}
    return result


def _development_labels(record, *, absolute=False):
    case, payload = record['case'], record['payload']
    if case['role'] not in ('decision_fit', 'risk_calibration', 'selection_normal_fit'):
        raise ValueError('Clean development quality is forbidden for this role')
    if 'clean' not in record:
        raise ValueError('Explicit development-only clean reference is required')
    raw, corrected = np.asarray(payload['raw']), np.asarray(payload['corrected'])
    clean = np.asarray(record['clean'])
    if clean.shape != raw.shape:
        raise ValueError('Development clean reference is not aligned')
    targets = np.stack([corrected[j, :, j] for j in range(raw.shape[1])])
    positive, usable, benefit = selection.development_labels(
        role=case['role'], family=case.get('family', 'unchanged'),
        target_index=case.get('target_index'), observed_window=raw[-60:],
        corrected_targets=targets, clean_window=clean[-60:], valid=payload['valid'],
        control_eligible=case.get('control_eligible', True))
    if not absolute and case.get('family') not in ('drift', 'stuck'):
        positive[:] = 0
    return positive, usable, benefit


def _relative_risk(records, profiles, metadata, model, admission_name):
    rows = []
    for record in _role(records, 'risk_calibration'):
        features, _ = observable_features(record['payload'], record['frame'], profiles,
                                          metadata=metadata, context_status=_status(record))
        probability = ranker.predict(features, model)
        winner = unique_top(probability, record['payload']['valid'])
        others = probability[np.asarray(record['payload']['valid'], bool) &
                             (np.arange(len(probability)) != winner)] if winner is not None else []
        margin = float(probability[winner] - (np.max(others) if len(others) else 0.)) if winner is not None else 0.
        positive, _, _ = _development_labels(record)
        admission = event_admissions(record['payload'], record['frame'], profiles,
                                     metadata=metadata, context_status=_status(record))[admission_name]
        rows.append({'case_id': record['case']['case_id'],
                     'normal': record['case'].get('family') == 'unchanged',
                     'event_alarm': bool(admission['numeric_alarm']), 'margin': margin,
                     'winner': winner, 'correct_beneficial': bool(winner is not None and positive[winner]),
                     'family': record['case'].get('family', 'unchanged')})
    return rows


def _choose_relative(records, profiles, metadata, model, admission_name):
    risk = _relative_risk(records, profiles, metadata, model, admission_name)
    if not risk:
        return _disabled('missing_risk_calibration')
    return {**relative.choose_threshold(risk), 'risk_n': len(risk),
            'normal_n': sum(r['normal'] for r in risk),
            'risk_case_ids': [r['case_id'] for r in risk],
            'process_and_context_guards_during_threshold_fit': False}


def fit_relative_ranker(records, profiles, *, metadata, seed=23):
    """Fit fixed linear/tree alternatives, then select by risk-only usefulness."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    fitted = _role(records, 'decision_fit')
    if profiles.get('bank') is None or profiles.get('reconstruction') is None or not fitted:
        reason = 'incomplete_normal_candidate_calibration' if profiles.get('reconstruction') is None else 'missing_dictionary_or_decision_fit'
        return {'model': None, 'eligible': False, 'disabled_reason': reason,
                **{'selection_' + method: _disabled(reason)
                   for method in ('original_calibrated', 'reference', 'context')}}
    xx, yy = [], []
    for record in fitted:
        features, _ = observable_features(record['payload'], record['frame'], profiles,
                                          metadata=metadata, context_status=_status(record))
        positive, usable, _ = _development_labels(record)
        if usable.any():
            xx.append(features[usable]); yy.append(positive[usable])
    x = np.concatenate(xx) if xx else np.empty((0, 12))
    y = np.concatenate(yy) if yy else np.empty(0, dtype=np.uint8)
    receipt = {'candidate_n': len(y), 'positive_n': int(y.sum()), 'fit_seed': seed,
               'role': 'decision_fit', 'case_ids': [r['case']['case_id'] for r in fitted]}
    if len(np.unique(y)) != 2:
        reason = 'single_class_or_empty_development_labels'
        return {'model': None, 'eligible': False, 'disabled_reason': reason, 'fit_receipt': receipt,
                **{'selection_' + method: _disabled(reason)
                   for method in ('original_calibrated', 'reference', 'context')}}
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    with threadpool_limits(limits=4):
        linear = LogisticRegression(C=1., class_weight='balanced', max_iter=2000,
                                    random_state=31017).fit(z, y)
        tree_estimator = HistGradientBoostingClassifier(**RANKER_PARAMETERS).fit(z, y)
    common = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
              'features': list(relative.FEATURES), 'fit_seed': seed,
              'fit_candidate_n': len(y), 'fit_positive_n': int(y.sum())}
    linear_model = {**common, 'kind': 'linear', 'coefficient': linear.coef_[0].tolist(),
                    'intercept': float(linear.intercept_[0])}
    tree = {**ranker.export_tree_model(tree_estimator, scaler.mean_, scaler.scale_), **common}
    error = float(np.max(np.abs(ranker.predict(x, tree) - tree_estimator.predict_proba(z)[:, 1])))
    if error > 1e-12:
        raise ValueError('Portable relative ranker differs from fitted estimator')
    alternatives = []
    for model in (linear_model, tree):
        chosen = _choose_relative(records, profiles, metadata, model, 'original_calibrated')
        selected = next((r for r in chosen['grid'] if r['threshold'] == chosen['threshold']), None)
        alternatives.append({'model': model, 'selection': chosen,
                             'good': selected['good'] if selected and chosen['eligible'] else -1,
                             'wrong': selected['wrong_or_not_beneficial'] if selected and chosen['eligible'] else 10**9})
    eligible = [a for a in alternatives if a['selection']['eligible']]
    chosen = min(eligible, key=lambda a: (-a['good'], a['wrong'], a['model']['kind'] != 'linear')) if eligible else alternatives[0]
    model = chosen['model']
    receipt.update(feature_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                   label_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                   portable_max_abs_error=error)
    return {'model': model, 'eligible': True, 'fit_receipt': receipt,
            'selection_original_calibrated': chosen['selection'],
            'selection_reference': _choose_relative(records, profiles, metadata, model, 'reference'),
            'selection_context': _choose_relative(records, profiles, metadata, model, 'context'),
            'alternatives': [{'kind': a['model']['kind'], 'selection': a['selection']} for a in alternatives]}


def fit_profiles(records, *, metadata, seed=23, out=None, ranker_bundle=None,
                 pooled_selector=None, fit_level='full'):
    """Fit normal profiles and development-only rules for one configuration/seed.

    Supply the seed-23 ``ranker_bundle`` to subsequent checkpoints to retain its
    learned ranker and policy thresholds. ``pooled_selector`` may be attached
    after all configurations' seed-23 relative models have been fitted.
    """
    records = _records(records)
    reference, calibration = _normal_inputs(records, metadata)
    d = len(metadata['features'])
    if d < 2:
        raise ValueError('At least two AE sensor channels are required')
    normal_dictionary = [r for r in _role(records, 'dictionary')
                         if r['case'].get('family', 'unchanged') == 'unchanged']
    if not normal_dictionary:
        raise ValueError('Original calibration requires unchanged dictionary periods')
    signed_dictionary = np.stack([r['payload']['signed'] for r in normal_dictionary])
    dictionary_labels = [phases(r['frame'], metadata['context_unit']) for r in normal_dictionary]
    calibration_labels = [phases(r['frame'], metadata['context_unit']) for r in calibration]
    native_values = [float(r.get('native_threshold', metadata.get('native_threshold', float('nan'))))
                     for r in records]
    native_threshold = native_values[0]
    if not np.isfinite(native_threshold) or native_threshold <= 0 or not np.allclose(native_values, native_threshold, rtol=0, atol=0):
        raise ValueError('One positive frozen native threshold is required per model')
    noise = relative.fit_noise(signed_dictionary, dictionary_labels)
    components = np.stack([relative.event_components(z, p, noise, native_threshold)
                           for z, p in zip(signed_dictionary, dictionary_labels)])
    noise['component_scale'] = np.maximum(relative.high_quantile(components), 1e-6).tolist()
    normal_events = np.stack([relative.event_scores(r['payload']['signed'], labels, noise, native_threshold)
                              for r, labels in zip(calibration, calibration_labels)])
    refs, cals = _contexts(reference, metadata), _contexts(calibration, metadata)
    a = np.stack([r['payload']['signed'] for r in reference])
    b = np.stack([r['payload']['signed'] for r in calibration])
    shared = history.fit_profile(a, [r['history_class'] for r in refs],
                                 b, [r['history_class'] for r in cals])
    conditional = context.fit_context_profile(a, refs, list(map(_group, reference)),
        calibration_signed=b, calibration_contexts=cals,
        calibration_groups=list(map(_group, calibration)), level=fit_level)
    std = np.asarray(metadata.get('training_std', records[0].get('training_std')), float)
    ranges = np.asarray(metadata.get('training_ranges', records[0].get('training_ranges')), float)
    if std.shape != (d,) or ranges.shape != (d,) or not np.isfinite(std).all() or not np.isfinite(ranges).all() or (std < 0).any() or (ranges < 0).any():
        raise ValueError('Ordered training-only channel standard deviations and ranges are required')
    bank = fit_bank(_role(records, 'dictionary'))
    profiles = {'schema': 'safeprocess-fresh-profiles-v1', 'seed': int(seed),
                'feature_order': list(metadata['features']), 'metadata_binding': metadata_binding(metadata),
                'native_threshold': native_threshold, 'training_std': std.tolist(),
                'training_ranges': ranges.tolist(), 'noise': noise,
                'event_limits': relative.fit_event_limits(normal_events, [x[-1] for x in calibration_labels]),
                'reconstruction': fit_reconstruction_profile(calibration, calibration_labels),
                'bank': bank, 'reference': shared, 'context': conditional,
                'process': fit_process_profile(reference, calibration, metadata=metadata),
                'dominance': fit_dominance_profile(reference), 'ranker_bundle': {'model': None},
                'source_roles': dict(Counter(r['case']['role'] for r in records)),
                'normal_candidate_failures': [r['case']['case_id'] for r in calibration
                                              if not np.asarray(r['payload']['valid'], bool).all()],
                'assessment_labels_used': False, 'fresh_profile': True}
    profiles['ranker_bundle'] = deepcopy(ranker_bundle) if ranker_bundle is not None else fit_relative_ranker(
        records, profiles, metadata=metadata, seed=seed)
    profiles['ranker_bundle'].setdefault('selector', None)
    for method in ('selection', 'combined'):
        profiles['ranker_bundle'].setdefault('selection_' + method, _disabled('pooled_selector_not_fitted'))
    if pooled_selector is not None:
        profiles = attach_pooled_selector(profiles, records, metadata=metadata, selector=pooled_selector)
    profiles['profile_id'] = _identity(profiles)
    if out is not None:
        path = Path(out)
        if path.suffix != '.json':
            path = path / 'profiles.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profiles, indent=2, allow_nan=False) + '\n')
    return profiles


def _identity(profiles):
    return hashlib.sha256(json.dumps({k: v for k, v in profiles.items() if k != 'profile_id'},
                                    sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _unit(record):
    case = record['case']
    value = case.get('unit', case.get('dataset', record.get('unit')))
    if value is None:
        raise ValueError('Pooled selection requires a configuration identity on every development case')
    return str(value)


def fit_pooled_selector(records, metadata_by_dataset, *, profiles_by_unit):
    """Fit one absolute scorer across declared development configurations."""
    from threadpoolctl import threadpool_limits
    records = _records(records)
    xx, yy, configurations, families, groups, roles = [], [], [], [], [], []
    excluded = Counter()
    for record in records:
        role = record['case']['role']
        if role not in ('decision_fit', 'selection_normal_fit'):
            continue
        unit = _unit(record)
        profile, metadata = profiles_by_unit[unit], metadata_by_dataset[unit]
        if profile['ranker_bundle'].get('model') is None or profile.get('reconstruction') is None:
            excluded[unit] += 1
            continue
        features, probability = observable_features(record['payload'], record['frame'], profile,
                                                     metadata=metadata, context_status=_status(record))
        x = absolute_features(record['payload'], features, probability, profile)
        positive, usable, _ = _development_labels(record, absolute=True)
        for j in np.flatnonzero(usable):
            xx.append(x[j]); yy.append(positive[j]); configurations.append(unit)
            families.append(record['case'].get('family', 'unchanged'))
            groups.append(_group(record)); roles.append(role)
    if len(set(map(int, yy))) != 2:
        return {'kind': 'unavailable', 'eligible': False,
                'disabled_reason': 'single_class_or_empty_pooled_development_labels',
                'fit_receipt': {'candidate_n': len(yy), 'positive_n': int(sum(yy)),
                                'excluded_unfittable_configuration_cases': dict(excluded)}}
    with threadpool_limits(limits=4):
        model = selection.fit_selector(xx, yy, configurations, families, groups, roles)
    model['eligible'] = True
    model['fit_receipt']['excluded_unfittable_configuration_cases'] = dict(excluded)
    return model


def attach_pooled_selector(profiles, records, *, metadata, selector):
    """Attach a fitted pooled scorer and fit this configuration's risk cutoffs."""
    records = _records(records)
    result = deepcopy(profiles)
    bundle = result['ranker_bundle']
    if selector is None or selector.get('kind') == 'unavailable' or not selector.get('eligible', True):
        bundle['selector'] = None
        for method in ('selection', 'combined'):
            bundle['selection_' + method] = _disabled('pooled_selector_unfittable')
    elif bundle.get('model') is None or result.get('reconstruction') is None:
        bundle['selector'] = deepcopy(selector)
        for method in ('selection', 'combined'):
            bundle['selection_' + method] = _disabled('relative_features_unavailable')
    else:
        bundle['selector'] = deepcopy(selector)
        for method, admission_name in (('selection', 'reference'), ('combined', 'context')):
            cases = []
            for record in _role(records, 'risk_calibration'):
                features, probability = observable_features(record['payload'], record['frame'], result,
                                                             metadata=metadata, context_status=_status(record))
                scores = selection.predict_scores(absolute_features(record['payload'], features, probability, result), selector)
                positive, _, _ = _development_labels(record, absolute=True)
                admission = event_admissions(record['payload'], record['frame'], result,
                                             metadata=metadata, context_status=_status(record))[admission_name]
                cases.append({'case_id': record['case']['case_id'], 'role': 'risk_calibration',
                              'unit': record['case'].get('unit', metadata.get('dataset', metadata['context_unit'])),
                              'normal': record['case'].get('family') == 'unchanged',
                              'event_alarm': admission['numeric_alarm'], 'scores': scores.tolist(),
                              'valid': np.asarray(record['payload']['valid'], bool).tolist(),
                              'positive': positive.tolist()})
            bundle['selection_' + method] = selection.choose_absolute_threshold(cases) if cases else _disabled('missing_risk_calibration')
    result['profile_id'] = _identity(result)
    return result


def fit_history_ladder(records, *, metadata, profiles):
    """Fit four auxiliary normalizers on cached normal/development evidence.

    Both comparison tracks retain the primary ranker, guards and candidate
    evidence. Event admission changes with normalization in both tracks.
    """
    records = _records(records)
    reference, calibration = _normal_inputs(records, metadata)
    a = np.stack([r['payload']['signed'] for r in reference])
    b = np.stack([r['payload']['signed'] for r in calibration])
    refs, cals = _contexts(reference, metadata), _contexts(calibration, metadata)
    model = profiles['ranker_bundle'].get('model')
    variants = {}
    full_gap = deepcopy(profiles['ranker_bundle']['selection_context'])
    for level in context.HISTORY_LEVELS:
        fitted = context.fit_context_profile(a, refs, list(map(_group, reference)),
                    calibration_signed=b, calibration_contexts=cals,
                    calibration_groups=list(map(_group, calibration)), level=level)
        temporary = deepcopy(profiles)
        temporary['context'] = fitted
        refitted = _choose_relative(records, temporary, metadata, model, 'context') if model is not None else _disabled('unfittable_primary_ranker')
        variants[level] = {'normalizer': fitted, 'risk_refitted_gap': refitted,
                           'fixed_full_gap': deepcopy(full_gap)}
    return {'schema': 'safeprocess-fresh-history-ladder-v1', 'primary_profile_id': profiles['profile_id'],
            'metadata_binding': metadata_binding(metadata), 'base_profiles': deepcopy(profiles),
            'variants': variants, 'cpu_only': True, 'additional_model_inference': False,
            'interpretation': 'Explanatory development-data analysis. Normalization changes event admission in both tracks; neither is an isolated causal feature effect.'}


def predict_history_ladder(payload, frame, ladder, *, metadata, context_status='available'):
    """Replay four normalizers/two gap tracks, sharing observable ranker/guard work."""
    if ladder['metadata_binding'] != metadata_binding(metadata):
        raise ValueError('History ladder metadata binding differs')
    profile = ladder['base_profiles']
    _, probabilities = observable_features(payload, frame, profile, metadata=metadata,
                                            context_status=context_status)
    process, _ = process_evidence(payload, profile, metadata=metadata)
    flag = bool(process['process_flag'])
    native = bool((np.asarray(payload['residuals']).mean(axis=1) > profile['native_threshold']).any())
    component = context.context_features(frame, metadata['context_unit'], context_status)
    result = {}
    for level, variant in ladder['variants'].items():
        result[level] = {}
        admission = context.score_context(payload['signed'], component, variant['normalizer'])
        for comparison, gap_key in (('risk_refitted', 'risk_refitted_gap'), ('fixed_full_gap', 'fixed_full_gap')):
            gap = variant[gap_key]
            pred = relative_decision(probabilities, payload['valid'],
                event_alarm=admission['numeric_alarm'], threshold=gap['threshold'],
                eligible=gap['eligible'], context_valid=not admission['invalid_context'], process_flag=flag)
            prediction = _finish(pred, native_alarm=native, admission=admission,
                                  process=flag, preview=pred['preview'])
            prediction.update(history_level=level, comparison=comparison, method='context',
                              graph_status=process['graph_status'],
                              persistent_peaks=process['persistent_peaks'], status='completed')
            result[level][comparison] = prediction
    return result
