"""Observable-only implementation of the seven SafeProcess paper policies.

All fitting and evaluator truth live outside this module. A prediction can
recommend a reconstruction for review. It never authorizes a plant action.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import numpy as np
from . import context
from .signatures import bank_arrays, signature_features, signature_scores, envelope_veto
from .kernels import history, relative, prominence, dominance, ranker, selection
from .kernels.common import unique_top

METHODS = ('original_calibrated', 'residual_dominance', 'envelope_screened',
           'reference', 'context', 'selection', 'combined')


def metadata_binding(metadata):
    """Hash channel order, ownership and declared connections, not local paths."""
    keys = ('features', 'owners', 'connections', 'context_unit')
    return hashlib.sha256(json.dumps({k: metadata.get(k) for k in keys},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def phases(frame, unit):
    """Five 80%-occupancy operating-state labels used by Original calibrated."""
    if unit.startswith('hai_'):
        groups = {}
        for col in frame:
            if '.state_' in col and col.endswith('.active'):
                groups.setdefault(col.split('.state_', 1)[0], []).append(col)
        if not groups:
            return ['unavailable'] * 5
        components = []
        for component, columns in sorted(groups.items()):
            values = frame[columns].to_numpy(float)
            valid = np.isin(values, [0, 1]).all(axis=1) & (values.sum(axis=1) == 1)
            names = [c.split('.state_', 1)[1].removesuffix('.active') for c in columns]
            components.append([names[int(j)] if ok else None
                               for j, ok in zip(values.argmax(axis=1), valid)])
        rows = ['unavailable' if None in values else values[0] if len(values) == 1 else
                '|'.join(f'{name}={value}' for name, value in zip(sorted(groups), values))
                for values in zip(*components)]
    elif unit == 'four_tank':
        if 'CuStepNo ValueY' not in frame:
            return ['unavailable'] * 5
        rows = [f'step_{int(v)}' if np.isfinite(v) else 'unavailable'
                for v in frame['CuStepNo ValueY']]
    elif unit == 'swat':
        columns = ['MV301', 'MV302', 'P301', 'P302', 'P401', 'P402']
        if any(c not in frame for c in columns):
            return ['unavailable'] * 5
        values = frame[columns].to_numpy(float)
        rows = ['state_' + '_'.join(str(int(v)) for v in row)
                if np.isfinite(row).all() else 'unavailable' for row in values]
    else:
        raise ValueError('Unknown context adapter: ' + str(unit))
    result = []
    for start in context.STARTS:
        label, count = Counter(rows[start:start + 60]).most_common(1)[0]
        result.append(label if count >= 48 else 'transition')
    return result


def observable_features(payload, frame, profiles, *, metadata, context_status='available'):
    """The twelve candidate features and original probability, without labels."""
    labels = phases(frame, metadata['context_unit'])
    if context_status != 'available':
        labels = ['unavailable'] * 5
    if profiles.get('bank') is None or profiles.get('reconstruction') is None:
        return None, None
    bank = bank_arrays(profiles['bank'])
    scores = signature_scores(signature_features(payload['signed'], bank['whitener']), bank)
    features = relative.candidate_features(
        np.asarray(payload['raw'])[-60:], np.asarray(payload['signed']), scores,
        np.asarray(payload['corrected']), np.asarray(payload['after']),
        np.asarray(payload['fractions']), np.asarray(payload['valid']), labels[-1],
        profiles['reconstruction'], np.asarray(profiles['training_std']),
        labels[-1] in profiles['noise']['groups'])
    model = profiles['ranker_bundle'].get('model')
    probabilities = ranker.predict(features, model) if model is not None else None
    return features, probabilities


def absolute_features(payload, features, probabilities, profiles):
    corrected = np.asarray(payload['corrected'])
    d = corrected.shape[0]
    target_values = np.stack([corrected[j, :, j] for j in range(d)])
    correction = selection.masked_correction(np.asarray(payload['raw'])[-60:], target_values)
    return selection.build_candidate_features(features, probabilities, correction,
                                               profiles['training_ranges'], payload['valid'])


def event_admissions(payload, frame, profiles, *, metadata, context_status='available'):
    """Normalizer-specific A and U states, independent of candidate quality."""
    labels = phases(frame, metadata['context_unit'])
    if context_status != 'available':
        labels = ['unavailable'] * 5
    component = context.context_features(frame, metadata['context_unit'], context_status)
    cls = component['history_class'] if not component['invalid_context'] else 'unavailable'
    original_scores = relative.event_scores(payload['signed'], labels, profiles['noise'],
                                           profiles['native_threshold'])
    original_alarm = relative.event_alarm(original_scores, labels[-1], profiles['event_limits'])
    shared = history.admission(payload['signed'], cls, profiles['reference'])
    shared['invalid_context'] = bool(shared['invalid_context'] or component['invalid_context'])
    conditional = context.score_context(payload['signed'], component, profiles['context'])
    return {'original_calibrated': {'numeric_alarm': original_alarm,
                                   'invalid_context': component['invalid_context'],
                                   'event_scores': original_scores.tolist()},
            'reference': shared, 'context': conditional}


def process_evidence(payload, profiles, *, metadata):
    """Residual-weighted absolute Jacobians and the induced declared graph."""
    guard = profiles['process']
    sensitivity = prominence.excess(prominence.weighted_sensitivity(
        np.asarray(payload['residuals']), np.asarray(payload['influence']),
        guard['residual_profile']), guard['sensitivity_profile'])
    owners = metadata.get('owners', [])
    if isinstance(owners, dict):
        owners = [owners.get(f) for f in metadata['features']]
    missing = len(owners) != len(metadata['features']) or any(o is None or o == '' for o in owners)
    graph = {'owners': ['__unknown_owner__'] * len(metadata['features']) if missing else owners,
             'system_connections': metadata.get('connections', [])}
    result = prominence.peak_route(sensitivity, guard['sensitivity_threshold'], graph)
    if missing:
        result.update(process_flag=False, graph_status='missing_component_ownership')
    elif 'connections' not in metadata:
        result.update(process_flag=False, graph_status='missing_declared_connection_information')
    return result, sensitivity


def _finish(prediction, *, native_alarm, admission, process, preview=None):
    result = dict(prediction)
    event = bool(result.get('numeric_alarm', False))
    invalid = bool(admission.get('invalid_context', False))
    concern = bool(event or process or invalid or native_alarm)
    target = result.get('target_index_recommended') if result.get('proposed') else None
    result.update(detected=event, target=target, target_index_recommended=target,
                  preview=preview, preview_target=preview, preview_target_index=preview,
                  operator_preview=preview is not None, concern=concern, review=concern,
                  F=bool(process), process_flag=bool(process), U=not invalid,
                  context_valid=not invalid, original_alarm=bool(native_alarm),
                  invalid_context=invalid, admission=admission,
                  route='reconstruction_assessment' if result.get('proposed') else
                        'user_review' if concern else 'accept',
                  replacement_authorized=False, operator_confirmation_required=bool(result.get('proposed')),
                  applied_action=None)
    reasons = list(result.get('reasons', result.get('review_reasons', [])))
    if native_alarm and 'native_ae_concern' not in reasons:
        reasons.append('native_ae_concern')
    if invalid and 'invalid_or_untrusted_context' not in reasons:
        reasons.append('invalid_or_untrusted_context')
    if process and 'process_veto' not in reasons:
        reasons.append('process_veto')
    result['reasons'] = reasons
    return result


def relative_decision(probabilities, valid, *, event_alarm, threshold,
                      context_valid=True, process_flag=False, eligible=True):
    """Strict relative-gap K with all-candidate validity and deterministic ties."""
    valid = np.asarray(valid, bool)
    winner, margin = None, 0.
    if probabilities is not None:
        p = np.asarray(probabilities, float)
        if p.shape != valid.shape or p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Finite aligned candidate probabilities required')
        winner = unique_top(p, valid)
        if winner is not None:
            other = p[valid & (np.arange(len(p)) != winner)]
            margin = float(p[winner] - (other.max() if len(other) else 0.))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Relative confidence threshold outside [0,1]')
    qualified = bool(eligible and probabilities is not None and winner is not None
                     and valid.all() and margin > threshold and context_valid)
    proposed = bool(event_alarm and qualified and not process_flag)
    preview = winner if event_alarm and qualified and process_flag else None
    reasons = []
    if event_alarm:
        reasons.append('numeric_event')
    if not eligible or probabilities is None:
        reasons.append('ineligible_development_calibration')
    if not valid.all():
        reasons.append('relative_requires_all_candidates_valid')
    if winner is None:
        reasons.append('no_unique_relative_winner')
    elif margin <= threshold:
        reasons.append('relative_gap_not_strictly_above_threshold')
    return {'numeric_alarm': bool(event_alarm), 'proposed': proposed,
            'target_index_recommended': winner if proposed else None, 'ranked_target': winner,
            'margin': margin, 'confidence_threshold': float(threshold),
            'valid_all': bool(valid.all()), 'qualified': qualified,
            'preview': preview, 'reasons': reasons}


def predict_all(payload, frame, profiles, *, metadata, context_status='available'):
    """Predict all seven policies. This API has no clean/label/target inputs."""
    if metadata_binding(metadata) != profiles['metadata_binding']:
        raise ValueError('Channel order, context or ontology binding differs from calibration')
    d = len(metadata['features'])
    required = {'raw': (300, d), 'signed': (5, 60, d), 'residuals': (5, d),
                'influence': (5, d, d), 'corrected': (d, 60, d), 'valid': (d,),
                'after': (d, d), 'fractions': (d,)}
    for key, shape in required.items():
        a = np.asarray(payload[key])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('Nonfinite or misaligned observable payload: ' + key)
    if len(frame) != 300:
        raise ValueError('Context must align with the 300-row observation')
    admissions = event_admissions(payload, frame, profiles, metadata=metadata, context_status=context_status)
    process, sensitivity = process_evidence(payload, profiles, metadata=metadata)
    flag = bool(process['process_flag'])
    residuals = np.asarray(payload['residuals'])
    active = residuals.mean(axis=1) > profiles['native_threshold']
    native = bool(active.any())
    features, probabilities = observable_features(payload, frame, profiles, metadata=metadata,
                                                  context_status=context_status)
    bundle = profiles['ranker_bundle']
    result = {}
    for method, admission_name in (('original_calibrated', 'original_calibrated'),
                                   ('reference', 'reference'), ('context', 'context')):
        admission = admissions[admission_name]
        gap = bundle.get('selection_' + method, {'threshold': 1., 'eligible': False})
        pred = relative_decision(probabilities, payload['valid'], event_alarm=admission['numeric_alarm'],
                                 threshold=gap['threshold'], eligible=gap['eligible'],
                                 context_valid=not admission['invalid_context'], process_flag=flag)
        result[method] = _finish(pred, native_alarm=native, admission=admission,
                                 process=flag, preview=pred['preview'])
    absolute_scores = None
    selector = bundle.get('selector')
    if selector is not None and features is not None and probabilities is not None:
        absolute_scores = selection.predict_scores(absolute_features(payload, features, probabilities, profiles), selector)
    for method, admission_name in (('selection', 'reference'), ('combined', 'context')):
        admission = admissions[admission_name]
        threshold = bundle.get('selection_' + method, {'threshold': 1., 'eligible': False})
        if absolute_scores is None or not threshold['eligible']:
            pred = {'numeric_alarm': admission['numeric_alarm'], 'proposed': False,
                    'target_index_recommended': None, 'ranked_target': None,
                    'qualifying_count': 0, 'reasons': ['ineligible_development_calibration']}
            preview = None
        else:
            status = 'invalid' if admission['invalid_context'] else 'available'
            pred = selection.predict_absolute(absolute_scores, payload['valid'], threshold['threshold'],
                                              admission['numeric_alarm'], flag, status, native)
            preview = pred['absolute_candidate'] if (admission['numeric_alarm'] and flag and
                       not admission['invalid_context'] and pred['qualifying_count'] == 1) else None
        result[method] = _finish(pred, native_alarm=native, admission=admission,
                                 process=flag, preview=preview)
    route = dominance.residual_route(residuals, active, variant='normal_calibrated_residual',
                                     q99=np.asarray(profiles['dominance']['q99']),
                                     scale=np.asarray(profiles['dominance']['scale']))
    # These two baselines have their own selection contracts. The learned-policy
    # context/all-candidate guards are not silently added to their kernels.
    baseline = {'numeric_alarm': native, 'proposed': route['route'] == 'reconstruction',
                'target_index_recommended': route['target_index'], 'reasons': [], 'trace': route}
    result['residual_dominance'] = _finish(baseline, native_alarm=native,
                                         admission={'invalid_context': False}, process=False)
    confirmed = prominence.confirm_probe(process, np.asarray(payload['fractions']),
                                         np.asarray(payload['valid']), profiles['process']['probe_threshold'])
    envelope = envelope_veto(confirmed, np.asarray(payload['after']), np.asarray(payload['valid']), profiles['process'])
    baseline = {'numeric_alarm': bool(process['alarm']),
                'proposed': envelope['route'] == 'reconstruction',
                'target_index_recommended': envelope['target_index'], 'reasons': [], 'trace': envelope}
    result['envelope_screened'] = _finish(baseline, native_alarm=native,
                                        admission={'invalid_context': False}, process=flag)
    for method in result:
        result[method]['method'] = method
        result[method]['graph_status'] = process['graph_status']
        result[method]['persistent_peaks'] = process['persistent_peaks']
        result[method]['model_scores'] = probabilities.tolist() if probabilities is not None else None
        result[method]['absolute_scores'] = absolute_scores.tolist() if absolute_scores is not None else None
        result[method]['status'] = 'completed'
    return {method: result[method] for method in METHODS}
