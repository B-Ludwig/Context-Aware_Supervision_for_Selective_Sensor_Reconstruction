"""Scientific boundary checks independent of recordings and the results artifact."""
from copy import deepcopy
import inspect
import json
import numpy as np
import pandas as pd
import pytest
from safeprocess import context
from safeprocess.calibration import (fit_profiles, fit_pooled_selector,
    attach_pooled_selector, fit_history_ladder, predict_history_ladder)
from safeprocess.policies import (METHODS, predict_all, relative_decision,
                                  metadata_binding, process_evidence)
from safeprocess.signatures import TEMPLATES
from safeprocess.kernels import prominence, dominance, selection


def _frame(transition=False):
    active = np.zeros(300, int)
    if transition:
        active[250:] = 1
    return pd.DataFrame({'unit.state_idle.active': 1-active,
                         'unit.state_work.active': active})


def _record(identity, role, family='unchanged', target=None, *, amplitude=0., base_id=None,
            transition=False):
    seed = sum(identity.encode())
    rng = np.random.default_rng(seed)
    clean = rng.normal(.5, .02, (300, 2))
    raw = clean.copy()
    signed = rng.normal(0, .005, (5, 60, 2))
    if target is not None:
        offset = .5 if amplitude == 0 else float(amplitude)/100
        raw[60:, target] += offset
        signed[:, :, target] += offset
    corrected = np.repeat(raw[-60:][None], 2, axis=0)
    for j in range(2):
        corrected[j, :, j] = clean[-60:, j] + (0. if j == target else .002)
    after = np.repeat(np.mean(signed[-1]**2, axis=0)[None], 2, axis=0)
    for j in range(2):
        after[j, j] = .00001
    case = {'case_id': identity, 'role': role, 'unit': 'hai_demo', 'family': family,
            'target_index': target, 'amplitude': amplitude}
    if base_id is not None:
        case['base_id'] = base_id
    return {'case': case, 'frame': _frame(transition), 'clean': clean,
            'source_group': identity, 'native_threshold': .001,
            'training_std': np.array([.02, .02]), 'training_ranges': np.ones(2),
            'payload': {'raw': raw, 'signed': signed,
                        'residuals': (signed**2).mean(axis=1),
                        'influence': np.repeat(np.eye(2)[None], 5, axis=0),
                        'corrected': corrected, 'valid': np.ones(2, bool),
                        'after': after, 'fractions': np.array([
                            .8 if j == target else -.1 for j in range(2)])}}


@pytest.fixture(scope='module')
def fitted():
    metadata = {'dataset': 'hai_demo', 'context_unit': 'hai_demo',
                'features': ['sensor_a', 'sensor_b'], 'owners': ['a', 'b'],
                'connections': [['a', 'b']], 'training_std': [.02, .02],
                'training_ranges': [1., 1.], 'native_threshold': .001}
    records = [_record('dictionary-base', 'dictionary')]
    for j in range(2):
        for i, (family, amp) in enumerate(TEMPLATES):
            records.append(_record(f'template-{j}-{i}', 'dictionary', family, j,
                                   amplitude=amp, base_id='dictionary-base'))
    records += [_record(f'ref-{i}', 'normal_reference', transition=bool(i % 2)) for i in range(6)]
    records += [_record(f'cal-{i}', 'normal_calibration', transition=bool(i % 2)) for i in range(4)]
    for role, n in [('decision_fit', 30), ('risk_calibration', 8)]:
        for i in range(n):
            records.append(_record(f'{role}-{i}', role, 'drift', i % 2,
                                   amplitude=10 + i, transition=bool(i % 2)))
        records += [_record(f'{role}-normal-{i}', role) for i in range(2)]
    profiles = fit_profiles(records, metadata=metadata)
    selector = fit_pooled_selector(records, {'hai_demo': metadata},
                                   profiles_by_unit={'hai_demo': profiles})
    profiles = attach_pooled_selector(profiles, records, metadata=metadata, selector=selector)
    return records, metadata, profiles


def test_relative_strict_gap_ties_and_candidate_failure():
    opts = dict(event_alarm=True, threshold=.5, context_valid=True)
    assert not relative_decision([.75, .25], [True, True], **opts)['proposed']
    assert not relative_decision([.8, .8], [True, True], **opts)['proposed']
    assert not relative_decision([.9, .1], [True, False], **opts)['proposed']
    assert relative_decision([.9, .1], [True, True], **opts)['proposed']
    assert not relative_decision([.9, .1], [True, True], event_alarm=False, threshold=.5)['proposed']


def test_process_veto_preserves_preview_and_context_contract():
    pred = relative_decision([.9, .1], [True, True], event_alarm=True,
                             threshold=.5, process_flag=True)
    assert pred['preview'] == 0 and not pred['proposed']
    pred = relative_decision([.9, .1], [True, True], event_alarm=True,
                             threshold=.5, process_flag=True, context_valid=False)
    assert pred['preview'] is None


def test_absolute_excludes_invalid_alternative_but_rejects_ambiguity():
    assert selection.predict_absolute([.9, .9], [True, False], .5, True)['proposed']
    assert not selection.predict_absolute([.9, .9], [True, True], .5, True)['proposed']
    assert not selection.predict_absolute([.5, .1], [True, True], .5, True)['proposed']
    assert not selection.predict_absolute([.9, .1], [True, True], .5, False)['proposed']


def test_graph_requires_connected_induced_peak_owners():
    scores = np.ones((5, 3)) * 10
    graph = {'owners': ['a', 'middle', 'b'], 'system_connections': [['a', 'middle'], ['middle', 'b']]}
    scores[:, 1] = 0
    result = prominence.peak_route(scores, 1, graph)
    assert result['graph_status'] == 'disconnected_or_incomplete_graph'
    assert not result['process_flag']
    graph['system_connections'].append(['a', 'b'])
    assert prominence.peak_route(scores, 1, graph)['process_flag']
    graph['owners'] = ['same'] * 3
    assert not prominence.peak_route(scores, 1, graph)['process_flag']


def test_weighted_jacobian_preserves_output_normalization():
    r = np.array([[5., 3.]])
    profile = {'q95': [1., 1.], 'scale': [2., 1.]}
    influence = np.array([[[9., 1.], [1., 1.]]])
    np.testing.assert_allclose(prominence.weighted_sensitivity(r, influence, profile), [[2.8, 1.2]])


def test_context_is_causal_and_unknown_support_is_not_invalid(fitted):
    _, _, profiles = fitted
    frame = _frame(True)
    before = context.context_features(frame, 'hai_demo')
    frame.iloc[270:, :] = [1, 0]
    after = context.context_features(frame, 'hai_demo')
    assert before['windows'][0] == after['windows'][0]
    unseen = deepcopy(before)
    for window in unseen['windows']:
        window['components']['unit']['current_state'] = 'unseen'
        window['components']['unit']['occupancy'] = {'unseen': 1.}
    scored = context.score_context(np.ones((5, 60, 2)), unseen, profiles['context'])
    assert scored['unknown_state'] and scored['reference_fallback']
    assert not scored['invalid_context'] and scored['proposal_eligible']
    invalid = context.context_features(_frame(), 'hai_demo', 'untrusted')
    assert context.score_context(np.ones((5, 60, 2)), invalid, profiles['context'])['invalid_context']


def test_json_profiles_seven_methods_and_truth_free_boundary(fitted):
    records, metadata, profiles = fitted
    profile = json.loads(json.dumps(profiles, allow_nan=False))
    record = next(r for r in records if r['case']['role'] == 'risk_calibration')
    result = predict_all(record['payload'], record['frame'], profile, metadata=metadata)
    assert tuple(result) == METHODS
    assert all(r['status'] == 'completed' and not r['replacement_authorized'] for r in result.values())
    assert not {'clean', 'family', 'target_index'} & set(inspect.signature(predict_all).parameters)
    changed = deepcopy(metadata); changed['features'].reverse()
    with pytest.raises(ValueError, match='binding'):
        predict_all(record['payload'], record['frame'], profile, metadata=changed)


def test_native_concern_does_not_admit_a_proposal(fitted):
    records, metadata, profiles = fitted
    record = next(r for r in records if r['case']['role'] == 'risk_calibration')
    profile = deepcopy(profiles)
    profile['native_threshold'] = 1e-20
    for gap in ('selection_reference', 'selection_context', 'selection_original_calibrated',
                'selection_selection', 'selection_combined'):
        profile['ranker_bundle'][gap] = {'threshold': 1., 'eligible': False}
    result = predict_all(record['payload'], record['frame'], profile, metadata=metadata)
    for name in ('reference', 'context', 'original_calibrated', 'selection', 'combined'):
        assert result[name]['concern'] and not result[name]['proposed']


def test_disabled_labels_are_explicit_and_reference_calibration_is_disjoint(fitted):
    records, metadata, _ = fitted
    data = deepcopy(records)
    for record in data:
        if record['case']['role'] == 'decision_fit':
            record['case']['family'] = 'unchanged'; record['case']['target_index'] = None
    profile = fit_profiles(data, metadata=metadata)
    assert profile['ranker_bundle']['model'] is None
    assert profile['ranker_bundle']['disabled_reason'] == 'single_class_or_empty_development_labels'
    data = deepcopy(records)
    next(r for r in data if r['case']['role'] == 'normal_calibration')['source_group'] = 'ref-0'
    with pytest.raises(ValueError, match='overlap'):
        fit_profiles(data, metadata=metadata)
    data = deepcopy(records)
    data[0]['case']['role'] = 'assessment'
    with pytest.raises(ValueError, match='Forbidden'):
        fit_profiles(data, metadata=metadata)


def test_ladder_retains_trust_and_support_across_feature_levels(fitted):
    records, metadata, profiles = fitted
    ladder = fit_history_ladder(records, metadata=metadata, profiles=profiles)
    assert tuple(ladder['variants']) == context.HISTORY_LEVELS
    record = next(r for r in records if r['case']['role'] == 'risk_calibration')
    predictions = predict_history_ladder(record['payload'], record['frame'], ladder,
                                          metadata=metadata, context_status='untrusted')
    for comparisons in predictions.values():
        for prediction in comparisons.values():
            assert prediction['concern'] and not prediction['proposed'] and not prediction['U']
    full_gap = profiles['ranker_bundle']['selection_context']
    assert all(v['fixed_full_gap'] == full_gap for v in ladder['variants'].values())


def test_residual_dominance_requires_all_five_active_windows():
    residual = np.tile([100., 1.], (5, 1))
    kwargs = dict(variant='normal_calibrated_residual', q99=np.ones(2), scale=np.ones(2))
    assert dominance.residual_route(residual, np.ones(5, bool), **kwargs)['route'] == 'reconstruction'
    assert dominance.residual_route(residual, [1, 1, 1, 1, 0], **kwargs)['route'] == 'abstain'


def test_seed_transfer_reuses_fixed_ranker_and_all_policy_thresholds(fitted):
    records, metadata, profiles = fitted
    numerical_only = [r for r in records if r['case']['role'] in
                      ('normal_reference', 'normal_calibration', 'dictionary')]
    transferred = fit_profiles(numerical_only, metadata=metadata, seed=42,
                                ranker_bundle=profiles['ranker_bundle'])
    assert transferred['seed'] == 42
    assert transferred['ranker_bundle'] == profiles['ranker_bundle']


def test_normal_candidate_failure_disables_fitting_without_disappearing(fitted):
    records, metadata, _ = fitted
    data = deepcopy(records)
    failed = next(r for r in data if r['case']['role'] == 'normal_calibration')
    failed['payload']['valid'][0] = False
    profile = fit_profiles(data, metadata=metadata)
    assert failed['case']['case_id'] in profile['normal_candidate_failures']
    assert profile['reconstruction'] is None
    assert profile['ranker_bundle']['model'] is None
    assert profile['ranker_bundle']['disabled_reason'] == 'incomplete_normal_candidate_calibration'
