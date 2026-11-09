"""Explain an existing Reference/Context pair from portable closed evidence.

No model execution, optimization, fitting, threshold search or case selection.
The case is the already published illustrative example selected on 28 October.
The standalone package kernels replay the fixed numerical evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CASE = '292fd580e6f4676f9bfd'
SLOT = 'four_tank-seed23'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from safeprocess.kernels.history import admission
from safeprocess.context import score_context, conditional_normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPOSITORY)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assert root != out and root not in out.parents, 'Write outside retained evidence.'
    inputs = {}

    def bound(relative):
        path = root / relative
        inputs[relative] = sha(path)
        return path

    def load(relative):
        return json.loads(bound(relative).read_text())

    trace = load('results/traces/trace_useful_observable.json')
    assert trace['case_id'] == CASE and trace['slot'] == SLOT
    reference = load(f'results/replay/profiles/reference/{SLOT}.json')
    context = load(f'results/replay/profiles/context/{SLOT}.json')
    reference_gap = load('results/replay/profiles/reference_gap.json')['threshold']
    context_gap = load('results/replay/profiles/context_gap.json')['threshold']
    for key in ['checkpoint_sha256', 'scaler_sha256', 'feature_sha256']:
        assert reference[key] == context[key]
    with np.load(bound('results/traces/trace_useful_inference.npz'),
                 allow_pickle=False) as arrays:
        signed = arrays['signed']
        raw = arrays['raw']
        candidates = arrays['candidate_trajectories']
        probabilities = arrays['candidate_scores']
        valid = arrays['candidate_valid']
        saved_center, saved_scale = arrays['center'], arrays['scale']
    record = trace['context']
    ref_event = admission(signed, record['history_class'], reference)
    ctx_event = score_context(signed, record, context)
    center, scale, flags = conditional_normalizer(record, context)
    np.testing.assert_allclose(center, saved_center, rtol=0, atol=1e-14)
    np.testing.assert_allclose(scale, saved_scale, rtol=0, atol=1e-14)
    for key in ['event_score', 'persistent_score', 'accumulated_score', 'limit']:
        assert np.isclose(ctx_event[key], trace['event'][key], rtol=1e-12, atol=1e-12)
    order = np.argsort(-probabilities, kind='stable')
    winner = int(order[0])
    gap = float(probabilities[order[0]] - probabilities[order[1]])
    assert winner == trace['selected_or_ranked'] == 6
    assert np.isclose(gap, trace['gap']) and reference_gap == context_gap
    assert valid.all() and trace['routes']['U'] and not trace['routes']['F']
    assert gap > reference_gap and ctx_event['numeric_alarm'] and not ref_event['numeric_alarm']
    assert ctx_event['limit'] > ref_event['limit']

    # These are the same recorded candidate/rank/trust/veto inputs in both policies.
    # Event admission is the boundary that changes for this particular pair.
    accepted = bool(valid.all() and gap > reference_gap and trace['routes']['U'])
    ref_proposed = bool(ref_event['numeric_alarm'] and accepted and not trace['routes']['F'])
    ctx_proposed = bool(ctx_event['numeric_alarm'] and accepted and not trace['routes']['F'])
    ref_norm = reference['groups'][record['history_class']]
    ref_standardized = (signed - np.asarray(ref_norm['center'])) / np.asarray(ref_norm['scale'])
    ctx_standardized = (signed - center[:, None, :]) / scale[:, None, :]
    states = trace['observable_component_states']['CuStepNo ValueY']
    changes = [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    assert len(changes) == 1 and changes[0] < 200
    shared = {
        'case_id': CASE, 'slot': SLOT, 'selected_channel_index': winner,
        'selected_channel': trace['channels'][winner],
        'history_class': record['history_class'], 'controller_change_offsets': changes,
        'current_controller_state': record['windows'][-1]['components']['CuStepNo ValueY']['current_state'],
        'candidate_gap': gap, 'required_gap': context_gap,
        'all_candidates_valid': True, 'context_valid': True, 'process_flag': False,
        'case_selection': trace['rule'], 'illustrative_only': True,
    }
    rows = []
    for label, event, proposed, z, mu, sigma in [
            ('Reference', ref_event, ref_proposed, ref_standardized,
             np.tile(ref_norm['center'], (5, 1)), np.tile(ref_norm['scale'], (5, 1))),
            ('Context', ctx_event, ctx_proposed, ctx_standardized, center, scale)]:
        rows.append({'policy': label, 'event_score': event['event_score'],
                     'event_limit': event['limit'], 'numeric_event': bool(event['numeric_alarm']),
                     'proposal': proposed, 'candidate_gap': gap, 'required_gap': context_gap,
                     'last_window_signed_mean': float(z[-1, :, winner].mean()),
                     'last_window_center': float(mu[-1, winner]),
                     'last_window_scale': float(sigma[-1, winner])})

    # Quality is evaluator-only. It is joined after the two decisions above.
    closed = pd.read_parquet(bound('results/data/refinement.parquet'),
                             filters=[('case_id', '=', CASE), ('seed', '=', 23)])
    assert not closed.duplicated(['variant', 'case_id', 'seed']).any()
    for row in rows:
        stored = closed.loc[closed.variant.eq(row['policy'].lower())].iloc[0]
        assert bool(stored.detected) == row['numeric_event']
        assert bool(stored.proposed) == row['proposal']
    quality = load('results/traces/trace_useful_evaluator.json')
    assert quality['case_id'] == CASE and quality['true_target_index'] == winner
    with np.load(bound('results/traces/trace_useful_evaluator.npz'),
                 allow_pickle=False) as arrays:
        clean = arrays['clean']
    training_range = trace['training_ranges'][winner]
    before = float(np.abs(raw[-60:, winner] - clean[-60:, winner]).mean() / training_range)
    after = float(np.abs(candidates[winner] - clean[-60:, winner]).mean() / training_range)
    assert np.isclose(before, quality['before_fraction'])
    assert np.isclose(after, quality['after_fraction']) and after < before
    rows[0]['post_policy_mae_pct_range'] = 100 * before
    rows[1]['post_policy_mae_pct_range'] = 100 * after
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'matched_case.csv', index=False)
    result = {'status': 'passed', 'shared_inference_inputs': shared, 'policies': rows,
              'evaluator_only_quality': {'before_fraction': before, 'candidate_after_fraction': after},
              'input_sha256': inputs, 'script_sha256': sha(Path(__file__)),
              'new_inference': False, 'new_fitting': False, 'new_threshold_selection': False,
              'decision_difference': 'event admission; same ranking, candidate evidence and required gap'}
    (out / 'matched_case.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [r'\begin{tabular}{lrr}', r'\hline', r'Quantity & \textit{Reference} & \textit{Context} \\', r'\hline',
             rf'Last-window signed mean $m$ & {rows[0]["last_window_signed_mean"]:.2f} & {rows[1]["last_window_signed_mean"]:.2f} \\',
             rf'Event score $S$ & {rows[0]["event_score"]:.2f} & {rows[1]["event_score"]:.2f} \\',
             rf'Calibrated event limit & {rows[0]["event_limit"]:.2f} & {rows[1]["event_limit"]:.2f} \\',
             rf'Candidate gap $\Delta$ & {gap:.2f} & {gap:.2f} \\',
             rf'Required gap $\tau$ & {context_gap:.2f} & {context_gap:.2f} \\',
             r'Event $A$ / proposal $P$ & 0 / 0 & 1 / 1 \\',
             r'\hline',
             rf'Post-policy MAE (\% range) & {100 * before:.2f} & {100 * after:.2f} \\',
             r'\hline', r'\end{tabular}']
    (out / 'matched_case.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'passed', 'case_id': CASE, 'reference_event': ref_event['event_score'],
                      'context_event': ctx_event['event_score'], 'out': str(out)}))


if __name__ == '__main__':
    main()
