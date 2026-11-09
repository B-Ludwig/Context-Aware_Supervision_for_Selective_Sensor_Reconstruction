#!/usr/bin/env python3
"""Replay 31 fixed observable fixtures with the standalone policy kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from safeprocess.context import score_context
from safeprocess.policies import relative_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPOSITORY)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if root == out or root in out.parents:
        parser.error('Write replay outputs outside the code/results repository.')
    fixtures_path = root / 'results/replay/fixtures/scoring.json'
    fixtures = json.loads(fixtures_path.read_text())
    checked = []
    with np.load(root / 'results/replay/fixtures/signed_residuals.npz',
                 allow_pickle=False) as arrays:
        for row in fixtures:
            profile_path = root / f'results/replay/profiles/context/{row["slot"]}.json'
            profile = json.loads(profile_path.read_text())
            event = score_context(arrays[row['id']], row['context_record'], profile)
            invalid = bool(event['invalid_context'] or row['context_status'] != 'available')
            prediction = relative_decision(
                row['candidate_probabilities'], row['candidate_valid'],
                event_alarm=event['numeric_alarm'], threshold=row['confidence_threshold'],
                context_valid=not invalid, process_flag=row['process_flag'])
            values = {
                'event_score': event['event_score'],
                'numeric_alarm': event['numeric_alarm'],
                'concern': bool(event['numeric_alarm'] or row['process_flag'] or invalid
                                or row['original_ae_alarm']),
                'proposed': prediction['proposed'],
                'operator_preview': prediction['preview'] is not None,
                'target_index_recommended': prediction['target_index_recommended'],
            }
            for key, expected in row['expected'].items():
                matches = (np.isclose(values[key], expected, rtol=1e-12, atol=1e-9)
                           if key == 'event_score' else values[key] == expected)
                if not matches:
                    raise AssertionError((row['id'], key, values[key], expected))
            checked.append({'id': row['id'], 'slot': row['slot'],
                            'event_score': float(event['event_score'])})
    assert len(checked) == 31
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        'status': 'passed',
        'scope': '31 observable residual/context/candidate fixtures replayed by standalone kernels',
        'fixtures': checked,
        'source_fixture_sha256': hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
        'inference': False, 'fitting': False,
    }
    (out / 'OBSERVABLE_REPLAY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'fixtures': len(checked), 'out': str(out)}))


if __name__ == '__main__':
    main()
