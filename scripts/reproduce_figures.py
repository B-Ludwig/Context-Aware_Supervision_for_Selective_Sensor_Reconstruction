#!/usr/bin/env python3
"""Regenerate the published figures from closed tables and illustrative traces.

This command performs no model execution, training or profile fitting. The
optional --tex flag uses TeX typography; the default needs only Matplotlib.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from figures import decision_layer, matched_traces, severity, tradeoff, window_traces
from figures.typography import set_use_tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tex', action='store_true', help='Use TeX-rendered Computer Modern labels')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out == root or root in out.parents:
        parser.error('Write figures outside the supplied repository.')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'mplconfig'))
    set_use_tex(args.tex)
    data = root / 'results/data'
    traces = []
    for kind in ['useful', 'adverse']:
        folder = root / 'results/traces'
        record = json.loads((folder / f'trace_{kind}_observable.json').read_text())
        evaluator = json.loads((folder / f'trace_{kind}_evaluator.json').read_text())
        assert evaluator['quality_available'], kind
        with np.load(folder / f'trace_{kind}_inference.npz', allow_pickle=False) as values:
            record.update(raw=values['raw'], targets=values['candidate_trajectories'],
                          energy=values['window_energy'], cum=values['window_accumulated'])
        with np.load(folder / f'trace_{kind}_evaluator.npz', allow_pickle=False) as values:
            record['clean'] = values['clean']
        traces.append(record)
    subprocess.run([sys.executable, '-B', str(root / 'scripts/matched_case.py'),
                    '--root', str(root), '--out', str(out / 'matched-case')], check=True)
    matched = json.loads((out / 'matched-case/matched_case.json').read_text())
    assert matched['status'] == 'passed'
    tradeoff.plot(pd.read_csv(data / 'margin_pooled.csv'),
                  pd.read_csv(data / 'fixed_policies.csv'),
                  pd.read_csv(data / 'control_policies.csv'), out / 'tradeoff.pdf')
    window_traces.plot(traces, out / 'window_detail.pdf')
    matched_traces.plot(traces, matched, out / 'matched_trajectories.pdf')
    compact = pd.read_parquet(data / 'refinement.parquet',
                             columns=['variant', 'family', 'unit', 'seed', 'magnitude',
                                      'n', 'detected', 'beneficial'])
    severity.plot(compact, pd.read_csv(data / 'severity_reference.csv'), out,
                  out / 'severity.pdf')
    pd.testing.assert_frame_equal(pd.read_csv(out / 'severity_checkpoint_curves.csv'),
                                  pd.read_csv(data / 'severity_checkpoint_curves.csv'),
                                  check_exact=True)
    decision_layer.render(out / 'decision_layer.pdf')
    receipt = {
        'status': 'passed', 'inference': False, 'fitting': False,
        'font': 'TeX-rendered Computer Modern' if args.tex else 'Matplotlib serif with Computer Modern mathematics',
        'figures': ['decision_layer', 'tradeoff', 'matched_trajectories', 'window_detail', 'severity'],
        'severity_checkpoint_curves_exact': True,
        'matched_score_to_limit_ratios': {
            row['policy']: row['event_score'] / row['event_limit'] for row in matched['policies']},
        'trace_selection': 'Deterministic retained illustrative cases; see results/traces and docs/RESULTS.md',
    }
    (out / 'FIGURE_REPRODUCTION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
