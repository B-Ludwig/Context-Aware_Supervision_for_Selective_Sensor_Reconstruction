"""Render closed numerical evidence without fitting or model execution."""
from pathlib import Path
import numpy as np
import pandas as pd
from .typography import configure_plot_fonts, method_label, plain_text, save_figure

def plot(traces, matched, path):
    """Use closed traces and a verified matched pair, without model execution."""
    configure_plot_fonts(8)
    import matplotlib.pyplot as plt

    path = Path(path)
    assert len(traces) == 2
    assert matched['status'] == 'passed'
    assert traces[0]['case_id'] == matched['shared_inference_inputs']['case_id']
    assert [row['policy'] for row in matched['policies']] == ['Reference', 'Context']
    assert not matched['policies'][0]['proposal'] and matched['policies'][1]['proposal']
    for row in matched['policies']:
        assert np.isfinite(row['event_score'])
        assert np.isfinite(row['event_limit']) and row['event_limit'] > 0

    fig, axes = plt.subplots(2, 2, figsize=(7, 3.10))
    for i, record in enumerate(traces):
        ax = axes[0, i]
        j = record['selected_or_ranked']
        scale = record['training_ranges'][j]
        rows = np.arange(300) + record['source']['start']
        ax.plot(rows[200:], record['raw'][200:, j]/scale, color='#9f6038', lw=1.1, label='Observed')
        ax.plot(rows[-60:], record['targets'][j]/scale, color='#087f83', lw=1.2, label='Candidate')
        ax.plot(rows[200:], record['clean'][200:, j]/scale, color='#26364d', lw=1, ls='--', label='Clean (evaluator)')
        ax.set_title(['Useful proposal', 'Harmful wrong-target proposal'][i], fontsize=9)
        ax.set_xlabel('Source row')
        ax.set_ylabel('Value / training range')
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    ax = axes[1, 0]
    colors = ['#537eaa', '#087f83']
    ratios = []
    for row, y, color in zip(matched['policies'], [1, 0], colors):
        ratio = row['event_score'] / row['event_limit']
        ratios.append(ratio)
        ax.barh(y, ratio, color=color, alpha=.85, height=.42)
        ax.text(ratio, y + .27, f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
    ax.axvline(1, color='#a64e3a', ls='--', lw=1)
    ax.set_yticks([1, 0], [method_label('Reference'), method_label('Context')])
    ax.set_xlim(0, 2.1)
    ax.set_ylim(-.65, 1.65)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel('Event score / own calibrated limit')
    ax.set_title('Same candidate, different event admission', fontsize=8)
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax = axes[1, 1]
    record = traces[1]
    x = np.arange(1, 6)
    ax.plot(x, record['energy'].max(1), 'o-', ms=3, color='#537eaa', label='Max window energy')
    ax.plot(x, record['cum'].max(1), 's-', ms=3, color='#087f83', label='Max signed accumulation')
    ax.axhline(record['event']['limit'], color='#a64e3a', ls='--', lw=1, label='Event limit')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xlabel('Overlapping window / prefix')
    ax.set_ylabel('Context score')
    ax.set_title('Context residual evidence', fontsize=8)
    ax.grid(alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.52, 1.03))
    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.52, -.02))
    fig.tight_layout(rect=[0, .06, 1, .96], h_pad=1.5, w_pad=1.2)
    save_figure(fig, path)
    plt.close(fig)
    return fig
