"""Pure kernels from original/prominence_panel.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np
from .common import unique_top

def fit_scale(values):
    values = np.asarray(values, float)
    q = np.quantile(values, .95, axis=0, method='higher')
    med = np.median(values, axis=0)
    return {'q95': q.tolist(), 'scale': np.maximum.reduce([q-med, 1e-6*np.abs(q), np.full_like(q, 1e-12)]).tolist()}

def excess(values, profile):
    return np.maximum(0., (np.asarray(values)-profile['q95'])/profile['scale'])

def weighted_sensitivity(r, influence, residual_profile):
    """Per-output normalized absolute Jacobian; abnormal residuals supply weights."""
    influence = np.asarray(influence, float)
    p = influence/np.maximum(influence.sum(axis=-1, keepdims=True), 1e-30)
    return np.einsum('bi,bij->bj', excess(r, residual_profile), p)

def graph_evidence(peaks, graph):
    owners = sorted({graph['owners'][j] for j in peaks})
    if len(peaks) < 2:
        return 'single_peak'
    if len(owners) < 2:
        return 'unavailable_coarse_shared_owner'
    edges = {tuple(sorted(e)) for e in graph['system_connections']}
    reached = {owners[0]}
    while True:
        new = {b for a in reached for b in owners if tuple(sorted((a,b))) in edges}
        if new <= reached:
            break
        reached |= new
    return 'connected_declared_components' if len(reached)==len(owners) else 'disconnected_or_incomplete_graph'

def peak_route(scores, threshold, graph=None):
    """Five overlapping native windows; labels and target identity are excluded."""
    scores = np.asarray(scores, float)
    if scores.ndim!=2 or len(scores)!=5 or not np.isfinite(scores).all():
        raise ValueError('expected five finite score windows')
    peaks = scores > threshold
    active = peaks.any(axis=1)
    persistent = np.flatnonzero(peaks.sum(axis=0)>=4).tolist()
    result = {'alarm': bool(active.any()), 'alarm_windows': int(active.sum()),
              'persistent_peaks': persistent, 'window_peaks': [np.flatnonzero(p).tolist() for p in peaks],
              'route': 'accept', 'target_index': None, 'process_flag': False, 'graph_status': 'not_consulted'}
    if not active.any():
        return result
    result['route']='abstain'
    if not active.all() or not persistent:
        return result
    if len(persistent)==1:
        result.update(route='reconstruction', target_index=persistent[0])
    elif graph is None:
        result.update(route='diagnosis', process_flag=True)
    else:
        status=graph_evidence(persistent, graph);result['graph_status']=status
        if status=='connected_declared_components':
            result.update(route='diagnosis', process_flag=True)
    return result

def confirm_probe(result, fractions, valid, threshold):
    """Normal-calibrated, in-sample consistency veto; never fault-origin proof."""
    result=dict(result);result['pre_probe_route']=result['route'];result['probe_sensor_flag']=False
    winner=unique_top(fractions, valid)
    confirmed=(bool(np.all(valid)) and winner is not None and fractions[winner]>threshold
               and winner in result['persistent_peaks'] and result['alarm_windows']==5)
    result.update(probe_winner=winner, probe_confirmed=bool(confirmed))
    if confirmed:
        result['probe_sensor_flag']=True
        if not result['process_flag']:
            result.update(route='reconstruction', target_index=int(winner))
    elif result['route']=='reconstruction':
        result.update(route='abstain', target_index=None)
    return result
