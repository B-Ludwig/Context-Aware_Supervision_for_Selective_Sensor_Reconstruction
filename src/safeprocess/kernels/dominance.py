"""Pure kernels from original/residual_drift.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np
from collections import Counter

SETTINGS={'dominance':.7,'material_share':.2,'minimum_standardized_excess':1.,'top_second_ratio':3.,
          'minimum_event_windows':5,'branch_support':.8,'target_support':.8,'allowed_gap':0}

def residual_route(residuals,active,*,variant,q99,scale,settings=SETTINGS):
    """Only residuals, alarm flags and normal calibration enter this policy.

    Sensor index is output identity, not an input-specific exception. No fault
    label, target truth, magnitude, dataset name, mode or ontology is accepted.
    """
    r=np.asarray(residuals,dtype=float);active=np.asarray(active,dtype=bool)
    if r.ndim!=2 or len(active)!=len(r) or r.shape[1]<2 or not np.isfinite(r).all() or (r<0).any():raise ValueError('invalid residual matrix')
    if variant=='raw_residual':evidence=r
    elif variant=='normal_calibrated_residual':
        q99=np.asarray(q99);scale=np.asarray(scale)
        if q99.shape!=(r.shape[1],) or scale.shape!=q99.shape or not np.isfinite(q99).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('invalid normal residual profile')
        evidence=np.maximum(0,(r-q99)/scale)
    else:raise ValueError('unknown variant')
    windows=[]
    for i,(values,alarm) in enumerate(zip(evidence,active)):
        order=np.argsort(-values,kind='stable');top,second=order[:2];total=float(values.sum());share=float(values[top]/total) if total else 0.
        ties=bool(values[top]==values[second]);material=int((values/total>=settings['material_share']).sum()) if variant=='raw_residual' and total else int((values>=settings['minimum_standardized_excess']).sum()) if variant!='raw_residual' else 0
        localized=(material==1 and share>=settings['dominance']) if variant=='raw_residual' else (material>=1 and share>=settings['dominance'] and (values[second]==0 or values[top]>=settings['top_second_ratio']*values[second]))
        if not alarm:route='accept';target=None;reason='no_trained_detector_alarm'
        elif localized and not ties:route='reconstruction';target=int(top);reason='localized_residual_evidence'
        elif material>=2:route='diagnosis';target=None;reason='distributed_residual_evidence'
        else:route='abstain';target=None;reason='residual_evidence_not_decisive'
        windows.append({'window_index':i,'route':route,'target_index':target,'top_share':share,'material_channels':material,'reason':reason})
    events=[];i=0
    while i<len(active):
        if not active[i]:i+=1;continue
        end=i+1
        while end<len(active) and active[end]:end+=1
        ws=windows[i:end];counts=Counter(w['route'] for w in ws);route='abstain';target=None
        if len(ws)>=settings['minimum_event_windows']:
            for branch in ('reconstruction','diagnosis'):
                if counts[branch]/len(ws)<settings['branch_support']:continue
                if branch=='reconstruction':
                    targets=Counter(w['target_index'] for w in ws if w['route']==branch)
                    candidate,n=targets.most_common(1)[0]
                    if n/counts[branch]<settings['target_support']:continue
                    target=candidate
                route=branch
        events.append({'first_window':i,'last_window':end-1,'windows':len(ws),'route':route,'target_index':target});i=end
    actions={(e['route'],e['target_index']) for e in events if e['route'] in ('reconstruction','diagnosis')}
    if not events:route,target='accept',None
    elif len(actions)==1:route,target=next(iter(actions))
    else:route,target='abstain',None
    return {'route':route,'target_index':target,'windows':windows,'events':events,'conflicting_actionable_events':len(actions)>1,
            'reconstruction_executed':False,'diagnosis_executed':False,'replacement_authorized':False}
