"""Pure kernels from history/history_core.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np

HISTORY_ROWS=300

REFERENCE_SHRINKAGE=8.

CLASS_CALIBRATION_MIN=4

SCALE_FLOOR=1e-6

def observed_labels(frame,unit):
    if unit.startswith('hai_'):
        groups={}
        for col in frame:
            if '.state_' in col and col.endswith('.active'):groups.setdefault(col.split('.state_',1)[0],[]).append(col)
        if not groups:return [None]*len(frame)
        parts=[]
        for component,cols in sorted(groups.items()):
            values=frame[cols].to_numpy(float);valid=np.isin(values,[0,1]).all(axis=1)&(values.sum(axis=1)==1)
            names=[c.split('.state_',1)[1].removesuffix('.active') for c in cols]
            parts.append([names[int(j)] if ok else None for j,ok in zip(values.argmax(axis=1),valid)])
        return [None if None in x else '|'.join(x) for x in zip(*parts)]
    if unit=='four_tank':
        values=frame['CuStepNo ValueY'].to_numpy(float)
        return [str(int(v)) if np.isfinite(v) and v==int(v) else None for v in values]
    if unit=='swat':
        values=frame[['MV301','MV302','P301','P302','P401','P402']].to_numpy(float)
        return ['|'.join(str(int(v)) for v in row) if np.isfinite(row).all() and np.equal(row,np.floor(row)).all() else None for row in values]
    raise ValueError('Unknown original context adapter: '+unit)

def history(labels):
    if len(labels)!=300 or any(x is None for x in labels):return {'history_class':'unavailable','change_count':None,'age_rows':None}
    changes=[i for i in range(1,300) if labels[i]!=labels[i-1]]
    return {'history_class':'transition_settling' if changes else 'steady','change_count':len(changes),'age_rows':299-changes[-1] if changes else 300}

def fit_normalizer(signed):
    a=np.asarray(signed,float)
    if a.ndim!=4 or a.shape[1:3]!=(5,60) or not len(a) or not np.isfinite(a).all():raise ValueError('Finite nonempty reference contexts x5x60xchannels required')
    center=np.median(a.reshape(-1,a.shape[-1]),axis=0)
    scale=np.maximum(np.sqrt(np.mean((a-center)**2,axis=(0,1,2))),SCALE_FLOOR)
    return {'center':center.tolist(),'scale':scale.tolist(),'n_reference':len(a)}

def fit_reference(signed,classes):
    a=np.asarray(signed,float);classes=np.asarray(classes)
    if len(a)!=len(classes):raise ValueError('Reference history alignment differs')
    good=classes!='unavailable'
    pooled=fit_normalizer(a[good]);result={'pooled':pooled,'groups':{}}
    for cls in ['steady','transition_settling']:
        selected=a[classes==cls];n=len(selected);w=n/(n+REFERENCE_SHRINKAGE)
        base=fit_normalizer(selected) if n else pooled
        result['groups'][cls]={'n_reference':n,'weight':w,'center':(w*np.array(base['center'])+(1-w)*np.array(pooled['center'])).tolist(),'scale':np.maximum(np.sqrt(w*np.array(base['scale'])**2+(1-w)*np.array(pooled['scale'])**2),SCALE_FLOOR).tolist()}
    return result

def event_score(signed,normalizer):
    a=np.asarray(signed,float)
    if a.ndim!=3 or a.shape[:2]!=(5,60) or not np.isfinite(a).all():raise ValueError('Five finite60-row signed windows required')
    z=(a-np.array(normalizer['center']))/np.array(normalizer['scale'])
    persistent=float(np.sort((z*z).mean(axis=1),axis=0)[-3].max())
    accumulated=float(np.max(np.abs(np.cumsum(z.mean(axis=1),axis=0))/np.sqrt(np.arange(1,6))[:,None]))**2
    return {'event_score':max(persistent,accumulated),'persistent_score':persistent,'accumulated_score':accumulated}

def empirical_limit(scores):
    a=np.asarray(scores,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Missing usable finite normal calibration')
    return float(np.quantile(a,.95,method='higher'))

def fit_profile(reference,reference_classes,calibration,calibration_classes):
    result=fit_reference(reference,reference_classes);scores=[];validclasses=[]
    for signed,cls in zip(calibration,calibration_classes):
        if cls=='unavailable':continue
        scores.append(event_score(signed,result['groups'].get(cls,result['pooled']))['event_score']);validclasses.append(cls)
    result['calibration']={'pooled':{'limit':empirical_limit(scores),'n_calibration':len(scores)},'groups':{}}
    for cls in ['steady','transition_settling']:
        a=[v for v,c in zip(scores,validclasses) if c==cls]
        result['calibration']['groups'][cls]={'n_calibration':len(a),'limit':empirical_limit(a) if len(a)>=CLASS_CALIBRATION_MIN else None,'uses_class_limit':len(a)>=CLASS_CALIBRATION_MIN}
    result['calibration_scores']=scores;result['calibration_classes']=validclasses
    return result

def admission(signed,cls,profile):
    norm=profile['groups'].get(cls,profile['pooled']);cell=profile['calibration']['groups'].get(cls,{'n_calibration':0,'uses_class_limit':False})
    fallback=not cell['uses_class_limit'];cal=profile['calibration']['pooled'] if fallback else cell
    s=event_score(signed,norm)
    nref=profile['groups'].get(cls,{}).get('n_reference',0)
    return {**s,'numeric_alarm':bool(s['event_score']>cal['limit']),'limit':cal['limit'],'calibration_n':cal['n_calibration'],'class_calibration_n':cell['n_calibration'],'reference_class_n':nref,'reference_weight':profile['groups'].get(cls,{}).get('weight',0.),'calibration_fallback':fallback,'reference_fallback':nref==0,'low_support':bool(cal['n_calibration']<20 or cell['n_calibration']<4 or nref<8),'invalid_context':cls=='unavailable'}
