"""Pure kernels from original/final_calibrator.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np

_CACHE={}

def export_tree_model(estimator,mean,scale):
    trees=[]
    for stage in estimator._predictors:
        if len(stage)!=1:raise ValueError('binary decision calibration required')
        nodes=stage[0].nodes
        if np.any(nodes['is_categorical']):raise ValueError('only fixed numerical features are allowed')
        trees.append({k:nodes[k].tolist() for k in ('value','feature_idx','num_threshold','left','right','is_leaf')})
    return {'kind':'hist_gradient_boosting','mean':np.asarray(mean).tolist(),'scale':np.asarray(scale).tolist(),
            'baseline':float(estimator._baseline_prediction[0,0]),'trees':trees,
            'parameters':estimator.get_params(),'qualification':'Fixed small nonlinear decision calibrator; AE architecture/checkpoints unchanged.'}

def predict(features,model):
    x=(np.asarray(features,float)-np.asarray(model['mean']))/np.asarray(model['scale'])
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite candidate features required')
    if model.get('kind','linear')=='linear':
        logits=x@np.asarray(model['coefficient'])+model['intercept']
    elif model['kind']=='hist_gradient_boosting':
        key=id(model)
        if key not in _CACHE:
            if len(_CACHE) >= 32:
                _CACHE.pop(next(iter(_CACHE)))
            # Retain the source object too, so Python cannot reuse its identity.
            _CACHE[key]=(model,[{k:np.asarray(v) for k,v in tree.items()} for tree in model['trees']])
        logits=np.full(len(x),model['baseline'],float)
        for tree in _CACHE[key][1]:
            node=np.zeros(len(x),dtype=int)
            for _ in range(32):
                active=~tree['is_leaf'][node].astype(bool)
                if not active.any():break
                rows=np.flatnonzero(active);n=node[rows]
                left=x[rows,tree['feature_idx'][n].astype(int)]<=tree['num_threshold'][n]
                node[rows]=np.where(left,tree['left'][n],tree['right'][n]).astype(int)
            else:raise ValueError('tree traversal exceeded declared depth')
            logits+=tree['value'][node]
    else:raise ValueError('unknown frozen decision calibrator')
    return 1/(1+np.exp(-np.clip(logits,-40,40)))
