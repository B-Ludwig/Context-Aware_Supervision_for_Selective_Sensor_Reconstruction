"""Pure kernels from original/attribution_probe.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
import numpy as np

def unique_top(values,valid=None):
    values=np.asarray(values,float);valid=np.ones(len(values),bool) if valid is None else np.asarray(valid,bool)
    indices=np.flatnonzero(valid & np.isfinite(values))
    if not len(indices):return None
    order=indices[np.argsort(-values[indices],kind='stable')]
    if len(order)>1 and np.isclose(values[order[0]],values[order[1]],rtol=1e-6,atol=1e-12):return None
    return int(order[0])

def cross_gains(before,after):
    before=np.asarray(before);after=np.asarray(after);f=len(before)
    keep=~np.eye(f,dtype=bool)
    b=np.sum(np.where(keep,before[None],0.),axis=1)
    a=np.sum(np.where(keep,after,0.),axis=1)
    return (b-a)/(f-1),(b-a)/np.maximum(b,1e-12)
