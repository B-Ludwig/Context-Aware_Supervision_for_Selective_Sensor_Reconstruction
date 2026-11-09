"""Pure kernels from refinement/selection_refinement.py in the retained SafeProcess implementation.

Campaign orchestration and storage dependencies are intentionally excluded.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np

ORIGINAL_FEATURE_NAMES = ('log_normal_distance','relative_signature_fit','signature_improvement',
 'signature_margin','own_residual_share','cross_gain','log_repaired_residual_ratio',
 'log_change_to_error_ratio','log_observed_variation_ratio','residual_rank',
 'signature_rank','context_supported')

FEATURE_NAMES = ORIGINAL_FEATURE_NAMES + ('original_candidate_score','original_score_gap',
                                         'normalized_masked_target_correction')

PARAMETERS = dict(max_iter=100,max_leaf_nodes=7,max_depth=3,min_samples_leaf=20,
 learning_rate=.1,l2_regularization=1.,random_state=31017,early_stopping=False)

THRESHOLD_GRID = tuple(float(x) for x in np.linspace(0.,1.,101))

BAD_CONTEXT = frozenset(('missing','misaligned','untrusted','unavailable','invalid'))

PRIMARY = frozenset(('drift','stuck','noisy_stuck','quantized_stuck'))

def feature_manifest():
    return {'schema':'absolute-candidate-selection-v1','features':list(FEATURE_NAMES),
     'parameters':PARAMETERS,'pooled_across_configurations':True,
     'original_score':'Frozen original candidate probability; no reranking input from new context.',
     'gap':'p_j - max(p_k for k != j), all original candidates; sole candidate comparison baseline zero.',
     'correction':'mean(abs(corrected_target[j,:]-observed[-60:,j])) / max(training_range[j],1e-12); no log/percent conversion.',
     'weights':'Equal config, then family within config, then source period within config/family, then candidate rows within period; normalized mean one.',
     'fit_roles':['decision_fit','selection_normal_fit'],'threshold_role':'risk_calibration',
     'threshold_grid':list(THRESHOLD_GRID),'threshold_scope':'configuration and event variant separately; same grid/constraints everywhere',
     'constraints':{'max_incorrect_or_nonbeneficial':.05,'max_normal_referral':.05},
     'threshold_objective':'Most correct-beneficial, then fewest incorrect/nonbeneficial, then larger threshold; no eligible -> threshold1.',
     'comparison':'Strict score > threshold. Exactly one valid candidate qualifies; multiple -> ambiguous review; zero -> abstain.',
     'validity':'Candidate-level validity; invalid candidates excluded. Process veto and invalid context always prevent proposals.',
     'negatives':'Unchanged normal and eligible non-single-sensor development controls; all incorrect/nonbeneficial valid candidates negative.',
     'assessment_truth_input':False,'new_assessment_TTO':False}

def build_candidate_features(original_features, original_scores, correction, reference_scale,
                             candidate_valid=None):
    """Correction is an observable per-candidate masked-target MAE, not recovery quality."""
    x=np.asarray(original_features,float); p=np.asarray(original_scores,float)
    c=np.asarray(correction,float); scale=np.asarray(reference_scale,float)
    if x.ndim!=2 or x.shape[1]!=12 or p.shape!=(len(x),) or c.shape!=p.shape or scale.shape!=p.shape:
        raise ValueError('Expected F x12 original features and F aligned score/correction/range arrays')
    if not len(x) or not all(np.isfinite(a).all() for a in (x,p,c,scale)):
        raise ValueError('Candidate observables must be finite and nonempty')
    if np.any((p<0)|(p>1)) or np.any(c<0) or np.any(scale<0):
        raise ValueError('Invalid original score, correction magnitude, or training range')
    gap=np.array([p[j]-(np.delete(p,j).max() if len(p)>1 else 0.) for j in range(len(p))])
    result=np.column_stack((x,p,gap,c/np.maximum(scale,1e-12)))
    if candidate_valid is not None and np.asarray(candidate_valid,bool).shape!=p.shape:
        raise ValueError('Candidate validity order/shape differs')
    return result

def masked_correction(observed_window, corrected_targets):
    raw=np.asarray(observed_window,float); corrected=np.asarray(corrected_targets,float)
    if raw.ndim!=2 or corrected.shape!=(raw.shape[1],raw.shape[0]) or raw.shape[0]!=60:
        raise ValueError('Expected observed60 xF and corrected F x60')
    if not np.isfinite(raw).all() or not np.isfinite(corrected).all():
        raise ValueError('Nonfinite masked correction')
    return np.mean(np.abs(corrected-raw.T),axis=1)

def development_labels(*, role, family, target_index, observed_window, corrected_targets,
                        clean_window, valid, control_eligible=True):
    """Controller-only truth boundary. Never call while predicting assessment decisions."""
    if role not in ('decision_fit','risk_calibration','selection_normal_fit'):
        raise ValueError('Forbidden label role: '+str(role))
    if role=='selection_normal_fit' and (family!='unchanged' or target_index is not None):
        raise ValueError('Extra normal fitting admits only unchanged negative controls')
    raw=np.asarray(observed_window,float); corrected=np.asarray(corrected_targets,float)
    clean=np.asarray(clean_window,float); valid=np.asarray(valid,bool)
    if raw.ndim!=2 or raw.shape!=clean.shape or corrected.shape!=(raw.shape[1],raw.shape[0]) or valid.shape!=(raw.shape[1],):
        raise ValueError('Development clean/candidate shape mismatch')
    quality=np.isfinite(raw).all(axis=0)&np.isfinite(clean).all(axis=0)&np.isfinite(corrected).all(axis=1)
    benefit=np.mean(np.abs(raw-clean),axis=0)-np.mean(np.abs(corrected-clean.T),axis=1)
    usable=valid&quality
    if family not in PRIMARY and family!='unchanged' and not control_eligible:
        usable[:]=False
    positive=np.zeros(len(valid),bool)
    if family in PRIMARY:
        if target_index is None or not 0<=int(target_index)<len(valid):
            raise ValueError('Development single-sensor target absent/out of bounds')
        j=int(target_index);positive[j]=bool(usable[j] and benefit[j]>0)
    return positive.astype(np.uint8),usable,benefit

def hierarchical_weights(configurations, families, source_periods):
    keys=list(zip(map(str,configurations),map(str,families),map(str,source_periods)))
    if not keys:raise ValueError('No fitting candidates')
    if len(keys)!=len(configurations) or len(keys)!=len(families) or len(keys)!=len(source_periods):
        raise ValueError('Weight metadata lengths differ')
    counts=Counter(keys);fam=defaultdict(set);period=defaultdict(set)
    for u,f,g in keys:fam[u].add(f);period[(u,f)].add(g)
    w=np.array([1/(len(fam)*len(fam[u])*len(period[(u,f)])*counts[(u,f,g)]) for u,f,g in keys])
    return w/w.mean()

def export_model(estimator):
    trees=[]
    for stage in estimator._predictors:
        if len(stage)!=1:raise ValueError('Binary classifier required')
        nodes=stage[0].nodes
        if np.any(nodes['is_categorical']):raise ValueError('Only fixed numerical features allowed')
        trees.append({k:nodes[k].tolist() for k in ('value','feature_idx','num_threshold','left','right','is_leaf')})
    return {'kind':'hist_gradient_boosting','mean':[0.]*15,'scale':[1.]*15,
     'baseline':float(estimator._baseline_prediction[0,0]),'trees':trees,
     'parameters':PARAMETERS,'features':list(FEATURE_NAMES),'feature_manifest':feature_manifest()}

def fit_selector(features, labels, configurations, families, source_periods, roles):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x=np.asarray(features,float);y=np.asarray(labels)
    if x.ndim!=2 or x.shape!=(len(y),15) or not np.isfinite(x).all():
        raise ValueError('Finite N x15 fit features required')
    if len(roles)!=len(y) or any(r not in ('decision_fit','selection_normal_fit') for r in roles):
        raise ValueError('Only decision_fit plus selected normal fit records may train selector')
    if any(len(v)!=len(y) for v in (configurations,families,source_periods)) or set(np.unique(y))!={0,1}:
        raise ValueError('Two binary classes and aligned weight metadata required')
    if any(r=='selection_normal_fit' and int(v)!=0 for r,v in zip(roles,y)):
        raise ValueError('Extra normal fitting candidates must all be negative')
    weights=hierarchical_weights(configurations,families,source_periods)
    estimator=HistGradientBoostingClassifier(**PARAMETERS).fit(x,y,sample_weight=weights)
    model=export_model(estimator);portable=predict_scores(x,model)
    error=float(np.max(np.abs(portable-estimator.predict_proba(x)[:,1])))
    if error>2e-14:raise AssertionError('Portable predictor mismatch: '+str(error))
    model['fit_receipt']={'candidate_n':len(y),'positive_n':int(y.sum()),
     'configurations':sorted(set(map(str,configurations))),'roles':dict(Counter(roles)),
     'weight_sum':float(weights.sum()),'portable_max_abs_error':error,
     'feature_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
     'label_sha256':hashlib.sha256(np.ascontiguousarray(y.astype(np.uint8)).tobytes()).hexdigest(),
     'weight_sha256':hashlib.sha256(weights.tobytes()).hexdigest()}
    return model

def predict_scores(features, model):
    x=np.asarray(features,float)
    if x.ndim!=2 or x.shape[1]!=15 or not np.isfinite(x).all():raise ValueError('Finite N x15 observables required')
    if model.get('kind')!='hist_gradient_boosting' or model.get('features')!=list(FEATURE_NAMES):
        raise ValueError('Wrong frozen selector schema')
    logits=np.full(len(x),model['baseline'],float)
    for tree in model['trees']:
        a={k:np.asarray(v) for k,v in tree.items()};node=np.zeros(len(x),int)
        for _ in range(5):
            active=~a['is_leaf'][node].astype(bool)
            if not active.any():break
            rows=np.flatnonzero(active);n=node[rows]
            left=x[rows,a['feature_idx'][n].astype(int)]<=a['num_threshold'][n]
            node[rows]=np.where(left,a['left'][n],a['right'][n]).astype(int)
        if not np.all(a['is_leaf'][node]):raise ValueError('Tree exceeds fixed depth')
        logits+=a['value'][node]
    return 1/(1+np.exp(-np.clip(logits,-700,700)))

def predict_absolute(scores, valid, threshold, event_alarm, process_veto=False,
                     context_status='available', original_alarm=False):
    p=np.asarray(scores,float);v=np.asarray(valid,bool)
    if p.ndim!=1 or p.shape!=v.shape or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Finite aligned candidate probabilities/validity required')
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Threshold outside[0,1]')
    qualifying=np.flatnonzero(v&(p>threshold)); count=len(qualifying)
    candidate=int(qualifying[0]) if count==1 else None
    invalid_context=context_status!='available'
    concern=bool(event_alarm or process_veto or invalid_context)
    proposed=bool(event_alarm and count==1 and not process_veto and not invalid_context)
    ranked=int(np.flatnonzero(v)[np.argmax(p[v])]) if v.any() else None
    sorted_scores=np.sort(p[v]);margin=float(sorted_scores[-1]-(sorted_scores[-2] if len(sorted_scores)>1 else 0.)) if v.any() else 0.
    reasons=[]
    if event_alarm:reasons.append('numeric_event')
    if process_veto:reasons.append('saved_process_concern')
    if invalid_context:reasons.append('invalid_or_untrusted_context')
    if concern and count>1:reasons.append('ambiguous_multiple_absolute_candidates')
    if concern and count==0:reasons.append('absolute_abstention')
    if concern and not v.all():reasons.append('invalid_candidates_excluded')
    return {'numeric_alarm':bool(event_alarm),'original_alarm':bool(original_alarm),
     'concern':concern,'review':concern,'proposed':proposed,'process_flag':bool(process_veto),
     'context_status':context_status,'invalid_context':invalid_context,'ranked_target':ranked,
     'target_index_recommended':candidate if proposed else None,'absolute_candidate':candidate,
     'qualifying_candidates':qualifying.tolist(),'qualifying_count':count,'ambiguous':count>1,
     'abstained':count==0,'confidence_threshold':float(threshold),'margin':margin,
     'low_confidence':count==0,'valid_all':bool(v.all()),'invalid_candidate':not bool(v.all()),
     'context_only_review':bool(concern and not event_alarm and not process_veto),
     'non_numeric_review':bool(concern and not event_alarm),'review_reasons':reasons,
     'route':'reconstruction_assessment' if proposed else 'user_review' if concern else 'accept',
     'replacement_authorized':False,'operator_confirmation_required':proposed,'applied_action':None}

def choose_absolute_threshold(risk_cases, *, max_wrong=.05, max_normal=.05):
    """One config/event variant at a time; only original risk cases permitted."""
    cases=list(risk_cases)
    if not cases:raise ValueError('Risk cases required, even if sparse')
    if any(c.get('role')!='risk_calibration' for c in cases):raise ValueError('Threshold selection is risk-only')
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate risk case')
    configs={c.get('unit') for c in cases}
    if len(configs)>1:raise ValueError('Select one configuration/event threshold at a time')
    normal_n=sum(bool(c['normal']) for c in cases);grid=[]
    for threshold in THRESHOLD_GRID:
        selected=[];ambiguous=0
        for c in cases:
            p=np.asarray(c['scores'],float);valid=np.asarray(c['valid'],bool);y=np.asarray(c['positive'],bool)
            if p.ndim!=1 or p.shape!=valid.shape or y.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                raise ValueError('Invalid risk candidate arrays')
            q=np.flatnonzero(valid&(p>threshold))
            if bool(c['event_alarm']) and len(q)>1:ambiguous+=1
            # Like frozen threshold fitting, guards are applied after numeric/confidence fitting.
            if bool(c['event_alarm']) and len(q)==1:selected.append((c,int(q[0])))
        good=sum(bool(c['positive'][j]) for c,j in selected);wrong=len(selected)-good
        normal=sum(bool(c['normal']) for c,j in selected)
        acceptable=bool(selected and wrong/len(selected)<=max_wrong and (normal_n==0 or normal/normal_n<=max_normal))
        grid.append({'threshold':threshold,'selected':len(selected),'good':good,
          'wrong_or_not_beneficial':wrong,'normal_referrals':normal,'normal_n':normal_n,
          'ambiguous_on_event':ambiguous,'acceptable':acceptable})
    eligible=[r for r in grid if r['acceptable']]
    chosen=min(eligible,key=lambda c:(-c['good'],c['wrong_or_not_beneficial'],-c['threshold'])) if eligible else None
    return {'threshold':chosen['threshold'] if chosen else 1.,'eligible':chosen is not None,
     'grid':grid,'risk_n':len(cases),'normal_n':normal_n,'selected':chosen,
     'process_and_context_guards_during_threshold_fit':False,
     'qualification':'Empirical seed23 risk-only constraint; small support is a limitation, not a stop.'}
