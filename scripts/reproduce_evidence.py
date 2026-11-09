"""Deterministic report-only aggregation of the supplied SafeProcess outcomes.
No campaign, fitter, model, policy, torch or sklearn modules are imported.
Dependencies: Python>=3.10, numpy, pandas, pyarrow. --root is the repository
containing results/. Native graph status is a closed text payload.
All table benefits are fractions of training range unless named *_pct_range.
"""
from pathlib import Path
import argparse, json, platform, sys
import numpy as np
import pandas as pd

PRIMARY=['drift','stuck','noisy_stuck','quantized_stuck']
LEVELS=['intercept','sharedcausal','sharedcurrentstate','fullcomponenthistory']
BIND=[]
ROOT=None

def bind(p):
 p=Path(p)
 try:loc=str(p.relative_to(ROOT))
 except ValueError:loc=p.name
 if loc not in BIND:BIND.append(loc)
 return p

def read(rel):
 p=bind(ROOT/rel)
 if p.suffix=='.parquet':return pd.read_parquet(p)
 if p.suffix=='.csv':return pd.read_csv(p)
 return json.loads(p.read_text())

def js(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def metrics(d):
 n=len(d);p=int(d.proposed.sum());h=int(d.harmful.sum());b=int(d.beneficial.sum())
 v=d.benefit_fraction
 q=int((d.proposed & v.notna()).sum())
 return {'N':n,'D':int(d.numeric_alarm.sum()),'P':p,'L':int(d.correct.sum()),'B':b,'H':h,'Q_proposal':q,'benefit_available_N':int(v.notna().sum()),'coverage_pct':100*b/n if n else np.nan,'proposal_pct':100*p/n if n else np.nan,'strict_harm_per_all_pct':100*h/n if n else np.nan,'strict_harm_per_quality_proposal_pct':100*h/q if q else np.nan,'target_precision_pct':100*int(d.correct.sum())/p if p else np.nan,'signed_benefit_sum_range':float(v.sum()),'signed_benefit_mean_range':float(v.mean()),'G_pct_range':100*float(v.mean()),'positive_benefit_sum_range':float(v[v>0].sum()),'negative_benefit_sum_range':float(v[v<0].sum()),'worst_benefit_range':float(v.min())}

def grouped(d,keys):
 rows=[]
 for k,g in d.groupby(keys,sort=True,dropna=False):
  if not isinstance(k,tuple):k=(k,)
  rows.append({**dict(zip(keys,k)),**metrics(g)})
 return pd.DataFrame(rows)

def save(df,out,name):df.to_csv(out/(name+'.csv'),index=False,float_format='%.17g')

def pair_cases(d,compare='Context',ref='Reference',keys=['case_id','seed']):
 a=d[d.method.eq(compare)].copy();b=d[d.method.eq(ref)].copy()
 assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
 m=a.merge(b,on=keys,how='outer',suffixes=('_new','_ref'),validate='one_to_one',indicator=True)
 assert m._merge.eq('both').all()
 for col in ['unit','family','source_sha256','source_start','source_end']:
  if col+'_new' in m:
   assert m[col+'_new'].equals(m[col+'_ref'])
   m[col]=m[col+'_new']
 for col in ['proposed','beneficial','harmful']:
  m['delta_'+col]=m[col+'_new'].astype(int)-m[col+'_ref'].astype(int)
 m['delta_benefit']=m.benefit_fraction_new-m.benefit_fraction_ref
 return m

def paired(m,keys,compare='Context',ref='Reference'):
 rows=[]
 for k,g in m.groupby(keys,sort=True,dropna=False):
  if not isinstance(k,tuple):k=(k,)
  n=len(g);row={**dict(zip(keys,k)),'comparison':compare+' minus '+ref,'N':n}
  for suff,lab in [('new',compare),('ref',ref)]:
   z=pd.DataFrame({c:g[c+'_'+suff].to_numpy() for c in ['numeric_alarm','proposed','correct','beneficial','harmful','benefit_fraction']})
   row.update({lab+'_'+a:b for a,b in metrics(z).items()})
  row.update(delta_B=int(g.delta_beneficial.sum()),delta_H=int(g.delta_harmful.sum()),delta_P=int(g.delta_proposed.sum()),delta_coverage_pp=100*g.delta_beneficial.mean(),delta_harm_per_all_pp=100*g.delta_harmful.mean(),delta_G_pct_range=100*g.delta_benefit.mean(),delta_signed_benefit_sum_range=g.delta_benefit.sum(),benefit_count_gains=int(g.delta_beneficial.gt(0).sum()),benefit_count_losses=int(g.delta_beneficial.lt(0).sum()),harm_count_increases=int(g.delta_harmful.gt(0).sum()),harm_count_reductions=int(g.delta_harmful.lt(0).sum()),signed_improved_n=int(g.delta_benefit.gt(0).sum()),signed_worsened_n=int(g.delta_benefit.lt(0).sum()),signed_equal_n=int(g.delta_benefit.eq(0).sum()))
  rows.append(row)
 return pd.DataFrame(rows)

def errors(d,keys):
 rows=[]
 for k,g in d.groupby(keys,sort=True,dropna=False):
  if not isinstance(k,tuple):k=(k,)
  for scope,s in [('all',g),('proposed',g[g.proposed]),('harmful',g[g.proposed & g.harmful]),('beneficial',g[g.beneficial])]:
   v=s.benefit_fraction
   row={**dict(zip(keys,k)),'error_subset':scope,'panel_N':len(g),'subset_N':len(s),'available_N':int(v.notna().sum()),'mean_benefit_range':v.mean(),'sum_benefit_range':v.sum(),'min_benefit_range':v.min()}
   for q in [0,.01,.05,.5,.95,.99,1]:row['benefit_quantile_'+str(q)]=v.quantile(q)
   for c in ['selected_before','selected_after','remaining_target_error','wrong_channel_damage']:
    if c in s:
     vcol=s[c].dropna();row[c+'_available_N']=len(vcol);row[c+'_mean']=vcol.mean() if len(vcol) else np.nan;row[c+'_median']=vcol.median() if len(vcol) else np.nan;row[c+'_q95']=vcol.quantile(.95) if len(vcol) else np.nan
   rows.append(row)
 return pd.DataFrame(rows)

def main():
 global ROOT
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--native-cache',type=Path,default=None);a=ap.parse_args()
 ROOT=a.root.resolve();out=a.out.resolve();assert out!=ROOT and out!=ROOT/'results' and ROOT/'results' not in out.parents,'Write outside the supplied results'
 out.mkdir(parents=True,exist_ok=True)
 checks={};native=json.loads(bind(a.native_cache or ROOT/'results/metadata/native_graph.json').read_text())
 f=read('results/data/primary_outcomes.parquet');h=read('results/data/history_outcomes.parquet');refinement=read('results/data/refinement.parquet');c=read('results/data/context_quality.parquet')
 assert not f.duplicated(['method','case_id','seed']).any();assert not h.duplicated(['level','mode','case_id','seed']).any();assert not refinement.duplicated(['variant','case_id','seed']).any();assert not c.duplicated(['case_id','seed']).any()
 for name,z in f.groupby('method'):
  assert len(z)==194685 and z.family.isin(PRIMARY).sum()==193350
  assert len(z[['unit','seed']].drop_duplicates())==60
 fp=f[f.family.isin(PRIMARY)].copy();hp=h[h.primary].copy()
 assert fp.benefit_fraction.notna().all() and hp.benefit_fraction.notna().all()
 assert (fp.loc[~fp.proposed,'benefit_fraction']==0).all()
 ladder=grouped(hp,['mode','level']);save(ladder,out,'history_ladder')
 oldlad=read('results/data/history_summary.csv').rename(columns={'n':'N','beneficial':'B','strict_harm':'H'})
 mm=ladder.merge(oldlad,on=['mode','level'],validate='one_to_one',suffixes=('','_old'))
 for x in ['N','B','H']:assert (mm[x]==mm[x+'_old']).all()
 assert np.allclose(mm.signed_benefit_mean_range,mm.mean_signed_benefit,rtol=0,atol=1e-14)
 save(grouped(hp,['mode','level','unit','seed']),out,'history_configuration_checkpoint')
 save(grouped(h[~h.primary],['mode','level','family']),out,'history_controls')
 z=ladder.set_index(['mode','level']);lad_delta=[]
 for mode in z.index.get_level_values(0).unique():
  for i in range(1,4):
   x=z.loc[(mode,LEVELS[i])];y=z.loc[(mode,LEVELS[i-1])]
   lad_delta.append({'mode':mode,'from':LEVELS[i-1],'to':LEVELS[i],'delta_B':int(x.B-y.B),'delta_H':int(x.H-y.H),'delta_G_pct_range':x.G_pct_range-y.G_pct_range})
 save(pd.DataFrame(lad_delta),out,'history_adjacent_deltas');checks['ladder_matches_retained_8_rows']=True
 pooled=grouped(fp,['method']);save(pooled,out,'original_pooled')
 anchors={'Context':(57966,8653,66619),'Reference':(53590,9154,62744),'Dominance':(37890,1058,None)}
 for method,(b,hh,p) in anchors.items():
  row=pooled.set_index('method').loc[method];assert int(row.B)==b and int(row.H)==hh
  if p is not None and method=='Context':assert int(row.P)==p
 checks['original_pooled_anchors']=True
 grouping={'configuration':['unit'],'checkpoint':['seed'],'configuration_checkpoint':['unit','seed'],'source_period':['unit','source_sha256','source_start','source_end'],'configuration_checkpoint_source_period':['unit','seed','source_sha256','source_start','source_end'],'family':['family'],'configuration_family':['unit','family'],'checkpoint_family':['seed','family'],'configuration_checkpoint_family_source_period':['unit','seed','family','source_sha256','source_start','source_end']}
 pairframes={}
 for name,keys in grouping.items():
  save(grouped(fp,['method']+keys),out,'original_'+name)
 for compare in ['Context','Dominance','Selection','Combined']:
  pm=pair_cases(fp,compare);pairframes[compare]=pm
  for name,keys in grouping.items():save(paired(pm,keys,compare),out,'paired_'+compare.lower()+'_'+name)
 pm=pairframes['Context'];conf=paired(pm,['unit']);seed=paired(pm,['seed']);period=paired(pm,grouping['source_period'])
 save(errors(fp,['method']),out,'original_error_magnitudes')
 save(errors(fp,['method','unit']),out,'original_configuration_error_magnitudes')
 save(grouped(f[f.family.eq('unchanged')],['method']),out,'original_normal_pooled')
 control_rows=[]
 for k,g in c[~c.family.isin(PRIMARY)].groupby(['family','qualification_group'],dropna=False):
  sel=g[g.proposed.astype(bool)];v=sel.benefit_mae_pct_range
  control_rows.append({'family':k[0],'qualification_group':k[1],'N':len(g),'P':len(sel),'available_selected_quality_N':int(v.notna().sum()),'selected_improved_N':int(v.gt(0).sum()),'selected_harmed_N':int(v.lt(0).sum()),'selected_benefit_mean_pct_range':v.mean(),'selected_benefit_sum_pct_range':v.sum(min_count=1)})
 save(pd.DataFrame(control_rows),out,'context_control_quality')
 control_lookup=pd.DataFrame(control_rows).groupby('family')[['P','selected_improved_N','selected_harmed_N']].sum()
 assert control_lookup.loc['witness_corruption'].tolist()==[78,74,4]
 assert control_lookup.loc['common_mode'].tolist()==[55,54,1]
 checks['witness_78_74_4_common_mode_55_54_1_selected_channel_only']=True
 cfg=grouped(fp,['method','unit']);family=grouped(fp,['method','family']);cfgfam=grouped(fp,['method','unit','family']);weights=[]
 for label,g in [('pooled',pooled),('configuration_equal',cfg),('family_equal',family),('configuration_then_family_equal',cfgfam)]:
  for method,t in g.groupby('method'):
   weights.append({'weighting':label,'method':method,'groups_N':len(t),'coverage_pct':t.coverage_pct.mean(),'G_pct_range':t.G_pct_range.mean(),'strict_harm_per_quality_proposal_pct':t.strict_harm_per_quality_proposal_pct.mean(),'target_precision_pct':t.target_precision_pct.mean()})
 save(pd.DataFrame(weights),out,'original_weight_sensitivity')
 concentration={'configurations':conf[['unit','N','delta_B','delta_H','delta_signed_benefit_sum_range','delta_G_pct_range']].to_dict('records'),'checkpoints':seed[['seed','N','delta_B','delta_H','delta_G_pct_range']].to_dict('records'),'source_period_count':len(period),'source_period_B_increase_equal_decrease':[int(period.delta_B.gt(0).sum()),int(period.delta_B.eq(0).sum()),int(period.delta_B.lt(0).sum())],'source_period_H_increase_equal_decrease':[int(period.delta_H.gt(0).sum()),int(period.delta_H.eq(0).sum()),int(period.delta_H.lt(0).sum())],'source_period_G_increase_equal_decrease':[int(period.delta_G_pct_range.gt(0).sum()),int(period.delta_G_pct_range.eq(0).sum()),int(period.delta_G_pct_range.lt(0).sum())],'total_delta_B':int(pm.delta_beneficial.sum()),'total_delta_H':int(pm.delta_harmful.sum()),'total_delta_signed_benefit_sum_range':float(pm.delta_benefit.sum())}
 js(out/'original_concentration.json',concentration)
 # Bind U and support to the same Context rows; support never defines U.
 ctx=refinement[refinement.variant.eq('context')].merge(c[['case_id','seed','context_status','invalid_context','valid_all','operator_preview']],on=['case_id','seed'],suffixes=('','_quality'),validate='one_to_one')
 assert (ctx.invalid_context==ctx.invalid_context_quality).all()
 ctx['U']=ctx.context_status.eq('available') & ~ctx.invalid_context.astype(bool)
 ctx['scope']=np.where(ctx.family.isin(PRIMARY),'primary','control')
 supportkeys=['scope','family','U','low_support','unknown_state','reference_fallback','calibration_fallback']
 sr=[]
 for k,g in ctx.groupby(supportkeys,dropna=False,sort=True):
  sr.append({**dict(zip(supportkeys,k)),'N':len(g),'P':int(g.proposed.sum()),'V':int(g.operator_preview.sum()),'B_primary_labels_only':int(g.beneficial.sum()),'H_primary_labels_only':int(g.harmful.sum())})
 save(pd.DataFrame(sr),out,'context_support_joint')
 support={'primary_N':int(ctx.scope.eq('primary').sum()),'primary_P':int(ctx[ctx.scope.eq('primary')].proposed.sum()),'primary_low_support_P':int(ctx[ctx.scope.eq('primary') & ctx.low_support.astype(bool)].proposed.sum()),'all_unknown_state_N':int(ctx.unknown_state.sum()),'primary_unknown_state_N':int(ctx.loc[ctx.scope.eq('primary'),'unknown_state'].sum()),'invalid_N':int((~ctx.U).sum()),'invalid_P':int(ctx.loc[~ctx.U,'proposed'].sum()),'invalid_V':int(ctx.loc[~ctx.U,'operator_preview'].sum())}
 assert support['primary_low_support_P']==56085 and support['primary_P']==66619
 assert support['invalid_P']==support['invalid_V']==0
 js(out/'context_support_summary.json',support)
 # Native status: record absent graph detail as absent, never infer a reason from F=0.
 nat=c[c.family.isin(['native_process','native_attack'])].copy();ne=pd.DataFrame(native['records'])
 nat=nat.merge(ne,on=['case_id','seed'],how='left',validate='one_to_one',indicator=True);assert nat._merge.eq('both').all()
 traces=[];arrays=[];nativew=None
 # Independently confirm the originally specified example identities by sorted closed outcomes.
 contextf=fp[fp.method.eq('Context')].merge(fp[fp.method.eq('Reference')][['case_id','seed','proposed']],on=['case_id','seed'],suffixes=('','_ref'),validate='one_to_one')
 useful=contextf[contextf.beneficial & contextf.proposed & ~contextf.proposed_ref].sort_values(['unit','seed','case_id']).iloc[0]
 adverse=contextf[contextf.harmful & contextf.proposed].sort_values(['unit','seed','case_id']).iloc[0]
 veto=c[c.family.isin(PRIMARY) & c.operator_preview.astype(bool)].sort_values(['unit','seed','case_id']).iloc[0]
 nn=c[c.family.isin(['native_process','native_attack']) & c.qualification_group.eq('response_supported')].sort_values(['unit','seed','case_id']).iloc[0]
 selected={'useful':useful,'adverse':adverse,'veto':veto,'native':nn}
 for kind in ['useful','adverse','veto','native']:
  rel=f'results/traces/trace_{kind}_observable.json';j=read(rel)
  assert j['case_id']==selected[kind].case_id and j['seed']==int(selected[kind].seed)
  ev=read(f'results/traces/trace_{kind}_evaluator.json')
  npzrel=f'results/traces/trace_{kind}_inference.npz';npz=bind(ROOT/npzrel)
  with np.load(npz,allow_pickle=False) as ar:
   assert ar['center'].shape==ar['scale'].shape==(5,13)
   # Verify cached normalization only; no prediction module is invoked.
   assert np.allclose((ar['signed']-ar['center'][:,None,:])/ar['scale'][:,None,:],ar['standardized'],rtol=1e-12,atol=1e-12)
   for key in ar.files:arrays.append({'kind':kind,'path':npzrel,'key':key,'shape':str(list(ar[key].shape)),'dtype':str(ar[key].dtype)})
  traces.append({'kind':kind,'case_id':j['case_id'],'slot':j['slot'],'selection_rule':j['rule'],'observable_json':rel,'context_locator':rel+'#/context','states_locator':rel+'#/observable_component_states','center_locator':npzrel+'::center[window,channel]','scale_locator':npzrel+'::scale[window,channel]','evaluator_json':f'results/traces/trace_{kind}_evaluator.json','graph_status':j['process_flag_details'].get('graph_status'),'persistent_peak_N':len(j['process_flag_details']['persistent_peaks']),**j['routes']})
  if kind=='native':nativew=j
 native_rows=[]
 for r in nat.itertuples():
  fields=r.cached_graph_status_fields
  row={k:getattr(r,k) for k in ['case_id','seed','unit','family','qualification_group','source_sha256','source_start','source_end','proposed','process_flag','quality_n','context_status','valid_all','cached_record_available']}
  row['cached_graph_status_fields_json']=json.dumps(fields,separators=(',',':'));row['graph_status']=fields[0]['value'] if fields else 'missing_not_cached';row['graph_status_source']='original_cached_prediction' if fields else 'missing'
  row['persistent_peak_N']=np.nan
  if r.case_id==nativew['case_id'] and int(r.seed)==nativew['seed']:
   row['graph_status']=nativew['process_flag_details']['graph_status'];row['graph_status_source']='retained_illustrative_trace';row['persistent_peak_N']=len(nativew['process_flag_details']['persistent_peaks'])
  native_rows.append(row)
 native_df=pd.DataFrame(native_rows);save(native_df,out,'native_graph_status_by_observation')
 save(native_df.groupby(['qualification_group','graph_status','graph_status_source'],dropna=False).agg(N=('case_id','size'),P=('proposed','sum'),F=('process_flag','sum'),Q_P=('quality_n','sum')).reset_index(),out,'native_graph_status_summary')
 supported=native_df[native_df.qualification_group.eq('response_supported')];assert len(supported)==25 and supported.process_flag.sum()==0 and supported.proposed.sum()==15
 assert nativew['process_flag_details']['graph_status']=='disconnected_or_incomplete_graph' and len(nativew['process_flag_details']['persistent_peaks'])==13 and not nativew['routes']['F']
 js(out/'trace_catalog.json',traces);save(pd.DataFrame(arrays),out,'trace_array_catalog');checks['trace_selection_and_normalization_verified']=True
 # Temporal closed outcomes. Evaluator metadata only supplies family/amplitude/target labels.
 tr='results/data/'
 t=read(tr+'temporal_outcomes.parquet').rename(columns={'strict_harm':'harmful','normalized_benefit':'benefit_fraction','id':'case_id'})
 t['method']=t.method.replace({'FrozenReference':'Reference','dominance':'Dominance'});t['seed']=t.slot.str.extract('seed(\d+)',expand=False).astype(int);t['unit']='swat'
 assert not t.duplicated(['method','case_id','seed']).any()
 truth=pd.DataFrame(read(tr+'temporal_labels.json'))
 meta=truth[['id','target','amplitude','context_id','source_end']].rename(columns={'id':'case_id'})
 assert not meta.case_id.duplicated().any();t=t.merge(meta,on='case_id',how='left',validate='many_to_one')
 blocks=read(tr+'temporal_blocks.csv');bm=blocks[['source_start','source_sha256','fault_block']]
 t=t.merge(bm,on='source_start',how='left',validate='many_to_one');assert t.source_sha256.notna().all()
 for method,g in t.groupby('method'):
  assert len(g)==712 and g.family.isin(PRIMARY).sum()==616 and g.family.eq('unchanged').sum()==96
 assert t.benefit_fraction.notna().all()
 tf=t[t.family.isin(PRIMARY)].copy();tn=t[t.family.eq('unchanged')].copy()
 assert tf.source_start.nunique()==4 and tn.source_start.nunique()==48
 temporalgroup={'pooled':['method'],'checkpoint':['method','seed'],'family':['method','family'],'checkpoint_family':['method','seed','family'],'stratum':['method','history_class'],'fault_block':['method','source_start','history_class'],'fault_block_checkpoint':['method','source_start','seed','history_class'],'fault_block_checkpoint_family':['method','source_start','seed','history_class','family'],'drift_magnitude_sign':['method','amplitude'],'target_block_checkpoint':['method','source_start','seed','target']}
 for name,keys in temporalgroup.items():save(grouped(tf[tf.family.eq('drift')] if name=='drift_magnitude_sign' else tf,keys),out,'temporal_'+name)
 for name,keys in {'pooled':['method'],'checkpoint':['method','seed'],'stratum':['method','history_class'],'source_period':['method','source_start','history_class'],'checkpoint_source_period':['method','source_start','seed','history_class']}.items():save(grouped(tn,keys),out,'temporal_normal_'+name)
 save(errors(tf,['method']),out,'temporal_fault_error_magnitudes');save(errors(tn,['method']),out,'temporal_normal_error_magnitudes')
 tp=pair_cases(t)
 for col in ['history_class','target','amplitude','context_id']:
  tp[col]=tp[col+'_new']
 faulttp=tp[tp.family.isin(PRIMARY)].copy();ntp=tp[tp.family.eq('unchanged')].copy()
 for name,keys in {'pooled':['unit'],'checkpoint':['seed'],'family':['family'],'source_period':['source_start','history_class'],'fault_block_checkpoint':['source_start','seed','history_class'],'fault_block_checkpoint_family':['source_start','seed','history_class','family'],'mechanism':['source_start','seed','target'],'mechanism_family':['source_start','seed','target','family']}.items():save(paired(faulttp,keys),out,'temporal_paired_'+name)
 save(paired(ntp,['seed','source_start','history_class']),out,'temporal_paired_normal_source_period')
 delta_cols=['case_id','seed','source_start','history_class','family','target','amplitude','proposed_new','proposed_ref','beneficial_new','beneficial_ref','harmful_new','harmful_ref','benefit_fraction_new','benefit_fraction_ref','delta_beneficial','delta_harmful','delta_benefit']
 save(tp[delta_cols],out,'temporal_paired_case_outcomes')
 changed=faulttp[faulttp.delta_beneficial.ne(0) | faulttp.delta_benefit.ne(0)].sort_values(['seed','source_start','target','family','amplitude'])
 save(changed[delta_cols],out,'temporal_incremental_fault_cases')
 save(tn[tn.proposed][['method','case_id','seed','source_start','history_class','selected_target','selected_before','selected_after','benefit_fraction','harmful']],out,'temporal_normal_proposals')
 sums=grouped(tf,['method']).set_index('method');ns=grouped(tn,['method']).set_index('method')
 for method,b,hh in [('Context',77,8),('Reference',65,8),('Dominance',6,0)]:assert int(sums.loc[method,'B'])==b and int(sums.loc[method,'H'])==hh
 assert int(ns.loc['Context','P'])==2 and int(ns.loc['Reference','P'])==1 and int(ns.loc['Dominance','P'])==0
 added=faulttp[faulttp.delta_beneficial.gt(0)];stuckadded=added[added.family.ne('drift')]
 temporal={'fault_N_per_method':616,'normal_N_per_method':96,'fault_blocks':sorted(map(int,tf.source_start.unique())),'added_B_cases':len(added),'added_drift_B_cases':int(added.family.eq('drift').sum()),'added_stuck_B_cases':len(stuckadded),'added_stuck_checkpoints':sorted(map(int,stuckadded.seed.unique())),'added_stuck_source_target_patterns':stuckadded[['source_start','target']].drop_duplicates().to_dict('records'),'total_delta_signed_benefit_range':float(faulttp.delta_benefit.sum()),'added_stuck_delta_signed_benefit_range':float(stuckadded.delta_benefit.sum()),'added_drift_delta_signed_benefit_range':float(added[added.family.eq('drift')].delta_benefit.sum()),'normal_delta_signed_benefit_range':float(ntp.delta_benefit.sum()),'Context_normal_mean_benefit_range':float(ns.loc['Context','signed_benefit_mean_range']),'Reference_normal_mean_benefit_range':float(ns.loc['Reference','signed_benefit_mean_range'])}
 js(out/'temporal_concentration.json',temporal)
 checks.update({'all_joins_one_to_one_or_explicit_many_to_one':True,'temporal_712_616_96_and_B_H_P_anchors':True,'native_supported_25_F0_P15':True,'U_low_support_separated':True,'no_inference_no_fit_no_threshold_selection_no_policy_changes':True})
 js(out/'CHECKS.json',checks)
 js(out/'REPRODUCTION.json',{'command':'python scripts/reproduce_evidence.py --root REPOSITORY --out NEW_OUTPUT_DIR','python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'inference':False,'fitting':False,'new_threshold_selection':False,'rows':{'original':len(f),'history':len(h),'temporal':len(t)},'checks_passed':len(checks),'table_files':len(list(out.glob('*.csv'))),'inputs':sorted(BIND)})
 print(json.dumps({'status':'passed','out':str(out),'checks':checks,'ladder':ladder[['mode','level','B','H','G_pct_range']].to_dict('records'),'original_concentration':concentration,'temporal_concentration':temporal}))
if __name__=='__main__':main()
