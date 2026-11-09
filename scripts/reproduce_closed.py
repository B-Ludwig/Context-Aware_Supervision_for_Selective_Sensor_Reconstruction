"""Regenerate the original fixed comparison from closed portable inputs (no inference)."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.out.resolve();assert out!=root and out!=root/'results' and root/'results' not in out.parents,'Write outside the supplied results';out.mkdir(parents=True,exist_ok=True)
primary=['drift','stuck','noisy_stuck','quantized_stuck']; cols=['n','detected','concern','proposed','correct','beneficial','harmful','quality_n'];rows=[]
ref=pd.read_parquet(root/'results/data/refinement.parquet');old=pd.read_parquet(root/'results/data/baselines.parquet')
for d,names in [(ref,['reference','context','selection','combined']),(old,['baseline_70','probe_envelope','calibrated_selective'])]:
 assert not d.duplicated(['variant','case_id','seed']).any()
 for name in names:
  z=d[d.variant.eq(name)];q=z[z.family.isin(primary)];assert len(z)==194685 and len(q)==193350
  assert len(q[['unit','seed']].drop_duplicates())==60
  v={k:int(q[k].sum()) for k in cols};g=q.groupby('unit')[cols].sum();normal=z[z.family.eq('unchanged')]
  labels={'baseline_70':'Residual dominance','probe_envelope':'Envelope screened','calibrated_selective':'Original calibrated','reference':'Reference','context':'Context','selection':'Selection','combined':'Combined'}
  rows.append({'variant':name,'method':labels[name],**v,'normal_N':int(normal.n.sum()),'normal_proposals':int(normal.proposed.sum()),'coverage_pct':100*v['beneficial']/v['n'],'target_precision_pct':100*v['correct']/v['proposed'],'strict_harm_pct':100*v['harmful']/v['quality_n'],'configuration_balanced_coverage_pct':float(100*(g.beneficial/g.n).mean()),'configuration_balanced_precision_pct':float(100*(g.correct/g.proposed).mean()),'configuration_balanced_harm_pct':float(100*(g.harmful/g.quality_n).mean())})
assert len(rows)==7 and len({r['variant'] for r in rows})==7
df=pd.DataFrame(rows);df.to_csv(out/'fixed_policy_metrics.csv',index=False)
c=df[df.variant.eq('context')].iloc[0];assert [int(c[k]) for k in ['detected','proposed','correct','beneficial','harmful']]==[128841,66619,58935,57966,8653]
review=pd.read_parquet(root/'results/data/concern.parquet'); review=review[review.variant.eq('context_review')]; assert len(review)==194685
review_totals={}
for scope,g in [('primary',review[review.family.isin(primary)]),('controls',review[~review.family.isin(primary)])]:
 review_totals[scope]={k:int(g[k].sum()) for k in ['n','detected','concern','proposed','preview_n','quality_n','quality_missing']}
assert review_totals['primary']['concern']==136201 and review_totals['primary']['preview_n']==40696
q=pd.read_parquet(root/'results/data/context_quality.parquet');ctrl=[]
for family,expected in [('witness_corruption',(78,74,4)),('common_mode',(55,54,1))]:
 z=q[q.family.eq(family)&q.proposed.astype(bool)];v=(len(z),int(z.benefit_mae_pct_range.gt(0).sum()),int(z.benefit_mae_pct_range.lt(0).sum()));assert v==expected
 ctrl.append({'family':family,'proposals':v[0],'improve':v[1],'harm':v[2]})
z=q[q.family.isin(primary)];sel=z.proposed.astype(bool);b=z.benefit_mae_pct_range.where(sel,0);assert b.notna().all();wrong=sel & ~z.correct.astype(bool)
assert np.allclose(z.loc[wrong,'keep_mae_pct_range'],0)

supported=q[q.family.isin(['native_process','native_attack']) & q.qualification_group.eq('response_supported')]
assert len(supported)==25 and int(supported.process_flag.sum())==0
temporal=pd.read_parquet(root/'results/data/temporal_outcomes.parquet')
assert not temporal.duplicated(['id','slot','method']).any()
assert temporal.groupby('method').size().eq(712).all() and temporal.method.nunique()==3
terminal={str(method):{str(status):int(n) for status,n in z.groupby('status').size().items()} for method,z in temporal.groupby('method')}
receipt={'status':'passed','mode':'closed-result regeneration; no fitting, raw-data access, AE or TTO','methods':rows,'supported_native_process_flags':{'N':len(supported),'F':int(supported.process_flag.sum())},'temporal_terminal_outcomes':terminal,'context_review_outputs':review_totals,'control_quality':ctrl,'context_signed_benefit_pct_range':float(b.mean()),'wrong_channel_proposals':int(wrong.sum()),'mean_remaining_total_error_pct_range':float((z.initial_error-b).mean())}
(out/'CLOSED_REPRODUCTION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'passed','out':str(out),'context_beneficial':int(c.beneficial)}))
