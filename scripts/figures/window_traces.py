"""Render closed numerical evidence without fitting or model execution."""
from pathlib import Path
import numpy as np
import pandas as pd
from .typography import configure_plot_fonts, method_label, plain_text, save_figure

def plot(traces,p):
 configure_plot_fonts(8)
 import matplotlib.pyplot as plt
 n=len(traces);fig,axes=plt.subplots(2,n,figsize=(3.5*n,3.05),squeeze=False)
 for i,r in enumerate(traces):
  ax=axes[0,i];j=r['selected_or_ranked'];rng=r['training_ranges'][j];raw=r['raw'][:,j]/rng;target=r['targets'][j]/rng;rows=np.arange(300)+r['source']['start'];ax.plot(rows[200:],raw[200:],color='#9f6038',lw=1.1,label='Observed');ax.plot(rows[-60:],target,color='#087f83',lw=1.2,label='Candidate')
  if r['clean'] is not None:ax.plot(rows[200:],r['clean'][200:,j]/rng,color='#26364d',lw=1,ls='--',label='Clean (evaluator)')
  state=r['observable_component_states'];changes=[t for t in range(201,300) if any(v[t]!=v[t-1] for v in state.values())]
  for t in changes:ax.axvline(rows[t],color='#8c9eaa',alpha=.5,lw=.5)
  ax.set_title(r['kind'].capitalize()+': '+r['unit'].replace('four_tank','FourTank')+', seed '+str(r['seed'])+f', j={j}',fontsize=9);ax.set_ylabel('Value / training range');ax.set_xlabel('Source row');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
  ax=axes[1,i];x=np.arange(1,6);ax.plot(x,r['energy'].max(1),'o-',ms=3,color='#537eaa',label='Max window energy');ax.plot(x,r['cum'].max(1),'s-',ms=3,color='#087f83',label='Max signed accumulation');ax.axhline(r['event']['limit'],color='#a64e3a',ls='--',lw=1,label='Event limit');ax.set_yscale('log');ax.set_xticks(x);ax.set_xlabel('Overlapping window / prefix');ax.set_ylabel('Normalized score');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);flags=r['routes'];txt=' '.join(f'{k}={int(flags[k])}' for k in ['U','F','P','V','C']);comparison=r"$>$" if r["gap"]>r["threshold"] else r"$\leq$";ax.set_title(txt+f"; gap {r['gap']:.2f} {comparison} {r['threshold']:.2f}",fontsize=8)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,1.03));handles,labels=axes[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.02));fig.tight_layout(rect=[0,.06,1,.96]);save_figure(fig,p);plt.close(fig);return fig
