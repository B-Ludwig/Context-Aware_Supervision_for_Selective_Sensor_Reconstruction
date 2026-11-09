"""Render closed numerical evidence without fitting or model execution."""
from pathlib import Path
import numpy as np
import pandas as pd
from .typography import configure_plot_fonts, method_label, plain_text, save_figure

def plot(d, fixed, control, output):
 configure_plot_fonts(8)
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(7.1,2.45));colors={'Reference':'#a26938','Context':'#087f83'}
 for method,color in colors.items():
  p=d[(d.method==method)&(d.scope=='primary')].set_index('threshold_label');normal=d[(d.method==method)&(d.scope=='unchanged')].set_index('threshold_label');z=p.drop('original').sort_index();n=normal.loc[z.index];xs=[100*z.harm_per_proposal,100*n.normal_or_all_proposal_rate,100*z.mean_signed_benefit]
  for i,ax in enumerate(axes):
   ax.plot(xs[i],100*z.coverage,'o-',ms=2.5,lw=1,color=color,label=method_label(method))
   x=[100*p.loc['original','harm_per_proposal'],100*normal.loc['original','normal_or_all_proposal_rate'],100*p.loc['original','mean_signed_benefit']][i];ax.plot(x,100*p.loc['original','coverage'],'*',ms=10,color=color,markeredgecolor='white',markeredgewidth=.5)
 for method,mark in [('Dominance','D'),('Selection','s'),('Combined','^')]:
  r=fixed.set_index('method').loc[method];n=control[(control.method==method)&(control.family=='unchanged')].iloc[0];xs=[100*r.strict_harm_per_proposal,100*n.proposals/n.n,100*r.mean_signed_benefit]
  for i,ax in enumerate(axes):ax.plot(xs[i],100*r.coverage,mark,ms=5,color={'Dominance':'#253750','Selection':'#999999','Combined':'#71549a'}[method],label=method_label(method))
 for ax,label in zip(axes,[plain_text(r'Strict harm / proposals (\%)'),plain_text(r'Normal proposal rate (\%)'),'Mean signed MAE reduction\n'+plain_text(r'(\% training range)')]):ax.set_xlabel(label);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel(plain_text(r'Correct-beneficial coverage (\%)'));handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.52,1.04));fig.tight_layout(rect=[0,0,1,.9]);save_figure(fig,output);plt.close(fig);return fig
