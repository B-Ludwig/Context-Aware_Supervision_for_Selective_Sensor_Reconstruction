"""Render closed numerical evidence without fitting or model execution."""
from pathlib import Path
import numpy as np
import pandas as pd
from .typography import configure_plot_fonts, method_label, plain_text, save_figure

def plot(compact, curves, out, output):
    configure_plot_fonts(8)
    import matplotlib.pyplot as plt
    z = compact[compact.variant.eq('context') & compact.family.eq('drift')]
    z = z.groupby(['unit','seed','magnitude'], observed=True)[['n','detected','beneficial']].sum().reset_index()
    z['detection'] = 100*z.detected/z.n;z['coverage'] = 100*z.beneficial/z.n
    z['group'] = np.where(z.unit.str.startswith('hai_'), 'HAI', z.unit.map({'four_tank':'FourTank','swat':'SWaT'}))
    seed_curves = z.groupby(['group','seed','magnitude'], observed=True)[['detection','coverage']].mean().reset_index()
    seed_curves.to_csv(out/'severity_checkpoint_curves.csv', index=False)
    fig, axes = plt.subplots(2,3,figsize=(7,2.8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.995,bottom=.19,top=.80,hspace=.15,wspace=.12)
    colors={'calibrated_selective':'#9f6e3b','shared_history_empirical':'#657487','context':'#097a72'}
    names={'calibrated_selective':method_label('Original calibrated'),'shared_history_empirical':method_label('Reference'),'context':method_label('Context')}
    for col, group in enumerate(['HAI','FourTank','SWaT']):
        band=seed_curves[seed_curves.group.eq(group)].groupby('magnitude')[['detection','coverage']].agg(['min','max'])
        for row, metric in enumerate(['detection','coverage']):
            axes[row,col].fill_between(band.index, band[(metric,'min')], band[(metric,'max')],color=colors['context'],alpha=.14,lw=0)
        for method,color in colors.items():
            a=curves[(curves.group==group)&(curves.family=='drift')&(curves.variant==method)].sort_values('magnitude')
            for row, metric in enumerate(['detection_macro','beneficial_coverage_macro']):
                axes[row,col].plot(a.magnitude,100*a[metric],color=color,marker='o',ms=3,lw=1.1,label=names[method])
        axes[0,col].set_title(group+(' (macro)' if group=='HAI' else ''),fontsize=9)
        for ax in axes[:,col]:
            ax.set_xscale('log');ax.set_xticks([1,2,5,10,20,50],labels=['1','2','5','10','20','50'])
            ax.set_ylim(-2,102);ax.set_yticks([0,50,100]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel(plain_text(r'$D/N$ (\%)'));axes[1,0].set_ylabel(plain_text(r'$B/N$ (\%)'))
    fig.supxlabel(plain_text(r'Drift magnitude (\% training range)'),fontsize=9,y=.015)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,1.02),ncol=3,frameon=False,fontsize=8)
    save_figure(fig, output, svg=True, dpi=220)
    plt.close(fig)
    return fig
