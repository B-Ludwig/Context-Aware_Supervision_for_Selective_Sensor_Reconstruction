"""Render closed numerical evidence without fitting or model execution."""
from pathlib import Path
import numpy as np
import pandas as pd
from .typography import configure_plot_fonts, method_label, plain_text, save_figure

from . import typography

def render(output):
    """Write the decision-layer overview as PDF, SVG and PNG."""
    configure_plot_fonts(8)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.2,3.10))
    ax.set_xlim(0,16);ax.set_ylim(0,8.5);ax.axis('off')
    ink='#253750';arrow='#667b87'

    def box(x,y,w,h,face,edge):
     ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=0.10',facecolor=face,edgecolor=edge,linewidth=.8,zorder=1))
    def text(x,y,s,size=8,color=ink,weight=None,ha='center'):
     if not typography.USE_TEX: size *= .90
     if not typography.USE_TEX and s.startswith(r'\textbf{') and s.endswith('}'):
      s=s[len(r'\textbf{'):-1];weight='bold'
     ax.text(x,y,plain_text(s),fontsize=size,color=color,fontweight=weight,ha=ha,va='center',zorder=4)
    def link(x0,y0,x1,y1):
     ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=8,linewidth=.8,color=arrow,zorder=2))

    # Observations and frozen model are available to the evidence branches.
    box(.20,7.55,15.60,.65,'#f6f8fa','#788a97')
    text(8,7.88,r'Observation $X$ \qquad operating history $c$ \qquad fixed AE $f_\theta$',9)
    columns=[(.20,5.00),(5.50,5.00),(10.80,5.00)]
    centers=[x+w/2 for x,w in columns]
    for x,w in columns:link(x+w/2,7.49,x+w/2,7.10)

    # Each box ends in a Section 3 predicate; the event limit is not tau.
    box(.20,4.75,5.00,2.30,'#f4f5f6','#788a97')
    text(2.70,6.70,r'\textbf{Interpret residuals}',8.7)
    text(2.70,6.23,r'$c\longrightarrow (\mu_{wj},\sigma_{wj})$',8.8)
    text(2.70,5.75,r'$z_{wtj}=(e_{wtj}-\mu_{wj})/\sigma_{wj}$',8.5)
    text(2.70,5.24,r'$S\longrightarrow A$',9.5)
    text(2.70,4.94,'Exceeds the calibrated event limit',7.4)

    box(5.50,4.75,5.00,2.30,'#f4f5f6','#788a97')
    text(8.00,6.70,r'\textbf{Compare candidate targets}',8.7)
    text(8.00,6.24,r'TTO candidates $\longrightarrow$ ranking scores $p_j$',8.1)
    text(8.00,5.79,r'Unique winner $j^*$ and gap $\Delta$',8.1)
    text(8.00,5.35,r'Context check $U=1$, all candidates valid',8.0)
    text(8.00,4.97,r'Eligibility $K$: $\Delta>\tau$',9.1)

    box(10.80,4.75,5.00,2.30,'#f4f5f6','#788a97')
    text(13.30,6.70,r'\textbf{Check distributed evidence}',8.7)
    text(13.30,6.23,'Normal-excess residuals',8)
    text(13.30,5.81,'AE input derivatives + declared graph',8)
    text(13.30,5.39,'Persistent, connected component peaks',8)
    text(13.30,4.98,r'Process flag $F$',9.1)

    # A common routing band avoids implying a one-branch/one-output correspondence.
    for x in centers:link(x,4.69,x,4.22)
    box(.20,3.70,15.60,.47,'#f6f8fa','#788a97')
    text(8,3.935,r'Route the candidate using $A$, $K$ and $F$',8.3)
    for x in centers:link(x,3.64,x,3.23)

    outputs=[(.20,'#eaf5f4','#087f83',r'\textbf{Proposal $P$}',r'$P=A\land K\land\neg F$','Estimate for operator confirmation'),
     (5.50,'#fbf4e8','#a87535',r'\textbf{Preview $V$}',r'$V=A\land K\land F$','Withheld estimate for investigation'),
     (10.80,'#f1f3f5','#788a97',r'\textbf{Abstain}',r'$P=V=0$ \quad if $\neg A\lor\neg K$','No proposal or preview')]
    for x,face,edge,heading,formula,detail in outputs:
     box(x,1.72,5.00,1.45,face,edge)
     text(x+2.50,2.85,heading,8.8)
     text(x+2.50,2.40,formula,8.8)
     text(x+2.50,1.99,detail,7.4)

    # Concern is not one of the mutually exclusive candidate outcomes.
    box(.20,.16,15.60,1.03,'#f1f3f5','#788a97')
    text(8,.92,r'\textbf{Retain concern independently of candidate acceptance}',8.2)
    text(7.0,.49,r'$C=A\lor F\lor\neg U\lor A_{\rm native}$',10)
    text(12.6,.49,r'($A_{\rm native}$: original AE alarm)',7.1)
    fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995)
    save_figure(fig, output, svg=True, dpi=220)
    plt.close(fig)
    return fig
