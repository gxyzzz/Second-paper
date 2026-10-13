from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import *

FIGSIZE=(3.3,2.55)

def grouped_gain(csv_path,outbase,total=False):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    x=np.arange(len(d)); w=.33
    ax.bar(x-w/2,d.iloc[:,1],w,label='CoLiftRec',color=COLIFT,hatch='//',edgecolor='white',linewidth=.4)
    ax.bar(x+w/2,d.iloc[:,2],w,label='PDBR',color=PDBR,hatch='..',edgecolor='white',linewidth=.4)
    ax.set_xticks(x); ax.set_xticklabels(d.iloc[:,0]); ax.set_ylabel('Relative Improvement (%)'); ax.set_ylim(bottom=0)
    if total and d.shape[1]>3:
        top=np.maximum(d.iloc[:,1].to_numpy(),d.iloc[:,2].to_numpy())
        for xi,y,t in zip(x,top,d.iloc[:,3]): ax.text(xi,y+.28,f'Total +{t:.2f}%',ha='center',va='bottom',fontsize=7.2)
        ax.set_ylim(0,max(top)*1.28)
    ax.legend(frameon=False,ncol=2,loc='upper left'); finish(ax); save(fig,outbase)

def sensitivity(csv_path,outbase,xlabel):
    apply_style(); d=pd.read_csv(csv_path).sort_values('parameter_value'); fig,ax=plt.subplots(figsize=FIGSIZE)
    ax.plot(d.parameter_value,d.validation_u4,marker='o',label='Validation',color=VALIDATION)
    ax.plot(d.parameter_value,d.test_u4,marker='s',linestyle='--',label='Test',color=TEST)
    ax.axhline(0,color=GRID,linewidth=.7); ax.set_xlabel(xlabel); ax.set_ylabel('U4 Improvement over B0 (%)')
    ax.legend(frameon=False); finish(ax); save(fig,outbase)

def training(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    ax.plot(d.epoch,d.baseline_colift_r20,label='Baseline',color=BASELINE)
    ax.plot(d.epoch,d.pdbr_colift_r20,label='PDBR',color=PDBR)
    be=int(d.loc[d.baseline_colift_r20.idxmax(),'epoch']); pe=int(d.loc[d.pdbr_colift_r20.idxmax(),'epoch'])
    ax.axvline(be,color=BASELINE,linestyle=':',linewidth=.8,alpha=.7); ax.axvline(pe,color=PDBR,linestyle=':',linewidth=.8,alpha=.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation R@20'); ax.legend(frameon=False); finish(ax); save(fig,outbase)

def preference_alignment(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    ax.plot(d.epoch,d.a_true_median,label='TRUE user',color=TRUE,marker='o',markevery=max(1,len(d)//8))
    ax.plot(d.epoch,d.a_shuf_median,label='SHUFFLED user',color=SHUFFLED,linestyle='--',marker='s',markevery=max(1,len(d)//8))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Alignment with Preference Direction'); ax.legend(frameon=False); finish(ax); save(fig,outbase)

def probability(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    ax.plot(d.epoch,d.p_delta_positive,color=PDBR,marker='o',markevery=max(1,len(d)//8))
    ax.axhline(.5,color=BASELINE,linestyle='--',linewidth=.9,alpha=.65)
    ax.set_ylim(0,1); ax.set_xlabel('Epoch'); ax.set_ylabel(r'$P(A_{true} > A_{shuffled})$'); finish(ax); save(fig,outbase)

def geometry(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    x=np.arange(len(d)); w=.23
    for off,col,label,color,hatch in [(-w,'applied_refinement_fraction','Applied',PDBR,'//'),(0,'target_reachable_fraction','Reachable',COLIFT,'..'),(w,'cap_hit_fraction','Cap hit',BASELINE,'xx')]:
        ax.bar(x+off,100*d[col],w,label=label,color=color,hatch=hatch,edgecolor='white',linewidth=.3)
    ax.set_xticks(x); ax.set_xticklabels(d.dataset); ax.set_ylabel('Fraction (%)'); ax.set_ylim(bottom=0); ax.legend(frameon=False,ncol=3,fontsize=6.6); finish(ax); save(fig,outbase)

def efficiency(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE)
    order=['Baby','Sports','Electronics']; x=np.arange(3); w=.33
    b=[float(d[(d.dataset==z)&(d.branch=='Baseline')].epochs_run.iloc[0]) for z in order]
    p=[float(d[(d.dataset==z)&(d.branch=='PDBR')].epochs_run.iloc[0]) for z in order]
    ax.bar(x-w/2,b,w,label='Baseline',color=BASELINE,hatch='//',edgecolor='white',linewidth=.3)
    ax.bar(x+w/2,p,w,label='PDBR',color=PDBR,hatch='..',edgecolor='white',linewidth=.3)
    ax.axhline(60,color=GRID,linestyle='--',linewidth=.8); ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel('Training Epochs'); ax.set_ylim(0,65); ax.legend(frameon=False); finish(ax); save(fig,outbase)

def band_alignment(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE); x=np.arange(len(d)); w=.34
    ax.bar(x-w/2,d.A_true_mean,w,label='TRUE user',color=TRUE,hatch='//',edgecolor='white',linewidth=.3)
    ax.bar(x+w/2,d.A_shuf_mean,w,label='SHUFFLED user',color=SHUFFLED,hatch='..',edgecolor='white',linewidth=.3)
    ax.axhline(0,color=GRID,linewidth=.7); ax.set_xticks(x); ax.set_xticklabels(d.band); ax.set_ylabel('Mean Preference Alignment'); ax.legend(frameon=False); finish(ax); save(fig,outbase)

def band_shift(csv_path,outbase):
    apply_style(); d=pd.read_csv(csv_path); fig,ax=plt.subplots(figsize=FIGSIZE); x=np.arange(len(d))
    ax.bar(x,d.actual_score_shift_mean,width=.52,color=PDBR,hatch='//',edgecolor='white',linewidth=.3)
    ax.set_xticks(x); ax.set_xticklabels(d.band); ax.set_ylabel('Mean Actual Score Shift'); ax.set_ylim(bottom=0); finish(ax); save(fig,outbase)
