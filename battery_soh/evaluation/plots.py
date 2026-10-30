"""Generate research figures; only from actual experiment outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_figures(input_dir, out_dir=None):
    inp=Path(input_dir)
    out=Path(out_dir) if out_dir else inp/'figures'
    out.mkdir(parents=True,exist_ok=True)
    generated=[]
    mp=inp/'metrics.csv'
    if mp.exists() and mp.stat().st_size:
        scores=pd.read_csv(mp)
        if not scores.empty and 'MAE' in scores:
            for model,g in scores.groupby('model'):
                if 'cross' in set(g.experiment):
                    pivot=g.pivot_table(index='source',columns='target',values='MAE',aggfunc='mean')
                    fig,ax=plt.subplots(figsize=(7,5))
                    im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='viridis')
                    ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=30,ha='right')
                    ax.set_yticks(range(len(pivot.index)),pivot.index)
                    ax.set_xlabel('Target dataset');ax.set_ylabel('Training dataset')
                    ax.set_title(f'Cross-dataset MAE (SOH pp) — {model}')
                    fig.colorbar(im,ax=ax,label='MAE (percentage points)')
                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            v=pivot.iloc[i,j]
                            if np.isfinite(v): ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white')
                    fig.tight_layout()
                    filename=out/f'transfer_{model}.png';fig.savefig(filename,dpi=170);plt.close(fig)
                    generated.append(str(filename))
            pair_summary=scores.groupby(['model','source','target','experiment'],as_index=False).agg(
                mae_mean=('MAE','mean'),mae_std=('MAE','std'),rmse_mean=('RMSE','mean'),
                mape_mean=('MAPE','mean'),runs=('seed','nunique'))
            pair_summary.to_csv(out/'per_pair_seed_summary.csv',index=False)
            ranking=scores.groupby('model',as_index=False)['MAE'].mean().sort_values('MAE')
            fig,ax=plt.subplots(figsize=(7,4));ax.barh(ranking.model,ranking.MAE)
            ax.invert_yaxis();ax.set_xlabel('Mean MAE over available runs (SOH pp)')
            ax.set_title('Model ranking — compare only same experiment coverage')
            fig.tight_layout();filename=out/'ranking.png';fig.savefig(filename,dpi=170);plt.close(fig)
            generated.append(str(filename))
            summary=scores.groupby(['experiment','model'],as_index=False)['MAE'].agg(['mean','std','count'])
            summary.to_csv(out/'score_summary.csv')
    ap=inp/'adaptation_metrics.csv'
    if ap.exists() and ap.stat().st_size:
        ad=pd.read_csv(ap)
        ad=ad.dropna(subset=['MAE']) if 'MAE' in ad else pd.DataFrame()
        if not ad.empty:
            fig,ax=plt.subplots(figsize=(7,4))
            for method,g in ad.groupby('method'):
                curve=g.groupby('k')['MAE'].mean()
                ax.plot(curve.index,curve.values,marker='o',label=method)
            ax.set_xlabel('Number of target training batteries');ax.set_ylabel('Target MAE (SOH pp)')
            ax.set_title('Few-shot adaptation (disjoint target test batteries)')
            ax.legend();fig.tight_layout()
            filename=out/'adaptation_curve.png';fig.savefig(filename,dpi=170);plt.close(fig)
            generated.append(str(filename))
    pp=inp/'predictions.csv'
    if pp.exists() and pp.stat().st_size:
        pred=pd.read_csv(pp)
        if 'temperature_c' in pred:
            sub=pred.dropna(subset=['temperature_c','absolute_error'])
            if len(sub)>=5:
                fig,ax=plt.subplots(figsize=(6,4))
                ax.scatter(sub.temperature_c,sub.absolute_error,s=6,alpha=.3)
                ax.set_xlabel('Temperature (°C)');ax.set_ylabel('Absolute error (SOH pp)')
                ax.set_title('Error vs measured temperature (associational)')
                fig.tight_layout();p=out/'error_temperature.png';fig.savefig(p,dpi=170);plt.close(fig)
                generated.append(str(p))
        if 'charge_rate_c' in pred:
            sub=pred.dropna(subset=['charge_rate_c','absolute_error'])
            if len(sub)>=5:
                fig,ax=plt.subplots(figsize=(6,4))
                ax.scatter(sub.charge_rate_c,sub.absolute_error,s=6,alpha=.3)
                ax.set_xlabel('Charging C-rate');ax.set_ylabel('Absolute error (SOH pp)')
                ax.set_title('Error vs measured charge rate (associational)')
                fig.tight_layout();p=out/'error_charge_rate.png';fig.savefig(p,dpi=170);plt.close(fig)
                generated.append(str(p))
        if 'chemistry' in pred:
            sub=pred.dropna(subset=['chemistry','absolute_error'])
            if not sub.empty:
                agg=sub.groupby('chemistry')['absolute_error'].mean()
                fig,ax=plt.subplots(figsize=(5,4));ax.bar(agg.index,agg.values)
                ax.set_xlabel('Chemistry');ax.set_ylabel('Mean absolute error (SOH pp)')
                ax.set_title('Error by reported chemistry (confounded by dataset)')
                fig.tight_layout();p=out/'error_chemistry.png';fig.savefig(p,dpi=170);plt.close(fig)
                generated.append(str(p))
    return generated
