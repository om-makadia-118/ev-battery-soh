"""Disjoint target-support evaluation; no target-test information during training."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.features import checked_features
from ..data.splits import battery_split, target_support_test
from ..models.deep import DEEP_NAMES
from ..models.classical import make_classical
from .engine import model_arrays,train_model,evaluate_model

class MeanVarianceAligner:
    """Target-support-only per-feature normalization transfer."""
    def fit(self, source, support):
        source, support=np.asarray(source,dtype=float),np.asarray(support,dtype=float)
        self.ms,self.ss=source.mean(0),source.std(0).clip(min=1e-6)
        self.mt,self.st=support.mean(0),support.std(0).clip(min=1e-6)
        return self
    def source_to_target(self,source):
        return ((np.asarray(source)-self.ms)/self.ss*self.st+self.mt).astype(np.float32)

class CoralAligner:
    """Whiten source covariance, recolor using *support-only* target covariance."""
    def __init__(self, reg=1e-3): self.reg=reg
    @staticmethod
    def _sqrtm_psd(M, inverse=False):
        w,U=np.linalg.eigh(M)
        power= -0.5 if inverse else 0.5
        return (U * np.maximum(w,1e-10)**power) @ U.T
    def fit(self, source, support):
        source=np.atleast_2d(np.asarray(source,dtype=float))
        support=np.atleast_2d(np.asarray(support,dtype=float))
        if len(source)<2 or len(support)<2: raise ValueError('CORAL needs >=2 observations per domain')
        self.mean_s,self.mean_t=source.mean(0),support.mean(0)
        cs=np.atleast_2d(np.cov(source,rowvar=False))
        ct=np.atleast_2d(np.cov(support,rowvar=False))
        eye=np.eye(source.shape[1])
        self.align=self._sqrtm_psd(cs+self.reg*eye,True)@self._sqrtm_psd(ct+self.reg*eye)
        return self
    def source_to_target(self,source):
        return ((np.asarray(source)-self.mean_s)@self.align+self.mean_t).astype(np.float32)


def adapt_pair(df, *, source: str, target: str, model: str='ridge',
               counts=(0,1,3,5,10), methods=('none','pooled','scratch','normalization','coral','finetune'),
               seed=42, features=None, window=4, deep_config=None,
               output='results/adaptation'):
    features=features or ['cycle']
    checked_features(df,features)
    src=df[df.dataset==source].copy()
    tgt=df[df.dataset==target].copy()
    if source==target or src.empty or tgt.empty: raise ValueError('Source/target must be distinct present datasets')
    train,val,_=battery_split(src,seed)
    support_pool,test=target_support_test(tgt,seed)
    source_model=train_model(model,train,val,features,window,seed,deep_config)
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(support_pool.battery_id.unique())
    rows=[]; pred_rows=[]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for k in counts:
        k=int(k)
        if k>len(shuffled):
            rows.append(dict(source=source,target=target,model=model,seed=seed,k=k,
                             method='not_feasible',reason=f'{len(shuffled)} support batteries available'))
            continue
        support=support_pool[support_pool.battery_id.isin(shuffled[:k])].copy()
        active=(['none'] if k==0 else
                [m for m in methods if m in (('scratch','finetune') if model in DEEP_NAMES
                                                else ('pooled','scratch','normalization','coral'))])
        for method in active:
            try:
                fitted=source_model
                if method=='finetune':
                    if model not in DEEP_NAMES:
                        raise ValueError('Parameter fine-tuning applies to PyTorch models only')
                    x,y,_=model_arrays(support,model,features,window)
                    fitted=source_model.finetune(x,y)
                elif method=='scratch':
                    if model in DEEP_NAMES:
                        # Disjoint support train/val by battery; small k cannot support splitting.
                        if k<3: raise ValueError('deep scratch training requires >=3 support batteries')
                        st,sv,_=battery_split(support,seed)
                        fitted=train_model(model,st,sv,features,window,seed,deep_config)
                    else:
                        fitted=make_classical(model,seed)
                        x,y,_=model_arrays(support,model,features,window)
                        fitted.fit(x,y)
                elif method=='pooled':
                    combined=pd.concat([train,support],ignore_index=True)
                    fitted=make_classical(model,seed)
                    x,y,_=model_arrays(combined,model,features,window)
                    fitted.fit(x,y)
                elif method in ('coral','normalization'):
                    xs,ys,_=model_arrays(train,model,features,window)
                    xt,_,_=model_arrays(support,model,features,window)
                    align=(CoralAligner() if method=='coral' else MeanVarianceAligner()).fit(xs,xt)
                    fitted=make_classical(model,seed).fit(align.source_to_target(xs),ys)
                elif method!='none':
                    raise ValueError(f'Unknown adaptation method {method}')
                if method in ('coral','normalization'):
                    x,truth,p=model_arrays(test,model,features,window)
                    predictions=fitted.predict(x)
                    from ..evaluation.metrics import regression_metrics
                    scores=regression_metrics(truth,predictions)
                    p=p[['dataset','battery_id','cycle']].copy()
                    p['actual_soh'],p['predicted_soh']=truth,predictions
                else:
                    scores,p=evaluate_model(fitted,model,test,features,window)
                rows.append(dict(source=source,target=target,model=model,seed=seed,k=k,
                                 method=method,reason='',**scores,test_batteries=test.battery_id.nunique()))
                p['source'],p['target'],p['model'],p['seed'],p['k'],p['method']=source,target,model,seed,k,method
                pred_rows.append(p)
            except Exception as exc:
                rows.append(dict(source=source,target=target,model=model,seed=seed,k=k,
                                 method=method,reason=f'{type(exc).__name__}: {exc}'))
    result=pd.DataFrame(rows)
    result.to_csv(out/'adaptation_metrics.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(out/'adaptation_predictions.csv',index=False)
    return result
