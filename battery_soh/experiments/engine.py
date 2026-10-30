"""Single research protocol for within, pairwise transfer and LODO."""
from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.features import checked_features, tabular, sequence_windows
from ..data.splits import battery_split
from ..evaluation.metrics import regression_metrics
from ..models.classical import make_classical
from ..models.deep import DEEP_NAMES, TorchRegressor

CLASSICAL_NAMES = ('mean','ridge','elasticnet','xgboost')
ALL_MODELS = CLASSICAL_NAMES + DEEP_NAMES

def model_arrays(df, model_name, features, window):
    if model_name in DEEP_NAMES:
        x, y, positions = sequence_windows(df, features, window)
        return x, y, df.iloc[positions].copy().reset_index(drop=True)
    x,y = tabular(df, features)
    return x,y,df.reset_index(drop=True)

def train_model(name, train, val, features, window, seed, deep_config=None):
    X,y,_ = model_arrays(train,name,features,window)
    Xv,yv,_ = model_arrays(val,name,features,window)
    if not len(X) or not len(Xv): raise ValueError('Too few samples to train/validate')
    if name in CLASSICAL_NAMES:
        model=make_classical(name,seed)
        model.fit(X,y)
    elif name in DEEP_NAMES:
        model=TorchRegressor(name,seed=seed,**(deep_config or {}))
        model.fit(X,y,Xv,yv)
    else: raise ValueError(f'Unknown model: {name}')
    return model

def evaluate_model(model, name, test, features, window):
    x,y,rows = model_arrays(test,name,features,window)
    if not len(x): raise ValueError('No test sequences / rows')
    preds = model.predict(x)
    metrics = regression_metrics(y,preds)
    record = rows[['dataset','battery_id','cycle','chemistry','temperature_c','charge_rate_c']].copy()
    record['actual_soh'],record['predicted_soh'] = y,preds
    record['absolute_error'] = np.abs(y-preds)
    return metrics, record

def benchmark(df: pd.DataFrame, *, experiment: str='cross',
              models: list[str] | None=None, seeds: list[int] | None=None,
              features: list[str] | None=None, window: int=4,
              deep_config: dict | None=None, output='results/benchmark', strict=False):
    """Any failure explicitly logged; never silently invent a score."""
    models = list(models or CLASSICAL_NAMES)
    seeds = list(seeds or [42])
    features = list(features or ['cycle'])
    checked_features(df,features)
    bad_models = set(models)-set(ALL_MODELS)
    if bad_models: raise ValueError(f'Unknown models: {bad_models}')
    if experiment not in ('within','cross','matrix','lodo'): raise ValueError(experiment)
    datasets = sorted(df.dataset.unique().tolist())
    if len(datasets)<2 and experiment != 'within': raise ValueError('Need >=2 datasets for transfer')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    metrics_rows, pred_rows, failures = [],[],[]
    for seed, name in itertools.product(seeds,models):
        if experiment=='within':
            tasks=[(d,d) for d in datasets]
        elif experiment in ('cross','matrix'):
            tasks=[(a,b) for a,b in itertools.product(datasets,datasets) if experiment=='matrix' or a!=b]
        else:
            tasks=[('+'.join([a for a in datasets if a!=b]),b) for b in datasets]
        for source,target in tasks:
            run = dict(seed=int(seed), model=name,
                       experiment=('within' if experiment=='matrix' and source==target else
                                   'cross' if experiment=='matrix' else experiment),
                       source=source, target=target)
            try:
                tr_all=df.loc[df.dataset.isin(source.split('+'))]
                tr, val, source_holdout=battery_split(tr_all,seed)
                # For within, evaluate only source held-out batteries;
                # for cross/LODO, target dataset was never used to fit.
                te=source_holdout if run['experiment']=='within' else df.loc[df.dataset==target].copy()
                fitted=train_model(name,tr,val,features,window,seed,deep_config)
                scores, p=evaluate_model(fitted,name,te,features,window)
                metrics_rows.append({**run,**scores,'train_batteries':len(tr[['dataset','battery_id']].drop_duplicates()),
                                     'test_batteries':len(te[['dataset','battery_id']].drop_duplicates()),
                                     'test_rows':len(p)})
                for k,v in run.items(): p[k]=v
                pred_rows.append(p)
                print(f"OK {experiment} {source} -> {target} {name} seed={seed}: MAE={scores['MAE']:.3f}",flush=True)
            except Exception as exc:
                failures.append({**run,'error':f'{type(exc).__name__}: {exc}'})
                print(f'FAILED {run}: {exc}',flush=True)
                if strict: raise
    metrics=pd.DataFrame(metrics_rows)
    metrics.to_csv(out/'metrics.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    else: pd.DataFrame().to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(failures,columns=['seed','model','experiment','source','target','error']).to_csv(out/'failures.csv',index=False)
    (out/'config.json').write_text(json.dumps({'experiment':experiment,'models':models,'seeds':seeds,
            'features':features,'window':window,'deep_config':deep_config or {},
            'datasets':datasets},indent=2))
    return metrics
