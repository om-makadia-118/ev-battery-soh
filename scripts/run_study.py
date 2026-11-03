"""Multi-stage experiment launcher after all five real sources have been validated.

Examples:
  python scripts/run_study.py --phase benchmark --models ridge xgboost --seeds 11
  python scripts/run_study.py --phase adaptation --models ridge --seeds 11
  python scripts/run_study.py --phase all --models mean ridge elasticnet xgboost --seeds 11 22 33 44 55
  Deep models: use notebooks/colab_deep_learning.ipynb, or pass --device cuda from a GPU runtime.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import argparse
from pathlib import Path
import pandas as pd
from battery_soh.data.schema import harmonize
from battery_soh.experiments.engine import benchmark, ALL_MODELS
from battery_soh.experiments.adaptation import adapt_pair
from battery_soh.evaluation.plots import make_figures

p=argparse.ArgumentParser()
p.add_argument('--input',default='data/processed/unified.csv')
p.add_argument('--phase',choices=['benchmark','adaptation','all'],default='benchmark')
p.add_argument('--models',nargs='+',default=list(ALL_MODELS))
p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
p.add_argument('--features',nargs='+',default=['cycle'])
p.add_argument('--window',type=int,default=4)
p.add_argument('--epochs',type=int,default=30)
p.add_argument('--device',default='cpu')
p.add_argument('--allow-cpu-deep',action='store_true',help='Allow slow deep-learning runs on CPU; not recommended')
p.add_argument('--output',default='results/full_study')
args=p.parse_args()
if set(args.models) & {'lstm','cnn','transformer'} and args.device == 'cpu' and not args.allow_cpu_deep:
    raise SystemExit('Deep models are configured for Google Colab GPU. Open notebooks/colab_deep_learning.ipynb, or pass --device cuda. For an intentional slow CPU run, add --allow-cpu-deep.')
df=harmonize(pd.read_csv(args.input, low_memory=False))
if df.dataset.nunique()!=5: raise SystemExit('The report requires exactly five confirmed source datasets.')
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
deep_cfg={'epochs':args.epochs,'device':args.device}
if args.phase in ('benchmark','all'):
    for mode in ('matrix','lodo'):
        folder=out/mode
        benchmark(df,experiment=mode,models=args.models,seeds=args.seeds,
                  features=args.features,window=args.window,deep_config=deep_cfg,output=folder)
        make_figures(folder)
if args.phase in ('adaptation','all'):
    all_results=[]
    names=sorted(df.dataset.unique())
    for src in names:
        for dst in names:
            if src==dst: continue
            for name in args.models:
                for seed in args.seeds:
                    folder=out/'adaptation'/f'{src}_to_{dst}'/name/f'seed_{seed}'
                    r=adapt_pair(df,source=src,target=dst,model=name,seed=seed,
                                 features=args.features,window=args.window,
                                 deep_config=deep_cfg,output=folder)
                    all_results.append(r)
    aggregate=out/'adaptation'
    combined=pd.concat(all_results,ignore_index=True)
    combined.to_csv(aggregate/'adaptation_metrics.csv',index=False)
    make_figures(aggregate)
print('Completed requested runs. Inspect metrics.csv and failures.csv, and document infeasible settings.')
