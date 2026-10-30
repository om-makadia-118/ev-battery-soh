"""Run `python -m battery_soh --help`. Real runs NEVER create dummy data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data.loaders import inspect,load_raw
from .data.schema import harmonize,quality_report
from .data.demo import synthetic_datasets
from .experiments.engine import benchmark
from .experiments.adaptation import adapt_pair
from .evaluation.plots import make_figures


def get_parser():
    parser=argparse.ArgumentParser(prog='python -m battery_soh',description='EV battery cross-dataset SOH benchmark')
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('inspect',help='Print input columns or MATLAB root fields')
    p.add_argument('--path',required=True)
    p=subs.add_parser('prepare',help='Load, validate and harmonize real dataset inputs')
    p.add_argument('--raw',default='data/raw');p.add_argument('--mapping',default=None)
    p.add_argument('--out',default='data/processed/unified.csv')
    p.add_argument('--truncate-at-eol',action='store_true');p.add_argument('--eol',type=float,default=80.)
    p=subs.add_parser('demo',help='Run SYNTHETIC smoke test only, not research results')
    p.add_argument('--output',default='results/SYNTHETIC_DEMO')
    p=subs.add_parser('benchmark',help='Within / cross / LODO evaluation on existing data')
    p.add_argument('--input',default='data/processed/unified.csv')
    p.add_argument('--experiment',choices=['within','cross','matrix','lodo'],default='cross')
    p.add_argument('--models',nargs='+',default=['mean','ridge','elasticnet','xgboost'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--features',nargs='+',default=['cycle'])
    p.add_argument('--window',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='results/benchmark')
    p.add_argument('--strict',action='store_true',help='Abort on first failed experiment')
    p=subs.add_parser('adapt',help='Few-shot, pooled training, fine-tuning and CORAL')
    p.add_argument('--input',default='data/processed/unified.csv')
    p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--model',default='ridge');p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--counts',nargs='+',type=int,default=[0,1,3,5,10])
    p.add_argument('--features',nargs='+',default=['cycle']);p.add_argument('--window',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='results/adaptation')
    p=subs.add_parser('report',help='Create plots from saved experiment CSV outputs')
    p.add_argument('--input-dir',required=True)
    p.add_argument('--out',default=None)
    return parser

def main(argv=None):
    args=get_parser().parse_args(argv)
    if args.command=='inspect':
        print(inspect(args.path));return
    if args.command=='prepare':
        frame=load_raw(args.raw,args.mapping)
        frame=harmonize(frame,eol_threshold=args.eol,truncate_at_eol=args.truncate_at_eol)
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        if out.suffix.lower()=='.parquet': frame.to_parquet(out,index=False)
        elif out.suffix.lower()=='.csv': frame.to_csv(out,index=False)
        else: raise ValueError('Output extension must be .csv or .parquet')
        qr=quality_report(frame);qr.to_csv(out.with_name('data_quality.csv'),index=False)
        print(qr.to_string(index=False));print('Saved real prepared data:',out)
        print('IMPORTANT: validate rated capacities and chemistry per selected source collection.')
        return
    if args.command=='demo':
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        df=synthetic_datasets();df.to_csv(out/'synthetic_dataset.csv',index=False)
        print('SYNTHETIC SMOKE TEST — not real battery data; not usable for research claims.',flush=True)
        benchmark(df,experiment='cross',models=['mean','ridge','elasticnet'],
                  seeds=[42],features=['cycle'],output=out)
        make_figures(out)
        print('Synthetic check completed:',out)
        return
    if args.command=='report':
        print('Generated:',*make_figures(args.input_dir,args.out),sep='\n');return
    inp=Path(args.input)
    if not inp.exists(): raise FileNotFoundError(f'Missing {inp}. Run prepare after adding REAL battery files.')
    df=pd.read_parquet(inp) if inp.suffix=='.parquet' else pd.read_csv(inp)
    # Recalculate truth to reject untrusted saved SOH columns.
    df=harmonize(df)
    deep_config={'epochs':args.epochs,'device':args.device}
    if args.command=='benchmark':
        benchmark(df,experiment=args.experiment,models=args.models,seeds=args.seeds,
                  features=args.features,window=args.window,deep_config=deep_config,
                  output=args.output,strict=args.strict)
        make_figures(args.output)
    elif args.command=='adapt':
        parts=[]
        for seed in args.seeds:
            # Keep each seed in a separate folder; then aggregate all seed CSVs.
            subset=Path(args.output)/f'seed_{seed}'
            r=adapt_pair(df,source=args.source,target=args.target,model=args.model,
                         seed=seed,counts=args.counts,features=args.features,
                         window=args.window,deep_config=deep_config,output=subset)
            parts.append(r)
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        pd.concat(parts,ignore_index=True).to_csv(output/'adaptation_metrics.csv',index=False)
        make_figures(output)
        print('Saved adaptation result',output/'adaptation_metrics.csv')

if __name__=='__main__': main()
