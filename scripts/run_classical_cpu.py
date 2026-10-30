"""Run the CPU-friendly classical baseline matrix on prepared real data.

Usage:
  python scripts/run_classical_cpu.py
  python scripts/run_classical_cpu.py --seeds 11 22 33 44 55
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from battery_soh.data.schema import harmonize
from battery_soh.experiments.engine import benchmark, CLASSICAL_NAMES
from battery_soh.evaluation.plots import make_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/unified.csv')
    parser.add_argument('--output', default='results/classical_cpu_matrix')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--features', nargs='+', default=['cycle'])
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f'Missing {path}. First run: python -m battery_soh prepare --raw data/raw')
    df = pd.read_csv(path)
    df = harmonize(df)
    datasets = sorted(df.dataset.unique())
    if len(datasets) < 2:
        raise SystemExit(f'Need at least two real datasets; found {datasets}')
    print('Datasets:', datasets)
    print('Batteries by dataset:')
    print(df.groupby('dataset').battery_id.nunique().to_string())
    print('\nRunning CPU classical models only:', list(CLASSICAL_NAMES))
    benchmark(
        df, experiment='matrix', models=list(CLASSICAL_NAMES), seeds=args.seeds,
        features=args.features, output=args.output, strict=False,
    )
    make_figures(args.output)
    print(f'Finished. Inspect {args.output}/failures.csv before interpreting metrics.')


if __name__ == '__main__':
    main()
