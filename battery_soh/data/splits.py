"""Splits are by full battery, never by cycling observation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def battery_split(df: pd.DataFrame, seed: int, test_fraction: float=.25,
                  val_fraction: float=.2):
    ids = df['dataset'].astype(str) + '::' + df['battery_id'].astype(str)
    batteries = ids.unique()
    if len(batteries) < 3:
        raise ValueError(f'Need >=3 distinct batteries for train/val/test; got {len(batteries)}')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(batteries)
    n_test = max(1, round(test_fraction * len(batteries)))
    n_val = max(1, round(val_fraction * (len(batteries)-n_test)))
    if len(batteries)-n_test-n_val < 1:
        raise ValueError('Not enough training batteries')
    test_ids, val_ids, train_ids = shuffled[:n_test], shuffled[n_test:n_test+n_val], shuffled[n_test+n_val:]
    a, b, c = (df.loc[ids.isin(group)].copy().reset_index(drop=True) for group in (train_ids, val_ids, test_ids))
    assert not ({*train_ids} & {*val_ids} or {*train_ids} & {*test_ids} or {*val_ids} & {*test_ids})
    return a,b,c


def target_support_test(df: pd.DataFrame, seed: int, max_support: int=10,
                        test_fraction: float=.3):
    """Reserve target test cells first. Return support pool and unseen test set."""
    batteries = df.battery_id.unique()
    if len(batteries) < 3: raise ValueError('Need >=3 target batteries')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(batteries)
    n_test = max(2, int(np.ceil(test_fraction * len(batteries))))
    n_test = min(n_test, len(batteries)-1)
    test_set = set(shuffled[:n_test])
    return (df.loc[~df.battery_id.isin(test_set)].copy().reset_index(drop=True),
            df.loc[df.battery_id.isin(test_set)].copy().reset_index(drop=True))
