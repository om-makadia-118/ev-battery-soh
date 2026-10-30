"""Leakage-aware feature extraction and chronological feature windows."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_OR_PROXY = {'soh', 'capacity_ah', 'rated_capacity_ah', 'eol_reached',
                   'battery_id', 'dataset', 'source_file', 'chemistry'}


def checked_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    if not features or set(features) & TARGET_OR_PROXY:
        raise ValueError(f'Feature list is empty or contains target/proxy/meta: {set(features) & TARGET_OR_PROXY}')
    missing = [f for f in features if f not in df.columns or not pd.api.types.is_numeric_dtype(df[f])]
    if missing:
        raise ValueError(f'Missing/non-numeric features: {missing}')
    if df[features].isna().any().any():
        breakdown = df.groupby('dataset')[features].apply(lambda a: a.isna().mean()).to_string()
        raise ValueError('Missing input observations. Use explicitly common features, not invented values.\n'+breakdown)
    return features


def tabular(df: pd.DataFrame, features: list[str]):
    checked_features(df, features)
    return df[features].to_numpy(np.float32), df.soh.to_numpy(np.float32)


def sequence_windows(df: pd.DataFrame, features: list[str], window: int):
    """Only past/current input measurements; never pads with future values or target.

    Returns X, y, corresponding row positions in the *supplied* frame.
    """
    checked_features(df, features)
    if window < 1: raise ValueError('window must be >=1')
    X, y, indices = [], [], []
    # work with stable positional indexes (caller indexes might not be contiguous)
    for (_, _), g in df.reset_index(drop=True).groupby(['dataset','battery_id'], sort=False):
        if not g.cycle.is_monotonic_increasing:
            raise ValueError('Sort per-battery cycle order before constructing windows')
        vals = g[features].to_numpy(np.float32)
        for j in range(window-1, len(g)):
            X.append(vals[j-window+1:j+1])
            y.append(float(g.soh.iloc[j]))
            indices.append(int(g.index[j]))
    if not X:
        return np.empty((0, window, len(features)), np.float32), np.empty(0,np.float32), np.empty(0,int)
    return np.stack(X), np.asarray(y,np.float32), np.asarray(indices,int)
