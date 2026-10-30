"""Canonical one-row-per-observed-capacity-check schema (NOT interpolated cycles)."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ('dataset', 'battery_id', 'cycle', 'capacity_ah', 'rated_capacity_ah')
OPTIONAL_NUMERIC = ('temperature_c', 'charge_rate_c', 'charge_duration_s',
                    'charge_voltage_mean_v', 'charge_current_mean_a',
                    'discharge_voltage_mean_v', 'discharge_duration_s',
                    'discharge_current_mean_a', 'discharge_voltage_min_v',
                    'discharge_voltage_max_v', 'internal_resistance_ohm',
                    'discharge_energy_wh')
META = ('chemistry', 'source_file')
DATASET_SPECS = {
    'nasa': {'rated_capacity_ah': 2.0, 'chemistry': 'LCO'},
    'calce': {'rated_capacity_ah': 1.1, 'chemistry': 'LCO'},
    'xjtu': {'rated_capacity_ah': 2.0, 'chemistry': 'NCM'},
    'oxford': {'rated_capacity_ah': 0.74, 'chemistry': 'LCO'},
    'matr': {'rated_capacity_ah': 1.1, 'chemistry': 'LFP'},
}

def harmonize(frame: pd.DataFrame, *, eol_threshold: float = 80.0,
              truncate_at_eol: bool = False) -> pd.DataFrame:
    """Validate units/keys, derive SOH; preserve only actually observed cycles."""
    df = frame.copy()
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing canonical columns: {sorted(missing)}')
    df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
    df['battery_id'] = df['battery_id'].astype(str).str.strip()
    if df['battery_id'].eq('').any():
        raise ValueError('Empty battery_id')
    for col in ('cycle', 'capacity_ah', 'rated_capacity_ah', *OPTIONAL_NUMERIC):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[list(REQUIRED[2:])].isna().any().any():
        raise ValueError('Cycle, capacity_ah and rated_capacity_ah must be numeric/non-null')
    if (df['cycle'] < 0).any() or (df['capacity_ah'] <= 0).any() or (df['rated_capacity_ah'] <= 0).any():
        raise ValueError('Invalid cycle or nonpositive battery capacity')
    if df.duplicated(['dataset', 'battery_id', 'cycle']).any():
        examples = df[df.duplicated(['dataset','battery_id','cycle'], keep=False)][list(REQUIRED[:3])].head(6)
        raise ValueError(f'Duplicate dataset/battery/cycle records; inspect: {examples.to_dict("records")}')
    if 'chemistry' not in df:
        df['chemistry'] = df['dataset'].map(lambda d: DATASET_SPECS.get(d, {}).get('chemistry'))
    if 'source_file' not in df:
        df['source_file'] = ''
    df = df.sort_values(['dataset', 'battery_id', 'cycle']).reset_index(drop=True)
    df['soh'] = 100.0 * df['capacity_ah'] / df['rated_capacity_ah']
    if not df['soh'].between(1, 130).all():
        raise ValueError('SOH outside 1–130%. Check Ah/mAh and rated-capacity assumptions.')
    df['eol_reached'] = df['soh'] <= eol_threshold
    if truncate_at_eol:
        # Keep all observations up to and including first crossing; never extrapolate.
        g = df.groupby(['dataset', 'battery_id'], sort=False)
        after = g['eol_reached'].transform(lambda s: s.cumsum().shift(fill_value=0) > 0)
        df = df.loc[~after].copy()
    return df.reset_index(drop=True)

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby('dataset'):
        row = {'dataset': name, 'batteries': g.battery_id.nunique(), 'observations': len(g),
               'min_cycle': int(g.cycle.min()), 'max_cycle': int(g.cycle.max()),
               'soh_min': round(float(g.soh.min()), 2), 'soh_max': round(float(g.soh.max()), 2),
               'ever_eol_batteries': g.loc[g.eol_reached, 'battery_id'].nunique()}
        row.update({f'coverage_{c}': round(float(g[c].notna().mean()), 3) for c in OPTIONAL_NUMERIC})
        rows.append(row)
    return pd.DataFrame(rows)
