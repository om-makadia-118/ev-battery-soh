"""Native battery dataset readers with source-specific capacity extraction.

CALCE Arbin and CADEX logs, XJTU standardized pickles, NASA MAT files, Oxford
MAT files and MATR HDF5/MAT files use separate validated readers. Never infer
capacity from an arbitrary current trace without charge/discharge segmentation.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .schema import DATASET_SPECS, REQUIRED, OPTIONAL_NUMERIC, harmonize

ALIASES = {
    'battery_id': ['battery_id', 'cell_id', 'cell', 'battery', 'batteryname'],
    'cycle': ['cycle', 'cycle_index', 'cycle_number', 'cyclenumber', 'cyc'],
    'capacity_ah': ['capacity_ah', 'discharge_capacity_ah', 'qd_ah', 'capacity'],
    'temperature_c': ['temperature_c', 'ambient_temperature', 'temperature'],
    'charge_rate_c': ['charge_rate_c', 'charge_c_rate'],
}

def read_table(path: str | Path, *, sheet=0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in ('.pkl', '.pickle'):
        # Only unpickle files downloaded from a trusted dataset source.
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise ValueError(f'{path}: pickle contains {type(obj).__name__}, not a DataFrame; use inspect first')
    raise ValueError(f'Unsupported table format {path.name}')

def mapped_table(path: str | Path, dataset: str, *, column_map: dict | None = None,
                 capacity_unit: str = 'Ah', rated_capacity_ah: float | None = None,
                 battery_id: str | None = None, sheet=0,
                 cycle_from_row: bool = False, battery_id_from_filename: bool = False) -> pd.DataFrame:
    """Map a verified per-cycle discharge-capacity table to canonical schema.

    column_map maps canonical names -> exact source column names. `capacity_unit`
    is mandatory/explicit to prevent silent 1000x Ah-vs-mAh errors.
    """
    dataset = dataset.lower()
    raw = read_table(path, sheet=sheet)
    cols = {str(c).strip().lower(): c for c in raw.columns}
    rename = {}
    extra_columns = {field: [field] for field in OPTIONAL_NUMERIC if field not in ALIASES}
    for canonical, aliases in {**ALIASES, **extra_columns}.items():
        src = (column_map or {}).get(canonical)
        if src is None:
            for alias in aliases:
                if alias in cols:
                    src = cols[alias]
                    break
        if src is not None:
            if src not in raw:
                raise ValueError(f'{path}: mapped column {src!r} is absent. Available: {list(raw.columns)}')
            rename[src] = canonical
    df = raw.rename(columns=rename).copy()
    if cycle_from_row and 'cycle' not in df:
        df['cycle'] = np.arange(1, len(df)+1)
    if 'capacity_ah' not in df or 'cycle' not in df:
        raise ValueError(f'{path}: need a cycle-level DISCHARGE capacity and cycle number. '
                         f'Columns: {list(raw.columns)}. Map them via loaders.json.')
    if 'battery_id' not in df:
        if battery_id is None and battery_id_from_filename:
            battery_id = Path(path).stem
        if battery_id is None:
            raise ValueError(f'{path}: no battery_id column; supply battery_id in loaders.json')
        df['battery_id'] = battery_id
    unit = capacity_unit.lower()
    if unit not in ('ah', 'mah'):
        raise ValueError('capacity_unit must be Ah or mAh')
    df['capacity_ah'] = pd.to_numeric(df['capacity_ah'], errors='coerce') / (1000 if unit == 'mah' else 1)
    df['rated_capacity_ah'] = rated_capacity_ah if rated_capacity_ah is not None else DATASET_SPECS[dataset]['rated_capacity_ah']
    df['dataset'] = dataset
    df['chemistry'] = DATASET_SPECS.get(dataset, {}).get('chemistry')
    df['source_file'] = str(path)
    return df

def _attr(obj, key, default=None):
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def _scalar(x):
    a = np.asarray(x).reshape(-1)
    return float(a[0]) if a.size else float('nan')

def nasa_mat(path: str | Path) -> pd.DataFrame:
    """NASA PCoE Battery Data Set: top-level Bxxxx.cycle operations."""
    path = Path(path)
    mat = loadmat(path, simplify_cells=True)
    root = mat.get(path.stem)
    if root is None:
        keys = [k for k in mat if re.match(r'^B\d+', k)]
        root = mat[keys[0]] if keys else None
    if root is None:
        raise ValueError(f'{path}: NASA Bxxxx structure not found: {list(mat)}')
    operations = _attr(root, 'cycle', [])
    if isinstance(operations, dict): operations = [operations]
    out = []
    discharge_idx = 0
    last_charge = {}
    for operation in np.atleast_1d(operations).flat:
        op_type = str(_attr(operation, 'type', '')).strip().lower()
        if op_type == 'charge':
            ch = _attr(operation, 'data', {})
            times = np.asarray(_attr(ch, 'Time', []), dtype=float).ravel()
            volts = np.asarray(_attr(ch, 'Voltage_measured', []), dtype=float).ravel()
            current = np.asarray(_attr(ch, 'Current_measured', []), dtype=float).ravel()
            last_charge = {
                'charge_duration_s': float(np.nanmax(times)-np.nanmin(times)) if times.size else np.nan,
                'charge_voltage_mean_v': float(np.nanmean(volts)) if volts.size else np.nan,
                'charge_current_mean_a': float(np.nanmean(np.abs(current))) if current.size else np.nan,
            }
            continue
        if op_type != 'discharge':
            continue
        discharge_idx += 1
        data = _attr(operation, 'data', {})
        cap = _attr(data, 'Capacity')
        if cap is None or not np.isfinite(_scalar(cap)):
            continue
        voltage = np.asarray(_attr(data, 'Voltage_measured', []), dtype=float).ravel()
        temperature = np.asarray(_attr(data, 'Temperature_measured', []), dtype=float).ravel()
        out.append({'dataset': 'nasa', 'battery_id': path.stem, 'cycle': discharge_idx,
                    'capacity_ah': _scalar(cap), 'rated_capacity_ah': 2.0,
                    'temperature_c': float(np.nanmean(temperature)) if temperature.size else np.nan,
                    'discharge_voltage_mean_v': float(np.nanmean(voltage)) if voltage.size else np.nan,
                    **last_charge, 'chemistry': 'LCO', 'source_file': str(path)})
        last_charge = {}
    if not out: raise ValueError(f'{path}: no discharge events with Capacity found')
    return pd.DataFrame(out)

def _oxford_records_from_mapping(path: Path, cells: dict) -> pd.DataFrame:
    """Convert CellN/cycNNNN/C1dc structures to observed capacity records."""
    out = []
    for key, cell in cells.items():
        if not re.fullmatch(r'Cell\d+', str(key), flags=re.I):
            continue
        if not isinstance(cell, dict):
            continue
        for cyc_name, characterization in cell.items():
            match = re.match(r'cyc(\d+)$', str(cyc_name), re.I)
            if not match or not isinstance(characterization, dict):
                continue
            dc = _attr(characterization, 'C1dc')
            if not isinstance(dc, dict):
                continue
            q = np.asarray(_attr(dc, 'q', []), dtype=float).ravel()
            q = q[np.isfinite(q)]
            if q.size < 2:
                continue
            capacity_ah = float(np.ptp(q) / 1000.0)  # q is mAh in Oxford release
            if not np.isfinite(capacity_ah) or capacity_ah <= 0:
                continue
            ch = _attr(characterization, 'C1ch', {})
            cht = np.asarray(_attr(ch, 't', []), dtype=float).ravel()
            chv = np.asarray(_attr(ch, 'v', []), dtype=float).ravel()
            v = np.asarray(_attr(dc, 'v', []), dtype=float).ravel()
            temp = np.asarray(_attr(dc, 'T', []), dtype=float).ravel()
            out.append({
                'dataset': 'oxford', 'battery_id': str(key),
                'cycle': int(match.group(1)), 'capacity_ah': capacity_ah,
                'rated_capacity_ah': 0.74,
                'discharge_voltage_mean_v': float(np.nanmean(v)) if v.size else np.nan,
                'temperature_c': float(np.nanmean(temp)) if temp.size else np.nan,
                'charge_duration_s': float(np.nanmax(cht)-np.nanmin(cht)) if cht.size else np.nan,
                'charge_voltage_mean_v': float(np.nanmean(chv)) if chv.size else np.nan,
                'chemistry': 'LCO', 'source_file': str(path),
            })
    if not out:
        raise ValueError(f'{path}: no CellN/cycNNNN/C1dc/q records found')
    return pd.DataFrame(out)

def _h5_resolve(f, node, depth=0):
    """Resolve the small MATLAB v7.3 struct subset needed by the Oxford reader."""
    import h5py
    if depth > 12:
        raise ValueError('Unexpectedly deep MATLAB HDF5 reference nesting')
    if isinstance(node, h5py.Reference):
        return _h5_resolve(f, f[node], depth + 1) if node else None
    if isinstance(node, h5py.Group):
        return {k: _h5_resolve(f, v, depth + 1)
                for k, v in node.items() if k != '#refs#'}
    if isinstance(node, h5py.Dataset):
        raw = node[()]
        if isinstance(raw, np.ndarray) and (raw.dtype.kind == 'O' or h5py.check_dtype(ref=raw.dtype) is not None):
            values = [_h5_resolve(f, ref, depth + 1) if ref else None for ref in raw.ravel()]
            return values[0] if len(values) == 1 else values
        return np.asarray(raw).squeeze()
    return node

def oxford_mat(path: str | Path) -> pd.DataFrame:
    """Oxford Dataset 1 reader for MATLAB v5 and v7.3 files.

    Only actual C1 discharge characterization checkpoints are emitted. The
    Oxford q vector is in mAh; its range is converted to Ah.
    """
    path = Path(path)
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            cells = {}
            for key in f.keys():
                if re.fullmatch(r'Cell\d+', key, flags=re.I):
                    cells[key] = _h5_resolve(f, f[key])
            return _oxford_records_from_mapping(path, cells)
    mat = loadmat(path, simplify_cells=True)
    cells = {k: v for k, v in mat.items() if re.fullmatch(r'Cell\d+', k, flags=re.I)}
    return _oxford_records_from_mapping(path, cells)

def _inspect_pickle(path: Path) -> str:
    """Summarize a trusted research pickle without dumping large arrays."""
    import pandas as pd
    obj = pd.read_pickle(path)  # Pickle can execute code: only use trusted sources.
    lines = [f'File: {path.name}', f'Top-level type: {type(obj).__name__}']

    def describe(value, label='root', depth=0):
        indent = '  ' * depth
        if isinstance(value, pd.DataFrame):
            lines.append(f'{indent}{label}: DataFrame shape={value.shape}')
            lines.append(f'{indent}columns={list(value.columns)}')
            lines.append(value.head(3).to_string(index=False))
        elif isinstance(value, pd.Series):
            lines.append(f'{indent}{label}: Series length={len(value)}, dtype={value.dtype}')
            lines.append(value.head(5).to_string())
        elif isinstance(value, dict):
            keys = list(value.keys())
            lines.append(f'{indent}{label}: dict keys={keys[:30]}')
            if depth < 2:
                for key in keys[:12]:
                    describe(value[key], str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            lines.append(f'{indent}{label}: {type(value).__name__} length={len(value)}')
            if value and depth < 2:
                describe(value[0], '[0]', depth + 1)
        elif isinstance(value, np.ndarray):
            lines.append(f'{indent}{label}: ndarray shape={value.shape}, dtype={value.dtype}')
            if value.size and value.dtype.kind in 'biuf':
                flat = value.ravel()
                lines.append(f'{indent}sample={flat[:8].tolist()}')
        else:
            text = repr(value)
            lines.append(f'{indent}{label}: {type(value).__name__} {text[:240]}')

    describe(obj)
    return '\n'.join(lines)

def _numeric_values(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)]

def xjtu_pickle(path: str | Path) -> pd.DataFrame:
    """Read a BatteryLife-standardized XJTU cell pickle.

    Expected top-level keys include cell_id, cycle_data and
    nominal_capacity_in_Ah. Each cycle_data entry contains a cumulative
    discharge_capacity_in_Ah trace. The observed capacity label is its range.
    """
    path = Path(path)
    data = pd.read_pickle(path)  # Only load trusted, official dataset pickles.
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a dict pickle, got {type(data).__name__}')
    required = {'cycle_data', 'nominal_capacity_in_Ah'}
    missing = required - set(data)
    if missing:
        raise ValueError(
            f'{path}: not a recognized BatteryLife standardized pickle; missing {sorted(missing)}. '
            'Run `python -m battery_soh inspect --path <file>` and share its output.'
        )
    cycle_data = data['cycle_data']
    if isinstance(cycle_data, dict):
        cycle_items = list(cycle_data.values())
    elif isinstance(cycle_data, (list, tuple)):
        cycle_items = list(cycle_data)
    else:
        raise ValueError(f'{path}: cycle_data must be a list or dict, got {type(cycle_data).__name__}')
    battery_id = str(data.get('cell_id') or path.stem)
    rated = float(np.asarray(data['nominal_capacity_in_Ah']).reshape(-1)[0])
    if not np.isfinite(rated) or rated <= 0:
        raise ValueError(f'{path}: invalid nominal_capacity_in_Ah={rated}')
    rows = []
    for idx, cycle in enumerate(cycle_items, start=1):
        if not isinstance(cycle, dict):
            continue
        qd = _numeric_values(cycle.get('discharge_capacity_in_Ah'))
        # BatteryLife stores cumulative discharge capacity through the cycle.
        # Keep the measured range; do not use the current SOH/target as a feature.
        qd = qd[qd >= 0]
        if qd.size < 2:
            continue
        capacity = float(np.max(qd) - np.min(qd))
        if not np.isfinite(capacity) or capacity <= 0:
            continue
        cycle_number = cycle.get('cycle_number', idx)
        cycle_number = float(np.asarray(cycle_number).reshape(-1)[0])
        temp = _numeric_values(cycle.get('temperature_in_C'))
        rows.append({
            'dataset': 'xjtu', 'battery_id': battery_id, 'cycle': cycle_number,
            'capacity_ah': capacity, 'rated_capacity_ah': rated,
            'temperature_c': float(np.mean(temp)) if temp.size else np.nan,
            'chemistry': 'NCM', 'source_file': str(path),
        })
    if not rows:
        raise ValueError(f'{path}: no usable discharge-capacity traces in cycle_data')
    return pd.DataFrame(rows)

def matr_mat(path: str | Path) -> pd.DataFrame:
    """MATR Severson HDF5 (.mat v7.3) batch summary (observed capacities).

    Identifiers are batch-file-specific; review duplicate continuing cells across
    batches before using the combined dataset in a publishable experiment.
    """
    import h5py
    path=Path(path)
    out=[]
    with h5py.File(path,'r') as f:
        if 'batch' not in f or 'summary' not in f['batch']:
            raise ValueError(f'{path}: missing batch/summary in MATR HDF5 file')
        refs=f['batch']['summary'][()]
        for idx,ref in enumerate(refs.reshape(-1)):
            summary=f[ref]
            def vect(name):
                if name not in summary: return np.array([])
                raw=summary[name][()]
                if isinstance(raw,np.ndarray) and raw.dtype.kind=='O':
                    arrays=[np.asarray(f[r][()]).ravel() for r in raw.ravel()]
                    return np.concatenate(arrays) if arrays else np.array([])
                return np.asarray(raw,dtype=float).ravel()
            q=vect('QDischarge')
            t=vect('Tavg')
            duration=vect('chargetime')
            cycles=vect('cycle')
            for j,c in enumerate(q):
                if not np.isfinite(c) or c<=0 or c>1.43: continue
                cyc=float(cycles[j]) if j<len(cycles) else j+1
                out.append({'dataset':'matr','battery_id':f'{path.stem}_cell{idx:03d}',
                            'cycle':cyc, 'capacity_ah':float(c), 'rated_capacity_ah':1.1,
                            'temperature_c':float(t[j]) if j<len(t) else np.nan,
                            'charge_duration_s':float(duration[j]) * 60 if j<len(duration) else np.nan,
                            'chemistry':'LFP','source_file':str(path)})
    if not out: raise ValueError(f'{path}: no QDischarge cycle records read')
    return pd.DataFrame(out)



CALCE_PROTOCOLS = {
    3: 'Type 3: alternating discharge current',
    5: 'Type 5: low-regime partial cycling with full-capacity checks',
    6: 'Type 6: high-regime partial cycling with full-capacity checks',
    7: 'Type 4: variable discharge cutoff',
    8: 'Type 1: 0.5C discharge',
    21: 'Type 1: 0.5C discharge',
    33: 'Type 1: 0.5C discharge',
    34: 'Type 1: 0.5C discharge',
    35: 'Type 2: 1C discharge',
    36: 'Type 2: 1C discharge',
    37: 'Type 2: 1C discharge',
    38: 'Type 2: 1C discharge',
    9: 'Type 3: alternating discharge current',
    24: 'Type 6: high-regime partial cycling with full-capacity checks',
    25: 'Type 6: high-regime partial cycling with full-capacity checks',
}

def _calce_battery_id(path: Path) -> str:
    """Infer CS2 cell identity from the CALCE filename or its parent folders."""
    candidates = [path.stem, *reversed(path.parts[:-1])]
    for value in candidates:
        match = re.search(r'CS2[_-]?(\d+)', str(value), flags=re.I)
        if match:
            return f'CS2_{int(match.group(1))}'
    raise ValueError(
        f'{path}: cannot infer a CS2 battery ID. Keep the original CALCE filename '
        'or place the file inside a folder named CS2_<cell number>.'
    )

def _calce_channel_frames(path: Path) -> pd.DataFrame:
    """Read all Arbin Channel_* sheets from one CALCE workbook."""
    book = pd.ExcelFile(path)
    channel_sheets = [s for s in book.sheet_names if str(s).lower().startswith('channel_')]
    if not channel_sheets:
        raise ValueError(
            f'{path}: no Channel_* sheet found. Expected an Arbin workbook with '
            f'Info, Channel_* and optionally Statistics_* sheets; found {book.sheet_names}.'
        )
    frames = []
    for sheet in channel_sheets:
        frame = pd.read_excel(book, sheet_name=sheet)
        frame.columns = [str(c).strip() for c in frame.columns]
        required = {
            'Date_Time', 'Cycle_Index', 'Current(A)', 'Voltage(V)',
            'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{path} / {sheet}: missing required columns {sorted(missing)}')
        frame['_source_file'] = str(path)
        frame['_channel_sheet'] = str(sheet)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def calce_excel_file(path: str | Path, *, rated_capacity_ah: float = 1.1, _raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extract full-discharge capacity checks from one CALCE Arbin workbook.

    Arbin capacity columns are cumulative counters, so the per-cycle capacity is
    the maximum discharge-capacity counter during the discharge step minus the
    minimum counter observed in that cycle. Partial-discharge cycling is excluded
    from SOH labels: a qualifying check must include both a full-charge voltage
    (>=4.15 V somewhere in the cycle) and a discharge endpoint (<=2.85 V).
    """
    path = Path(path)
    battery_id = _calce_battery_id(path)
    raw = _calce_channel_frames(path) if _raw is None else _raw.copy()
    raw['Date_Time'] = pd.to_datetime(raw['Date_Time'], errors='coerce')
    for col in ('Cycle_Index', 'Current(A)', 'Voltage(V)',
                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                'Test_Time(s)', 'Charge_Energy(Wh)', 'Discharge_Energy(Wh)',
                'Internal_Resistance(Ohm)'):
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=['Cycle_Index', 'Current(A)', 'Voltage(V)',
                             'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'])
    if raw.empty:
        raise ValueError(f'{path}: no numeric channel measurements found')
    sort_cols = ['Date_Time']
    if 'Test_Time(s)' in raw:
        sort_cols.append('Test_Time(s)')
    sort_cols.extend(['_source_file', 'Data_Point'] if 'Data_Point' in raw else ['_source_file'])
    raw = raw.sort_values(sort_cols, kind='stable').reset_index(drop=True)

    # CALCE distributes one cell's test history over date-named workbooks.
    # Cycle_Index may restart at 1 in a later workbook. Rebase each decrease
    # chronologically so cycle numbers remain unique across the full history.
    local_cycles = raw['Cycle_Index'].to_numpy(dtype=float)
    global_cycles = np.empty(len(raw), dtype=float)
    offset = 0.0
    last_global = 0.0
    previous_local = None
    for i, local in enumerate(local_cycles):
        if previous_local is not None and local < previous_local:
            offset = last_global
        global_value = local + offset
        global_cycles[i] = global_value
        last_global = global_value
        previous_local = local
    raw['_global_cycle'] = global_cycles

    cell_number = int(re.search(r'_(\d+)$', battery_id).group(1))
    protocol_type = {
        3: 3, 9: 3, 7: 4, 5: 5, 6: 5, 24: 6, 25: 6,
        8: 1, 21: 1, 33: 1, 34: 1, 35: 2, 36: 2, 37: 2, 38: 2,
    }.get(cell_number)
    protocol_name = CALCE_PROTOCOLS.get(cell_number, 'CALCE CS2 protocol')
    rows = []
    for cycle, group in raw.groupby('_global_cycle', sort=True):
        current = group['Current(A)']
        voltage = group['Voltage(V)']
        discharge = group.loc[current < -0.01]
        charge = group.loc[current > 0.01]
        if discharge.empty:
            continue

        v_cycle = voltage.dropna()
        if v_cycle.empty:
            continue
        v_cycle_max = float(v_cycle.max())

        step_starts = (
            discharge['Step_Index'].ne(discharge['Step_Index'].shift())
            if 'Step_Index' in discharge else pd.Series(False, index=discharge.index)
        )
        dis_runs = step_starts.cumsum()

        for _, dis_step in discharge.groupby(dis_runs, sort=True):
            discharge_q = dis_step['Discharge_Capacity(Ah)'].dropna()
            if discharge_q.empty:
                continue
            qd_before = group.loc[group.index < dis_step.index[0], 'Discharge_Capacity(Ah)'].dropna()
            start_q = float(qd_before.iloc[-1]) if not qd_before.empty else float(discharge_q.min())
            capacity_ah = float(discharge_q.max() - start_q)
            if not np.isfinite(capacity_ah) or capacity_ah <= 0.05 or capacity_ah > rated_capacity_ah * 1.3:
                continue

            v_dis = dis_step['Voltage(V)'].dropna()
            if v_dis.empty:
                continue
            v_min = float(v_dis.min())

            # Type 5/6 deliberately contain partial cycling. Keep only explicit
            # full-capacity reference checks, not partial operating cycles.
            if v_cycle_max < 4.10 or v_min > 2.85:
                continue

            def duration(frame):
                if 'Test_Time(s)' not in frame or frame.empty:
                    return np.nan
                values = frame['Test_Time(s)'].dropna()
                return float(values.max() - values.min()) if len(values) else np.nan

            resistance = group.get('Internal_Resistance(Ohm)', pd.Series(dtype=float))
            resistance = resistance[(resistance > 0) & np.isfinite(resistance)]
            rows.append({
                'dataset': 'calce',
                'battery_id': battery_id,
                'cycle': int(round(cycle)),
                'capacity_ah': capacity_ah,
                'rated_capacity_ah': float(rated_capacity_ah),
                'temperature_c': np.nan,
                'charge_rate_c': (
                    float(np.percentile(
                        charge.loc[charge['Current(A)'] > 0.01, 'Current(A)'].abs(), 95
                    ) / rated_capacity_ah)
                    if not charge.empty and (charge['Current(A)'] > 0.01).any()
                    else np.nan
                ),
                'charge_duration_s': duration(charge),
                'charge_voltage_mean_v': float(charge['Voltage(V)'].mean()) if not charge.empty else np.nan,
                'charge_current_mean_a': float(charge['Current(A)'].mean()) if not charge.empty else np.nan,
                'discharge_voltage_mean_v': float(v_dis.mean()),
                'discharge_duration_s': duration(dis_step),
                'discharge_current_mean_a': float(dis_step['Current(A)'].mean()),
                'discharge_voltage_min_v': v_min,
                'discharge_voltage_max_v': float(v_dis.max()),
                'internal_resistance_ohm': float(resistance.median()) if len(resistance) else np.nan,
                'discharge_energy_wh': (
                    float(dis_step['Discharge_Energy(Wh)'].max() - dis_step['Discharge_Energy(Wh)'].min())
                    if 'Discharge_Energy(Wh)' in dis_step and dis_step['Discharge_Energy(Wh)'].notna().any()
                    else np.nan
                ),
                'protocol_type': f'Type {protocol_type}' if protocol_type else 'Unknown',
                'protocol_name': protocol_name,
                'chemistry': 'LCO',
                'source_file': ';'.join(sorted(group['_source_file'].astype(str).unique())),
            })
    if not rows:
        raise ValueError(
            f'{path}: no full-discharge capacity checks found. Check the channel data, '
            'current sign, voltage cutoffs and cycle-index continuity.'
        )
    out_df = pd.DataFrame(rows)
    if out_df.duplicated('cycle').any():
        out_df['cycle'] = np.arange(1, len(out_df) + 1)
    return out_df


def _calce_cadex_frame(path: Path) -> pd.DataFrame:
    """Read a tab-separated CADEX log and validate its measurement columns."""
    raw = pd.read_csv(path, sep='\t')
    raw.columns = [str(c).replace('\ufeff', '').strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.astype(str).str.match(r'^Unnamed')]
    required = {
        'Status category', 'mV', 'mA', 'Temperature', 'Duration',
        'Charge count', 'Discharge count', 'Capacity',
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f'{path}: not a recognized CALCE CADEX tab-separated log; '
            f'missing columns {sorted(missing)}. Found {list(raw.columns)}'
        )
    for col in required | {'Time', 'Status code', 'Pgm step', 'Pgm para', 'Pgm cycle'}:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=[
        'Status category', 'mV', 'mA', 'Duration', 'Discharge count'
    ]).copy()
    if raw.empty:
        raise ValueError(f'{path}: no numeric CADEX measurements found')
    raw = raw.sort_values('Duration', kind='stable').reset_index(drop=True)
    return raw


def calce_cadex_file(
    path: str | Path, *, rated_capacity_ah: float = 1.1, allow_empty: bool = False
) -> pd.DataFrame:
    """Extract full discharge capacity from one CALCE CADEX .txt log.

    CADEX records current in mA, voltage in mV and elapsed time in seconds.
    The `Capacity` field in this release behaves like a 0--100 step-progress
    indicator; it is NOT an Ah capacity measurement. Actual Ah is
    calculated by integrating |current| over elapsed time. Full discharge steps
    must start from a charged state (>=3.8 V) and discharge to cutoff (<=3.0 V).
    """
    path = Path(path)
    battery_id = _calce_battery_id(path)
    cell_number = int(re.search(r'_(\d+)$', battery_id).group(1))
    protocol_type = {
        8: 1, 21: 1,
    }.get(cell_number)
    protocol_name = CALCE_PROTOCOLS.get(cell_number, 'CALCE CS2 protocol')
    raw = _calce_cadex_frame(path)

    is_discharge = (
        raw['Status category'].eq(6)
        & raw['Discharge count'].gt(0)
    )
    if not is_discharge.any():
        if allow_empty:
            return pd.DataFrame()
        raise ValueError(f'{path}: no discharge records found')

    # A new segment begins when the status changes into discharge or the
    # tester's discharge counter changes. This avoids grouping unrelated
    # steps solely because they share a counter value.
    previous_discharge = is_discharge.shift(fill_value=False)
    previous_count = raw['Discharge count'].shift()
    starts = is_discharge & (
        ~previous_discharge | raw['Discharge count'].ne(previous_count)
    )
    raw['_discharge_run'] = starts.cumsum()
    rows = []

    for _, discharge in raw.loc[is_discharge].groupby('_discharge_run', sort=True):
        discharge = discharge.sort_values('Duration', kind='stable')
        local_cycle = float(discharge['Discharge count'].iloc[0])
        if not np.isfinite(local_cycle) or local_cycle <= 0:
            continue

        seconds = discharge['Duration'].to_numpy(dtype=float)
        current_a = np.abs(discharge['mA'].to_numpy(dtype=float)) / 1000.0
        voltage_v = discharge['mV'].to_numpy(dtype=float) / 1000.0
        valid = np.isfinite(seconds) & np.isfinite(current_a) & np.isfinite(voltage_v)
        seconds, current_a, voltage_v = seconds[valid], current_a[valid], voltage_v[valid]
        if len(seconds) < 2:
            continue
        order = np.argsort(seconds, kind='stable')
        seconds, current_a, voltage_v = seconds[order], current_a[order], voltage_v[order]
        # Drop repeated timestamps; integration over zero-width intervals adds
        # nothing, but unique timestamps make the numerical behavior explicit.
        keep = np.r_[True, np.diff(seconds) > 0]
        seconds, current_a, voltage_v = seconds[keep], current_a[keep], voltage_v[keep]
        if len(seconds) < 2:
            continue

        v_min = float(np.min(voltage_v))
        v_max = float(np.max(voltage_v))
        dur_s = float(seconds[-1] - seconds[0])
        if v_min > 3.0 or v_max < 3.8 or dur_s < 300:
            continue

        capacity_ah = float(
            np.sum((current_a[:-1] + current_a[1:]) * 0.5 * np.diff(seconds)) / 3600.0
        )
        if not np.isfinite(capacity_ah) or capacity_ah <= 0.05 or capacity_ah > rated_capacity_ah * 1.3:
            continue

        start_s = float(seconds[0])
        charge_candidates = raw.loc[
            raw['Status category'].eq(3)
            & raw['mA'].gt(0)
            & raw['Duration'].le(start_s)
        ]
        charge = pd.DataFrame()
        if not charge_candidates.empty:
            charge_count = charge_candidates['Charge count'].iloc[-1]
            charge = charge_candidates.loc[
                charge_candidates['Charge count'].eq(charge_count)
            ]
        charge_current_a = (
            charge['mA'].to_numpy(dtype=float) / 1000.0 if not charge.empty
            else np.array([], dtype=float)
        )
        charge_current_a = charge_current_a[np.isfinite(charge_current_a)]
        charge_rate_c = (
            float(np.percentile(charge_current_a, 95) / rated_capacity_ah)
            if charge_current_a.size and rated_capacity_ah > 0 else np.nan
        )
        charge_duration = (
            float(charge['Duration'].max() - charge['Duration'].min())
            if not charge.empty else np.nan
        )
        discharge_duration = dur_s
        energy_wh = float(
            np.sum(
                (
                    (current_a[:-1] * voltage_v[:-1])
                    + (current_a[1:] * voltage_v[1:])
                ) * 0.5 * np.diff(seconds)
            ) / 3600.0
        )
        rows.append({
            'dataset': 'calce',
            'battery_id': battery_id,
            'cycle': int(round(local_cycle)),
            'capacity_ah': capacity_ah,
            'rated_capacity_ah': float(rated_capacity_ah),
            'temperature_c': float(pd.to_numeric(
                discharge['Temperature'], errors='coerce'
            ).mean()),
            'charge_rate_c': charge_rate_c,
            'charge_duration_s': charge_duration,
            'charge_voltage_mean_v': (
                float(pd.to_numeric(charge['mV'], errors='coerce').mean() / 1000.0)
                if not charge.empty else np.nan
            ),
            'charge_current_mean_a': (
                float(pd.to_numeric(charge['mA'], errors='coerce').mean() / 1000.0)
                if not charge.empty else np.nan
            ),
            'discharge_voltage_mean_v': float(np.mean(voltage_v)),
            'discharge_duration_s': discharge_duration,
            'discharge_current_mean_a': -float(np.mean(current_a)),
            'discharge_voltage_min_v': v_min,
            'discharge_voltage_max_v': v_max,
            'discharge_energy_wh': energy_wh,
            'protocol_type': f'Type {protocol_type}' if protocol_type else 'Unknown',
            'protocol_name': protocol_name,
            'chemistry': 'LCO',
            'source_file': str(path),
        })

    if not rows:
        if allow_empty:
            return pd.DataFrame()
        raise ValueError(
            f'{path}: no completed CADEX discharge steps found. Expected status '
            'category 6 with a positive Discharge count and valid discharge cutoffs.'
        )
    return pd.DataFrame(rows)


def _calce_file_date(path: Path):
    """Return the date encoded as CS2_<cell>_<month>_<day>_<year>, if present."""
    match = re.search(
        r'CS2[_-]?\d+[_-](\d{1,2})[_-](\d{1,2})[_-](\d{2,4})',
        path.stem, flags=re.I
    )
    if not match:
        return None
    month, day, year = map(int, match.groups())
    if year < 100:
        year += 2000 if year < 70 else 1900
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def calce_cadex_folder(
    files: list[Path], *, rated_capacity_ah: float = 1.1
) -> pd.DataFrame:
    """Merge date-named CADEX logs for one CS2 cell and rebase reset counters."""
    ordered = sorted(
        files,
        key=lambda p: (_calce_file_date(p) is None,
                       _calce_file_date(p) if _calce_file_date(p) is not None else pd.Timestamp.max,
                       p.name.lower())
    )
    frames = [calce_cadex_file(p, rated_capacity_ah=rated_capacity_ah, allow_empty=True) for p in ordered]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(f'No completed CADEX discharge steps found in {files[0].parent}')
    output = []
    offset = 0
    previous_local = None
    last_global = 0
    for frame in frames:
        frame = frame.copy()
        local_cycles = frame['cycle'].astype(int).tolist()
        if local_cycles and previous_local is not None and local_cycles[0] <= previous_local:
            offset = last_global
        frame['cycle'] = frame['cycle'].astype(int) + offset
        if not frame.empty:
            last_global = int(frame['cycle'].max())
            previous_local = local_cycles[-1]
        output.append(frame)
    result = pd.concat(output, ignore_index=True)
    if result.duplicated(['dataset', 'battery_id', 'cycle']).any():
        result = result.drop_duplicates(subset=['dataset', 'battery_id', 'cycle'])
        result['cycle'] = np.arange(1, len(result) + 1)
    return result.sort_values(['battery_id', 'cycle']).reset_index(drop=True)


def calce_excel_folder(root: str | Path, *, rated_capacity_ah: float = 1.1) -> pd.DataFrame:
    """Load extracted CALCE CS2 Arbin workbooks and CADEX text logs."""
    root = Path(root)
    excel_files = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in ('.xlsx', '.xls')
        and not p.name.startswith('~$')
        and re.search(r'CS2[_-]?\d+', str(p), flags=re.I)
    )
    txt_files = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() == '.txt'
        and re.search(r'CS2[_-]?\d+', str(p), flags=re.I)
    )
    files_by_battery: dict[str, dict[str, list[Path]]] = {}
    for path in excel_files:
        battery_id = _calce_battery_id(path)
        files_by_battery.setdefault(battery_id, {'excel': [], 'txt': []})['excel'].append(path)
    for path in txt_files:
        battery_id = _calce_battery_id(path)
        files_by_battery.setdefault(battery_id, {'excel': [], 'txt': []})['txt'].append(path)

    if not files_by_battery:
        raise FileNotFoundError(
            f'No extracted CALCE CS2 Excel or CADEX text files found under {root}. '
            'Extract each CS2_*.zip into data/raw/calce/ first.'
        )

    frames = []
    for battery_id, grouped in sorted(files_by_battery.items()):
        if grouped['excel'] and grouped['txt']:
            raise ValueError(
                f'{battery_id}: both Arbin Excel and CADEX text files were found. '
                'Keep only the source format supplied for this cell.'
            )
        if grouped['excel']:
            raw_pieces = []
            for path in sorted(grouped['excel']):
                raw_pieces.append(_calce_channel_frames(path))
            combined_raw = pd.concat(raw_pieces, ignore_index=True)
            frames.append(calce_excel_file(
                Path(f'{battery_id}.xlsx'),
                rated_capacity_ah=rated_capacity_ah,
                _raw=combined_raw,
            ))
        elif grouped['txt']:
            cell_number = int(re.search(r'_(\d+)$', battery_id).group(1))
            if cell_number not in (8, 21):
                raise ValueError(
                    f'{battery_id}: CADEX .txt parser is currently validated only '
                    'for the CALCE CS2_8 and CS2_21 source format.'
                )
            frames.append(calce_cadex_folder(
                grouped['txt'], rated_capacity_ah=rated_capacity_ah
            ))

    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['dataset', 'battery_id', 'cycle']).any():
        duplicates = result[result.duplicated(['dataset', 'battery_id', 'cycle'], keep=False)]
        raise ValueError(
            'Duplicate CALCE cycle records after merging files. This may indicate '
            'overlapping date files or multiple tester channels. Inspect: '
            f'{duplicates[["battery_id", "cycle", "source_file"]].head(8).to_dict("records")}'
        )
    return result.sort_values(['battery_id', 'cycle']).reset_index(drop=True)


def inspect(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() in ('.pkl', '.pickle'):
        return _inspect_pickle(path)
    if path.suffix.lower() == '.mat':
        import h5py
        if h5py.is_hdf5(path):
            with h5py.File(path) as f:
                parts = [f'MATLAB v7.3 / HDF5: {path.name}']
                for key, value in f.items():
                    if isinstance(value, h5py.Group):
                        parts.append(f'{key}: group keys={list(value.keys())[:20]}')
                    else:
                        parts.append(f'{key}: dataset shape={value.shape}, dtype={value.dtype}')
                return '\n'.join(parts)
        m = loadmat(path, variable_names=None, simplify_cells=True)
        parts = []
        for key, value in m.items():
            if key.startswith('__'):
                continue
            if isinstance(value, dict):
                descr = f'dict keys={list(value)[:20]}'
            else:
                descr = f'type={type(value).__name__} shape={np.asarray(value).shape}'
            parts.append(f'{key}: {descr}')
        return '\n'.join(parts)
    if path.suffix.lower() == '.txt':
        raw = _calce_cadex_frame(path)
        parts = [f'File: {path.name}', 'Format: tab-separated CADEX log',
                 f'Rows: {len(raw)}', f'Columns: {list(raw.columns)}']
        parts.append(raw.head(5).to_string(index=False))
        parts.append(f"Discharge counts: {sorted(raw.loc[raw['Status category'].eq(6), 'Discharge count'].dropna().unique().tolist())}")
        return '\n'.join(parts)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        book = pd.ExcelFile(path)
        parts = [f'File: {path.name}', f'Sheets: {book.sheet_names}']
        for sheet in book.sheet_names:
            frame = pd.read_excel(book, sheet_name=sheet, nrows=3)
            parts.append(f'\n[{sheet}] columns={list(frame.columns)}')
            if not frame.empty:
                parts.append(frame.to_string(index=False))
        return '\n'.join(parts)
    df = read_table(path)
    return f'Rows: {len(df)}\nColumns: {list(df.columns)}\n\n{df.head(3).to_string(index=False)}'

def load_raw(root: str | Path, mapping_path: str | Path | None = None) -> pd.DataFrame:
    root = Path(root)
    cfg = json.loads(Path(mapping_path).read_text()) if mapping_path else {}
    frames = []
    for name in DATASET_SPECS:
        folder = root / name
        if not folder.exists(): continue
        if name == 'calce':
            calce_files = sorted(
                p for p in folder.rglob('*')
                if p.is_file() and p.suffix.lower() in ('.xlsx', '.xls', '.txt')
                and not p.name.startswith('~$')
                and re.search(r'CS2[_-]?\d+', str(p), flags=re.I)
            )
            if calce_files:
                frames.append(calce_excel_folder(folder))
                continue
        native_files = []
        if name == 'nasa': native_files = sorted(folder.rglob('B*.mat'))
        if name == 'oxford': native_files = sorted(folder.rglob('Oxford_Battery_Degradation_Dataset_1.mat'))
        if name == 'matr': native_files = sorted(folder.rglob('*batchdata*.mat'))
        if name == 'xjtu': native_files = sorted(folder.rglob('XJTU*.pkl'))
        for p in native_files:
            if name == 'nasa':
                frames.append(nasa_mat(p))
            elif name == 'oxford':
                frames.append(oxford_mat(p))
            elif name == 'matr':
                frames.append(matr_mat(p))
            elif name == 'xjtu':
                frames.append(xjtu_pickle(p))
        if native_files: continue  # Prefer native source and avoid duplicate CSV exports.
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() not in ('.csv','.xlsx','.xls','.parquet'): continue
            if p.name.startswith('~$'): continue
            # Explicit per-file config permits differing batteries and sheet headers.
            options = cfg.get(name, {}).get(p.name, cfg.get(name, {}).get('default', {}))
            try:
                frames.append(mapped_table(p, name, **options))
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Could not import {p}. Do not use raw tester logs as cycle summaries. {exc}') from exc
    if not frames:
        raise FileNotFoundError(f'No supported battery files under {root}. See docs/DATASETS.md')
    return harmonize(pd.concat(frames, ignore_index=True))
