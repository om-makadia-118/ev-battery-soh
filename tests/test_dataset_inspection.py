import numpy as np
import pandas as pd
import h5py

from battery_soh.data.loaders import inspect, oxford_mat


def test_inspect_trusted_pickle_dataframe(tmp_path):
    path = tmp_path / 'XJTU_3C_battery-1.pkl'
    pd.DataFrame({'cycle': [1, 2], 'capacity': [1.9, 1.8]}).to_pickle(path)
    text = inspect(path)
    assert 'DataFrame shape=(2, 2)' in text
    assert "'cycle'" in text
    assert 'capacity' in text


def test_oxford_hdf5_matlab_style_group(tmp_path):
    path = tmp_path / 'Oxford_Battery_Degradation_Dataset_1.mat'
    with h5py.File(path, 'w') as f:
        cell = f.create_group('Cell1')
        cyc = cell.create_group('cyc0000')
        dc = cyc.create_group('C1dc')
        dc.create_dataset('q', data=np.array([0.0, 250.0, 500.0, 740.0]))
        dc.create_dataset('v', data=np.array([4.1, 3.9, 3.7]))
        dc.create_dataset('T', data=np.array([40.0, 40.2, 40.1]))
        ch = cyc.create_group('C1ch')
        ch.create_dataset('t', data=np.array([0.0, 100.0, 200.0]))
        ch.create_dataset('v', data=np.array([3.5, 3.8, 4.2]))
    result = oxford_mat(path)
    assert len(result) == 1
    assert result.loc[0, 'battery_id'] == 'Cell1'
    assert result.loc[0, 'cycle'] == 0
    assert np.isclose(result.loc[0, 'capacity_ah'], .74)
    assert np.isclose(result.loc[0, 'temperature_c'], 40.1)


def test_oxford_hdf5_matlab_reference_dataset(tmp_path):
    path = tmp_path / 'Oxford_Battery_Degradation_Dataset_1.mat'
    with h5py.File(path, 'w') as f:
        refs = f.create_group('#refs#')
        cell_obj = refs.create_group('cell1')
        cyc = cell_obj.create_group('cyc0100')
        dc = cyc.create_group('C1dc')
        dc.create_dataset('q', data=np.array([0.0, 300.0, 740.0]))
        cell_ref = f.create_dataset('Cell1', shape=(1, 1), dtype=h5py.ref_dtype)
        cell_ref[0, 0] = cell_obj.ref
    result = oxford_mat(path)
    assert len(result) == 1
    assert result.loc[0, 'cycle'] == 100
    assert np.isclose(result.loc[0, 'capacity_ah'], .74)


def test_xjtu_batterylife_pickle_to_cycle_capacity(tmp_path):
    from battery_soh.data.loaders import xjtu_pickle
    path = tmp_path / 'XJTU_3C_battery-1.pkl'
    payload = {
        'cell_id': 'XJTU_3C_battery-1',
        'nominal_capacity_in_Ah': 2.0,
        'cycle_data': [
            {'cycle_number': 1, 'discharge_capacity_in_Ah': [0.0, 0.4, 0.9],
             'temperature_in_C': [25.0, 25.2]},
            {'cycle_number': 2, 'discharge_capacity_in_Ah': [0.0, 0.3, 0.85],
             'temperature_in_C': [25.0, 25.4]},
        ],
    }
    pd.to_pickle(payload, path)
    result = xjtu_pickle(path)
    assert len(result) == 2
    assert result.battery_id.nunique() == 1
    assert np.allclose(result.capacity_ah, [0.9, 0.85])
    assert np.allclose(result.rated_capacity_ah, 2.0)
    assert np.isclose(result.loc[0, 'temperature_c'], 25.1)


def test_load_raw_discovers_xjtu_pickle_files(tmp_path):
    from battery_soh.data.loaders import load_raw
    folder = tmp_path / 'xjtu'
    folder.mkdir()
    pd.to_pickle({
        'cell_id': 'XJTU_3C_battery-1', 'nominal_capacity_in_Ah': 2.0,
        'cycle_data': [
            {'cycle_number': 1, 'discharge_capacity_in_Ah': [0.0, 0.8, 1.9]},
            {'cycle_number': 2, 'discharge_capacity_in_Ah': [0.0, 0.7, 1.8]},
        ],
    }, folder / 'XJTU_3C_battery-1.pkl')
    result = load_raw(tmp_path)
    assert set(result.dataset) == {'xjtu'}
    assert result.battery_id.nunique() == 1
    assert np.allclose(result.soh, [95.0, 90.0])
