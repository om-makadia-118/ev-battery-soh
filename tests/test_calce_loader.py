import numpy as np
import pandas as pd

from battery_soh.data.loaders import calce_excel_file, calce_excel_folder, inspect


def _write_calce_workbook(path, cycles, start_time):
    rows = []
    point = 1
    time_s = 0.0
    qd_total = 0.0
    qc_total = 0.0
    start = pd.Timestamp(start_time)
    for cycle_no, profile in cycles:
        full_charge, full_discharge, capacity = profile
        charge_v = 4.2 if full_charge else 3.77
        discharge_end_v = 2.7 if full_discharge else 3.77
        # Rest before charge.
        for voltage in [3.8, 3.85]:
            rows.append([point, time_s, start + pd.Timedelta(seconds=time_s),
                         30.0, 1, cycle_no, 0.0, voltage,
                         qc_total, qd_total, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
            point += 1
            time_s += 30
        # Charge step.
        qc_total += 0.1
        rows.append([point, time_s, start + pd.Timedelta(seconds=time_s),
                     30.0, 2, cycle_no, 0.55, charge_v,
                     qc_total, qd_total, 0.4, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
        point += 1
        time_s += 30
        # Rest at charge voltage.
        rows.append([point, time_s, start + pd.Timedelta(seconds=time_s),
                     30.0, 3, cycle_no, 0.0, charge_v,
                     qc_total, qd_total, 0.4, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
        point += 1
        time_s += 30
        # Discharge step: cumulative counter increases from its prior value.
        for frac, voltage in zip([0.1, 0.5, 1.0], [4.0, 3.4, discharge_end_v]):
            qd_total += capacity * (frac - (0 if frac == 0.1 else (0.1 if frac == 0.5 else 0.5)))
            rows.append([point, time_s, start + pd.Timedelta(seconds=time_s),
                         30.0, 4, cycle_no, -0.55, voltage,
                         qc_total, qd_total, 0.4, qd_total * 3.6,
                         0.0, 0.1, 0, 0.0, 0.0])
            point += 1
            time_s += 30
    columns = [
        'Data_Point', 'Test_Time(s)', 'Date_Time', 'Step_Time(s)', 'Step_Index',
        'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)',
        'Discharge_Capacity(Ah)', 'Charge_Energy(Wh)', 'Discharge_Energy(Wh)',
        'dV/dt(V/s)', 'Internal_Resistance(Ohm)', 'Is_FC_Data',
        'AC_Impedance(Ohm)', 'ACI_Phase_Angle(Deg)',
    ]
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame([['test']]).to_excel(writer, sheet_name='Info', index=False, header=False)
        pd.DataFrame(rows, columns=columns).to_excel(
            writer, sheet_name='Channel_1-006', index=False
        )


def test_calce_extracts_full_discharge_capacity_not_cumulative_counter(tmp_path):
    path = tmp_path / 'CS2_33_1_10_11.xlsx'
    _write_calce_workbook(
        path,
        [
            (1, (True, True, 0.90)),
            (2, (False, True, 0.60)),  # Low-regime partial: starts below 4.15 V.
            (3, (True, False, 0.30)),  # High-regime partial: does not reach 2.85 V.
            (4, (True, True, 0.85)),
        ],
        '2011-01-01',
    )
    result = calce_excel_file(path)
    assert result.cycle.tolist() == [1, 4]
    assert np.allclose(result.capacity_ah, [0.90, 0.85])
    assert np.allclose(result.rated_capacity_ah, 1.1)
    assert result.protocol_type.eq('Type 1').all()


def test_calce_folder_rebases_cycle_index_across_date_files(tmp_path):
    _write_calce_workbook(
        tmp_path / 'CS2_33_1_10_11.xlsx',
        [(1, (True, True, 0.90)), (2, (True, True, 0.89))],
        '2011-01-01',
    )
    _write_calce_workbook(
        tmp_path / 'CS2_33_1_17_11.xlsx',
        [(1, (True, True, 0.88)), (2, (True, True, 0.87))],
        '2011-01-02',
    )
    result = calce_excel_folder(tmp_path)
    assert result.cycle.tolist() == [1, 2, 3, 4]
    assert np.allclose(result.capacity_ah, [0.90, 0.89, 0.88, 0.87])


def test_inspect_xlsx_reports_channel_sheet_columns(tmp_path):
    path = tmp_path / 'CS2_33_1_10_11.xlsx'
    _write_calce_workbook(path, [(1, (True, True, 0.9))], '2011-01-01')
    output = inspect(path)
    assert 'Channel_1-006' in output
    assert 'Discharge_Capacity(Ah)' in output


def test_load_raw_discovers_calce_workbooks(tmp_path):
    from battery_soh.data.loaders import load_raw
    folder = tmp_path / 'calce' / 'CS2_33'
    folder.mkdir(parents=True)
    _write_calce_workbook(
        folder / 'CS2_33_1_10_11.xlsx',
        [(1, (True, True, 0.90))],
        '2011-01-01',
    )
    result = load_raw(tmp_path)
    assert result.dataset.eq('calce').all()
    assert result.battery_id.unique().tolist() == ['CS2_33']
    assert np.isclose(result.capacity_ah.iloc[0], 0.90)



def _write_cadex_log(path, cycle_numbers):
    rows = []
    duration = 0
    for cycle in cycle_numbers:
        charge_count = 2 * cycle - 1
        # A charge phase. Current is in mA, voltage in mV, duration in seconds.
        for dt, current, voltage, progress in [
            (0, 550, 4000, 0), (3600, 550, 4100, 50), (7200, 100, 4200, 100)
        ]:
            rows.append([
                duration + dt, 8, 3, 3, 0, 1, 2, 2, voltage, current, 25,
                duration + dt, charge_count, cycle - 1, progress,
                *([0] * 14),
            ])
        duration += 7300
        for dt, current, voltage, progress in [
            (0, -550, 4100, 0), (3600, -550, 3500, 50), (7200, -550, 2800, 100)
        ]:
            rows.append([
                duration + dt, 8, 6, 3, 0, 2, 7, 1, voltage, current, 26,
                duration + dt, charge_count, cycle, progress,
                *([0] * 14),
            ])
        duration += 7300
    columns = [
        'Time', 'Status code', 'Status category', 'Status color', 'Pgm code',
        'Pgm step', 'Pgm para', 'Pgm cycle', 'mV', 'mA', 'Temperature',
        'Duration', 'Charge count', 'Discharge count', 'Capacity',
        'Analog input 1', 'Analog input 2', 'Analog input 3', 'Analog input 4',
        'Digital input 1', 'Digital input 2', 'Digital input 3', 'Digital input 4',
        'Digital output 1', 'Digital output 2', 'Digital output 3', 'Digital output 4',
        'Analog output 1', 'Analog output 2',
    ]
    pd.DataFrame(rows, columns=columns).assign(
        Time=lambda x: x['Duration'] / 60.0
    ).to_csv(path, sep='\t', index=False)


def test_calce_cadex_integrates_current_and_does_not_use_capacity_percent(tmp_path):
    from battery_soh.data.loaders import calce_cadex_file
    path = tmp_path / 'CS2_8_1_19_10.txt'
    _write_cadex_log(path, [1, 2])
    result = calce_cadex_file(path)
    assert result.cycle.tolist() == [1, 2]
    assert np.allclose(result.capacity_ah, [1.1, 1.1])
    assert np.allclose(result.rated_capacity_ah, 1.1)
    assert result.protocol_type.eq('Type 1').all()
    assert np.allclose(result.charge_rate_c, 0.5)


def test_calce_cadex_folder_rebases_reset_discharge_counts(tmp_path):
    from battery_soh.data.loaders import calce_cadex_folder
    first = tmp_path / 'CS2_8_1_19_10.txt'
    second = tmp_path / 'CS2_8_1_21_10.txt'
    _write_cadex_log(first, [1, 2])
    _write_cadex_log(second, [1, 2])
    result = calce_cadex_folder([second, first])
    assert result.cycle.tolist() == [1, 2, 3, 4]


def test_calce_folder_and_load_raw_support_cadex_txt(tmp_path):
    from battery_soh.data.loaders import calce_excel_folder, load_raw
    folder = tmp_path / 'calce' / 'CS2_8'
    folder.mkdir(parents=True)
    _write_cadex_log(folder / 'CS2_8_1_19_10.txt', [1, 2])
    direct = calce_excel_folder(folder)
    assert direct.battery_id.unique().tolist() == ['CS2_8']
    result = load_raw(tmp_path)
    assert result.dataset.eq('calce').all()
    assert result.battery_id.unique().tolist() == ['CS2_8']
    assert np.allclose(result.capacity_ah, [1.1, 1.1])


def test_inspect_cadex_reports_tabular_structure(tmp_path):
    from battery_soh.data.loaders import inspect
    path = tmp_path / 'CS2_8_1_19_10.txt'
    _write_cadex_log(path, [1])
    output = inspect(path)
    assert 'tab-separated CADEX log' in output
    assert 'Status category' in output
    assert 'Discharge counts' in output
