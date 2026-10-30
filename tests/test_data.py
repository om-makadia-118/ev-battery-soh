import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from battery_soh.data.demo import synthetic_datasets
from battery_soh.data.schema import harmonize, quality_report
from battery_soh.data.features import checked_features, sequence_windows
from battery_soh.data.splits import battery_split, target_support_test
from battery_soh.data.loaders import nasa_mat, oxford_mat, mapped_table


def test_soh_and_eol():
    df=harmonize(pd.DataFrame({'dataset':['nasa']*2,'battery_id':['B5']*2,
      'cycle':[1,2],'capacity_ah':[2.0,1.5],'rated_capacity_ah':[2.,2.]}))
    assert df.soh.tolist()==[100.,75.]
    assert df.eol_reached.tolist()==[False,True]


def test_bad_units_rejected():
    with pytest.raises(ValueError,match='SOH outside'):
        harmonize(pd.DataFrame({'dataset':['calce'],'battery_id':['c1'],
          'cycle':[1],'capacity_ah':[1100],'rated_capacity_ah':[1.1]}))


def test_no_target_leakage():
    df=synthetic_datasets()
    for name in ['capacity_ah','soh','rated_capacity_ah']:
        with pytest.raises(ValueError): checked_features(df,[name])
    checked_features(df,['cycle'])


def test_splits_disjoint():
    df=synthetic_datasets()
    a,b,c=battery_split(df[df.dataset=='toy_a'],seed=42)
    assert not set(a.battery_id)&set(b.battery_id)
    assert not set(a.battery_id)&set(c.battery_id)
    assert not set(b.battery_id)&set(c.battery_id)
    pool,test=target_support_test(df[df.dataset=='toy_b'],seed=42)
    assert not set(pool.battery_id)&set(test.battery_id)
    assert len(test.battery_id.unique())>=2


def test_windows_do_not_cross_cells():
    df=synthetic_datasets(batteries=2,observations=6)
    x,y,index=sequence_windows(df,['cycle'],window=4)
    assert x.shape==(3*2*3,4,1)
    assert all(np.array_equal(row.ravel(), [1,2,3,4]) for row in x[::3])
    assert all(np.array_equal(row.ravel(), [3,4,5,6]) for row in x[2::3])


def test_nasa_mat_parser(tmp_path):
    p=tmp_path/'B0005.mat'
    ops=np.array([
      {'type':'charge','data':{'Voltage_measured':[4.0]}},
      {'type':'discharge','data':{'Capacity':1.85,'Voltage_measured':[3.6,3.2],
                                          'Temperature_measured':[23.,24.]}}
    ],dtype=object)
    savemat(p,{'B0005':{'cycle':ops}})
    df=nasa_mat(p)
    assert len(df)==1
    assert df.capacity_ah.iloc[0]==pytest.approx(1.85)
    assert df.cycle.iloc[0]==1


def test_oxford_mat_parser(tmp_path):
    p=tmp_path/'Oxford_Battery_Degradation_Dataset_1.mat'
    savemat(p,{'Cell1':{'cyc0100':{'C1dc':{'q':[0,700], 'v':[4.,3.],
                                              't':[0,3000], 'T':[40,41]}}}})
    df=oxford_mat(p)
    assert len(df)==1
    assert df.capacity_ah.iloc[0]==pytest.approx(.7)
    assert df.cycle.iloc[0]==100


def test_mapping_per_cycle_and_mAh(tmp_path):
    path=tmp_path/'c1.csv'
    pd.DataFrame({'cap_mAh':[990,960]}).to_csv(path,index=False)
    df=mapped_table(path,'calce',column_map={'capacity_ah':'cap_mAh'},
                    battery_id='CS2_35',capacity_unit='mAh',cycle_from_row=True)
    assert df.capacity_ah.tolist()==pytest.approx([.99,.96])

def test_matr_hdf5_parser(tmp_path):
    import h5py
    from battery_soh.data.loaders import matr_mat
    p=tmp_path/'2017-05-12_batchdata_updated_struct_errorcorrect.mat'
    with h5py.File(p,'w') as f:
        batch=f.create_group('batch')
        s=f.create_group('one_cell_summary')
        s.create_dataset('QDischarge',data=np.array([[1.04,1.02,.99]]))
        s.create_dataset('cycle',data=np.array([[1.,2.,3.]]))
        refs=batch.create_dataset('summary',shape=(1,1),dtype=h5py.ref_dtype)
        refs[0,0]=s.ref
    df=matr_mat(p)
    assert df.capacity_ah.tolist()==pytest.approx([1.04,1.02,.99])
    assert df.battery_id.nunique()==1
