import pandas as pd
import pytest
import numpy as np
from battery_soh.data.demo import synthetic_datasets
from battery_soh.experiments.engine import benchmark,train_model,evaluate_model
from battery_soh.experiments.adaptation import adapt_pair,CoralAligner
from battery_soh.data.splits import battery_split
from battery_soh.evaluation.plots import make_figures

@pytest.fixture(scope='module')
def data(): return synthetic_datasets(batteries=7,observations=20)

@pytest.mark.parametrize('mode,expected',[('within',3),('cross',6),('lodo',3)])
def test_benchmark(data,tmp_path,mode,expected):
    result=benchmark(data,experiment=mode,models=['mean','ridge'],seeds=[7],
                     features=['cycle'],output=tmp_path,strict=True)
    assert len(result)==expected*2
    assert result.MAE.notna().all()
    assert (tmp_path/'predictions.csv').exists()
    assert (tmp_path/'failures.csv').exists()


def test_adapt(data,tmp_path):
    result=adapt_pair(data,source='toy_a',target='toy_b',model='ridge',seed=7,
                      counts=[0,1,3,5,10],output=tmp_path)
    assert result.query("k==0 and method=='none'").MAE.notna().all()
    assert result.query("k==10 and method=='not_feasible'").shape[0]==1
    assert result.query("k==3 and method=='coral'").MAE.notna().all()
    assert (tmp_path/'adaptation_metrics.csv').exists()


def test_coral_one_feature():
    align=CoralAligner().fit(np.arange(10).reshape(-1,1),np.arange(10,20).reshape(-1,1))
    assert align.source_to_target(np.array([[5.]])).shape==(1,1)


def test_deep_cpu(data):
    pytest.importorskip('torch')
    from battery_soh.models.deep import TorchRegressor
    from battery_soh.experiments.engine import model_arrays
    src=data[data.dataset=='toy_a']
    tr,val,te=battery_split(src,8)
    fit=train_model('cnn',tr,val,['cycle'],4,8,{'epochs':2,'patience':1,'device':'cpu'})
    scores,preds=evaluate_model(fit,'cnn',te,['cycle'],4)
    assert len(preds)>0 and np.isfinite(scores['MAE'])
    ft=fit.finetune(*model_arrays(val,'cnn',['cycle'],4)[:2],epochs=1)
    assert len(ft.predict(model_arrays(te,'cnn',['cycle'],4)[0]))>0


def test_plots(data,tmp_path):
    benchmark(data,experiment='cross',models=['ridge'],seeds=[7],output=tmp_path,strict=True)
    figs=make_figures(tmp_path)
    assert any('transfer' in p for p in figs)

def test_five_by_five_matrix_shape(tmp_path):
    from battery_soh.data.demo import synthetic_datasets
    d=synthetic_datasets(batteries=5,observations=8)
    result=benchmark(d,experiment='matrix',models=['mean'],seeds=[42],output=tmp_path,strict=True)
    assert len(result)==9  # 3 x 3 including valid held-out diagonals
    assert len(result.query("experiment=='within'"))==3
    assert len(result.query("experiment=='cross'"))==6
    assert any('transfer_mean' in p for p in make_figures(tmp_path))

def test_normalization_adaptation(tmp_path):
    from battery_soh.experiments.adaptation import MeanVarianceAligner
    x=np.array([[0.],[1.],[2.]])
    y=np.array([[10.],[11.],[12.]])
    a=MeanVarianceAligner().fit(x,y)
    np.testing.assert_allclose(a.source_to_target(x),y,atol=1e-4)
    d=synthetic_datasets(batteries=7,observations=10)
    res=adapt_pair(d,source='toy_a',target='toy_b',model='ridge',counts=[1],output=tmp_path)
    assert res.query("method=='normalization'").MAE.notna().all()
