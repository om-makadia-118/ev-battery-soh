"""Models returned fresh for every experiment (avoid model state reuse)."""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def make_classical(name: str, seed: int):
    if name == 'mean': return DummyRegressor(strategy='mean')
    if name == 'ridge': return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == 'elasticnet':
        return make_pipeline(StandardScaler(), ElasticNet(alpha=.01,l1_ratio=.5,max_iter=10000,random_state=seed))
    if name == 'xgboost':
        return XGBRegressor(n_estimators=150, max_depth=4, learning_rate=.05,
                            subsample=.8, colsample_bytree=.9, random_state=seed, n_jobs=2)
    raise ValueError(f'Unknown classical model: {name}')
