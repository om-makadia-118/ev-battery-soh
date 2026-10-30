from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(truth, predicted):
    a, b = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if not len(a) or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid metric vectors')
    if (a == 0).any(): raise ValueError('MAPE is undefined for zero SOH')
    return {'MAE': float(mean_absolute_error(a,b)),
            'RMSE': float(np.sqrt(mean_squared_error(a,b))),
            'MAPE': float(np.mean(np.abs((a-b)/a))*100)}
