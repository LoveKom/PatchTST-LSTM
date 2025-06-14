from typing import Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(real: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs((real - pred) / real)) * 100
    r2 = r2_score(real, pred)
    return mae, rmse, mape, r2
