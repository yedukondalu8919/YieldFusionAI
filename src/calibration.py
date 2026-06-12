import numpy as np
from sklearn.linear_model import LinearRegression
from .config import CFG

class RegionalCalibrator:
    def __init__(self, lambda_corr=CFG.calibration_lambda, epsilon=CFG.calibration_epsilon):
        self.lambda_corr = lambda_corr
        self.epsilon = epsilon
        self.global_model = LinearRegression()
        self.region_models = {}
        self.region_residual_mean = {}

    def fit(self, y_pred, y_true, regions=None):
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        y_true = np.asarray(y_true).reshape(-1)
        self.global_model.fit(y_pred, y_true)
        if regions is not None:
            regions = np.asarray(regions)
            for r in np.unique(regions):
                idx = np.where(regions == r)[0]
                if len(idx) >= 4:
                    lr = LinearRegression().fit(y_pred[idx], y_true[idx])
                    self.region_models[r] = lr
                    self.region_residual_mean[r] = float(np.mean(y_true[idx] - lr.predict(y_pred[idx])))
        return self

    def predict(self, y_pred, regions=None):
        raw = np.asarray(y_pred).reshape(-1, 1)
        calibrated = self.global_model.predict(raw)
        if regions is not None:
            regions = np.asarray(regions)
            out = calibrated.copy()
            for i, r in enumerate(regions):
                if r in self.region_models:
                    out[i] = self.region_models[r].predict(raw[i].reshape(1, -1))[0]
                    delta = self.region_residual_mean.get(r, 0.0)
                    if abs(delta) > self.epsilon:
                        out[i] = out[i] + self.lambda_corr * delta
            return out.reshape(-1, 1)
        return calibrated.reshape(-1, 1)
