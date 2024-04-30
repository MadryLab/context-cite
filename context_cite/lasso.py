import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline


class LassoRegression:
    def __init__(self, lasso_alpha) -> None:
        self.lasso_alpha = lasso_alpha

    def fit(self, masks, outputs, num_output_tokens) -> tuple:
        weight, bias = self._fit_lasso(masks, outputs / num_output_tokens)
        return weight * num_output_tokens, bias * num_output_tokens

    def _fit_lasso(self, masks, outputs) -> tuple[np.ndarray, float]:
        X = masks.astype(np.float32)
        Y = outputs
        scaler = StandardScaler()
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        # Pipeline is ((X - scaler.mean_) / scaler.scale_) @ lasso.coef_.T + lasso.intercept_
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        return weight, bias
