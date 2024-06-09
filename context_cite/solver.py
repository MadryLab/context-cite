import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV

class BaseSolver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]: ...


class LassoRegression(BaseSolver):
    """
    A LASSO solver using the scikit-learn library.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, lasso_alpha: float = 0.01) -> None:
        self.lasso_alpha = lasso_alpha

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens
        scaler = StandardScaler()
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        # Pipeline is ((X - scaler.mean_) / scaler.scale_) @ lasso.coef_.T + lasso.intercept_
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        return weight * num_output_tokens, bias * num_output_tokens
    
class CosineSimLassoRegression(BaseSolver):
    """
    A LASSO solver using the scikit-learn library.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, lasso_alpha: float = 0.01) -> None:
        self.lasso_alpha = lasso_alpha

    def fit(
        self, masks: NDArray, outputs: NDArray
    ) -> Tuple[NDArray, NDArray]:
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
    
    def fit_cv(self, masks: NDArray, outputs: NDArray, alphas: List[float]) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs
        scaler = StandardScaler()
        lasso = LassoCV(alphas=alphas, random_state=0, fit_intercept=True)
        # Pipeline is ((X - scaler.mean_) / scaler.scale_) @ lasso.coef_.T + lasso.intercept_
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        print(f"Best alpha selected: {lasso.alpha_}")

        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        # bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        bias = 0
        return weight, bias
