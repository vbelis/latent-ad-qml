# A wrapper to the classical one-class SVM model of sklearn to define
# the predict() and score() methods.

from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import numpy as np


class CustomOneClassSVM(OneClassSVM):
    """
    Wrapper class of the `OneClassSVM` class, to redifine the `predict` and
    `score` to accommodate for the `train.py` and `test.py` scripts.
    """

    def __init__(
        self,
        *,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        tol=0.001,
        nu=0.5,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_data: bool = False,
        sample_weight: np.ndarray = None,
    ) -> float:
        """
        Return the mean accuracy on the given test data x and labels y.

        Args:
            x : array-like of shape (n_samples, n_features). Test samples.
            y : array-like of shape (n_samples,). True labels for `x`.

            sample_weight : array-like of shape (n_samples,), default=None.

        Returns: Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if train_data:
            y = np.ones(len(x))  # To compute the fraction of outliers in training.
            return accuracy_score(y, self.predict(x), sample_weight=sample_weight)

        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the label of a data vector X.
        Maps the prediction label of the one-class SVM from 1 -> 0
        and -1 -> 1 for inliers (background) and outliers
        (anomalies/signal), respectively.

        Args:
            x: Data vector array of shape (n_samples, n_features)

        Returns: The predicted labels of the input data vectors, of shape (n_samples).
        """
        y = super().predict(x)
        y[y == 1] = 0
        y[y == -1] = 1
        return y

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Signed distance to the separating hyperplane, positive for an inlier
        and negative for an outlier. The output of `super().decision_function`
        is multiplied by -1 in order to have the same sign convention between
        supervised and unsupervised kernel machines. For some reason the scores
        have the opposite sign for signal and background for SVC.decision_function
        and OneClassSVM.decision_function.
        """
        return -1.0*super().decision_function(x)
