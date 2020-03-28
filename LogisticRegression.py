from typing import Union
import numpy as np  
import math 

from MachineLearning.deep_learning.activation_functions import Sigmoid


class LogisticRegression:
    """ 
    Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    """
    def __init__(self, learning_rate: float = 0.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()
    
    def _initialize_parameters(self, X: np.ndarray) -> None:
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features ,))
    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = X.dot(self.param)
        h = self.sigmoid(z)
        gradient = X.T.dot(h - y) / y.shape[0]
        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray, n_iteriations: Union[int, float] = 10000) -> None:
        self._initialize_parameters(X)
        for i in range(n_iteriations):
            self.param = self.param - self.learning_rate * self.gradient(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred