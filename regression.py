from typing import Union
import numpy as np 
import math


class l1_regularization:
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)
    
    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization:
    """ Regularization for Ridge Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
    
    def grad(self, w):
        return self.alpha * w
    

class Regression:
    """Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: Union[int, float]
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: Union[int, float]
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations: Union[int, float], learning_rate: Union[int, float]): 
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_features):
        """
        Initialize weights randomly
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)

            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)

            grad_w  = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w = w - self.learning_rate * grad_w
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
