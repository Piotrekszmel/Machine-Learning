from typing import Union, Tuple, List
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
    """
    Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: Union[int, float]
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: Union[int, float]
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations: Union[int, float], learning_rate: Union[int, float]) -> None: 
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_features: Union[int, float]) -> None:
        """
        Initialize weights randomly
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X: Union[List, Tuple, np.ndarray], y: Union[List, Tuple, np.ndarray]) -> None:
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
            self.w = self.w -  self.learning_rate * grad_w
    
    def predict(self, X: Union[List, Tuple, np.ndarray]) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        print(type(y_pred))
        return y_pred


class LinearRegression(Regression):
    """
    Linear model.
    
    # Example: 
    
    ```python
        X = [[1,2,3,4,5], [10,11,12,13,14]]
        y = [6, 15]   

        Linear = LinearRegression()
        Linear.fit(np.array(X), np.array(y))
        print(Linear.predict([[6,7,8,9,10]]))

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations: float = 100, learning_rate: float = 0.001) -> None:
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)
    
    def fit(self, X: Union[List, Tuple, np.ndarray], y: Union[List, Tuple, np.ndarray]) -> np.ndarray:
        super().fit(X, y)
    

class LassoRegression(Regression):
    """
    Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super().__init__(n_iterations=n_iterations, 
                        learning_rate=learning_rate)