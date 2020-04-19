import numpy as np
from MachineLearning.utils.data_manipulation import normalize, make_diagonal


class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.):
        self.lr = lr 
        self.momentum = momentum
        self.v = None
    
    def update(self, w: np.ndarray, grad_w: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        # Optional use of momentum if set
        self.v = self.momentum * self.v + (1 - self.momentum) * grad_w
        # Move against the gradient to minimize loss
        return w - self.lr * self.v


class NesterovAcceleratedGradient:
    def __init__(self, lr: float = 0.001, momentum: float = 0.4):
        self.lr = lr
        self.momentum = momentum
        self.v = np.array([])
    
    def update(self, w: np.ndarray, grad_func) -> np.ndarray:
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.v), -1, 1)
        # Initialize on first update
        if not self.v.any():
            self.v = np.zeros(np.shape(w))
        
        self.v = self.momentum * self.v + self.lr * approx_future_grad
        # Move against the gradient to minimize loss
        return w - self.v
    

class Adagrad:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.F = None # Sum of squares of the gradients
        self.eps = 1e-8
    
    def update(self, w: np.ndarray, grad_w: np.ndarray):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.lr * grad_w / np.sqrt(self.G + self.eps)