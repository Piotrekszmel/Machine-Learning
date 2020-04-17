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