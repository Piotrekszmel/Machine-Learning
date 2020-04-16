import numpy as np
from MachineLearning.utils.data_manipulation import normalize, make_diagonal


class SGD:
    def __init__(self, lr=0.01, momentum=0.):
        self.lr = lr 
        self.momentum = momentum
        self.v = None
    
    def update(self, w, grad_w):
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        
        self.v = self.momentum * self.v + (1 - self.momentum) * grad_w
        return w = self.lr * self.lr