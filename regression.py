from __future__ import print_function, division
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