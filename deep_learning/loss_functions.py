import numpy as np 

from MachineLearning.deep_learning.activation_functions import Sigmoid
from MachineLearning.utils.data_operation import accuracy_score


class Loss:
    def loss(self, y, y_pred):
        return NotImplementedError()
    
    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self):
        pass
    
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))