import numpy as np 
import math


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power((y - y_pred), 2))
    return mse 
    