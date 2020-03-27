from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt 

from MachineLearning.utils.data_manipulation import normalize, make_diagonal, train_test_split
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
    def __init__(self, learning_rate = 0.1):
        self.param = param
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()
    
    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
        