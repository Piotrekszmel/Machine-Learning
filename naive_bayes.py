import numpy as np 
import math 

from MachineLearning.utils.data_manipulation import train_test_split, polynomial_features
from MachineLearning.utils.data_operation import accuracy_score


class NaiveBayes:
    """The Gaussian Naive Bayes classifier. """
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
             # Add the mean and variance for each feature (column)
             for col in X_where_c.T:
                 parameters = {"mean": col.mean(), "var": col.var()}
                 self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, x):
        """ Gaussian likelihood of the data x given mean and var """
         # Added in denominator to prevent division by zero
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        """ 
        Calculate the prior of class c
        (samples where class == c / total number of samples)
        """ 
        frequency = np.mean(self.y == c)
        return frequency