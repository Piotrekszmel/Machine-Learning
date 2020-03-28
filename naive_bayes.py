import numpy as np 
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
