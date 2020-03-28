import numpy as np 
import math


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power((y_true - y_pred), 2))
    return mse 
    

def calculate_variance(X: np.ndarray) -> np.ndarray:
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance


def calculate_std_dev(X: np.ndarray) -> np.ndarray:
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def calculate_covariance_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)
