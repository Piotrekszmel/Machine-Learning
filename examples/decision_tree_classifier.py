from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os

from MachineLearning.utils.data_manipulation import train_test_split, standardize
from MachineLearning.utils.data_operation import mean_squared_error, accuracy_score, calculate_variance
from MachineLearning.utils.misc import Plot
from MachineLearning.supervised_learning.decision_tree import ClassificationTree


def main():
    print("-- Classification Tree --")

    data = datasets  .load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)

    Plot().plot_in_2d(X_test, y_pred, 
        title="Decision Tree", 
        accuracy=accuracy, 
        legend_labels=data.target_names)


if __name__ == "__main__":
    main()