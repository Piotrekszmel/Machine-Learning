from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt 

from MachineLearning.utils.data_manipulation import train_test_split, normalize
from MachineLearning.utils.data_operation import accuracy_score
from MachineLearning.utils.misc import Plot
from MachineLearning.LogisticRegression import LogisticRegression


def main():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]

    y[y==1] = 0
    y[y==2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()


