from sklearn import datasets
import numpy as np 

from MachineLearning.utils.data_manipulation import train_test_split, normalize
from MachineLearning.utils.data_operation import accuracy_score
from MachineLearning.utils.misc import Plot
from MachineLearning.naive_bayes import NaiveBayes


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()