
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MachineLearning.regression import LassoRegression
from MachineLearning.utils.data_manipulation import k_fold_cross_validation_sets, normalize, train_test_split, polynomial_features
from MachineLearning.utils.data_operation import mean_squared_error


def main():

    # Load temperature data
    data = pd.read_csv("../data/TempLinkoping2016.txt", sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = data["temp"].values

    X = time # fraction of the year [0, 1]
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    poly_degree = 13

    model = LassoRegression(degree=15, 
                            reg_factor=0.05,
                            learning_rate=0.001,
                            n_iterations=4000)
    model.fit(X_train, y_train)

    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse} (given by reg. factor: 0.05)")

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Lasso Regression")
    plt.title(f"MSE: {mse:.2f}", fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.savefig("lasso_regression   ")
    plt.show()


if __name__ == "__main__":
    main()