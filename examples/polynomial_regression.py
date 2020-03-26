import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 


from MachineLearning.regression import PolynomialRidgeRegression
from MachineLearning.utils.data_manipulation import k_fold_cross_validation_sets, normalize, train_test_split, polynomial_features
from MachineLearning.utils.data_operation import mean_squared_error


def main():

     # Load temperature data
    data = pd.read_csv("../data/TempLinkoping2016.txt", sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = data["temp"].values

    X = time
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3)

    poly_degree = 15

    # Finding regularization constant using cross validation
    lowest_error = float("inf")
    best_reg_factor = None
    
    print("Finding regularization constant using cross validation")
    
    k = 10
    for reg_factor in np.arange(0, 0.1, 0.01):
        cross_validation_sets = k_fold_cross_validation_sets(X_train,
                                                            y_train,
                                                            k)

        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            model = PolynomialRidgeRegression(degree=poly_degree,
                                            reg_factor=reg_factor,
                                            n_iterations=10000,
                                            learning_rate=0.001)
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)
            _mse = mean_squared_error(_y_test, y_pred)
            mse += _mse
        mse /= k

        print(f"Mean Squared Error: {mse} (regularization: {reg_factor})")

        if mse < lowest_error:
            best_reg_factor = reg_factor
            lowest_error = mse

    model = PolynomialRidgeRegression(degree=poly_degree,
                                    reg_factor=best_reg_factor,
                                    n_iterations=10000,
                                    learning_rate=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {lowest_error} (given by reg. factor: {best_reg_factor})")

    y_pred_line = model.predict(X)

    cmap = plt.get_cmap("viridis")
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Ridge Regression")
    plt.title(f"MSE: {mse:.2f}", fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.savefig("polynomial_regression")
    plt.show()


if __name__ == "__main__":
    main()

