import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from MachineLearning.utils.data_manipulation import train_test_split, standardize
from MachineLearning.utils.data_operation import mean_squared_error, accuracy_score, calculate_variance
from MachineLearning.utils.misc import Plot
from MachineLearning.decision_tree import RegressionTree

def main():
    print("-- Regression Tree --")
    
    data = pd.read_csv("../data/TempLinkoping2016.txt", sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    X = standardize(time)
    y = temp[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RegressionTree()
    model.fit(X, y)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')
    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title(f"MSE: {mse:.2f}", fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()