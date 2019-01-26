import numpy as np
from numpy_based_linear_regression import NumPyBasedLinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def main():
    X, y = make_regression(n_samples=100, n_features=1, noise = 2)
    X, y = np.asarray(X), np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
    regressor = NumPyBasedLinearRegression()
    fitting_successs = regressor.fit(X=X_train, Y_actual=y_train, debug_mode=True, loss_plot_mode=True)
    if fitting_successs:
        y_predicted = regressor.predict(X=X_test, debug_mode=True)
        if y_predicted != None:
            coefficient_of_determination = regressor.get_coefficient_of_determination(X=X_test, Y_predicted=y_predicted, Y_actual=y_test)
            if coefficient_of_determination != None:
                print("Coefficient of determination on test set: " + str(coefficient_of_determination))
            else:
                print("Error: coefficient of determination calculation fails")
        else:
            print("Error: prediction fails")
    else:
        print("Error: fitting fails")

main()
