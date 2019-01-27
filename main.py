import numpy as np
from numpy_based_linear_regression import NumPyBasedLinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def main(debug_mode=True, loss_plot_mode=True):
    X, y = make_regression(n_samples=100, n_features=100, noise = 20)
    X, y = np.asarray(X), np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T.reshape((1, y_train.shape[0])), y_test.T.reshape((1, y_test.shape[0]))
    print("X_train.shape:\t " + str(X_train.shape))
    print("X_test.shape:\t " + str(X_test.shape))
    print("y_train.shape:\t " + str(y_train.shape))
    print("y_test.shape:\t " + str(y_test.shape))
    regressor = NumPyBasedLinearRegression()
    fitting_successs = regressor.fit(X=X_train, Y_actual=y_train, decay_rate=0.0, early_stopping_point=10000, convergence_tolerance = 0.001, batch_size=1, debug_mode=debug_mode, loss_plot_mode=loss_plot_mode)
    if fitting_successs:
        y_predicted = regressor.predict(X=X_test, debug_mode=debug_mode)
        if y_predicted != None:
            coefficient_of_determination = regressor.get_coefficient_of_determination(X=X_test, Y_predicted=y_predicted, Y_actual=y_test, debug_mode=debug_mode)
            if coefficient_of_determination != None:
                print("Coefficient of determination on test set: " + str(coefficient_of_determination))
            else:
                print("Error: coefficient of determination calculation fails")
        else:
            print("Error: prediction fails")
    else:
        print("Error: fitting fails")

main()
