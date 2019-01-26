import random
import numpy as np

class NumPyBasedLinearRegression(object):

    """
    Constructor: initialize member variables

    @return None
    """
    def __init__(self):
        self.__n_x = None
        self.__w = None
        self.__b = None

    """
    Get the predicted values of the output variable

    @param w: the NumPy array of the weights
    @param X: the NumPy array of the input variables
    @param b: the NumPay array of the bias term
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a NumPy array of the predicted values of the output variable
    """
    def __get_Y_predicted(self, w, X, b, debug_mode=True):
        Y_predicted = np.dot(w.T, X) + b
        return Y_predicted

    """
    Get the mean squared error of the model based on the provided data

    @param Y_predicted: the NumPy array of the predicted values of the output variable
    @param Y_actual: the NumPy array of the actual values of the output variable
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return the mean squared error of the model based on the provided data
    """
    def __get_mean_squared_error(self, Y_predicted, Y_actual, debug_mode=True):
        # check the dimension of Y_predicted and Y_actual
        if Y_predicted.shape != Y_actual.shape:
            if debug_mode:
                print("Error: Y_predicted.shape != Y_actual.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__get_mean_squared_error()")
            return None
        # calculate mean squared error (MSE)
        mean_squared_error = (1 / m) * np.sum(np.square(Y_predicted - Y_actual))
        return mean_squared_error

    """
    Get the gradients of the model based on the provided data

    @param Y_predicted: the NumPy array of the predicted values of the output variable
    @param Y_actual: the NumPy array of the actual values of the output variable
    @param X: the NumPy array of the input variables
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array of the gradient of the weights, and (2) a NumPy array of the gradient of the bias term
    """
    def __get_gradients(self, Y_predicted, Y_actual, X, debug_mode=True):
        # check the dimension of Y_predicted and Y_actual
        if Y_predicted.shape != Y_actual.shape:
            if debug_mode:
                print("Error: Y_predicted.shape != Y_actual.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__get_gradients()")
            return None
        # find the number of examples m
        m = Y_predicted.shape[1]
        # calcualte the residual
        dY = Y_predicted - Y_actual
        # calculate the gradient for the weights
        dw = (1 / m) * np.dot(X, dY.T)
        # calculate the gradient for the bias term
        db = (1 / m) * np.sum(dY)
        # return the gradients in a tuple
        return (dw, db)

    """
    Update the parameters of the model

    @param w: the NumPy array of the weights
    @param b: the NumPay array of the bias term
    @param learning_rate: the speed of gradient descent
    @param dw: the NumPy array of the gradient of the weights
    @param db: the NumPy array of the gradient of the bias term
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array of the updated weights, and (2) a NumPy array of the updated bias term
    """
    def __update_parameters(self, w, b, learning_rate, dw, db, debug_mode=True):
        # check the dimension of w and its gradient
        if w.shape != dw.shape:
            if debug_mode:
                print("Error: w.shape != dw.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__update_parameters()")
            return None
        # check the dimension of b and its gradient
        if b.shape != db.shape:
            if debug_mode:
                print("Error: b.shape != db.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__update_parameters()")
            return None
        # update w using gradient descent
        w -= learning_rate * dw
        b -= learning_rate * db
        return (w, b)

    """
    Fit the model to the provided data

    @param X: the NumPy array of the input variables
    @param Y_actual: the NumPy array of the actual values of the output variable
    @param learning_rate: (optional) the speed of gradient descent; the default value is 0.001
    @param iterations: (optional) the maximum number of iterations allowed; the default value is 1000
    @param convergence_tolerance: (optional) the threshold to decide whether the gradient descent converges; the default value is 0.001
    @param batch_size: the batch size of mini-batch gradient descent
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @return a boolean value that indicates whether the fitting is successful
    """
    def fit(self, X, Y_actual, learning_rate=0.001, iterations=1000, convergence_tolerance=0.001, batch_size=1, debug_mode=False):
        # reconfigure the model setting
        self.__n_x = n_x
        self.__w = np.random.randn(self.__n_x, 1)
        self.__b = np.random.randn(1, 1)
        # check the number of features
        if X.shape[0] != self.__n_x:
            if debug_mode:
                print("Error: X.shape[0] != self.__n_x")
                print("\tStack trace: NumPyBasedLinearRegression.fit()")
            return False
        # check the dimension of X and Y
        if X.shape[1] != Y_actual.shape[1]:
            if debug_mode:
                print("Error: X.shape[1] != Y.shape[1]")
                print("\tStack trace: NumPyBasedLinearRegression.fit()")
            return False
        # allocate batches
        m = Y_actual.shape[1]
        # iterations of gradient descent
        for i in range(iterations):
            # get the mean squared error (MSE)
            Y_predicted = self.__get_Y_predicted(self.__w, X, self.__b, debug_mode=debug_mode)
            mean_squared_error = self.__get_mean_squared_error(Y_predicted=Y_predicted, Y_actual=Y_actual, debug_mode=debug_mode)
            if debug_mode:
                print("Iteration " + str(i) + "\t | MSE = " + str(mean_squared_error))
            # check MSE against convergence tolerance
            if mean_squared_error < convergence_tolerance:
                if debug_mode:
                    print("Message: convergence_tolerance reached at iteration " + str(i))
                    print("\tStack trace: NumPyBasedLinearRegression.fit()")
                break
            # get the gradients
            dw, db = self.__get_gradients(Y_predicted=Y_predicted, Y_actual=Y_actual, X=X, debug_mode=debug_mode)
            # update the parameters
            self.__w, self.__b = self.__update_parameters(w=self.__w, b=self.__b, learning_rate=learning_rate, dw=dw, db=db, debug_mode=debug_mode)
        return True

    """
    Predict using the model based on the provided data

    @param X: the NumPy array of the input variables
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @return a NumPy array of the predicted values of the output variable
    """
    def predict(self, X, debug_mode=False):
        # check the validity of member variables
        if self.__n_x == None or self.__w == None or self.__b == None:
            if debug_mode:
                print("Error: the model has not been trained")
                print("\tStack trace: NumPyBasedLinearRegression.predict()")
            return None
        # check number of features
        if X.shape[0] != self.__n_x:
            if debug_mode:
                print("Error: X.shape[0] != self.__n_x")
                print("\tStack trace: NumPyBasedLinearRegression.predict()")
            return None
        # get prediction
        Y_predicted = self.__get_Y_predicted(w=self.__w, X=X, b=self.__b, debug_mode=debug_mode)
        return Y_predicted
