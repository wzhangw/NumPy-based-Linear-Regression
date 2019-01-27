import random
import numpy as np
import matplotlib.pyplot as plt

class NumPyBasedLinearRegression(object):

    """
    Constructor: declare member variables

    @return None
    """
    def __init__(self):
        # the number of features
        self.__n_x = None
        # the NumPy array of the weights
        self.__w = None
        # the NumPay array of the bias term
        self.__b = None
        # the log of epoch mean squared errors
        self.__epoch_mean_squared_errors = None
        # the log of iterative mean squared errors
        self.__iterative_mean_squared_errors = None

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
        # check the dimension of Y_predicted & Y_actual
        if Y_predicted.shape != Y_actual.shape:
            if debug_mode:
                print("Error: Y_predicted.shape != Y_actual.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__get_mean_squared_error()")
            return None
        # find the number of examples m
        m = Y_predicted.shape[1]
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
        # check the dimension of Y_predicted & Y_actual
        if Y_predicted.shape != Y_actual.shape:
            if debug_mode:
                print("Error: Y_predicted.shape != Y_actual.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__get_gradients()")
            return None
        # find the number of examples m
        m = Y_predicted.shape[1]
        # calcualte the residual
        dY = Y_predicted - Y_actual
        # calculate the gradients for the weights and the bias term
        dw = (1 / m) * np.dot(X, dY.T)
        db = (1 / m) * np.sum(dY, axis=1).reshape(1, 1)
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
        # check the dimension of w & its gradient
        if w.shape != dw.shape:
            if debug_mode:
                print("Error: w.shape != dw.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__update_parameters()")
            return None
        # check the dimension of b & its gradient
        if b.shape != db.shape:
            if debug_mode:
                print("Error: b.shape != db.shape")
                print("\tStack trace: NumPyBasedLinearRegression.__update_parameters()")
            return None
        # update the weights and the bias term using gradient descent
        w -= learning_rate * dw
        b -= learning_rate * db
        return (w, b)

    """
    Fit the model to the provided data

    @param X: the NumPy array of the input variables
    @param Y_actual: the NumPy array of the actual values of the output variable
    @param learning_rate: (optional) the speed of gradient descent; the default value is 0.001
    @param early_stopping_point: (optional) the maximum number of epoches allowed; the default value is 1000
    @param convergence_tolerance: (optional) the threshold to decide whether the gradient descent converges; the default value is 0.001
    @param batch_size: (optional) the batch size of mini-batch gradient descent; the default value is 1
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @param loss_plot_mode: (optional) a boolean value that indicates whether the loss plot mode is active; the default value is true
    @return a boolean value that indicates whether the fitting is successful
    """
    def fit(self, X, Y_actual, learning_rate=0.001, decay_rate=0.1, early_stopping_point=1000, convergence_tolerance=0.001, batch_size=1, debug_mode=False, loss_plot_mode=True):
        # reconfigure the model setting
        self.__n_x = X.shape[0]
        self.__w = np.random.randn(self.__n_x, 1)
        self.__b = np.random.randn(1, 1)
        self.__epoch_mean_squared_errors = []
        self.__iterative_mean_squared_errors = []
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
        num_batches = m // batch_size + (m % batch_size > 0)
        X_batches = np.array_split(X, num_batches, axis=1)
        Y_actual_batches = np.array_split(Y_actual, num_batches, axis=1)
        # epoches of gradient descent
        for epoch in range(early_stopping_point):
            # get the epoch mean squared error (MSE) and add to epoch fitting log
            Y_predicted = self.__get_Y_predicted(self.__w, X, self.__b, debug_mode=debug_mode)
            epoch_mean_squared_error = self.__get_mean_squared_error(Y_predicted=Y_predicted, Y_actual=Y_actual, debug_mode=debug_mode)
            if debug_mode:
                print("Epoch " + str(epoch) + "\t MSE = " + str(epoch_mean_squared_error))
            self.__epoch_mean_squared_errors.append(epoch_mean_squared_error)
            if epoch_mean_squared_error < convergence_tolerance:
                if debug_mode:
                    print("Message: convergence_tolerance reached at epoch " + str(epoch))
                    print("\tStack trace: NumPyBasedLinearRegression.fit()")
                break
            # iterate through batches
            for batch_index in range(num_batches):
                # get the batch based on batch index
                X_batch = X_batches[batch_index]
                Y_actual_batch = Y_actual_batches[batch_index]
                # get the iterative mean squared error (MSE) and add to iterative fitting log
                Y_predicted_batch = self.__get_Y_predicted(self.__w, X_batch, self.__b, debug_mode=debug_mode)
                iterative_mean_squared_error = self.__get_mean_squared_error(Y_predicted=Y_predicted_batch, Y_actual=Y_actual_batch, debug_mode=debug_mode)
                self.__iterative_mean_squared_errors.append(iterative_mean_squared_error)
                # get the gradients
                dw, db = self.__get_gradients(Y_predicted=Y_predicted_batch, Y_actual=Y_actual_batch, X=X_batch, debug_mode=debug_mode)
                # learning rate decay
                decayed_learning_rate = (1 / (1 + decay_rate * epoch)) * learning_rate
                # update the parameters
                self.__w, self.__b = self.__update_parameters(w=self.__w, b=self.__b, learning_rate=decayed_learning_rate, dw=dw, db=db, debug_mode=debug_mode)
        if loss_plot_mode:
            # plot epoch mean squared errors
            plt.plot(self.__epoch_mean_squared_errors)
            plt.title("NumPy-based Linear Regression, batch size = " + str(batch_size) + "\nEpoch Mean Squared Errors\nErnest Xu")
            plt.xlabel("Epoch")
            plt.ylabel("Mean Squared Error")
            plt.show()
            # plot iterative mean squared errors
            plt.plot(self.__iterative_mean_squared_errors)
            plt.title("NumPy-based Linear Regression, batch size = " + str(batch_size) + "\nIterative Mean Squared Errors\nErnest Xu")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Squared Error")
            plt.show()
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

    """
    Get the coefficient of determination of the model

    @param X: the NumPy array of the input variables
    @param Y_predicted: the NumPy array of the predicted values of the output variable
    @param Y_actual: the NumPy array of the actual values of the output variable
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @return the coefficient of determination of the model
    """
    def get_coefficient_of_determination(self, X, Y_predicted, Y_actual, debug_mode=False):
        # check the validity of member variables
        if self.__n_x == None or self.__w == None or self.__b == None:
            if debug_mode:
                print("Error: the model has not been trained")
                print("\tStack trace: NumPyBasedLinearRegression.predict()")
            return None
        # check the dimension of X & Y_predicted & Y_actual
        if X.shape[1] != Y_predicted.shape[1] or Y_predicted.shape[1] != Y_actual.shape[1]:
            if debug_mode:
                print("Error: X.shape[1] != Y_predicted.shape[1] or Y_predicted.shape[1] != Y_actual.shape[1]")
                print("\tStack trace: NumPyBasedLinearRegression.get_coefficient_of_determination()")
            return None
        # find the number of examples m
        m = Y_actual.shape[1]
        # calculate the mean of actual data
        Y_mean = (1 / m) * np.sum(Y_actual)
        # calculate the total sum of squares
        total_sum_of_squares = np.sum(np.square(Y_actual - Y_mean))
        # calculate the regression sum of squares
        regression_sum_of_squares = np.sum(np.square(Y_predicted - Y_mean))
        # calculate the coefficient of determination
        coefficient_of_determination = regression_sum_of_squares / total_sum_of_squares
        return coefficient_of_determination
