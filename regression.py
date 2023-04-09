from functions import *
import numpy as np


def ComputeCostGrad(X, y, theta, _lambda):
    """
    Calculates the current values of the cost function and the gradient
    :param X: the X array of the set
    :param y: the y vector (technically a numpy array of (y,1) shape) of the set
    :param theta: the current theta regression parameters
    :param _lambda: the lambda value used for L2 regularization (_lambda=0 for no regularization)
    :return: current values of cost, gradient
    """
    h = sigmoid(X.dot(theta))  # hypothesis h_theta

    # calculate cost function
    regc = 0
    if _lambda != 0:
        regc = (_lambda / (2.0)) * np.sum(theta ** 2)  # reguralization
    cur_j = ((y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) - regc) / X.shape[0]

    # calculate gradient
    regg = 0
    if _lambda != 0:
        regg = _lambda * theta
    grad = (X.T.dot(y - h) - regg) / y.shape[0]

    return cur_j, grad


def ComputeLogisticRegression(X, y, X_valid, y_valid, _lambda, tot_iter, alpha):
    """
    The main function that trains the logistic regression model by calculating the error (Cost) for
    the train and validation sets as well as the gradient for the train set, then adjusting the theta parameters
    based on the train gradient for a certain amount of iterations.
    :param X: the X array of the train set
    :param y: the y vector (technically a matrix of (y,1) shape) of the training set
    :param X_valid: the X array of the validation set
    :param y_valid: the y vector (technically a numpy array of (y,1) shape) of the validation set
    :param _lambda: the lambda value used for L2 normalization (default value at 0.0, no normalization)
    :param tot_iter: the number of iterations
    :param alpha: the learning rate at which the model's theta parameters change
    :return: two arrays that contain the error for the train and validation sets for all iteration
             and the final values of the theta parameters
    """
    theta = np.zeros(X.shape[1]).reshape((-1, 1))

    J_train = []
    J_valid = []

    for i in range(tot_iter):
        train_error, train_grad = ComputeCostGrad(X, y, theta, _lambda)
        valid_error, _ = ComputeCostGrad(X_valid, y_valid, theta, _lambda)

        # update parameters by subtracting gradient values
        theta = theta + alpha * train_grad

        # store current cost
        J_train.append(train_error[0])
        J_valid.append(valid_error[0])

    return J_train, J_valid, theta


def predict(theta, X, threshold=0.5 - 1e-6):
    """
    A function used to predict whether the label of a set's items is 0 or 1 using the regression parameters learned
    previously through the ComputeLogisticRegression function
    :param theta: the regression parameters
    :param X: the X array of the set
    :param threshold: the threshold for the prediction (if sigmoid(X*theta)>=threshold, predict 1)
    :return: two arrays of shape (m,1) where m is the amount of elements contained in X
             one of them contains the raw values of the sigmoid function and the other contains the prediction (0 or 1)
    """

    p = sigmoid(np.dot(X, theta))
    prob = p
    p = p > threshold

    return p, prob


def print_Predictions(theta, X, Y, name='Logistic Regression Classifier', threshold=0.5 - 1e-6):
    """
    A function used to print out the results of the predict function for a test set.
    :param theta: the regression parameters
    :param X: the X array of the set
    :param Y: the Y array of the set
    :param name: the name of the classifier
    :param threshold: the threshold for the prediction (if sigmoid(X*theta)>=threshold, predict 1)
    """
    p, prob = predict(theta, X, threshold)
    print('Model: ' + name + ', Accuracy of test set: ', np.mean(p.astype('int') == Y))
