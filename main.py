from regression import *

(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_prepare_data()  # load all data once


def run_Regression(_lambda=0.0, tot_iter=2000, alpha=0.1, threshold=0.5 - 1e-6, name='Logistic Regression Classifier'):
    """
    A function that creates and trains a Logistic Regression Classifier and then checks its accuracy for
    train, validation and test datasets
    :param _lambda: The lambda value used for L2 normalization (leave as is for no L2 normalization)
    :param tot_iter: The total amount of iterations
    :param alpha: The learning rate
    :param threshold: The threshold used for predicting
    :param name: The name of the model
    :return:
    """
    j_train, j_valid, theta = ComputeLogisticRegression(X_train, y_train, X_valid, y_valid, _lambda, tot_iter, alpha)
    print_Predictions(theta, X_test, y_test, name, threshold)

    return