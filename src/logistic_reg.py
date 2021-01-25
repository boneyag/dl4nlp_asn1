import numpy as np
from numpy import random
import random
import pickle

import matplotlib
from numpy.core.fromnumeric import shape
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def sigmoid(z):
    """
    Implementation of the sigmoid function. Purpose of this function
    is to keep y values [0, 1]. Works for scaler or vector inputs.
    """
    s = 1.0 / ( 1.0 + np.exp(-z ))

    return s

def regression(X, w, b):
    """
    Calcuate the linear regression for a given input vectors
    X - input matrix with dims (D,M) where D-# features and M-# instances
    w - weight vector with dims(D,1)
    b - scaler bias
    z - output vector with dims(D,1)
    """
    z = np.dot(w.T, X) + b

    return z

def gradient(y_expected, y_train):
    """
    Calculate the gradient
    """
    grad = np.multiply((y_expected - y_train), y_expected)
    grad = np.multiply(grad, (1 - y_expected))
    # grad = np.sum(grad)/len(y_expected)
    return grad

def get_accuracy(y_hat, y):
    """
    Calculate the accuracy on a set of predictions.
    """
    return np.sum(y_hat == y)/len(y_hat)


def training(X_train, y_train, X_val, y_val):
    """
    Training the logistic regression model with mini-batch gradient descent
    """
    

    # initialize the parameters
    w = np.random.uniform(-0.5, 0.5, 2000)
    b = random.uniform(-0.5, 0.5)

    alpha = 0.1 # fixed learning rate

    [_, instances] = X_train.shape

    error = list()
    training_accuracy = list()
    validation_accuracy = list()

    model = {'best_w': w,
             'best_b': b}

    for i in range(1,301,1):
        current_instance = 0
        for j in range(1, int(instances/20)+1, 1):

            # slice a mini-batch
            X = X_train[:, current_instance:current_instance+20]
            y = y_train[current_instance:current_instance+20]
            current_instance += 20

            y_expected = sigmoid(regression(X, w, b))
            grad = gradient(y_expected, y)

            w = w - alpha * np.sum(np.dot(X, grad.T))/len(y)
            b = b - alpha * np.sum(grad)/len(y)

        pred = sigmoid(regression(X_train, w, b))
        pred_c = (pred > 0.5).astype(int)

        training_accuracy.append(get_accuracy(pred_c, y_train))

        pred = sigmoid(regression(X_val, w, b))
        pred_c = (pred > 0.5).astype(int)

        val_accuracy = get_accuracy(pred_c, y_val)
        
        if i == 1:
            model["best_w"] = w
            model["best_b"] = b
        elif i > 1 and val_accuracy > validation_accuracy[-1]:
            model["best_w"] = w
            model["best_b"] = b
        
        validation_accuracy.append(val_accuracy)

    plot(training_accuracy)
    plot(validation_accuracy)

    pickle.dump((training_accuracy, validation_accuracy), open("../serialized/plot.pkl", "wb"))
    return model

def test_model(X_test, y_test):
    """
    Test the final model's accuracy. Model was selected based on the highest accuracy on the vaidation set. 
    """
    model = pickle.load(open("../serialized/log_model2.pkl", "rb"))
    y_expected = sigmoid(regression(X_test, model["best_w"], model["best_b"]))

    pred_c = (y_expected > 0.5).astype(int)

    accuracy = get_accuracy(pred_c, y_test)

    print(accuracy)

def plot(data):
    """
    Plot a graph.
    """

    plt.plot(data)
    plt.ylabel("some numbers")
    plt.show()
