import unittest
import sys
import numpy as np
import numpy.testing

from src.logistic_reg import sigmoid
from src.logistic_reg import regression
from src.logistic_reg import training
from src.logistic_reg import gradient
from src.logistic_reg import get_accuracy

class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        max_output = sigmoid(sys.maxsize)
        # this will retunrs a rumtime warning for overflow
        min_output = sigmoid(-1 * sys.maxsize -1)
        mid_output = sigmoid(0)

        self.assertAlmostEqual(1.0, max_output, 7)
        self.assertAlmostEqual(0.0, min_output, 7)
        self.assertEqual(0.5, mid_output)

        # test for np.array
        array_output = np.array([0.5, 0.5, 0.5])
        array_input = np.zeros(3)
        numpy.testing.assert_array_equal(array_output, sigmoid(array_input))

    def test_regression(self):
        # precalculated example to test regression equation
        w = np.array([2, 3, -1])
        X = np.array([4, -1, 2])
        b = -8

        self.assertEqual(-5, regression(X, w, b))

    def test_gradient(self):
        y_exp = np.array([0.8, 0.2, 0.7])
        y = np.array([1, 0, 1])

        grad = gradient(y_exp, y)

        print(grad)

    def test_accuracy(self):

        y_hat = np.array([1, 0 , 1, 0])
        y = np.array([1, 1, 1, 1])

        self.assertEqual(0.5, get_accuracy(y_hat, y))