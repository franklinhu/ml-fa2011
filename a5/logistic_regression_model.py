import math
import random

import numpy

ALPHA = 0.001 # Stochastic gradient descent step size

class LogisticRegressionModel:

    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [0] * num_features
        self.update_counter = 0

    def update_weights(self, example, cls):
        w = self.weights

        logit_grad_const = self._logistic_gradient_const(example, cls)
        
        for i in xrange(self.num_features):
            d = ALPHA * cls * example[i] * logit_grad_const
            w[i] += d

    def predict(self, example):
        return self._g(numpy.dot(self.weights, example))

    def _logistic_gradient_const(self, example, cls):
        return 1. - self._g(numpy.dot(self.weights, example))

    def _g(self, z):
        return 1. / (1 + math.e ** -z)

    def _normalize_weights(self):
        pass

