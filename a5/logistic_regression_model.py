
ALPHA = 0.5 # Stochastic gradient descent step size

def inner_product(a, b):
    assert len(a) == len(b)
    return [a[i] * b[i] for i in xrange(len(a))]

class LogisticRegressionModel:

    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [1] * num_features

    def update_weights(self, example):
        new_weights = [0] * self.num_features
        w = self.weights

        # The gradient of the logit function WRT x_i is:
        # -((w_i ** 2) * x_i * (e ** -(wx))) / (1 + (e ** -(wx))) ^ 2
        # Since only x_i * (w_i ** 2) is dependent on each weight/example
        # pair, we can compute the remaining portion just once
        logit_gradient_const = self._logistic_gradient_const(example)

        for i in xrange(self.num_features):
            new_weights[i] = w[i] + ALPHA * self._gradient(w[i], example[i],
                             logit_gradient_const)
        self.weights = new_weights

    def predict(self, example):
        prob = self._logistic(example)
        if prob == 0.5:
            return random.choice([0,1])
        elif prob > 0.5:
            return 1
        else: # prob < 0.5
            return 0

    def _gradient(self, w_i, x_i, logit_gradient_const):
        return (w_i ** 2) * x_i * logit_gradient_const

    def _logistic(self, example):
        return 1. / (1. + self._e_to_the(example))

    def _logistic_gradient_const(self, example):
        e_to_the = self._e_to_the(example)
        return -1. * e_to_the / (1 + e_to_the) ** 2

    def _e_to_the(self, example):
        return e ** -(inner_product(self.weights, example))


    def _normalize_weights(self):
        pass

