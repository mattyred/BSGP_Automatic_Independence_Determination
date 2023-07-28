# Credit to GPFlow.

import tensorflow as tf
import numpy as np
from .quadrature import ndiagquad
import sys
from gpflow.likelihoods.scalar_discrete import Bernoulli

class Gaussian(object):
    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu-x) / var)

    def __init__(self, variance=1.0, **kwargs):
        self.variance = tf.exp(tf.Variable(np.log(variance), dtype=tf.float64, name='lik_log_variance'))

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class BernoulliCustom(Bernoulli):
    def __init__(self, X):
        self.X = X
        super().__init__()

    def conditional_mean(self, F):
        return super()._conditional_mean(self, self.X, F)

    def conditional_variance(self, F):
        return super()._conditional_variance(self, self.X, F)

    def predict_density(self, Fmu, Fvar, Y):
        return super()._predict_log_density(self, self.X, Fmu, Fvar, Y)

    def predict_mean_and_var(self, Fmu, Fvar):
        return super()._predict_mean_and_var(self, self.X, Fmu, Fvar)