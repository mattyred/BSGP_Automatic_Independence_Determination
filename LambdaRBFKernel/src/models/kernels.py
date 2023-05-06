import tensorflow as tf # 2.11.0
import gpflow # 2.7.0
import tensorflow_probability as tfp
import numpy as np
import src.models.utils as utils

class LambdaRBF(gpflow.kernels.Kernel):  

    def __init__(self, **kwargs):
        randomized = kwargs["randomized"]
        d = kwargs["d"]
        variance = kwargs["variance"]
        if not randomized:
            L = utils.get_lower_triangular_from_diag(d)
        else:
            L = utils.get_lower_triangular_uniform_random(d)
        super().__init__()
        self.L = gpflow.Parameter(L, transform=None, dtype=tf.float64, name='L')
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), dtype=tf.float64, name='variance')

    def K(self, X, X2=None):
        """
            X: matrix NxD
            X2: matrix NxD
            ---
            Returns Kernel matrix as a 2D tensor
        """
        if X2 is None:
            X2 = X
        #N1 = X.shape[0] WORKS
        #N2 = X2.shape[0] WORKS
        N1 = tf.squeeze(tf.shape(X)[:-1])
        N2 = tf.squeeze(tf.shape(X2)[:-1])
        Lambda = self.precision() # recover LLᵀ

        # compute z, z2
        z = self._z(X, Lambda) # N1x1 array
        z2 = self._z(X2, Lambda) # N2x1 array
        # compute X(X2Λ)ᵀ
        X2Lambda = tf.linalg.matmul(X2, Lambda)
        XX2LambdaT = tf.linalg.matmul(X, tf.transpose(X2Lambda)) # N1xN2 matrix
        # compute z1ᵀ 
        ones_N2 = tf.ones(shape=(N2,1), dtype=tf.float64) # N2x1 array
        zcol = tf.linalg.matmul(z, tf.transpose(ones_N2)) # N1xN2 matrix
        # compute 1z2ᵀ 
        ones_N1 = tf.ones(shape=(N1,1), dtype=tf.float64) # N1x1 array
        zrow = tf.linalg.matmul(ones_N1, tf.transpose(z2)) # N1xN2 matrix

        exp_arg = zcol - 2*XX2LambdaT + zrow
        Kxx = tf.math.exp(-0.5 * exp_arg)
        return self.variance * Kxx
    
    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
    def _z(self, X, Lambda):
        XLambda = tf.linalg.matmul(X, Lambda)
        XLambdaX = tf.math.multiply(XLambda, X)
        return tf.math.reduce_sum(XLambdaX, axis=1, keepdims=True)
    
    def precision(self):
        L = tfp.math.fill_triangular(self.L) # recover L matrix from L array
        Lambda = tf.linalg.matmul(L, tf.transpose(L))
        return Lambda
    
    def __str__(self):
        Lambda = self.precision()
        return 'Variance: {}\nLambda: {}'.format(self.variance, Lambda)
    
    def precision_dict(self, reverse=True):
        precision = np.abs(self.precision().numpy())
        precision_dict = []
        d = precision.shape[0]
        for i in range(d):
            for j in range(d):
                precision_dict.append({'i': i, 'j': j, 'l': precision[i][j]}) if i <= j else None
        return sorted(precision_dict, key=lambda d: d['l'], reverse=reverse) 
    
    def K_Lambda(self, X, Lambda, X2=None):
        """
            X: matrix NxD
            X2: matrix NxD
            ---
            Returns Kernel matrix as a 2D tensor
        """
        if X2 is None:
            X2 = X
        N1 = tf.squeeze(tf.shape(X)[:-1])
        N2 = tf.squeeze(tf.shape(X2)[:-1])

        # compute z, z2
        z = self._z(X, Lambda) # N1x1 array
        z2 = self._z(X2, Lambda) # N2x1 array
        # compute X(X2Λ)ᵀ
        X2Lambda = tf.linalg.matmul(X2, Lambda)
        XX2LambdaT = tf.linalg.matmul(X, tf.transpose(X2Lambda)) # N1xN2 matrix
        # compute z1ᵀ 
        ones_N2 = tf.ones(shape=(N2,1), dtype=tf.float64) # N2x1 array
        zcol = tf.linalg.matmul(z, tf.transpose(ones_N2)) # N1xN2 matrix
        # compute 1z2ᵀ 
        ones_N1 = tf.ones(shape=(N1,1), dtype=tf.float64) # N1x1 array
        zrow = tf.linalg.matmul(ones_N1, tf.transpose(z2)) # N1xN2 matrix

        exp_arg = zcol - 2*XX2LambdaT + zrow
        Kxx = tf.math.exp(-0.5 * exp_arg)
        return self.variance * Kxx
    
    def covariance_dict(self, reverse=True):
        covariance = np.abs(tf.linalg.inv(self.precision()).numpy())
        covariance_dict = []
        d = covariance.shape[0]
        for i in range(d):
            for j in range(d):
                covariance_dict.append({'i': i, 'j': j, 'c': covariance[i][j]}) if i <= j else None
        return sorted(covariance_dict, key=lambda d: d['c'], reverse=reverse) 
    
class ARD_gpflow(gpflow.kernels.SquaredExponential):
    def __init__(self, **kwargs):
        randomized = kwargs["randomized"]
        d = kwargs["d"]
        variance = kwargs["variance"]
        if not randomized:
            lengthscales = tf.constant([d**0.5]*d, dtype=tf.float64)
        else:
            lengthscales = np.random.uniform(0.5,3,d)      
        super().__init__(variance, lengthscales)
    
    def precision(self) -> tf.Tensor:
        return tf.linalg.diag(self.lengthscales**(-2))  
    
    def lengthscales_dict(self, reverse=True):
        lengthscales = self.lengthscales.numpy()
        lengthscales_dict = []
        d = lengthscales.shape[0]
        for i in range(d):
            lengthscales_dict.append({'i': i, 'j': i, 'l': lengthscales[i]})
        return sorted(lengthscales_dict, key=lambda d: d['l'], reverse=reverse) 