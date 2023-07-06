import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy

def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = tf.linalg.cholesky(var + tf.eye(tf.shape(mean)[0], dtype=tf.float64)[None, :, :] * 1e-7)
        rnd = tf.transpose(tf.squeeze(tf.matmul(chol, tf.random.normal(tf.shape(tf.transpose(mean)), dtype=tf.float64)[:, :, None])))
        return mean + rnd
    return mean + tf.random.normal(tf.shape(mean), dtype=tf.float64) * tf.sqrt(var)

def get_upper_triangular_from_diag(d):
    """
    diag: diagonal of lengthscales parameter [D,]
    ---
    Σ=inv(Λ) -> diagonal matrix with lengthscales on the diagonal (RBF)
    The diagonal of Λ is obtained as 1/(l^2), l is a lengthscale
    returns: L, Λ=UᵀU
    """
    # Define the lengthscales according to the standard RBF kernel
    lengthscales = np.full((d,), d**0.5, dtype=np.float64) # lengthscales = tf.constant([d**0.5]*d, dtype=tf.float64)
    # Obtain the matrix U such that UᵀU=Λ and Λ=inv(diag(lengthscales))
    Lambda = np.diag(1/(lengthscales**2)) # Lambda = tf.linalg.diag(1/(lengthscales**2))
    Up = scipy.linalg.cholesky(Lambda, lower=False) # Up = Cholesky(inv(diag(lengthscales)))
    return tfp.math.fill_triangular_inverse(Up, upper=True) 

def get_upper_triangular_uniform_random(d):
    full_Up = np.random.uniform(-1,1,(d,d))
    Lambda = np.transpose(full_Up) @ full_Up # Λ=UᵀU
    Up = scipy.linalg.cholesky(Lambda, lower=False)
    return tfp.math.fill_triangular_inverse(Up, upper=True) 
