import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_lower_triangular_from_diag(d):
    """
    diag: diagonal of lengthscales parameter [D,]
    ---
    Σ=inv(Λ) -> diagonal matrix with lengthscales on the diagonal (RBF)
    The diagonal of Λ is obtained as 1/(l^2), l is a lengthscale
    returns: L, Λ=LLᵀ
    """
    # Define the lengthscales according to the standard RBF kernel
    lengthscales = tf.constant([d**0.5]*d, dtype=tf.float64)
    # Obtain the matrix L such that LLᵀ=Λ and Λ=inv(diag(lengthscales))
    Lambda = tf.linalg.diag(1/(lengthscales**2))
    L = tf.linalg.cholesky(Lambda) # L = Cholesky(inv(diag(lengthscales)))
    return tfp.math.fill_triangular_inverse(L) 

def get_lower_triangular_uniform_random(d):
    full_L = np.random.uniform(-1,1,(d,d))
    Lambda = full_L @ np.transpose(full_L) # Λ=LLᵀ
    L = tf.linalg.cholesky(Lambda)
    return tfp.math.fill_triangular_inverse(L)
