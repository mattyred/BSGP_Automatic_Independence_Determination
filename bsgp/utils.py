import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
import sys

def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = tf.linalg.cholesky(var + tf.eye(tf.shape(mean)[0], dtype=tf.float64)[None, :, :] * 1e-7)
        rnd = tf.transpose(tf.squeeze(tf.matmul(chol, tf.random.normal(tf.shape(tf.transpose(mean)), dtype=tf.float64)[:, :, None])))
        return mean + rnd
    return mean + tf.random.normal(tf.shape(mean), dtype=tf.float64) * tf.sqrt(var)

def get_lower_triangular_from_diag(d):
    """
    diag: diagonal of lengthscales parameter [D,]
    ---
    Σ=inv(Λ) -> diagonal matrix with lengthscales on the diagonal (RBF)
    The diagonal of Λ is obtained as 1/(l^2), l is a lengthscale
    returns: L, Λ=LLᵀ
    """
    # Define the lengthscales according to the standard RBF kernel
    lengthscales = np.full((d,), d**0.5, dtype=np.float64) # lengthscales = tf.constant([d**0.5]*d, dtype=tf.float64)
    # Obtain the matrix L such that LLᵀ=Λ and Λ=inv(diag(lengthscales))
    Lambda = np.diag(1/(lengthscales**2)) # Lambda = tf.linalg.diag(1/(lengthscales**2))
    L = scipy.linalg.cholesky(Lambda, lower=True) # L = Cholesky(inv(diag(lengthscales)))
    return tfp.math.fill_triangular_inverse(L, upper=False) 

def get_lower_triangular_uniform_random(d):
    L = np.random.uniform(-1,1,(d*(d+1)//2,))
    Lambda = L @ np.transpose(L) # Λ=LLᵀ
    L = scipy.linalg.cholesky(Lambda + np.eye(d) * 1e-7 , lower=True)
    return tfp.math.fill_triangular_inverse(L, upper=False)

def commutation_matrix(m, n):
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")
    return np.eye(m * n)[w, :]

def apply_pca(X, n_comp):
    N = X.shape[0]
    C = (1/N) * X.T @ X # N x N
    A, P = np.linalg.eigh(C) # C = PAPᵀ
    Pd = P[:, ::-1][:, 0:n_comp] # D x d
    Z = X @ Pd # N x d
    return Z, Pd