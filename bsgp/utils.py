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
    full_L = np.random.uniform(-1,1,(d,d))
    Lambda = full_L @ np.transpose(full_L)# Λ=LLᵀ
    L = scipy.linalg.cholesky(Lambda, lower=True)
    return tfp.math.fill_triangular_inverse(L, upper=False) 

def commutation_matrix(m, n):
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")
    return np.eye(m * n)[w, :]

"""
@tf.function
def logdet_jacobian(Kc, U):
    #tf.print({'vecU': U}, output_stream=sys.stderr)
    U = tfp.math.fill_triangular(U, upper=True)
    n = U.shape[0]
    UmT = tf.linalg.LinearOperatorFullMatrix([tf.transpose(U)])
    I = tf.linalg.LinearOperatorFullMatrix([tf.eye(n, dtype=tf.float64)])
    kron1 = tf.squeeze(tf.linalg.LinearOperatorKronecker([UmT, I]).to_dense())
    kron2 = tf.squeeze(tf.linalg.LinearOperatorKronecker([I, UmT]).to_dense())
    J = tf.linalg.matmul(kron1, Kc) + kron2
    eigs = tf.math.real(tf.linalg.eigvals(J)) 
    #tf.print({'eigs shape': tf.shape(eigs)}, output_stream=sys.stderr)
    eigs = tf.sort(eigs, direction='DESCENDING')
    eigs_nonzero = eigs[0:n*(n+1)//2]
    logdet = tf.reduce_sum(tf.math.log(tf.math.abs(eigs_nonzero)))
    return eigs, eigs_nonzero, logdet
"""
@tf.function
def logdet_jacobian(L, eps=1e-6):
    #tf.print({'vecL': L}, output_stream=sys.stderr)
    L = tfp.math.fill_triangular(L, upper=False)
    n = L.shape[0]
    diag_L = tf.linalg.tensor_diag_part(L) 
    exps = tf.cast(tf.reverse(tf.range(n) + 1, axis=[0]), dtype=L.dtype)
    #tf.print({'e': diag_L**exps}, output_stream=sys.stderr)
    return tf.cast(n*tf.math.log(2.0), dtype=L.dtype) + tf.reduce_sum(tf.math.multiply(exps,tf.math.log(tf.math.abs(diag_L) + eps))) #tf.math.log(tf.math.abs((2.0**n) * tf.reduce_prod(tf.pow(diag_L,exps))))