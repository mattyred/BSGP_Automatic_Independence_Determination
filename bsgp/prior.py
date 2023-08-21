import tensorflow as tf
import tensorflow_probability as tfp


def _integral_function_approx(u):
    gamma = tf.constant(0.5772156649015328606, dtype=tf.float64) # Euler's constant
    G = tf.math.exp(-gamma)
    b = tf.math.sqrt(2*(1 - G) / (G*(2-G)))
    hinf = ((1-G)*(tf.math.square(G) - 6*G + 12)) / (3*G*tf.math.square(2-G)*b)
    q = (20/47)*tf.math.pow(u, tf.cast(tf.math.sqrt(31/26), dtype=tf.float64))
    h = 1/(1+u*tf.math.sqrt(u)) + hinf*q/(1+q)
    return (tf.math.exp(-u)*tf.math.log((1 + G/u - (1-G)/tf.square(h+b*u)))) / (G + (1-G)*tf.math.exp(-u/(1-G)))

def logdet_jacobian(L, eps=1e-6):
    L = tfp.math.fill_triangular(L, upper=False)
    n = L.shape[0]
    diag_L = tf.linalg.tensor_diag_part(L) 
    exps = tf.cast(tf.reverse(tf.range(n) + 1, axis=[0]), dtype=L.dtype)
    return tf.cast(n*tf.math.log(2.0), dtype=L.dtype) + tf.reduce_sum(tf.math.multiply(exps,tf.math.log(tf.math.abs(diag_L)))) #tf.math.log(tf.math.abs((2.0**n) * tf.reduce_prod(tf.pow(diag_L,exps))))

def horseshoe_logprob(X, scale):
    u = tf.square(X) / (2*scale**2)
    return tf.reduce_sum(u + tf.math.log(_integral_function_approx(u)))

def horseshoe_logprob_tf(X, scale):
    X = tf.cast(X,  dtype=tf.float32)
    return tf.reduce_sum(tf.cast(tfp.distributions.Horseshoe(scale=scale).log_prob(X), dtype=tf.float64))

def matrix_normal_logprob(X):
    return -0.5 * tf.linalg.trace(tf.matmul(X, tf.transpose(X)))

def matrix_wishart_logprob(L, P):
    L = tfp.math.fill_triangular(L)
    n = L.shape[0]
    return -tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64),tf.math.abs(L))))) - tf.linalg.trace(n*P) / 2.0

def matrix_invwishart_logprob(L, P):
    L = tfp.math.fill_triangular(L)
    n = L.shape[0]
    return -(2*n + 1) * tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64),tf.math.abs(L))))) - tf.linalg.trace(tf.linalg.inv(P)) / 2.0

def laplace_logprob(P, b=0.01):
    return -tf.reduce_sum(tf.norm(P, ord=1) / b) 

def normal_logprob(X, m=0, v=1):
    return -tf.reduce_sum(tf.square((X-m)/v)) / 2.0