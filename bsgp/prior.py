import tensorflow as tf
import tensorflow_probability as tfp

def logdet_jacobian(L, eps=1e-6):
    L = tfp.math.fill_triangular(L, upper=False)
    n = L.shape[0]
    diag_L = tf.linalg.tensor_diag_part(L) 
    exps = tf.cast(tf.reverse(tf.range(n) + 1, axis=[0]), dtype=L.dtype)
    return tf.cast(n*tf.math.log(2.0), dtype=L.dtype) + tf.reduce_sum(tf.math.multiply(exps,tf.math.log(tf.math.abs(diag_L)))) #tf.math.log(tf.math.abs((2.0**n) * tf.reduce_prod(tf.pow(diag_L,exps))))

def horseshoe_logprob(hs, X):
    X = tf.cast(X, dtype=tf.float32) # to be input of hs.log_prob
    #tf.print({'X': X}, output_stream=sys.stderr)
    #X = X + 1e-1 #tf.where(X == 0, 1e-1, X)
    X = tf.boolean_mask(X, tf.not_equal(X, 0))
    hs_log_prob = hs.log_prob(X) # remove 0 elements
    #tf.print({'X': hs_log_prob, 'hs_log_prob': tf.reduce_sum(tf.cast(hs_log_prob, dtype=tf.float64))}, output_stream=sys.stderr)
    #hs_log_prob = tf.where(tf.math.is_inf(hs_log_prob), hs.log_prob(0.1), hs_log_prob)
    #tf.print({'hs_log_prob_noinf': tf.cast(hs_log_prob, dtype=tf.float64)}, output_stream=sys.stderr)
    return tf.reduce_sum(tf.cast(hs_log_prob, dtype=tf.float64)) # to be input of reduce_sum

def matrix_normal_logprob(X):
    return -0.5 * tf.linalg.trace(tf.matmul(X, tf.transpose(X)))

def matrix_wishart_logprob(L, P):
    L = tfp.math.fill_triangular(L)
    n = L.shape[0]
    return -tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64), tf.math.abs(L))))) - tf.linalg.trace(tf.linalg.matmul((1/n)*tf.eye(n, dtype = tf.float64),P)) / 2

def matrix_invwishart_logprob(L, P):
    L = tfp.math.fill_triangular(L)
    n = L.shape[0]
    return -(2*n + 1) * tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64),tf.math.abs(L))))) - tf.linalg.trace(tf.linalg.inv(P)) / 2

def laplace_logprob(P, b=0.01):
    return -tf.reduce_sum(tf.norm(P, ord=1) / b)

def normal_logprob(X, m=0, v=1):
    return -tf.reduce_sum(tf.square(X)) / 2.0