import tensorflow as tf # 2.11.0
import gpflow # 2.7.0
import tensorflow_probability as tfp

class LambdaRBF(gpflow.kernels.Kernel):  
    def __init__(self, Lambda_L, variance=1.0):
        super().__init__()
        # Create a Parameter associated to Lambda matrix
        # self.Lambda_L = gpflow.Parameter(Lambda_L, transform=gpflow.utilities.triangular(), dtype=tf.float64, name='KernelPrecision_L') OK
        self.Lambda_L = gpflow.Parameter(Lambda_L, transform=None, dtype=tf.float64, name='KernelPrecision_L')
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), dtype=tf.float64, name='KernelAmplitude')
        #self.Kxx = tf.Variable(np.empty((N, N), dtype=np.float64), name='KernelMatrix')

    def K(self, X, X2=None):
        """
            X: matrix NxD
            X2: matrix NxD
            ---
            Returns Kernel matrix as a 2D tensor
        """
        if X2 is None:
            X2 = X
        N1 = X.shape[0]
        N2 = X2.shape[0]
        Lambda_L = tfp.math.fill_triangular(self.Lambda_L) # TRY
        Lambda = tf.linalg.matmul(Lambda_L, tf.transpose(Lambda_L))
        # Lambda = tf.linalg.matmul(self.Lambda_L, tf.transpose(self.Lambda_L)) # recover LLᵀ

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
        N = X.shape[0]
        return tf.fill([N,], self.variance)  # this returns a 1D tensor
        #return tf.linalg.diag_part(self.K(X))
    
    def _z(self, X, Lambda):
        XLambda = tf.linalg.matmul(X, Lambda)
        XLambdaX = tf.math.multiply(XLambda, X)
        return tf.math.reduce_sum(XLambdaX, axis=1, keepdims=True)
    
    def _to_array(self, L):
        D = tf.shape(L).numpy()[0]
        return tf.reshape(L, [D*2,1])
    
    def _to_matrix(self, l):
        D = int(tf.shape(l).numpy()[0]/2)
        return tf.reshape(l, [D,D])
