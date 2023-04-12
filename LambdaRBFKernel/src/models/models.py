import tensorflow as tf
import gpflow

class GPRLasso(gpflow.models.GPR):
    """
    Basic Gaussian process regression, but L1 penalty term is added to 
    the loss. This model assumes that the underlying kernel is either 
    full Gaussian kernel or ARD kernel.
    """
    
    def __init__(self, **kwargs):

        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(GPRLasso, self).__init__((data[0], data[1]), kernel)

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]
        d = self.data[0].shape[1]

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return super().log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(self.kernel.precision()))
    
    def summary(self):
        print('Kernel variance: %1.1f'%(self.kernel.variance.numpy()))
        print('Lambda diagonal: ', tf.linalg.diag_part(self.kernel.precision()).numpy())
    
    def train(self):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.training_loss, self.trainable_variables)
    
class GPR_gpflow(gpflow.models.GPR):

    def __init__(self, **kwargs):
        data = kwargs["data"]
        kernel = kwargs["kernel"]
        super(GPR_gpflow, self).__init__(data, kernel)

    def train(self):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.training_loss, self.trainable_variables)
    