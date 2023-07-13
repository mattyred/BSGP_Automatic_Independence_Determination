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

class SVGPLasso(gpflow.models.SVGP):
    
    def __init__(self, **kwargs):
        self.data = kwargs["data"]
        self.kernel = kwargs["kernel"]
        self.num_inducing = kwargs["num_inducing"] # M
        self.max_iter = kwargs["max_iter"]
        self.minibatch_size = kwargs["minibatch_size"]
        self.likelihood = kwargs["likelihood"]

        Z = self.data[0][:self.num_inducing, :].copy()
        N = self.data[0].shape[0]
        super(SVGPLasso, self).__init__(self.kernel, self.likelihood, Z, num_data=N)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.data[0], self.data[1])).repeat().shuffle(self.data[0].shape[0])

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]
        d = self.data[0].shape[1]
    
    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data) - self.lasso*tf.math.reduce_sum(tf.abs(self.kernel.precision()))
    
    def summary(self):
        print('Kernel variance: %1.1f'%(self.kernel.variance.numpy()))
        print('Lambda diagonal: ', tf.linalg.diag_part(self.kernel.precision()).numpy())

    def _run_adam(self):
        """
        Utility function running the Adam optimizer
        """
        print('ADAM started...')
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(self.train_dataset.batch(self.minibatch_size))
        training_loss = self.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, self.trainable_variables)

        for step in range(self.max_iter):
            optimization_step()
            if step % 100 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        print('ADAM finished')
        return logf

    def train(self):
        self.logf = self._run_adam()
    