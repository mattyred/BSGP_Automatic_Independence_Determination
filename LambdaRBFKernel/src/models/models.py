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
        self.num_inducing = kwargs["num_inducing"]
        self.max_iter = kwargs["max_iter"]
        self.minibatch_size = kwargs["minibatch_size"]
        self.likelihood = kwargs["likelihood"]

        Z = self.data[0][:self.num_inducing, :].copy()
        self.gp_model = super(SVGPLasso, self).__init__(self.kernel, self.likelihood, Z, num_data=self.num_inducing)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.data[0], self.data[1])).repeat().shuffle(self.data[0].shape[0])

        self.lasso = 0 if "lasso" not in kwargs else kwargs["lasso"]
        d = self.data[0].shape[1]

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return super().log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(self.kernel.precision()))
    
    def summary(self):
        print('Kernel variance: %1.1f'%(self.kernel.variance.numpy()))
        print('Lambda diagonal: ', tf.linalg.diag_part(self.kernel.precision()).numpy())

    def _run_adam(self, model, train_dataset, minibatch_size, iterations):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        return logf

    def train(self):
        self.logf = self._run_adam(self.gp_model, self.train_dataset, self.minibatch_size, self.max_iter)
    