import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.metrics import mean_squared_error
import gpflow
from src.models.kernels import LambdaRBF, ARD_gpflow
from src.models.models import GPRLasso, SVGPLasso
import tensorflow as tf

def measure_mnll(model, X_train, Y_train, Ystd, X_test, Y_test):
    mean_train, var_train = model.predict_f(X_train)
    mean_test, var_test = model.predict_f(X_test)
    logps_train = norm.logpdf(np.repeat(Y_train[None, :, :]*Ystd, X_train.shape[0], axis=0), mean_train*Ystd, np.sqrt(var_train)*Ystd)
    train_mnll = -np.mean(logsumexp(logps_train, axis=0) - np.log(X_train.shape[0]))

    mean_test, var_test = model.predict_f(X_test)
    logps_test = norm.logpdf(np.repeat(Y_test[None, :, :]*Ystd, X_test.shape[0], axis=0), mean_test*Ystd, np.sqrt(var_test)*Ystd)
    test_mnll = -np.mean(logsumexp(logps_test, axis=0) - np.log(X_test.shape[0]))
    #train_mnll = -norm.logpdf(Y_train*Ystd, mean_train*Ystd, np.sqrt(var_train)*Ystd).mean()
    #test_mnll = -norm.logpdf(Y_test*Ystd, mean_test*Ystd, np.sqrt(var_test)*Ystd).mean()
    return train_mnll, test_mnll

def measure_rmse(model, X_train, Y_train, X_test, Y_test):
    y_pred_train, _ = model.predict_f(X_train)
    #train_rmse_stan = tf.sqrt(tf.reduce_mean((Y_train - y_pred_train)**2)).numpy()
    train_rmse_stan = mean_squared_error(Y_train, y_pred_train, squared=False)
    y_pred_test, _ = model.predict_f(X_test)
    #test_rmse_stan = tf.sqrt(tf.reduce_mean((Y_test - y_pred_test)**2)).numpy()
    test_rmse_stan = mean_squared_error(Y_test, y_pred_test, squared=False)
    return train_rmse_stan, test_rmse_stan

def sparsity_degree(L, tol=0.1):
    return (np.absolute(L.numpy()) < tol).sum() / L.shape[0]**2

def run_adam(model, train_dataset, minibatch_size, iterations):
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

def kfold_cv_model(model=None, X=None, Y=None, prior=None, kernel=None, k_folds=None, model_params=None, iprint=False):
    if iprint:
        print('-- Model: %s; Kernel: %s; --'%(model, kernel))
    results = {'train_rmse': [], 
                'test_rmse': [], 
                'train_mnll': [], 
                'test_mnll': [], 
                'Lambda': [],
                'sparsity_degree': [],
                'avg_sparsity_degree': [],
                'avg_train_rmse': 0.,
                'avg_test_rmse': 0.,
                'avg_train_mnll': 0.,
                'avg_test_mnll': 0.,}
    D = X.shape[1]
    for _ , (train_index, test_index) in enumerate(k_folds.split(X)):
        # Define X_train, Y_train, X_test, Y_test for fold i
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
        Y_train = (Y_train - Y_train_mean) / Y_train_std
        Y_test = (Y_test - Y_train_mean) / Y_train_std

        # Define the kernel: RBF(standard) or Lambda RBF
        if kernel == 'RBF-ARD':
            kernel = ARD_gpflow(variance=1.0, randomized=False, d=D)
        elif kernel == 'LRBF':
            kernel = LambdaRBF(variance=1.0, randomized=False, d=D)

        # Initialize the model: GPR or SVGP
        if model == 'GPR-Lasso':
            gp_model = GPRLasso(data=(X_train, Y_train), kernel=kernel, lasso=model_params['lasso'])
            gp_model.train()
            
        elif model == 'SVGP-Lasso':
            num_inducing = model_params['num_inducing']
            likelihood = model_params['likelihood']
            max_iter = model_params['max_iter']
            minibatch_size = model_params['minibatch_size']
            gp_model = SVGPLasso(data=(X_train, Y_train), kernel=kernel, lasso=model_params['lasso'], num_inducing=num_inducing, likelihood=likelihood, max_iter=max_iter, minibatch_size=minibatch_size)
            gp_model.train()

        # Measure performances
        train_rmse_stan, test_rmse_stan = measure_rmse(gp_model, X_train, Y_train, X_test, Y_test)
        train_mnll, test_mnll = measure_mnll(gp_model, X_train, Y_train, Y_train_std, X_test, Y_test)
        results['train_rmse'].append(train_rmse_stan)
        results['test_rmse'].append(test_rmse_stan)
        results['train_mnll'].append(train_mnll)
        results['test_mnll'].append(test_mnll)
        results['Lambda'].append(gp_model.kernel.precision())
        results['sparsity_degree'].append(sparsity_degree(gp_model.kernel.precision(), tol=model_params['tol_sparsity']))


    results['avg_train_rmse'] = np.mean(results['train_rmse'])
    results['avg_test_rmse'] = np.mean(results['test_rmse'])
    results['avg_train_mnll'] = np.mean(results['train_mnll'])
    results['avg_test_mnll'] = np.mean(results['test_mnll'])
    results['avg_sparsity_degree'] = np.mean(results['sparsity_degree'])
    if iprint:
        print('Average test RMSE: %5.3f\nAverage test MNLL: %5.3f\n'%(results['avg_test_rmse'], results['avg_test_mnll']))
    return results