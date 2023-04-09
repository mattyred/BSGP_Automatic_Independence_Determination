import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import norm
from scipy.special import logsumexp
import tensorflow_probability as tfp
from LambdaRBF import LambdaRBF
import gpflow # 2.7.0
plt.rcParams["figure.figsize"] = (12, 6)
plt.style.use("ggplot")

def plot_matrix(M=None, cmap='vlag', annot=True, correlation=False):
    """
    M: Matrix to be visualized with a color mapping
    correlation: plot the correlation matrix of the dataset M
    cmap: type of mapping
    ---
    It displays a matrix represented with a color scheme
    """
    if correlation:
        index_values = ['%d'%(i) for i in range(M.shape[0])]
        column_values = ['%d'%(i) for i in range(M.shape[1])]
        df = pd.DataFrame(data = M, index = index_values, columns = column_values)
        corr = df.corr()
        sns.heatmap(corr, annot=annot, cmap=cmap)
    else:
        min = np.min(M)
        max = np.max(M)
        center = (min+max)/2
        sns.heatmap(M, annot=annot, cmap=cmap, vmax=max, vmin=0, center=0, linewidth=.5)

def compare_matrix(M1, M2, cmap='vlag'):
    fig, axes = plt.subplots(1, 2, figsize=(24,6))
    min = np.min(M1)
    max = np.max(M1)
    center = (min+max)/2
    sns.heatmap(M1, ax=axes[0], annot=True, cmap=cmap, vmax=max, vmin=min, center=center, linewidth=.5)
    axes[0].set_title('LambdaRBF')

    min = np.min(M2)
    max = np.max(M2)
    center = (min+max)/2
    sns.heatmap(M2, ax=axes[1], annot=True, cmap=cmap, vmax=max, vmin=min, center=center, linewidth=.5)
    axes[1].set_title('ARD')
    plt.show()



def get_lower_triangular_from_diag(diag):
    """
    diag: diagonal of lengthscales parameter [D,]
    ---
    Σ=inv(Λ) -> diagonal matrix with lengthscales on the diagonal (RBF)
    The diagonal of Λ is obtained as 1/(l^2), l is a lengthscale
    returns: L, Λ=LLᵀ
    """
    Lambda = tf.linalg.diag(1/(diag**2))
    L = tf.linalg.cholesky(Lambda)
    return L

def create_dataset(dataset, fold, static_train_test=True):
    dataset_path = ('./data/' + dataset + '.pth')
    #logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy().astype('float64'), Y.numpy().astype('float64')
    if static_train_test:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
        Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
        Y_train = (Y_train - Y_train_mean) / Y_train_std
        Y_test = (Y_test - Y_train_mean) / Y_train_std
        return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std
    else:
        return X, Y


def measure_rmse(model, X_train, Y_train, X_test, Y_test):
    y_pred_train, _ = model.predict_f(X_train)
    train_rmse_stan = tf.sqrt(tf.reduce_mean((Y_train - y_pred_train)**2)).numpy()
    y_pred_test, _ = model.predict_f(X_test)
    test_rmse_stan = tf.sqrt(tf.reduce_mean((Y_test - y_pred_test)**2)).numpy()
    return train_rmse_stan, test_rmse_stan

def measure_mnll(model, X_train, Y_train, Ystd, X_test, Y_test):
    mean_train, var_train = model.predict_f(X_train)
    mean_test, var_test = model.predict_f(X_test)
    """
    logps_train = norm.logpdf(np.repeat(Y_train[None, :, :]*Ystd, X_train.shape[0], axis=0), mean_train*Ystd, np.sqrt(var_train)*Ystd)
    train_mnll = -np.mean(logsumexp(logps_train, axis=0) - np.log(X_train.shape[0]))

    mean_test, var_test = model.predict_f(X_test)
    logps_test = norm.logpdf(np.repeat(Y_test[None, :, :]*Ystd, X_test.shape[0], axis=0), mean_test*Ystd, np.sqrt(var_test)*Ystd)
    test_mnll = -np.mean(logsumexp(logps_test, axis=0) - np.log(X_test.shape[0]))
    """
    train_mnll = -norm.logpdf(Y_train*Ystd, mean_train*Ystd, np.sqrt(var_train)*Ystd).mean()
    test_mnll = -norm.logpdf(Y_test*Ystd, mean_test*Ystd, np.sqrt(var_test)*Ystd).mean()
    return train_mnll, test_mnll

def train_GPR_LRBF_model(X_train=None, Y_train=None, reg=-1, prior=None, iprint=True):
    D = X_train.shape[1]
    # Define the kernel 
    lengthscales = tf.constant([D**0.5]*D, dtype=tf.float64)
    Lambda_L = get_lower_triangular_from_diag(lengthscales)
    Lambda_L_array = tfp.math.fill_triangular_inverse(Lambda_L)
    LRBF = LambdaRBF(Lambda_L_array, 1.0)
    if prior is not None:
        LRBF.Lambda_L.prior = prior['Lambda_L_prior']
        LRBF.variance.prior = prior['variance_prior']
    # GPR model (no approx)
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=LRBF,
    )
    gpflow.utilities.print_summary(model, fmt="notebook")
    if iprint:
        print('--- Initial values ---')
        print('Variance: %.3f'%(LRBF.variance.numpy()))
        print('Lambda diagonal: ', tf.linalg.diag_part(LRBF.get_Lambda()).numpy())
    opt = gpflow.optimizers.Scipy()
    if reg > 0:
        def regularized_training_loss():
            return -(model.log_marginal_likelihood() + model.log_prior_density()) + reg * tf.norm(model.kernel.get_Lambda(), ord=1)
        opt.minimize(regularized_training_loss, model.trainable_variables)
    else:
        opt.minimize(model.training_loss, model.trainable_variables)
    if iprint:
        print('--- Final values ---')
        print('Variance: %.3f'%(LRBF.variance.numpy()))
        plot_matrix(LRBF.get_Lambda())
    return model, LRBF.get_Lambda()

def train_GPR_RBF_model(X_train=None, Y_train=None, prior=None, iprint=True):
    D = X_train.shape[1]
    RBF = gpflow.kernels.SquaredExponential(variance=1, lengthscales=(D**0.5)*np.ones(D))
    if prior is not None:
        RBF.lengthscales.prior = prior['lengthscales_prior']
        RBF.variance.prior = prior['variance_prior']
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=RBF,
    )
    gpflow.utilities.print_summary(model, fmt="notebook")
    if iprint:
        print('--- Initial values ---')
        print('Variance: %.3f'%(RBF.variance.numpy()))
        print('Lengthscales: ', RBF.lengthscales.numpy())
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    if iprint:
        print('--- Final values ---')
        print('Variance: %.3f'%(RBF.variance.numpy()))
        print('Lengthscales: ', RBF.lengthscales.numpy())
    Lambda_L_RBF = get_lower_triangular_from_diag(RBF.lengthscales.numpy())
    Lambda_RBF = tf.matmul(Lambda_L_RBF, tf.transpose(Lambda_L_RBF))
    plot_matrix(Lambda_RBF)
    return model, Lambda_L_RBF

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
    results = {'train_rmse': [], 
                'test_rmse': [], 
                'train_mnll': [], 
                'test_mnll': [], 
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
        if kernel == 'RBF':
            kernel = gpflow.kernels.SquaredExponential(variance=1, lengthscales=(D**0.5)*np.ones(D))
        elif kernel == 'LRBF':
            lengthscales = tf.constant([D**0.5]*D, dtype=tf.float64)
            Lambda_L = get_lower_triangular_from_diag(lengthscales)
            Lambda_L_array = tfp.math.fill_triangular_inverse(Lambda_L)
            kernel = LambdaRBF(Lambda_L_array, 1.0)

        # Initialize the model: GPR or SVGP
        if model == 'GPR':
            gp_model = gpflow.models.GPR(
                (X_train, Y_train),
                kernel=kernel,
            )
            opt = gpflow.optimizers.Scipy()
            reg = model_params['reg']
            if reg > 0:
                # Lambda RBF kernel + GPR model
                def regularized_training_loss():
                    return -(gp_model.log_marginal_likelihood() + gp_model.log_prior_density()) + reg * tf.norm(gp_model.kernel.get_Lambda(), ord=1)
                opt.minimize(regularized_training_loss, gp_model.trainable_variables)
            else:
                # RBF kernel + GPR model
                opt.minimize(gp_model.training_loss, gp_model.trainable_variables)
            
        elif model == 'SVGP':
            num_inducing = model_params['num_inducing']
            lik = model_params['likelihood']
            Z = X_train[:num_inducing, :].copy()
            maxiter = model_params['max_iter']
            minibatch_size = model_params['minibatch_size']
            model = gpflow.models.SVGP(kernel, lik, Z, num_data=num_inducing)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(X_train.shape[0])
            logf = run_adam(model, train_dataset, minibatch_size, maxiter)

        # Measure performances
        train_rmse_stan, test_rmse_stan = measure_rmse(gp_model, X_train, Y_train, X_test, Y_test)
        train_mnll, test_mnll = measure_mnll(gp_model, X_train, Y_train, Y_train_std, X_test, Y_test)
        results['train_rmse'].append(train_rmse_stan)
        results['test_rmse'].append(test_rmse_stan)
        results['train_mnll'].append(train_mnll)
        results['test_mnll'].append(test_mnll)


    results['avg_train_rmse'] = np.mean(results['train_rmse'])
    results['avg_test_rmse'] = np.mean(results['test_rmse'])
    results['avg_train_mnll'] = np.mean(results['train_mnll'])
    results['avg_test_mnll'] = np.mean(results['test_mnll'])
    if iprint:
        print('-- Model: %s; Kernel: %s; --'%(model, kernel))
        print('Average test RMSE: %5.3f\nAverage test MNLL: %5.3f\n'%(results['avg_test_rmse'], results['avg_test_mnll']))
    return results