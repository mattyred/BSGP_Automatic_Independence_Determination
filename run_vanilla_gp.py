#! /usr/local/bin/ipython --
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
import numpy as np
import argparse
#import tensorflow as tf
#tf.get_logger().setLevel('INFO')
import json
from sklearn.model_selection import KFold
from torch.utils.data import  TensorDataset
import torch
from pprint import pprint
import gpflow
from bsgp.kernels import FullPrecisionRBF
from scipy.stats import norm
from scipy.special import logsumexp

def next_path(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i / 2, i)
    while a + 1 < b:
        c = (a + b) / 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    directory = path_pattern % b
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def create_dataset(dataset, static, pca, fold):
    dataset_path = ('./data/' + dataset + '.pth')
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy().astype(np.float64), Y.numpy().astype(np.float64)
    if not static:
        Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-9
        return X, Y, Y_mean, Y_std
    else:
        X_train_indices_boolean = np.random.choice([1, 0], size=X.shape[0], p=[0.8, 0.2])
        X_train_indices = np.where(X_train_indices_boolean == 1)[0]
        X_test_indices = np.where(X_train_indices_boolean == 0)[0]
        X_train = X[X_train_indices]
        Y_train = Y[X_train_indices]
        X_test = X[X_test_indices]
        Y_test = Y[X_test_indices]
        Pd = None
        Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
        Y_train = (Y_train - Y_train_mean) / Y_train_std
        Y_test = (Y_test - Y_train_mean) / Y_train_std
        return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std, X_train_indices, X_test_indices, Pd

def assign_pathname(filepath, dataset, kernel_type):
    p = filepath + dataset + '_'
    return p + 'results'

def save_results_onefold(filepath, onefold_data, kernel_type):
    results = dict()
    results['model'] = args.model
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['pca'] = args.pca
    results['test_mnll'] = onefold_data['test_mnll']
    results['test_rmse'] = onefold_data['test_rmse']
    results['kernel_type'] = kernel_type

    pprint(results)
    npzfilepath = assign_pathname(filepath, args.dataset, kernel_type)
    np.savez(npzfilepath, test_mnll=np.array(results['test_mnll']), test_rmse=results['test_mnll'], kern_cov=onefold_data['kern_cov'])

def save_results_kfold(filepath, kfold_data, precise_kernel):
    results = dict()
    results['model'] = args.model
    results['kfold'] = args.kfold
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['pca'] = -1

    pprint(results)

    # Save kernel precision matrices
    npzfilepath = assign_pathname(filepath, args.dataset, precise_kernel)
    kern_cov_list = [kfold_data[i]['kern_cov'] for i in range(args.kfold)]
    test_mnll_list = [kfold_data[i]['test_mnll'] for i in range(args.kfold)]
    test_rmse_list = [kfold_data[i]['test_rmse'] for i in range(args.kfold)]
    kern_cov_folds = np.stack(kern_cov_list, axis=0)
    test_mnll_folds = np.stack(test_mnll_list, axis=0)
    test_rmse_folds = np.stack(test_rmse_list, axis=0)
    np.savez(npzfilepath, test_mnll=test_mnll_folds, test_rmse=test_rmse_folds, kern_cov=kern_cov_folds)


def main():
    #set_seed(0)
    filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')

    # static Train/Test split
    if args.kfold == -1:
        print('\n### Static Train/Test split ###')
        X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, X_train_indices, X_test_indices, Pd = create_dataset(args.dataset, True, args.pca, args.fold)
        if args.minibatch_size > len(X_train): args.minibatch_size = len(X_train)
        test_mnll, test_rmse, model = train_model(X_train, Y_train,  X_test, Y_test, Y_train_std, kernel_type=args.kernel_type) 
        kern_cov = model.kernel.precision().numpy() if args.kernel_type == 2 else 1/model.kernel.lengthscales.numpy()**2
        onefold_data = {'test_mnll': test_mnll, 
                        'test_rmse': test_rmse, 
                        'trained_model': model, 
                        'X_train_indices': X_train_indices, 
                        'X_test_indices': X_test_indices, 
                        'kern_cov': kern_cov,
                        'Pd': Pd} 
        save_results_onefold(filepath, onefold_data, args.kernel_type)
    # K-Fold Cross Validation
    else:
        kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
        X, Y, Y_mean, Y_std = create_dataset(args.dataset, False, args.pca, args.fold) # get full dataset
        kfold_data_ker1 = []
        current_fold_data = {'test_mnll': 0, 'test_rmse': 0, 'trained_model': 0} # For AID/ARD
        n_fold = 0
        for train_index, val_index in kfold.split(X):
            print('\n### Training fold: %d/%d ###'%(n_fold+1, args.kfold))
            X_train, X_test = X[train_index], X[val_index]
            Y_train, Y_test = Y[train_index], Y[val_index]
            # Standardize data
            Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
            Y_train = (Y_train - Y_train_mean) / Y_train_std
            Y_test = (Y_test - Y_train_mean) / Y_train_std
            # Train model on X_train, Y_train
            test_mnll, test_rmse, model = train_model(X_train, Y_train,  X_test, Y_test, Y_train_std, kernel_type=args.kernel_type)
            current_fold_data['test_mnll'] = test_mnll
            current_fold_data['test_rmse'] = test_rmse
            current_fold_data['trained_model'] = model
            current_fold_data['X_train_indices'] = train_index
            current_fold_data['X_test_indices'] = val_index
            print('Fold %d - kernel type: %d - test MNLL: %.3f' % (n_fold, args.kernel_type, current_fold_data['test_mnll']))
            # Store results current fold in 'kfold_data'
            kern_cov = model.kernel.precision().numpy() if args.kernel_type == 2 else 1/model.kernel.lengthscales.numpy()**2
            kfold_data_ker1.append({'test_mnll': current_fold_data['test_mnll'], 
                                    'test_rmse': current_fold_data['test_rmse'], 
                                    'trained_model': current_fold_data['trained_model'], 
                                    'X_train_indices': train_index, 
                                    'X_test_indices': val_index, 
                                    'kern_cov': kern_cov})
            n_fold += 1

        # Store results of all folds
        save_results_kfold(filepath, kfold_data_ker1, args.kernel_type)

def compute_mnll(model, X_test, Y_test, Y_train_std):
    ms, vs = model.predict_y(X_test)
    logps = norm.logpdf(Y_test[None, :, :]*Y_train_std, ms*Y_train_std, np.sqrt(vs)*Y_train_std)
    return -logsumexp(logps, axis=0).mean()

def compute_rmse(model, X_test, Y_test, Y_train_std=1.):
    ms, vs = model.predict_y(X_test)
    ps = np.mean((Y_test[None, :, :]*Y_train_std - ms*Y_train_std)**2)**0.5 / Y_train_std
    return ps.mean()

def train_model(X_train, Y_train,  X_test, Y_test, Y_train_std, kernel_type=0):
    D_in, D_out = X_train.shape[1], Y_train.shape[0]

    # Select kernel
    if kernel_type == 0:
        kernel = gpflow.kernels.SquaredExponential()
    elif kernel_type == 1:
        kernel = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=np.ones(D_in)**0.5)
    elif kernel_type ==2:
        prior_precision_info = {'type': 'wishart', 'parameters': None, 'parametrization': 'Lambda'}
        kernel = FullPrecisionRBF(variance=0.1, randomized=False, d=D_in, prior_precision_info=prior_precision_info)

    # Full GPR model
    model = gpflow.models.GPR((X_train, Y_train), kernel=kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    # Compute performances
    test_mnll = compute_mnll(model, X_test, Y_test, Y_train_std)
    test_rmse = compute_rmse(model, X_test, Y_test, Y_train_std)
    return test_mnll, test_rmse, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regression experiment')
    parser.add_argument('--minibatch_size', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--prior_type', choices=['determinantal', 'normal', 'strauss', 'uniform'], default='normal')
    parser.add_argument('--model', choices=['bsgp'], default='bsgp')
    parser.add_argument('--num_posterior_samples', type=int, default=512)
    parser.add_argument('--step_size', type=float, default=0.01) #0.01
    parser.add_argument('--kernel_type', type=int, default=0) # 0: RBF(no ARD) - 1: RBF(ARD) - 2: RBF(ACD)
    parser.add_argument('--kfold', type=int, default=-1) 
    parser.add_argument('--pca', type=int, default=-1)
    args = parser.parse_args()

    if args.model == 'bsgp':
        main()