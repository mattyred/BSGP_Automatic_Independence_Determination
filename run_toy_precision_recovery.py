#! /usr/local/bin/ipython --

import sys
import os
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np
from bsgp.models import RegressionModel
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import json
from sklearn.model_selection import train_test_split

from torch.utils.data import  TensorDataset

import torch
from pprint import pprint

import seaborn as sns
import pandas as  pd
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


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

def _z(X, Lambda):
        XLambda = X @ Lambda
        XLambdaX = np.multiply(XLambda, X)
        return np.sum(XLambdaX, axis=1, keepdims=True)
def K_lrbf(kernel_variance, X, Lambda, X2=None):
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

    # compute z, z2
    z = _z(X, Lambda) # N1x1 array
    z2 = _z(X2, Lambda) # N2x1 array
    # compute X(X2Λ)ᵀ
    X2Lambda = X2 @ Lambda
    XX2LambdaT = X @ X2Lambda.T # N1xN2 matrix
    # compute z1ᵀ 
    ones_N2 = np.ones(shape=(N2,1)) # N2x1 array
    zcol = z @ ones_N2.T # N1xN2 matrix
    # compute 1z2ᵀ 
    ones_N1 = np.ones(shape=(N1,1)) # N1x1 array
    zrow = ones_N1 @ z2.T # N1xN2 matrix

    exp_arg = zcol - 2*XX2LambdaT + zrow
    Kxx = np.exp(-0.5 * exp_arg)
    return kernel_variance * Kxx

def random_offdiag_precision(D, type_offdiag='norand', num_offdiag=0):
    diag = np.ones(D)
    L = np.diag(diag, 0)
    if num_offdiag > D-1:
        num_offdiag = D-1
    for n in range(num_offdiag):
        i = n+1
        # define the offdiagonal
        if type_offdiag == 'norand':
            offdiag = D**0.5 * np.ones(D-i)
        elif type_offdiag == 'uniform':
            offdiag = np.random.uniform(0, 1, D-i)
        elif type_offdiag == 'gaussian':
            offdiag = np.random.normal(-1, 1, D-i)
        # add it to the matrix L
        L += np.diag(offdiag, -i)
    # define the precision matrix
    precision = L @ L.T
    # normalize the precision matrix to have ones on the diagonal
    diag_m = np.diag(1/np.diag(precision))
    return (np.sqrt(diag_m) @ precision) @ np.sqrt(diag_m)

def generate_toy_dataset(precision, N=100, D=5, kernel_variance=1.0, noise_variance=0.1, random_state=0):
    # sampling from known covariance 
    X = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=N)
    K = K_lrbf(kernel_variance, X, precision)
    noise = np.random.normal(0, noise_variance, size=N)
    Y = (np.random.multivariate_normal(np.zeros(N), K, size=1) + noise).reshape(-1,1)
    # train-test split
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # center the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state, shuffle=True)
    Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
    Y_train = (Y_train - Y_train_mean) / Y_train_std
    Y_test = (Y_test - Y_train_mean) / Y_train_std
    return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std

def train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precision, precise_kernel=False):
    model = RegressionModel(args.prior_type)
    model.ARGS.num_inducing = args.num_inducing
    model.ARGS.minibatch_size = args.minibatch_size
    model.ARGS.iterations = args.iterations
    model.ARGS.n_layers = args.n_layers
    model.ARGS.num_posterior_samples = args.num_posterior_samples
    model.ARGS.prior_type = args.prior_type
    model.ARGS.full_cov = False
    model.ARGS.posterior_sample_spacing = 32
    logger.info('Number of inducing points: %d' % model.ARGS.num_inducing)
    model.ARGS.precise_kernel = precise_kernel 
    model.fit(X_train, Y_train, epsilon=args.step_size)
    test_mll = model.calculate_density(X_test, Y_test, Y_train_mean, Y_train_std).mean().tolist()
    save_results(filepath, test_mll, precision, precise_kernel, model.posterior_samples_kerncov, model.posterior_samples_kerlogvar) # kerncov: L matrix for LBRF / lengthscales for ARD   

def save_results(filepath, test_mll, precision, precise_kernel, posterior_samples_kerncov, posterior_samples_kerlogvar):
    results = dict()
    results['toy_dataset'] = {'N': args.N, 'D': args.D, 'num_offdiag': args.num_offdiag, 'type_offdiag': args.type_offdiag}
    results['model'] = args.model
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['test_mnll'] = -test_mll
    results['precise_kernel'] = precise_kernel

    pprint(results)
    if precise_kernel:
        jsonfilepath = filepath + 'LRBF_results.json'
        results['posterior_samples_L_precision'] = posterior_samples_kerncov
    else:
        jsonfilepath = filepath + 'ARD_results.json'
        results['posterior_samples_loglengthscales'] = posterior_samples_kerncov
    results['posterior_samples_kerlogvar'] = posterior_samples_kerlogvar
    results['underlying_precision'] = precision.tolist()
    with open(jsonfilepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    set_seed(0)
    # Define a custom precision matrix to be recovered
    precision = random_offdiag_precision(args.D, type_offdiag=args.type_offdiag, num_offdiag=args.num_offdiag)
    # Obtain the covariance matrix
    covariance = np.linalg.inv(precision) 
    # Generate toy dataset with underlying covariate structure
    X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std = generate_toy_dataset(precision, N=args.N, D=args.D, kernel_variance=1.0, noise_variance=0.1, random_state=args.fold)
    # Define minibatch size and filepath
    if args.minibatch_size > len(X_train): args.minibatch_size = len(X_train)
    filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    # Train
    if args.precise_kernel == 0 or args.precise_kernel == 1: # ARD or LRBF
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=args.precise_kernel)
    else: 
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precision, precise_kernel=False) # ARD
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precision, precise_kernel=True) # LRBF     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regression experiment')
    parser.add_argument('--N', type=int, default=100) # number of samples
    parser.add_argument('--D', type=int, default=5) # number of dimensions
    parser.add_argument('--num_offdiag', type=int, default=0) # number of non-zero off diagonals in the covariance
    parser.add_argument('--type_offdiag', type=str, default='gaussian')
    parser.add_argument('--num_inducing', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--prior_type', choices=['determinantal', 'normal', 'strauss', 'uniform'], default='normal')
    parser.add_argument('--model', choices=['bsgp'], default='bsgp')
    parser.add_argument('--num_posterior_samples', type=int, default=512)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--precise_kernel', type=int, default=0) # LRBF-MOD (0: ARD, 1: LRBF, 2: BOTH)
    
    args = parser.parse_args()

    if args.model == 'bsgp':
        main()