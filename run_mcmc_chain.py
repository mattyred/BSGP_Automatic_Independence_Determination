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
from sklearn.model_selection import KFold
tf.debugging.enable_check_numerics() 
from torch.utils.data import  TensorDataset

import torch
from pprint import pprint

import seaborn as sns
import pandas as  pd
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from bsgp.utils import apply_pca

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
    logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy(), Y.numpy()

    X_train_indices_boolean = np.random.choice([1, 0], size=X.shape[0], p=[0.8, 0.2])
    X_train_indices = np.where(X_train_indices_boolean == 1)[0]
    X_test_indices = np.where(X_train_indices_boolean == 0)[0]
    X_train = X[X_train_indices]
    Y_train = Y[X_train_indices]
    X_test = X[X_test_indices]
    Y_test = Y[X_test_indices]
    #print('FOLD TEST ', X_train[10:12,:]) # check random seed
    Pd = None
    if pca != -1:
        X_train, Pd = apply_pca(X_train, pca) # fit_transform X_train
        X_test = X_test @ Pd # transform X_test
    Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
    Y_train = (Y_train - Y_train_mean) / Y_train_std
    Y_test = (Y_test - Y_train_mean) / Y_train_std
    return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std, X_train_indices, X_test_indices, Pd

def save_results_onefold(filepath, onefold_data, precise_kernel):
    results = dict()
    results['model'] = args.model
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['pca'] = args.pca
    results['test_mnll'] = onefold_data['test_mnll']
    results['test_rmse'] = onefold_data['test_rmse']
    results['precise_kernel'] = precise_kernel

    #filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    pprint(results)
    npzfilepath = filepath + args.dataset + '_' + 'mcmc_results.npz'
    np.savez(npzfilepath, mcmc_samples_mean=onefold_data['trained_model'].samples_ms_iter, 
                          mcmc_samples_var=onefold_data['trained_model'].samples_vs_iter,
                          mcmc_samples_logps=onefold_data['trained_model'].samples_logps_iter)
    #with open(npzfilepath, 'w', encoding='utf-8') as f:
    #   json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    set_seed(args.fold)
    filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    print('\n### Static Train/Test split ###')
    X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, X_train_indices, X_test_indices, Pd = create_dataset(args.dataset, True, args.pca, args.fold)
    if args.minibatch_size > len(X_train): args.minibatch_size = len(X_train)

    test_mnll, test_rmse, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=args.precise_kernel) 
    onefold_data = {'test_mnll': test_mnll, 'test_rmse': test_rmse, 'trained_model': model, 'X_train_indices': X_train_indices, 'X_test_indices': X_test_indices, 'Pd': Pd} 
    save_results_onefold(filepath, onefold_data, args.precise_kernel)

def train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=False):
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
    model.ARGS.clip_by_value = args.clip_by_value
    model.ARGS.precise_kernel = precise_kernel 

    # Prior parameters for precise kernel
    model.ARGS.prior_precision_type = args.prior_precision_type
    model.ARGS.prior_precision_parameters = {'prior_laplace_b':  args.prior_laplace_b, 'prior_normal_mean':  args.prior_normal_mean, 'prior_normal_variance': args.prior_normal_variance, 'prior_horseshoe_globshrink': args.prior_horseshoe_globshrink, 'parametrization': args.prior_precision_select_param, 'init_random_L': args.init_random_L}
    
    # MCMC
    model.ARGS.mcmc_measures = True

    model.fit(X_train, Y_train, Xtest=X_test, Ytest=Y_test, Ystd=Y_train_std, epsilon=args.step_size)
    test_mnll = -model.calculate_density(X_test, Y_test, Y_train_mean, Y_train_std).mean().tolist()
    test_rmse = model.calculate_rmse(X_test, Y_test, Y_train_mean, Y_train_std).mean().tolist()
    return test_mnll, test_rmse, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regression experiment')
    parser.add_argument('--num_inducing', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--prior_type', choices=['determinantal', 'normal', 'strauss', 'uniform'], default='normal')
    parser.add_argument('--model', choices=['bsgp'], default='bsgp')
    parser.add_argument('--num_posterior_samples', type=int, default=512)
    parser.add_argument('--step_size', type=float, default=0.01) #0.01
    parser.add_argument('--precise_kernel', type=int, default=0)
    parser.add_argument('--prior_precision_type', choices=['normal', 'laplace+diagnormal', 'horseshoe+diagnormal', 'wishart', 'invwishart', 'laplace', 'horseshoe'], default='normal') # Prior on kernel precision matrix
    # Laplace prior
    parser.add_argument('--prior_laplace_b', type=float, default=0.01)
    # Default prior (Normal)
    parser.add_argument('--prior_normal_mean', type=float, default=0)
    parser.add_argument('--prior_normal_variance', type=float, default=1)
    # Horseshoe prior
    parser.add_argument('--prior_horseshoe_globshrink', type=float, default=0.1) 
    # Prior on L or Λ
    parser.add_argument('--prior_precision_select_param', choices=['Lambda', 'L'], default='Lambda')
    # PCA
    parser.add_argument('--pca', type=int, default=-1)
    # gradient clipping
    parser.add_argument('--clip_by_value', type=int, default=-1)
    # Random L initialization
    parser.add_argument('--init_random_L', type=int, default=1)

    args = parser.parse_args()

    if args.model == 'bsgp':
        main()