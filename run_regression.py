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

def create_dataset(dataset, static, fold):
    dataset_path = ('./data/' + dataset + '.pth')
    logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy(), Y.numpy()

    if static == False:
        Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-9
        Y = (Y - Y_mean) / Y_std
        return X, Y, Y_mean, Y_std
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
        Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
        Y_train = (Y_train - Y_train_mean) / Y_train_std
        Y_test = (Y_test - Y_train_mean) / Y_train_std
        return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std

def save_results_static(filepath, onefold_data, precise_kernel):
    results = dict()
    results['model'] = args.model
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['test_mnll'] = onefold_data['test_mnll']
    results['precise_kernel'] = precise_kernel

    #filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    pprint(results)
    if precise_kernel:
        jsonfilepath = filepath + 'LRBF_results.json'
        results['posterior_samples_U_precision'] = onefold_data['trained_model'].posterior_samples_kerncov
    else:
        jsonfilepath = filepath + 'ARD_results.json'
        results['posterior_samples_loglengthscales'] = onefold_data['trained_model'].posterior_samples_kerncov
    results['posterior_samples_kernlogvar'] = onefold_data['trained_model'].posterior_samples_kernlogvar
    with open(jsonfilepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def save_results_kfold(filepath, kfold_data, precise_kernel):
    results = dict()
    results['model'] = args.model
    results['kfold'] = args.kfold
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    #results['test_mnll'] = np.mean(results['test_mnll'])
    results['precise_kernel'] = precise_kernel

    #filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    pprint(results)

    # Save kernel precision matrices
    if precise_kernel == 1:
        jsonfilepath = filepath + 'LRBF_results.json'
        results['posterior_samples_U_precision'] = []
        for i in range(args.kfold):
            model = kfold_data[i]['trained_model'] # model of fold 'i'
            results['posterior_samples_U_precision'].append(model.posterior_samples_kerncov)
    elif precise_kernel == 0:
        jsonfilepath = filepath + 'ARD_results.json'
        results['posterior_samples_loglengthscales'] = []
        for i in range(args.kfold):
            model = kfold_data[i]['trained_model']
            results['posterior_samples_loglengthscales'].append(model.posterior_samples_kerncov)
    
    # Save kernel log variance and MNLL for each fold
    results['posterior_samples_kernlogvar'] = []
    results['test_mnll'] = []
    for i in range(args.kfold):
        model = kfold_data[i]['trained_model'] 
        results['posterior_samples_kernlogvar'].append(model.posterior_samples_kernlogvar)
        results['test_mnll'].append(kfold_data[i]['test_mnll'])

    with open(jsonfilepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    set_seed(0)
    filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    if args.kfold == -1: # static Train/Test split
        print('\n### Static Train/Test split ###')
        X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std = create_dataset(args.dataset, True, args.fold)
        if args.minibatch_size > len(X_train): args.minibatch_size = len(X_train)
        if args.precise_kernel == 0 or args.precise_kernel == 1:
            test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, 
            Y_train_mean, Y_train_std, precise_kernel=args.precise_kernel) 
            onefold_data = {'test_mnll': test_mnll, 'trained_model': model} 
            save_results_static(filepath, onefold_data, args.precise_kernel)
        else:
            test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=0)
            onefold_data = {'test_mnll': test_mnll, 'trained_model': model} 
            save_results_static(filepath, onefold_data, False)
            test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=1)
            onefold_data = {'test_mnll': test_mnll, 'trained_model': model}  
            save_results_static(filepath, onefold_data, True)
    else: # K-Fold Cross Validation
        kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
        X, Y, Y_mean, Y_std = create_dataset(args.dataset, False, args.fold) # get full dataset
        kfold_data_ker1 = []
        kfold_data_ker2 = []
        current_fold_data_ker1 = {'test_mnll': 0, 'trained_model': 0} # For LRBF/ARD
        current_fold_data_ker2 = {'test_mnll': 0, 'trained_model': 0} # When both LRBF and ARD are used
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
            if args.precise_kernel == 0 or args.precise_kernel == 1: # ARD or LRBF
                test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=args.precise_kernel)
                current_fold_data_ker1['test_mnll'] = test_mnll
                current_fold_data_ker1['trained_model'] = model
            else: # ARD and LRBF
                # ARD model
                test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=False) 
                current_fold_data_ker1['test_mnll'] = test_mnll
                current_fold_data_ker1['trained_model'] = model 
                # LRBF model
                test_mnll, model = train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=True) 
                current_fold_data_ker2['test_mnll'] = test_mnll
                current_fold_data_ker2['trained_model'] = model
            # Store results current fold in 'kfold_data'
            if args.precise_kernel == 0 or args.precise_kernel == 1: # LRBF or ARD
                kfold_data_ker1.append({'test_mnll': current_fold_data_ker1['test_mnll'], 'trained_model': current_fold_data_ker1['trained_model'], 'precise_kernel': args.precise_kernel})
            else:
                kfold_data_ker1.append({'test_mnll': current_fold_data_ker1['test_mnll'], 'trained_model': current_fold_data_ker1['trained_model'], 'precise_kernel': False})
                kfold_data_ker2.append({'test_mnll': current_fold_data_ker2['test_mnll'], 'trained_model': current_fold_data_ker2['trained_model'], 'precise_kernel': True})
            n_fold += 1
            #current_fold_data_ker1['test_mnll'] =  current_fold_data_ker1['trained_model'] = 0
            #current_fold_data_ker2['test_mnll'] =  current_fold_data_ker2['trained_model'] = 0
        # Store results of all folds
        if args.precise_kernel == 0 or args.precise_kernel == 1:
            save_results_kfold(filepath, kfold_data_ker1, args.precise_kernel)
        else:
            save_results_kfold(filepath, kfold_data_ker1, 0)
            save_results_kfold(filepath, kfold_data_ker2, 1)


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
    model.ARGS.precise_kernel = precise_kernel 
    model.fit(X_train, Y_train, epsilon=args.step_size)
    test_mnll = -model.calculate_density(X_test, Y_test, Y_train_mean, Y_train_std).mean().tolist()
    return test_mnll, model
    # save_results_static(filepath, test_mnll, precise_kernel, model.posterior_samples_kerncov, model.posterior_samples_kernlogvar) # kerncov: L matrix for LBRF / lengthscales for ARD

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
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--precise_kernel', type=int, default=0) # LRBF-MOD (0: ARD, 1: LRBF, 2: BOTH)
    parser.add_argument('--kfold', type=int, default=-1) # Number of folds for k-fold cv

    args = parser.parse_args()

    if args.model == 'bsgp':
        main()