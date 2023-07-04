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

def create_dataset(dataset, fold):
    dataset_path = ('./data/' + dataset + '.pth')
    logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy(), Y.numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
    Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
    Y_train = (Y_train - Y_train_mean) / Y_train_std
    Y_test = (Y_test - Y_train_mean) / Y_train_std

    return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std

def save_results(filepath, test_mll, precise_kernel, posterior_samples_kerncov, posterior_samples_kerlogvar):
    results = dict()
    results['model'] = args.model
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['test_mnll'] = -test_mll
    results['precise_kernel'] = precise_kernel

    #filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    pprint(results)
    if precise_kernel:
        jsonfilepath = filepath + 'LRBF_results.json'
        results['posterior_samples_L_precision'] = posterior_samples_kerncov
    else:
        jsonfilepath = filepath + 'ARD_results.json'
        results['posterior_samples_loglengthscales'] = posterior_samples_kerncov
    results['posterior_samples_kerlogvar'] = posterior_samples_kerlogvar
    with open(jsonfilepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    set_seed(0)
    X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std = create_dataset(args.dataset, args.fold)
    if args.minibatch_size > len(X_train): args.minibatch_size = len(X_train)
    filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    if args.precise_kernel == 0 or args.precise_kernel == 1: # ARD or LRBF
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=args.precise_kernel)
    else: 
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=False) # ARD
        train_model(filepath, X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std, precise_kernel=True) # LRBF

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
    test_mll = model.calculate_density(X_test, Y_test, Y_train_mean, Y_train_std).mean().tolist()
    save_results(filepath, test_mll, precise_kernel, model.posterior_samples_kerncov, model.posterior_samples_kerlogvar) # kerncov: L matrix for LBRF / lengthscales for ARD

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