#import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import argparse

DATASETS_FEATURES = {'boston': 13, 'powerplant': 4, 'kin8nm': 8, 'eeg': 14, 'wilt': 5, 'breast': 10}
DATASETS_REGRESSION = {'boston': True, 'powerplant': True, 'kin8nm': True, 'eeg': False, 'wilt': False, 'breast': False}

def process_results_onefold(filepath=None, dict=None, invsquare=False):
    if dict is None: 
        with open(filepath) as f:
            results = json.load(f)
    else:
        results = dict
    dataset = results['dataset']
    pca = results['pca']
    d = DATASETS_FEATURES[dataset] if pca == -1 else pca
    precise_kernel = results['precise_kernel']
    regression = DATASETS_REGRESSION[dataset]
    n_samples = len(results['posterior_samples_kern_logvar'])
    posterior_samples_kerlogvar = np.array(results['posterior_samples_kern_logvar'])
    X_train_indices = np.array(results['X_train_indices'])
    X_test_indices = np.array(results['X_test_indices'])

    # AID kernel
    if precise_kernel:
        posterior_samples_L = [np.array(results['posterior_samples_kern_L'][i]) for i in range(n_samples)]
        Pd = np.array(results['Pd']) if pca != -1 else None
        precisions_list = []
        precisions_rec_list = []
        for i in range(n_samples):
            L = tfp.math.fill_triangular(posterior_samples_L[i], upper=False)
            precision = tf.linalg.matmul(L, tf.transpose(L))
            if pca != -1:
                precision_rec = tf.linalg.matmul(tf.linalg.matmul(Pd, precision), tf.transpose(Pd))
                precisions_rec_list.append(precision_rec)
            precisions_list.append(precision.numpy())
        precisions_merged = np.empty((d, d), dtype=object)
        precisions_merged_mean = np.empty((d, d))
        precisions_merged_var = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                precisions_merged[i, j] = [mat[i, j] for mat in precisions_list]
                precisions_merged_mean[i, j] = np.mean(precisions_merged[i, j])
                precisions_merged_var[i, j] = np.var(precisions_merged[i, j])
        processed_results = {'precisions_merged': precisions_merged, 'precisions_merged_mean': precisions_merged_mean, 'precisions_merged_var': precisions_merged_var, 'posterior_samples_kerlogvar': posterior_samples_kerlogvar, 'posterior_samples_kern_L': posterior_samples_L, 'test_mnll': results['test_mnll'], 'X_train_indices': X_train_indices, 'X_test_indices': X_test_indices}

        if pca != -1:
            D = Pd.shape[0] # Original number of features
            precisions_rec_merged = np.empty((D, D), dtype=object)
            precisions_rec_merged_mean = np.empty((D, D))
            precisions_rec_merged_var = np.empty((D, D))  
            for i in range(D):
                for j in range(D):
                    precisions_rec_merged[i, j] = [mat[i, j] for mat in precisions_rec_list]
                    precisions_rec_merged_mean[i, j] = np.mean(precisions_rec_merged[i, j])
                    precisions_rec_merged_var[i, j] = np.var(precisions_rec_merged[i, j])  
            processed_results['precisions_rec_merged'] = precisions_rec_merged
            processed_results['precisions_rec_merged_mean'] = precisions_rec_merged_mean
            processed_results['precisions_rec_merged_var'] = precisions_rec_merged_var
            processed_results['Pd'] = Pd

        if regression:
            processed_results['test_rmse'] = results['test_rmse']
        else:
            processed_results['test_accuracy'] = results['test_accuracy']
        return processed_results
    
    # ARD kernel
    else:
        posterior_samples_loglengthscales = [np.array(results['posterior_samples_loglengthscales'][i]) for i in range(n_samples)]
        lengthscales_list = []
        for i in range(n_samples):
            if invsquare:
                lengthscales_list.append(tf.linalg.diag(1/tf.math.exp(posterior_samples_loglengthscales[i]**2)).numpy())
            else:
                lengthscales_list.append(tf.linalg.diag(tf.math.exp(posterior_samples_loglengthscales[i])).numpy())
        lengthscales_merged = np.empty((d, d), dtype=object)
        lengthscales_merged_mean = []
        lengthscales_merged_var = []
        for i in range(d):
            for j in range(d):
                lengthscales_merged[i, j] = [mat[i, j] for mat in lengthscales_list] if i==j else [0]*len(lengthscales_list)
                if i==j:
                    lengthscales_merged_mean.append(np.mean(lengthscales_merged[i, j]))
                    lengthscales_merged_var.append(np.var(lengthscales_merged[i, j]))
        processed_results = {'lengthscales_merged': lengthscales_merged, 'lengthscales_merged_mean': lengthscales_merged_mean, 'lengthscales_merged_var': lengthscales_merged_var, 'posterior_samples_kerlogvar': posterior_samples_kerlogvar, 'test_mnll': results['test_mnll'], 'X_train_indices': X_train_indices, 'X_test_indices': X_test_indices}
        if regression:
            processed_results['test_rmse'] = results['test_rmse']
        else:
            processed_results['test_accuracy'] = results['test_accuracy']
        return processed_results

def process_results_kfold(filepath=None, invsquare=False):
    #results_kfold = pd.read_json(filepath)
    with open(filepath) as f:
        results_kfold = json.load(f)
    merged_kfold = []
    mean_kfold = []
    var_kfold = []
    kerlogvar_kfold = []
    posterior_samples_kern_L_kfold = []
    test_mnll_kfold = []
    test_rmse_kfold = []
    test_accuracy_kfold = []
    X_train_indices_kfold = []
    X_test_indices_kfold = []
    dataset = results_kfold['dataset']
    kfold = results_kfold['kfold']
    precise_kernel = results_kfold['precise_kernel']
    regression = DATASETS_REGRESSION[dataset]
    for k in range(kfold):
      posterior_samples_kern_cov = 'posterior_samples_kern_L' if precise_kernel else 'posterior_samples_loglengthscales'
      dict_fold_k = {
          'model': results_kfold['model'],
          'kfold': results_kfold['kfold'],
          'num_inducing': results_kfold['num_inducing'],
          'minibatch_size': results_kfold['minibatch_size'],
          'n_layers': results_kfold['n_layers'],
          'prior_type': results_kfold['prior_type'],
          'fold': results_kfold['fold'],
          'dataset': results_kfold['dataset'],
          'pca': results_kfold['pca'] if 'pca' in results_kfold else -1, # to adapt to json files without 'pca' key
          'precise_kernel': results_kfold['precise_kernel'],
          #'prior_precision_type': results_kfold['prior_precision_type'],
           posterior_samples_kern_cov: results_kfold['posterior_samples_kern_L'][k] if precise_kernel else results_kfold['posterior_samples_loglengthscales'][k],
          'posterior_samples_kern_logvar': results_kfold['posterior_samples_kern_logvar'][k],
          'posterior_samples_U': results_kfold['posterior_samples_U'][k],
          'posterior_samples_Z': results_kfold['posterior_samples_Z'][k],
          'test_mnll': results_kfold['test_mnll'][k],
          'X_train_indices': results_kfold['X_train_indices'][k],
          'X_test_indices': results_kfold['X_test_indices'][k]
      }      
      if regression:
        dict_fold_k['test_rmse'] = results_kfold['test_rmse'][k]
      else:
        dict_fold_k['test_accuracy'] = results_kfold['test_accuracy'][k]
      processed_results = process_results_onefold(dict=dict_fold_k, invsquare=invsquare)
      merged_kfold.append(processed_results['precisions_merged']) if precise_kernel else merged_kfold.append(processed_results['lengthscales_merged'])
      mean_kfold.append(processed_results['precisions_merged_mean']) if precise_kernel else mean_kfold.append(processed_results['lengthscales_merged_mean'])
      var_kfold.append(processed_results['precisions_merged_var']) if precise_kernel else var_kfold.append(processed_results['lengthscales_merged_var'])
      kerlogvar_kfold.append(processed_results['posterior_samples_kerlogvar'])
      posterior_samples_kern_L_kfold.append(processed_results['posterior_samples_kern_L'])
      test_mnll_kfold.append(processed_results['test_mnll'])
      if regression:
          test_rmse_kfold.append(processed_results['test_rmse'])
      else:
          test_accuracy_kfold.append(processed_results['test_accuracy'])   
      X_train_indices_kfold.append(processed_results['X_train_indices'])
      X_test_indices_kfold.append(processed_results['X_test_indices'])
    mean_over_kfold  = np.mean(np.array(mean_kfold), axis=0)
    var_over_kfold = np.var(np.array(mean_kfold), axis=0)
    processed_results_kfold = {}
    if precise_kernel:
        processed_results_kfold['precisions_merged_kfold'] = merged_kfold
        processed_results_kfold['precisions_merged_mean_kfold'] = mean_kfold
        processed_results_kfold['precisions_mean_over_kfold'] = mean_over_kfold
        processed_results_kfold['precisions_merged_var_kfold'] = var_kfold 
        processed_results_kfold['posterior_samples_kern_L_kfold'] = posterior_samples_kern_L_kfold
    else:
        processed_results_kfold['lengthscales_merged_kfold'] = merged_kfold
        processed_results_kfold['lengthscales_merged_mean_kfold'] = mean_kfold
        processed_results_kfold['lengthscales_mean_over_kfold'] = mean_over_kfold
        processed_results_kfold['lengthscales_merged_var_kfold'] = var_kfold 
        processed_results_kfold['lengthscales_var_over_kfold'] = var_over_kfold
    processed_results_kfold['posterior_samples_kerlogvar_kfold'] = kerlogvar_kfold
    processed_results_kfold['test_mnll'] = test_mnll_kfold
    if regression:
        processed_results_kfold['test_rmse'] = test_rmse_kfold
    else:
        processed_results_kfold['test_accuracy'] = test_accuracy_kfold
    processed_results_kfold['X_train_indices_kfold'] = X_train_indices_kfold
    processed_results_kfold['X_test_indices_kfold'] = X_test_indices_kfold
    return processed_results_kfold
