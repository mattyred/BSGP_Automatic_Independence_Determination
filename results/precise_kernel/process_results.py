#import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import argparse

def process_results_onefold(filepath=None, dict=None, precise_kernel=0, invsquare=False, d=6, regression=True):
    #results = pd.read_json(filepath)
    if dict is None: 
        with open(filepath) as f:
            results = json.load(f)
    else:
        results = dict
    n_samples = len(results['posterior_samples_kern_logvar'])
    posterior_samples_kerlogvar = np.array(results['posterior_samples_kern_logvar'])
    test_mnll = results['test_mnll']
    if regression:
        test_rmse = results['test_rmse']
    else:
        test_accuracy = results['test_accuracy']
    X_train_indices = np.array(results['X_train_indices'])
    X_test_indices = np.array(results['X_test_indices'])
    if precise_kernel:
        posterior_samples_L = [np.array(results['posterior_samples_kern_L'][i]) for i in range(n_samples)]
        precisions_list = []
        for i in range(n_samples):
            L = tfp.math.fill_triangular(posterior_samples_L[i], upper=False)
            precision = tf.linalg.matmul(L, tf.transpose(L))
            precisions_list.append(precision.numpy())
        precisions_merged = np.empty((d, d), dtype=object)
        precisions_merged_mean = np.empty((d, d))
        precisions_merged_var = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                precisions_merged[i, j] = [mat[i, j] for mat in precisions_list]
                precisions_merged_mean[i, j] = np.mean(precisions_merged[i, j])
                precisions_merged_var[i, j] = np.var(precisions_merged[i, j])
        processed_results = {'precisions_merged': precisions_merged, 'precisions_merged_mean': precisions_merged_mean, 'precisions_merged_var': precisions_merged_var, 'posterior_samples_kerlogvar': posterior_samples_kerlogvar, 'test_mnll': test_mnll, 'X_train_indices': X_train_indices, 'X_test_indices': X_test_indices}
        if regression:
            processed_results['test_rmse'] = test_rmse
        else:
            processed_results['test_accuracy'] = test_accuracy
        return processed_results
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
        processed_results = {'lengthscales_merged': lengthscales_merged, 'lengthscales_merged_mean': lengthscales_merged_mean, 'lengthscales_merged_var': lengthscales_merged_var, 'posterior_samples_kerlogvar': posterior_samples_kerlogvar, 'test_mnll': test_mnll, 'X_train_indices': X_train_indices, 'X_test_indices': X_test_indices}
        if regression:
            processed_results['test_rmse'] = test_rmse
        else:
            processed_results['test_accuracy'] = test_accuracy
        return processed_results

def process_results_kfold(filepath=None, kfold=3, precise_kernel=0, invsquare=False, d=6, regression=True):
    #results_kfold = pd.read_json(filepath)
    with open(filepath) as f:
        results_kfold = json.load(f)
    merged_kfold = []
    mean_kfold = []
    var_kfold = []
    kerlogvar_kfold = []
    test_mnll_kfold = []
    test_rmse_kfold = []
    test_accuracy_kfold = []
    X_train_indices_kfold = []
    X_test_indices_kfold = []
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
      processed_results = process_results_onefold(dict=dict_fold_k, precise_kernel=precise_kernel, invsquare=invsquare, d=d)
      merged_kfold.append(processed_results['precisions_merged']) if precise_kernel else merged_kfold.append(processed_results['lengthscales_merged'])
      mean_kfold.append(processed_results['precisions_merged_mean']) if precise_kernel else mean_kfold.append(processed_results['lengthscales_merged_mean'])
      var_kfold.append(processed_results['precisions_merged_var']) if precise_kernel else var_kfold.append(processed_results['lengthscales_merged_var'])
      kerlogvar_kfold.append(processed_results['posterior_samples_kerlogvar'])
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
        processed_results_kfold['precisions_var_over_kfold'] = var_over_kfold
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

def heatmap_precision(precisions_mean, precisions_var, annot=True, fig_height=7, fig_width=5):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    d = precisions_mean.shape[0]
    min, max = np.min(precisions_mean), np.max(precisions_mean)
    h = sns.heatmap(precisions_mean, annot=annot, fmt='.3f', annot_kws={"size": 8}, cmap='vlag', vmax=max, vmin=-max, center=0, linewidth=.5, ax=ax) 
    if annot:
        for i in range(d):
            for j in range(d):
                mean_text = f'{precisions_mean[i, j]:.3f}'
                variance_text = f'({precisions_var[i, j]:.3f})'
                text = f'{mean_text}\n' + f'{variance_text}'
                h.texts[i * d + j].set_text(text)
    ax.set_title(r'$\Lambda_{i,j}$: mean(var) over samples')
    plt.show()

def heatmap_ard(lengthscales_merged_mean, lengthscales_merged_var, invsquare=False, annot=True, fig_height=5, fig_width=7):
  fig, ax = plt.subplots(figsize=(fig_width, fig_height))
  d = len(lengthscales_merged_mean)
  min, max = np.min(lengthscales_merged_mean), np.max(lengthscales_merged_mean)
  lengthscales_merged_mean_matrix = np.diag(lengthscales_merged_mean)
  lengthscales_merged_var_matrix = np.diag(lengthscales_merged_var)
  h = sns.heatmap(lengthscales_merged_mean_matrix, annot=True, fmt='.3f', annot_kws={"size": 8}, cmap='vlag', vmax=max, vmin=-max, center=0, linewidth=.5, ax=ax) 
  if annot:
    for i in range(d):
        for j in range(d):
            if i == j:
                mean_text = f'{lengthscales_merged_mean_matrix[i, j]:.3f}'
                variance_text = f'({lengthscales_merged_var_matrix[i, j]:.3f})'
                text = f'{mean_text}\n' + f'{variance_text}'
                h.texts[i * d + j].set_text(text)
  if invsquare:
    ax.set_title(r'$\Sigma^{-1}_{i,j}$: $\mathbf{\frac{1}{l^2}}$ mean(var) over samples')
  else:
    ax.set_title(r'$\Sigma_{i,j}$: $\mathbf{l}$ mean(var) over samples')
  plt.show()

def histograms_precision(precisios_merged=None, lengthscales_merged=None, d=6, fig_width=10, fig_height=10, bins=100):
    fig, axes = plt.subplots(d, d, figsize=(fig_width, fig_height))
    LRBF_min, LRBF_max = np.min(np.min(precisios_merged)), np.max(np.max(precisios_merged))
    if lengthscales_merged is not None:
        ARD_min, ARD_max = np.min(np.min(lengthscales_merged)), np.max(np.max(lengthscales_merged))
    else:
        ARD_min, ARD_max = LRBF_min, LRBF_max
    glob_min, glob_max = np.min([ARD_min, LRBF_min]), np.max([ARD_max, LRBF_max])
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            sns.histplot(precisios_merged[i, j], ax=ax, bins=bins, kde=True)
            if lengthscales_merged is not None and i==j:
                sns.histplot(lengthscales_merged[i, j], ax=ax, bins=bins, kde=True)
            ax.set_xlim(-glob_min-1e-3, glob_max+1e-3)
            ax.set_xlabel('')
            ax.set_ylabel('')
    fig.tight_layout()
    plt.show()

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

def main():
    processed_results_kfold = process_results_kfold(filepath=args.input_path, precise_kernel=args.precise_kernel, d=args.D, kfold=args.kfold)
    processed_results_kfold_aslist = convert_numpy_to_list(processed_results_kfold)
    with open(args.output_path, "w") as outfile:
        json.dump(processed_results_kfold_aslist, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regression experiment')
    parser.add_argument('--input_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--dataset', choices=['boston','kin8nm'], default=None)
    parser.add_argument('--D', type=int, default=None)
    parser.add_argument('--precise_kernel', type=int, choices=[0,1], default=1)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--type_data', choices=['precision','performance'], default=None)
    args = parser.parse_args()
    main()