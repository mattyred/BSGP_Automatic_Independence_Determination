import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

def process_results(filepath=None, precise_kernel=0, invsquare=False, d=6):
    results = pd.read_json(filepath)
    n_samples = len(results['posterior_samples_kernlogvar'])
    posterior_samples_kerlogvar = np.array(results['posterior_samples_kernlogvar'])
    if precise_kernel:
        posterior_samples_U = [np.array(results['posterior_samples_U_precision'][i]) for i in range(n_samples)]
        precisions_list = []
        for i in range(n_samples):
            U = tfp.math.fill_triangular(posterior_samples_U[i], upper=True)
            precision = tf.linalg.matmul(tf.transpose(U), U)
            precisions_list.append(precision.numpy())
        precisions_merged = np.empty((d, d), dtype=object)
        precisions_merged_mean = np.empty((d, d))
        precisions_merged_var = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                precisions_merged[i, j] = [mat[i, j] for mat in precisions_list]
                precisions_merged_mean[i, j] = np.mean(precisions_merged[i, j])
                precisions_merged_var[i, j] = np.var(precisions_merged[i, j])
        return precisions_merged, precisions_merged_mean, precisions_merged_var, posterior_samples_kerlogvar
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
        return lengthscales_merged, lengthscales_merged_mean, lengthscales_merged_var, posterior_samples_kerlogvar

def heatmap_precision(precisions_mean, precisions_var, annot=True, fig_height=7, fig_width=5):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    d = precisions_mean.shape[0]
    min, max = np.min(precisions_mean), np.max(precisions_mean)
    h = sns.heatmap(precisions_mean, annot=annot, fmt='.3f', annot_kws={"size": 8}, cmap='vlag', vmax=max, vmin=-max, center=0, linewidth=.5, ax=ax) 
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
