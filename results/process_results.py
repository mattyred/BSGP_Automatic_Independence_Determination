import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

def process_results(filepath=None, precise_kernel=0, d=6):
    results = pd.read_json(filepath)
    n_samples = len(results['posterior_samples_kernlogvar'])
    if precise_kernel:
        posterior_samples_U = [np.array(results['posterior_samples_U_precision'][i]) for i in range(n_samples)]
        posterior_samples_kerlogvar = np.array(results['posterior_samples_kernlogvar'])
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

def heatmap_precision(precisions_mean, precisions_var, fig_height=15, fig_width=5):
    fig, ax = plt.subplots(1, 2, figsize=(fig_height, fig_width))
    d = precisions_mean.shape[0]
    min, max = np.min(precisions_mean), np.max(precisions_mean)
    h = sns.heatmap(precisions_mean, annot=True, fmt='.3f', annot_kws={"size": 8}, cmap='vlag', vmax=max, vmin=-max, center=0, linewidth=.5, ax=ax[0]) 
    for i in range(d):
        for j in range(d):
            mean_text = f'{precisions_mean[i, j]:.3f}'
            variance_text = f'({precisions_var[i, j]:.3f})'
            text = f'{mean_text}\n' + f'{variance_text}'
            h.texts[i * d + j].set_text(text)
    ax[0].set_title(r'$\Lambda$ - mean(var) over samples')
    ax[1].set_title('RBF-ARD inv. squared lengthscales')
    plt.show()
