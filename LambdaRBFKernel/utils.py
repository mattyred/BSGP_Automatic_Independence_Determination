import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf

def plot_matrix(M=None, cmap='vlag', correlation=False):
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
        sns.heatmap(corr, annot=True, cmap=cmap)
    else:
        D = M.shape[0]
        min = np.min(M)
        max = np.max(M)
        center = (min+max)/2
        sns.heatmap(M, annot=True, cmap=cmap, vmax=max, vmin=min, center=center, linewidth=.5)

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