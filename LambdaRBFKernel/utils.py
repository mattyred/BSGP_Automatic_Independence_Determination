import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import torch

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

def create_dataset(dataset, fold):
    dataset_path = ('../data/' + dataset + '.pth')
    #logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy().astype('float64'), Y.numpy().astype('float64')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
    Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
    Y_train = (Y_train - Y_train_mean) / Y_train_std
    Y_test = (Y_test - Y_train_mean) / Y_train_std

    return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std

def measure_rmse(model, X_train, Y_train, X_test, Y_test):
    y_pred_train, _ = model.predict_f(X_train)
    train_rmse_stan = tf.sqrt(tf.reduce_mean((Y_train - y_pred_train)**2)).numpy()
    y_pred_test, _ = model.predict_f(X_test)
    test_rmse_stan = tf.sqrt(tf.reduce_mean((Y_test - y_pred_test)**2)).numpy()
    return train_rmse_stan, test_rmse_stan