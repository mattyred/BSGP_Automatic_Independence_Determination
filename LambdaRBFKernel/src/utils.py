import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 6)
plt.style.use("ggplot")

def plot_matrix(M=None, cmap='vlag', annot=True, correlation=False):
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
        sns.heatmap(corr, annot=annot, cmap=cmap)
    else:
        min = np.min(M)
        max = np.max(M)
        center = (min+max)/2
        sns.heatmap(M, annot=annot, cmap=cmap, vmax=max, vmin=-max, center=0, linewidth=.5)

def compare_matrix(M1, M2, cmap='vlag'):
    fig, axes = plt.subplots(1, 2, figsize=(24,6))
    min = np.min(M1)
    max = np.max(M1)
    center = (min+max)/2
    sns.heatmap(M1, ax=axes[0], annot=True, cmap=cmap, vmax=max, vmin=min, center=center, linewidth=.5)
    axes[0].set_title('LambdaRBF')

    min = np.min(M2)
    max = np.max(M2)
    center = (min+max)/2
    sns.heatmap(M2, ax=axes[1], annot=True, cmap=cmap, vmax=max, vmin=min, center=center, linewidth=.5)
    axes[1].set_title('ARD')
    plt.show()

def plot_matrix_cv(lambdas=None, cmap='vlag', annot=False, k=8, info={'dataset':None, 'lasso':0}):
    fig, axes = plt.subplots(4, 2, sharex=True, sharey=True)
    for i, Lambda in enumerate(lambdas):
        max = np.max(Lambda)
        row, col = i//2, i%2
        sns.heatmap(Lambda.numpy(), ax=axes[row,col], annot=annot, cmap=cmap, vmax=max, vmin=-max, center=0, linewidth=.5)
    fig.suptitle("CV Lambda optimized - %s - lasso=%1.1f"%(info['dataset'], info['lasso']), fontsize=16)
    plt.show()

def create_dataset(dataset, fold, static_train_test=True):
    dataset_path = ('./data/' + dataset + '.pth')
    #logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))
    X, Y = dataset.tensors
    X, Y = X.numpy().astype('float64'), Y.numpy().astype('float64')
    if static_train_test:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
        Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
        Y_train = (Y_train - Y_train_mean) / Y_train_std
        Y_test = (Y_test - Y_train_mean) / Y_train_std
        return X_train, Y_train, X_test, Y_test, Y_train_mean, Y_train_std
    else:
        return X, Y
