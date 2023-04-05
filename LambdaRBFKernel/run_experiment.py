import matplotlib.pyplot as plt # 3.6
import numpy as np # 1.22.4
import tensorflow as tf # 2.11.0
import gpflow # 2.7.0
import pandas as pd
import tensorflow_probability as tfp
from tensorflow import keras
import seaborn as sns
import torch
from tensorflow.python.ops.numpy_ops import np_config
from LambdaRBF import LambdaRBF
from utils import *
np_config.enable_numpy_behavior()
plt.rcParams["figure.figsize"] = (12, 6)
plt.style.use("ggplot")
print('tensorflow ', tf.__version__) 
print('pytorch ', torch.__version__) 
print('numpy ', np.__version__) 
print('gpflow ', gpflow.__version__) 
print('pandas ', pd.__version__) 

def main():
    X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std = create_dataset('boston', 0)
    D = X_train.shape[1]
    model_LRBF, Lambda = train_GPR_LRBF_model(X_train=X_train, Y_Train=Y_train, reg=0.1, iprint=True)
    train_rmse_stan, test_rmse_stan = measure_rmse(model_LRBF, X_train, Y_train, X_test, Y_test)
    print('Train RMSE (Standardised): %.3f'%(train_rmse_stan))
    print('Test RMSE (Standardised): %.3f'%(test_rmse_stan))
    train_mnll, test_mnll = measure_mnll(model_LRBF, X_train, Y_train, Y_train_std, X_test, Y_test)
    print('Train MNLL: %.3f'%(train_mnll))
    print('Test MNLL: %.3f'%(test_mnll))

main()