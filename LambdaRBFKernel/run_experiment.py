from utils import *

def main():
    print('Dataset: BOSTON')
    X, Y = create_dataset('boston', 0, False)
    kfolds = KFold(n_splits = 8)
    model_params = {'reg':-1,
                    'num_inducing': 100, 
                    'likelihood': gpflow.likelihoods.Gaussian(), 
                    'max_iter': 10000, 
                    'minibatch_size': 1000}
    results_RBF = kfold_cv_model(model='GPR', 
                                 X=X, 
                                 Y=Y, 
                                 kernel='RBF', 
                                 k_folds=kfolds, 
                                 model_params=model_params, 
                                 iprint=True)
    results_LRBF = kfold_cv_model(model='GPR', 
                                  X=X, 
                                  Y=Y, 
                                  kernel='LRBF', 
                                  k_folds=kfolds, 
                                  model_params=model_params, 
                                  iprint=True)
main()