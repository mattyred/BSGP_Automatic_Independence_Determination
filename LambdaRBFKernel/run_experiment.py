from src.models.models import GPRLasso
from src.models.kernels import LambdaRBF
from src.utils import create_dataset, plot_matrix
import gpflow


def main():
    X_train, Y_train,  X_test, Y_test, Y_train_mean, Y_train_std = create_dataset('yacht', 0)
    D = X_train.shape[1]
    LRBF = LambdaRBF(variance=1.0, randomized=False, d=D)
    gpr_lasso_LRBF = GPRLasso(data=(X_train, Y_train),kernel=LRBF,lasso=0.1)
    gpflow.utilities.print_summary(gpr_lasso_LRBF, fmt="notebook")
    print('--- Initial values ---')
    gpr_lasso_LRBF.summary()
    gpr_lasso_LRBF.train()
    print('--- Optimal values ---')
    gpr_lasso_LRBF.summary()
    plot_matrix(gpr_lasso_LRBF.kernel.precision())
main()