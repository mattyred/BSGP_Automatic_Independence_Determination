# Sparse Gaussian Processes Revisited: Bayesian Approaches to Inducing-Variable Approximations

```bash
>> python3 run_regression.py --help
usage: run_regression.py [-h] 
                         [--num_inducing NUM_INDUCING]
                         [--minibatch_size MINIBATCH_SIZE]
                         [--iterations ITERATIONS] 
                         [--n_layers N_LAYERS]
                         --dataset DATASET [--fold FOLD]
                         [--prior_type {determinantal,normal,strauss,uniform}]
                         [--model {bsgp}]
                         [--num_posterior_samples NUM_POSTERIOR_SAMPLES]
                         [--step_size STEP_SIZE]
                         [--precise_kernel USE_AID_KERNEL {0,1,2}]
                         [--kfold NUM_K_FOLDS]
                         [--prior_precision_type {normal, laplace+diagnormal, horseshoe+diagnormal, wishart, invwishart}]
                         [--prior_laplace_b LAPLACE_B]
                         [--prior_normal_mean NORMAL_MEAN]
                         [--prior_normal_variance NORMAL_VARIANCE]
                         [--prior_horseshoe_globshrink HORSESHOE_GLOBAL_SHRINKAGE]

>> python3 run_regression.py --help
usage: [same arguments as for regression, choose a proper dataset]
```

### Datasets

|            |      type      |   n.  | d-in |                                                                     |
|:----------:|:--------------:|:-----:|:----:|---------------------------------------------------------------------|
|     BOSTON |     regression |   506 |   13 | https://archive.ics.uci.edu/ml/datasets/Housing                     |
|     KIN8NM |     regression |  8192 |    8 |                                                                     |
| POWERPLANT |     regression |  9568 |    4 | https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant  |
|        EEG | classification | 14980 |   14 | https://archive.ics.uci.edu/dataset/264/eeg+eye+state               |
|       WILT | classification |  4839 |    5 | https://archive.ics.uci.edu/dataset/285/wilt                        |
|     BREAST | classification |   683 |   10 | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html |                                                                |

### Precise Kernel

|            |                                                         | parameters                                         | log-pdf                                                                                                   |
|------------|---------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Normal     | $p(\mathbf{\Lambda_{ij}}) = \mathcal{N}(\mu, \sigma^2)$ | --prior_normal_mean <br /> --prior_normal_variance | $\log C - \frac{1}{2\sigma^2}(\mathbf{\Lambda_{ij}} - \mu)^2$                                             |
| Laplace    | $p(\mathbf{\Lambda_{ij}}) = \mathcal{L}(m,b)$           | $m = 0$  <br /> --prior_laplace_b                  | $\log C - \frac{1}{b}\|\|\mathbf{\Lambda_{ij}} - m\|\|_1$                                                 |
| Horseshoe  | $p(\mathbf{\Lambda_{ij}}) = \mathcal{HS}(\tau)$         | --prior_horseshoe_globshrink                       | $\log C + \frac{1}{2\tau^2}\mathbf{\Lambda_{ij}}^2 + \log E_1(\frac{1}{2\tau^2}\mathbf{\Lambda_{ij}}^2)$  |
| Wishart    | $p(\mathbf{\Lambda}) = \mathcal{W}(\mathbf{V},K)$       | $K = D$ <br />  $\mathbf{V} = K^{-1}\mathbf{I}_D$  | $\log C - \sum_d{\log \|\mathbf{L}_{dd}\|} - \frac{1}{2}\text{Tr}[K\mathbf{\Lambda}]$                     |
| InvWishart | $p(\mathbf{\Lambda}) = \mathcal{IW}(\mathbf{V},K)$      | $K = D$ <br />  $\mathbf{V} = \mathbf{I}_D$        | $\log C - (2K + 1)\sum_d{\log \|\mathbf{L}_{dd}\|} - \frac{1}{2}\text{Tr}[\mathbf{V}\mathbf{\Lambda}^-1]$ |

### Reference
Rossi, S., Heinonen, M., Bonilla, E., Shen, Z. &amp; Filippone, M.. (2021).  Sparse Gaussian Processes Revisited: Bayesian Approaches to Inducing-Variable Approximations. <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:1837-1845 