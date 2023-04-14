## Wine dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time  |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :---: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    0.794     |  -   | 0.716 / 0.746 |             |            |            | 4m/5m |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    1.414     |  -   | 0.648 / 0.729 |             |            |            |  6m   |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |              | 0.1  | 0.689 / 0.751 |             |            |            |  6m   |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.718     |  1   | 0.691 / 0.756 |             |            |            |       |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    6.728     |  10  | 0.737 / 0.744 |             |            |            |       |

## Yacht dataset

|     $l$, $\sigma_f^2$     | $\sigma_f^2$ | Lasso | LRBF-RMSE-kfold | LRBF-MNLL-kfold | ARD-RMSE-kfold | ARD-MNLL-kfold |
| :-----------------------: | :----------: | :---: | :-------------: | :-------------: | :------------: | :------------: |
| $l=\sqrt{D},\sigma_f^2=1$ |              |   0   |      0.026      |      2.108      |     0.030      |     1.501      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |  0.1  |      0.031      |      1.510      |     0.030      |     1.502      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |   1   |      0.029      |      1.465      |     0.030      |     1.510      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |  10   |      0.034      |      2.568      |     0.035      |     2.090      |

## Boston dataset

|     $l$, $\sigma_f^2$     | $\sigma_f^2$ | Lasso | LRBF-RMSE-kfold | LRBF-MNLL-kfold | ARD-RMSE-kfold | ARD-MNLL-kfold |
| :-----------------------: | :----------: | :---: | :-------------: | :-------------: | :------------: | :------------: |
| $l=\sqrt{D},\sigma_f^2=1$ |              |   0   |      0.362      |      5.528      |     0.302      |     3.476      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |  0.1  |      0.314      |      5.325      |     0.302      |     3.484      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |   1   |      0.029      |      1.465      |     0.030      |     1.510      |
| $l=\sqrt{D},\sigma_f^2=1$ |              |  10   |      0.034      |      2.568      |     0.035      |     2.090      |

## Concrete dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :--: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    2.726     |  -   | 0.208 / 0.331 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    1.545     |  -   | 0.192 / 0.306 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.058     | 0.1  | 0.115 / 0.328 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.764     |  1   | 0.197 / 0.287 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    9.120     |  10  | 0.218 / 0.315 |             |            |            |      |
