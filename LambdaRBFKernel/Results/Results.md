## Wine dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time  |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :---: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    0.794     |  -   | 0.716 / 0.746 |             |            |            | 4m/5m |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    1.414     |  -   | 0.648 / 0.729 |             |            |            |  6m   |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |              | 0.1  | 0.689 / 0.751 |             |            |            |  6m   |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.718     |  1   | 0.691 / 0.756 |             |            |            |       |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    6.728     |  10  | 0.737 / 0.744 |             |            |            |       |

## Yacht dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :--: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    0.605     |  -   | 0.000 / 0.042 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    10.455    |  -   | 0.008 / 0.037 |             |   0.026    |   2.108    |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    4.834     | 0.1  | 0.007 / 0.022 |             |   0.031    |   1.510    |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    9.895     |  1   | 0.044 / 0.057 |             |   0.029    |   1.465    |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    4.838     |  10  | 0.011 / 0.029 |             |   0.034    |   2.568    |      |

## Boston dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :--: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    1.765     |  -   | 0.173 / 0.313 |             |   0.503    |   4.145    |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    1.959     |  -   | 0.113 / 0.327 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.058     | 0.1  | 0.115 / 0.328 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    3.851     |  1   | 0.184 / 0.302 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    13.844    |  10  | 0.211 / 0.355 |             |            |            |      |

## Concrete dataset

| Model |  Kernel  |     $l$, $\sigma_f^2$     | $\sigma_f^2$ | r-L1 |  RMSE-static  | MNLL-static | RMSE-kfold | MNLL-kfold | Time |
| :---: | :------: | :-----------------------: | :----------: | :--: | :-----------: | :---------: | :--------: | :--------: | :--: |
|  GPR  | RBF(ARD) | $l=\sqrt{D},\sigma_f^2=1$ |    2.726     |  -   | 0.208 / 0.331 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    1.545     |  -   | 0.192 / 0.306 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.058     | 0.1  | 0.115 / 0.328 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    2.764     |  1   | 0.197 / 0.287 |             |            |            |      |
|  GPR  |   LRBF   | $l=\sqrt{D},\sigma_f^2=1$ |    9.120     |  10  | 0.218 / 0.315 |             |            |            |      |
