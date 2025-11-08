## gp-pde-regression

> An unofficial implementation of the paper *“Gaussian Processes for Data Fulfilling Linear Differential Equations”* (Albert, 2019), created by [Boyuan Deng](https://bydeng01.github.io/) and available at: [https://arxiv.org/abs/1909.03447](https://arxiv.org/abs/1909.03447).

### Setup

- **Dependencies**
  - `numpy`
  - `scipy` (uses `scipy.optimize`, `scipy.linalg`, `scipy.special`)
  - `matplotlib`

Install:

```bash
pip install numpy scipy matplotlib
```

The experiments add `src` to `sys.path`, so you can run them directly.

### Run experiments

- **Laplace (Figure 1 style)**

```bash
python experiments/laplace_experiment.py
```

Outputs saved to `experiments/outputs/`:
- `laplace_pde_kernel.png`
- `laplace_se_kernel.png`
- `laplace_pde_violation.png`

- **Helmholtz (Figure 3 style)**

```bash
python experiments/helmholtz_experiment.py
```

Outputs saved to `experiments/outputs/`:
- `helmholtz_convergence.png`
- `helmholtz_likelihood.png`
- `helmholtz_sources.png`

- **Heat**

```bash
python experiments/heat_experiment.py
```

Outputs saved to `experiments/outputs/`:
- `heat_experiment.png`

Each script can also be imported and called programmatically via its `run_*_experiment(...)` function to customize parameters.

### Core API

- **Class** `gp_pde_regression.gp_pde.PDEGP(kernel, noise_variance=1e-4, source_locations=None, fundamental_solution=None)`
  - Trains a GP with a PDE-aware kernel. Optional point sources supported via a fundamental solution and source locations.
  - After `fit(...)` with sources, estimated source strengths and covariance are available as:
    - `b_mean` (array), `b_cov` (matrix)

  - **Methods**
    - `fit(X_train, y_train)`
    - `predict(X_test, return_std=False, return_cov=False)`
      - Returns `mean`, and optionally `std` or `cov`
    - `negative_log_likelihood()`
    - `optimize_hyperparameters(param_names, bounds, n_restarts=3)` → dict or `None`
      - Updates kernel in-place and refits on success

- **Kernels** (`gp_pde_regression.kernels`)
  - `LaplacePolarKernel(length_scale=1.0)`
  - `LaplacePolarKernelAlt(length_scale=1.0)`
  - `LaplaceLogKernel(length_scale=1.0)`
  - `LaplaceCartesianKernel(length_scale=1.0, rotation_angle=0.0)`
  - `HelmholtzKernel2D(wavenumber=1.0, regularization=1e-6)`
  - `HeatKernelNatural(diffusivity=1.0)`
  - `HeatKernelSourceFree(diffusivity=1.0, domain_bounds=(0.0, 1.0))`
  - `SquaredExponentialKernel(length_scale=1.0)`
  - All kernels support `update_hyperparams(**kwargs)` and callable evaluation `K = kernel(X1, X2)`

- **Fundamental solution**
  - `gp_pde_regression.kernels.HelmholtzFundamentalSolution2D(wavenumber=1.0)` is callable: `G(x, x_source)`

- **Utilities** (`gp_pde_regression.utils`)
  - `generate_grid_2d(bounds, n_points)` → `(X_grid, xx, yy)`
  - `generate_boundary_points(n_points, domain_size=1.0)` → `X_boundary`
  - `compute_metrics(y_true, y_pred, y_std=None)` → dict with `mae`, `max_error`, `rmse`, `relative_error`, optional `95_coverage`
  - Plot helpers:
    - `plot_laplace_results(...)`
    - `plot_helmholtz_convergence(sensor_counts, errors_pde, errors_se, save_path=None)`
    - `plot_likelihood_profile(k0_values, nll_values, k0_true, k0_opt, save_path=None)`
    - `plot_source_reconstruction(X_grid, xx, yy, y_pred, X_sources, q_mean, q_std, save_path=None)`

### Minimal usage examples

- **Standard GP with PDE-aware kernel**

```python
import numpy as np
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.kernels import LaplacePolarKernel

X_train = np.random.uniform(-1, 1, size=(20, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

kernel = LaplacePolarKernel(length_scale=2.0)
gp = PDEGP(kernel, noise_variance=0.01**2)
gp.fit(X_train, y_train)

X_test = np.random.uniform(-1, 1, size=(100, 2))
y_mean, y_std = gp.predict(X_test, return_std=True)
```

- **Helmholtz with source estimation and hyperparameter optimization**

```python
import numpy as np
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.kernels import HelmholtzKernel2D, HelmholtzFundamentalSolution2D

X_sources = np.array([[0.3, 0.4], [0.7, 0.8]])
k0_init = 9.0

kernel = HelmholtzKernel2D(wavenumber=k0_init)
G = HelmholtzFundamentalSolution2D(wavenumber=k0_init)
gp = PDEGP(kernel, noise_variance=0.01**2, source_locations=X_sources, fundamental_solution=G)

X_train = np.random.uniform([-1.5, -1.5], [1.5, 1.5], size=(20, 2))
y_train = np.random.randn(20)  # replace with measurements
gp.fit(X_train, y_train)

# Optimize wavenumber
opt = gp.optimize_hyperparameters(param_names=['wavenumber'], bounds=[(k0_init-2, k0_init+2)], n_restarts=3)
if opt:
    gp.fundamental_solution = HelmholtzFundamentalSolution2D(wavenumber=opt['wavenumber'])

# Predict
X_test = np.random.uniform([-1.5, -1.5], [1.5, 1.5], size=(50, 2))
y_pred = gp.predict(X_test)
q_mean, q_cov = gp.b_mean, gp.b_cov
```

- **Heat equation kernels**

```python
import numpy as np
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.kernels import HeatKernelNatural, HeatKernelSourceFree

# Data in (x, t)
X_train = np.column_stack([np.random.uniform(0, 1, 30), np.random.uniform(0.01, 0.5, 30)])
y_train = np.random.randn(30)

kernel = HeatKernelNatural(diffusivity=0.1)
gp = PDEGP(kernel, noise_variance=0.01**2)
gp.fit(X_train, y_train)

X_test = np.column_stack([np.linspace(0.05, 0.95, 20), np.linspace(0.01, 0.5, 20)])
y_mean, y_std = gp.predict(X_test, return_std=True)

# Source-free variant on (a, b)
kernel_sf = HeatKernelSourceFree(diffusivity=0.1, domain_bounds=(0.0, 1.0))
gp_sf = PDEGP(kernel_sf, noise_variance=0.01**2)
gp_sf.fit(X_train, y_train)
```