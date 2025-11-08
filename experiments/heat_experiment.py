"""
Heat equation experiment.
"""

import os, sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import matplotlib.pyplot as plt
from gp_pde_regression.kernels import HeatKernelNatural, HeatKernelSourceFree, SquaredExponentialKernel
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.utils import compute_metrics


def heat_analytical_solution(x, t, D=0.1, L=1.0):
    """
    Simple analytical solution for heat equation with initial condition.
    Using separation of variables: u(x,t) = sum of modes
    """
    u = 0.0
    n_modes = 20
    
    for n in range(1, n_modes + 1):
        # Sine series
        coeff = 2.0 / (n * np.pi) * (1 - (-1)**n)  # For u(x,0) = x
        u += coeff * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L)**2 * t)
    
    return u


def run_heat_experiment(D_true=0.1, domain_bounds=(0.0, 1.0),
                       n_train=30, noise_level=0.01, t_max=0.5):
    """
    Run heat equation experiment.
    
    Args:
        D_true: true diffusivity
        domain_bounds: (a, b) spatial domain
        n_train: number of training points
        noise_level: measurement noise std
        t_max: maximum time for observations
        
    Returns:
        results: dict with predictions and metrics
    """
    print("="*60)
    print("Heat Equation Experiment")
    print("="*60)
    print(f"True diffusivity: D = {D_true}")
    print(f"Domain: [{domain_bounds[0]}, {domain_bounds[1]}]")
    print(f"Time range: [0, {t_max}]")
    
    a, b = domain_bounds
    L = b - a
    
    # Generate training data (spacetime points)
    X_train_space = np.random.uniform(a + 0.1, b - 0.1, size=n_train)
    X_train_time = np.random.uniform(0.01, t_max, size=n_train)
    X_train = np.column_stack([X_train_space, X_train_time])
    
    y_train_true = heat_analytical_solution(X_train_space, X_train_time, 
                                           D=D_true, L=L)
    y_train = y_train_true + np.random.normal(0, noise_level, 
                                              size=y_train_true.shape)
    
    print(f"Training points: {n_train}")
    print(f"Noise level: {noise_level}")
    
    # Generate test grid
    n_space = 50
    n_time = 30
    x_test = np.linspace(a + 0.05, b - 0.05, n_space)
    t_test = np.linspace(0.01, t_max, n_time)
    
    X_test_list = []
    for t in t_test:
        for x in x_test:
            X_test_list.append([x, t])
    X_test = np.array(X_test_list)
    
    y_test_true = heat_analytical_solution(X_test[:, 0], X_test[:, 1], 
                                          D=D_true, L=L)
    
    # 1. Natural heat kernel
    print("\n1. Fitting GP with natural heat kernel...")
    kernel_natural = HeatKernelNatural(diffusivity=D_true)
    gp_natural = PDEGP(kernel_natural, noise_variance=noise_level**2)
    gp_natural.fit(X_train, y_train)
    
    y_pred_natural, y_std_natural = gp_natural.predict(X_test, return_std=True)
    
    metrics_natural = compute_metrics(y_test_true, y_pred_natural, y_std_natural)
    print(f"   MAE: {metrics_natural['mae']:.6f}")
    print(f"   RMSE: {metrics_natural['rmse']:.6f}")
    print(f"   95% coverage: {metrics_natural['95_coverage']:.4f}")
    
    # 2. Source-free domain kernel
    print("\n2. Fitting GP with source-free kernel...")
    kernel_sf = HeatKernelSourceFree(diffusivity=D_true, 
                                     domain_bounds=domain_bounds)
    gp_sf = PDEGP(kernel_sf, noise_variance=noise_level**2)
    gp_sf.fit(X_train, y_train)
    
    y_pred_sf, y_std_sf = gp_sf.predict(X_test, return_std=True)
    
    metrics_sf = compute_metrics(y_test_true, y_pred_sf, y_std_sf)
    print(f"   MAE: {metrics_sf['mae']:.6f}")
    print(f"   RMSE: {metrics_sf['rmse']:.6f}")
    print(f"   95% coverage: {metrics_sf['95_coverage']:.4f}")
    
    results = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test_true': y_test_true,
        'y_pred_natural': y_pred_natural,
        'y_std_natural': y_std_natural,
        'y_pred_sf': y_pred_sf,
        'y_std_sf': y_std_sf,
        'metrics_natural': metrics_natural,
        'metrics_sf': metrics_sf,
        'x_grid': x_test,
        't_grid': t_test
    }
    
    return results


def plot_heat_results(results, save_path=None):
    """Plot heat equation results as spacetime heatmaps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    n_space = len(results['x_grid'])
    n_time = len(results['t_grid'])
    
    # Reshape for visualization
    y_true_grid = results['y_test_true'].reshape(n_time, n_space)
    y_pred_nat_grid = results['y_pred_natural'].reshape(n_time, n_space)
    y_pred_sf_grid = results['y_pred_sf'].reshape(n_time, n_space)
    
    vmin = y_true_grid.min()
    vmax = y_true_grid.max()
    
    # True solution
    im = axes[0, 0].imshow(y_true_grid, aspect='auto', origin='lower',
                          extent=[results['x_grid'][0], results['x_grid'][-1],
                                 results['t_grid'][0], results['t_grid'][-1]],
                          vmin=vmin, vmax=vmax, cmap='hot')
    axes[0, 0].set_title('True Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Natural kernel prediction
    im = axes[0, 1].imshow(y_pred_nat_grid, aspect='auto', origin='lower',
                          extent=[results['x_grid'][0], results['x_grid'][-1],
                                 results['t_grid'][0], results['t_grid'][-1]],
                          vmin=vmin, vmax=vmax, cmap='hot')
    axes[0, 1].set_title('Natural Kernel Prediction')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Source-free kernel prediction
    im = axes[0, 2].imshow(y_pred_sf_grid, aspect='auto', origin='lower',
                          extent=[results['x_grid'][0], results['x_grid'][-1],
                                 results['t_grid'][0], results['t_grid'][-1]],
                          vmin=vmin, vmax=vmax, cmap='hot')
    axes[0, 2].set_title('Source-Free Kernel Prediction')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('t')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Errors
    error_nat = np.abs(y_true_grid - y_pred_nat_grid)
    error_sf = np.abs(y_true_grid - y_pred_sf_grid)
    
    im = axes[1, 0].imshow(error_nat, aspect='auto', origin='lower',
                          extent=[results['x_grid'][0], results['x_grid'][-1],
                                 results['t_grid'][0], results['t_grid'][-1]],
                          cmap='Reds')
    axes[1, 0].set_title('Natural Kernel Error')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(error_sf, aspect='auto', origin='lower',
                          extent=[results['x_grid'][0], results['x_grid'][-1],
                                 results['t_grid'][0], results['t_grid'][-1]],
                          cmap='Reds')
    axes[1, 1].set_title('Source-Free Kernel Error')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('t')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Training points
    axes[1, 2].scatter(results['X_train'][:, 0], results['X_train'][:, 1],
                      c=results['y_train'], cmap='hot', s=50, edgecolors='black')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('t')
    axes[1, 2].set_title('Training Data')
    axes[1, 2].set_xlim([results['x_grid'][0], results['x_grid'][-1]])
    axes[1, 2].set_ylim([results['t_grid'][0], results['t_grid'][-1]])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    # Set random seed
    np.random.seed(42)
    
    # Run experiment
    results = run_heat_experiment(
        D_true=0.1,
        domain_bounds=(0.0, 1.0),
        n_train=30,
        noise_level=0.01,
        t_max=0.5
    )
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    fig = plot_heat_results(
        results,
        save_path=os.path.join(outputs_dir, 'heat_experiment.png')
    )
    plt.close()
    
    print(f"\nResults saved to {outputs_dir}/")
    print("="*60)