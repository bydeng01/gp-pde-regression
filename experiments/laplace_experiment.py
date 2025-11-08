"""
Laplace equation experiment (reproduces Figure 1).
"""

import os, sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import matplotlib.pyplot as plt
from gp_pde_regression.kernels import LaplacePolarKernel, SquaredExponentialKernel
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.utils import (generate_grid_2d, generate_boundary_points, 
                   compute_metrics, plot_laplace_results)


def laplace_reference_solution(x, y):
    """
    Analytical solution from paper (Equation 31).
    u(x,y) = 0.5 * e^y * cos(x) + 2x * cos(2y)
    """
    return 0.5 * np.exp(y) * np.cos(x) + 2*x * np.cos(2*y)


def run_laplace_experiment(domain_size=1.0, n_train=8, noise_level=0.1,
                           length_scale=2.0, n_grid=100):
    """
    Run Laplace equation experiment.
    
    Args:
        domain_size: half-width of square domain
        n_train: number of training points (on boundary)
        noise_level: measurement noise std deviation
        length_scale: kernel length scale parameter s
        n_grid: number of grid points per dimension for evaluation
        
    Returns:
        results: dict with predictions and metrics
    """
    print("="*60)
    print("Laplace Equation Experiment")
    print("="*60)
    
    # Generate training data on boundary
    X_train = generate_boundary_points(n_train, domain_size)
    y_train_true = laplace_reference_solution(X_train[:, 0], X_train[:, 1])
    y_train = y_train_true + np.random.normal(0, noise_level, size=y_train_true.shape)
    
    print(f"Training points: {n_train} (boundary)")
    print(f"Noise level: {noise_level}")
    print(f"Length scale: {length_scale}")
    
    # Generate test grid
    bounds = ((-domain_size, domain_size), (-domain_size, domain_size))
    X_test, xx, yy = generate_grid_2d(bounds, n_grid)
    y_test_true = laplace_reference_solution(X_test[:, 0], X_test[:, 1])
    
    # PDE-aware kernel
    print("\n1. Fitting GP with PDE-aware (Laplace polar) kernel...")
    kernel_pde = LaplacePolarKernel(length_scale=length_scale)
    gp_pde = PDEGP(kernel_pde, noise_variance=noise_level**2)
    gp_pde.fit(X_train, y_train)
    
    y_pred_pde, y_std_pde = gp_pde.predict(X_test, return_std=True)
    
    metrics_pde = compute_metrics(y_test_true, y_pred_pde, y_std_pde)
    print(f"   MAE: {metrics_pde['mae']:.4f}")
    print(f"   Max error: {metrics_pde['max_error']:.4f}")
    print(f"   95% coverage: {metrics_pde['95_coverage']:.4f}")
    
    # Squared exponential kernel for comparison
    print("\n2. Fitting GP with squared exponential kernel...")
    kernel_se = SquaredExponentialKernel(length_scale=length_scale/2)
    gp_se = PDEGP(kernel_se, noise_variance=noise_level**2)
    gp_se.fit(X_train, y_train)
    
    y_pred_se, y_std_se = gp_se.predict(X_test, return_std=True)
    
    metrics_se = compute_metrics(y_test_true, y_pred_se, y_std_se)
    print(f"   MAE: {metrics_se['mae']:.4f}")
    print(f"   Max error: {metrics_se['max_error']:.4f}")
    print(f"   95% coverage: {metrics_se['95_coverage']:.4f}")
    
    # Check PDE satisfaction (compute Laplacian of prediction)
    print("\n3. Checking PDE satisfaction (∆u = 0)...")
    laplacian_pde = compute_laplacian_2d(y_pred_pde, xx, yy)
    laplacian_se = compute_laplacian_2d(y_pred_se, xx, yy)
    
    print(f"   PDE kernel - mean |∆u|: {np.mean(np.abs(laplacian_pde)):.6f}")
    print(f"   SE kernel - mean |∆u|: {np.mean(np.abs(laplacian_se)):.6f}")
    
    results = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'xx': xx,
        'yy': yy,
        'y_true': y_test_true,
        'y_pred_pde': y_pred_pde,
        'y_std_pde': y_std_pde,
        'y_pred_se': y_pred_se,
        'y_std_se': y_std_se,
        'metrics_pde': metrics_pde,
        'metrics_se': metrics_se,
        'laplacian_pde': laplacian_pde,
        'laplacian_se': laplacian_se
    }
    
    return results


def compute_laplacian_2d(values, xx, yy):
    """
    Compute Laplacian using finite differences.
    """
    ZZ = values.reshape(xx.shape)
    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]
    
    # Second derivatives
    d2_dx2 = np.zeros_like(ZZ)
    d2_dy2 = np.zeros_like(ZZ)
    
    # Interior points (central differences)
    d2_dx2[1:-1, 1:-1] = (ZZ[1:-1, 2:] - 2*ZZ[1:-1, 1:-1] + ZZ[1:-1, :-2]) / dx**2
    d2_dy2[1:-1, 1:-1] = (ZZ[2:, 1:-1] - 2*ZZ[1:-1, 1:-1] + ZZ[:-2, 1:-1]) / dy**2
    
    laplacian = d2_dx2 + d2_dy2
    
    return laplacian.ravel()


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiment
    results = run_laplace_experiment(
        domain_size=1.0,
        n_train=8,
        noise_level=0.1,
        length_scale=2.0,
        n_grid=100
    )
    
    # Plot results
    print("\n4. Generating plots...")
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Plot PDE-aware kernel results
    fig = plot_laplace_results(
        results['X_test'], 
        results['xx'], 
        results['yy'],
        results['y_true'],
        results['y_pred_pde'],
        results['y_std_pde'],
        results['X_train'],
        save_path=os.path.join(outputs_dir, 'laplace_pde_kernel.png')
    )
    plt.close()
    
    # Plot SE kernel results
    fig = plot_laplace_results(
        results['X_test'], 
        results['xx'], 
        results['yy'],
        results['y_true'],
        results['y_pred_se'],
        results['y_std_se'],
        results['X_train'],
        save_path=os.path.join(outputs_dir, 'laplace_se_kernel.png')
    )
    plt.close()
    
    # Plot Laplacian violations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    from gp_pde_regression.utils import plot_2d_field
    cf = plot_2d_field(axes[0], results['xx'], results['yy'], 
                       results['laplacian_pde'],
                       title='∆u for PDE-aware kernel', 
                       cmap='RdBu_r',
                       training_points=results['X_train'])
    plt.colorbar(cf, ax=axes[0])
    
    cf = plot_2d_field(axes[1], results['xx'], results['yy'], 
                       results['laplacian_se'],
                       title='∆u for SE kernel', 
                       cmap='RdBu_r',
                       training_points=results['X_train'])
    plt.colorbar(cf, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'laplace_pde_violation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {outputs_dir}/")
    print("="*60)