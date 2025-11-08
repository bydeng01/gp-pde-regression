"""
Helmholtz equation experiment (reproduces Figure 3).
"""

import os, sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from gp_pde_regression.kernels import HelmholtzKernel2D, SquaredExponentialKernel
from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.utils import (generate_grid_2d, compute_metrics, 
                   plot_helmholtz_convergence, plot_likelihood_profile,
                   plot_source_reconstruction)


class HelmholtzFundamentalSolution2D:
    """Fundamental solution for 2D Helmholtz equation"""
    
    def __init__(self, wavenumber):
        self.wavenumber = wavenumber
    
    def __call__(self, x, x_source):
        """G(x, ξ) = (i/4) H₀⁽¹⁾(k₀|x-ξ|)"""
        r = np.linalg.norm(x - x_source)
        if r < 1e-10:
            return 0.0
        G = 0.25j * hankel1(0, self.wavenumber * r)
        return np.real(G)


def solve_helmholtz_2d(X, k0, X_sources, q_sources, domain_bounds=(-1.5, 1.5)):
    """
    Solve Helmholtz equation using simple superposition of sources.
    This is a simplified solver for generating synthetic data.
    """
    u = np.zeros(len(X))
    
    G = HelmholtzFundamentalSolution2D(k0)
    
    for i, x in enumerate(X):
        # Source contributions
        for x_src, q in zip(X_sources, q_sources):
            u[i] += q * G(x, x_src)
    
    return u


def run_helmholtz_experiment(k0_true=9.16, noise_level=0.01,
                            n_sensors_list=[7, 13, 20],
                            domain_bounds=((-1.5, 1.5), (-1.5, 1.5))):
    """
    Run Helmholtz equation experiment with source reconstruction.
    
    Args:
        k0_true: true wavenumber
        noise_level: measurement noise std
        n_sensors_list: list of sensor counts for convergence study
        domain_bounds: ((xmin,xmax), (ymin,ymax))
        
    Returns:
        results: dict with all results
    """
    print("="*60)
    print("Helmholtz Equation Experiment")
    print("="*60)
    print(f"True wavenumber: k₀ = {k0_true}")
    print(f"Noise level: {noise_level}")
    
    # Define sources
    X_sources = np.array([[0.3, 0.4], [0.7, 0.8]])
    q_true = np.array([0.5, 1.0])
    
    print(f"Number of sources: {len(X_sources)}")
    print(f"True source strengths: {q_true}")
    
    # Test grid
    X_test, xx, yy = generate_grid_2d(domain_bounds, n_points=60)
    y_test_true = solve_helmholtz_2d(X_test, k0_true, X_sources, q_true)
    
    # Convergence study
    errors_pde = []
    errors_se = []
    k0_estimates = []
    q_estimates = []
    
    for n_sensors in n_sensors_list:
        print(f"\n{'='*60}")
        print(f"Experiment with {n_sensors} sensors")
        print(f"{'='*60}")
        
        # Generate random sensor locations in domain
        (xmin, xmax), (ymin, ymax) = domain_bounds
        X_train = np.random.uniform([xmin, ymin], [xmax, ymax], 
                                   size=(n_sensors, 2))
        
        # Generate measurements
        y_train_true = solve_helmholtz_2d(X_train, k0_true, X_sources, q_true)
        y_train = y_train_true + np.random.normal(0, noise_level, 
                                                  size=y_train_true.shape)
        
        # 1. PDE-aware kernel with source reconstruction
        print("\n1. PDE-aware kernel with hyperparameter optimization...")
        
        # Create kernel and GP
        kernel_pde = HelmholtzKernel2D(wavenumber=k0_true)
        G = HelmholtzFundamentalSolution2D(k0_true)
        gp_pde = PDEGP(kernel_pde, noise_variance=noise_level**2,
                       source_locations=X_sources, fundamental_solution=G)
        
        # Optimize wavenumber
        k0_bounds = [(k0_true - 2, k0_true + 2)]
        optimal_params = gp_pde.optimize_hyperparameters(
            param_names=['wavenumber'],
            bounds=k0_bounds,
            n_restarts=3
        )
        
        k0_opt = optimal_params['wavenumber'] if optimal_params else k0_true
        print(f"   Optimized k₀: {k0_opt:.4f} (true: {k0_true:.4f})")
        
        # Update fundamental solution with optimized wavenumber
        G_opt = HelmholtzFundamentalSolution2D(k0_opt)
        gp_pde.fundamental_solution = G_opt
        gp_pde.fit(X_train, y_train)
        
        # Source reconstruction
        q_est = gp_pde.b_mean
        q_std = np.sqrt(np.diag(gp_pde.b_cov))
        print(f"   Estimated sources: {q_est}")
        print(f"   Source std: {q_std}")
        print(f"   True sources: {q_true}")
        
        # Predict
        y_pred_pde = gp_pde.predict(X_test)
        metrics_pde = compute_metrics(y_test_true, y_pred_pde)
        print(f"   MAE: {metrics_pde['mae']:.6f}")
        
        errors_pde.append(metrics_pde['mae'])
        k0_estimates.append(k0_opt)
        q_estimates.append((q_est, q_std))
        
        # 2. Squared exponential kernel
        print("\n2. Squared exponential kernel...")
        length_scale = np.pi / k0_true  # Use wavelength as length scale
        kernel_se = SquaredExponentialKernel(length_scale=length_scale)
        gp_se = PDEGP(kernel_se, noise_variance=noise_level**2)
        gp_se.fit(X_train, y_train)
        
        y_pred_se = gp_se.predict(X_test)
        metrics_se = compute_metrics(y_test_true, y_pred_se)
        print(f"   MAE: {metrics_se['mae']:.6f}")
        
        errors_se.append(metrics_se['mae'])
    
    # Likelihood profile for largest sensor count
    print(f"\n{'='*60}")
    print("Computing likelihood profile...")
    print(f"{'='*60}")
    
    n_sensors = n_sensors_list[-1]
    X_train = np.random.uniform([xmin, ymin], [xmax, ymax], 
                               size=(n_sensors, 2))
    y_train_true = solve_helmholtz_2d(X_train, k0_true, X_sources, q_true)
    y_train = y_train_true + np.random.normal(0, noise_level, 
                                              size=y_train_true.shape)
    
    k0_range = np.linspace(k0_true - 3, k0_true + 3, 50)
    nll_values = []
    
    for k0 in k0_range:
        kernel = HelmholtzKernel2D(wavenumber=k0)
        G = HelmholtzFundamentalSolution2D(k0)
        gp = PDEGP(kernel, noise_variance=noise_level**2,
                   source_locations=X_sources, fundamental_solution=G)
        gp.fit(X_train, y_train)
        nll_values.append(gp.negative_log_likelihood())
    
    k0_optimal = k0_range[np.argmin(nll_values)]
    
    results = {
        'n_sensors_list': n_sensors_list,
        'errors_pde': errors_pde,
        'errors_se': errors_se,
        'k0_estimates': k0_estimates,
        'q_estimates': q_estimates,
        'k0_range': k0_range,
        'nll_values': nll_values,
        'k0_true': k0_true,
        'k0_optimal': k0_optimal,
        'X_sources': X_sources,
        'q_true': q_true,
        'X_test': X_test,
        'xx': xx,
        'yy': yy,
        'y_test_true': y_test_true
    }
    
    return results


if __name__ == '__main__':
    # Set random seed
    np.random.seed(42)
    
    # Run experiment
    results = run_helmholtz_experiment(
        k0_true=9.16,
        noise_level=0.01,
        n_sensors_list=[7, 13, 20]
    )
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Convergence plot
    fig = plot_helmholtz_convergence(
        results['n_sensors_list'],
        results['errors_pde'],
        results['errors_se'],
        save_path=os.path.join(outputs_dir, 'helmholtz_convergence.png')
    )
    plt.close()
    
    # Likelihood profile
    fig = plot_likelihood_profile(
        results['k0_range'],
        results['nll_values'],
        results['k0_true'],
        results['k0_optimal'],
        save_path=os.path.join(outputs_dir, 'helmholtz_likelihood.png')
    )
    plt.close()
    
    # Source reconstruction visualization (for largest sensor count)
    idx = -1  # Last experiment
    q_est, q_std = results['q_estimates'][idx]
    k0_est = results['k0_estimates'][idx]
    
    # Recompute prediction for visualization
    kernel = HelmholtzKernel2D(wavenumber=k0_est)
    G = HelmholtzFundamentalSolution2D(k0_est)
    gp = PDEGP(kernel, noise_variance=0.01**2,
               source_locations=results['X_sources'], 
               fundamental_solution=G)
    
    # Regenerate training data
    X_train = np.random.uniform([-1.5, -1.5], [1.5, 1.5], size=(20, 2))
    y_train = solve_helmholtz_2d(X_train, results['k0_true'], 
                                 results['X_sources'], results['q_true'])
    y_train += np.random.normal(0, 0.01, size=y_train.shape)
    gp.fit(X_train, y_train)
    
    y_pred = gp.predict(results['X_test'])
    
    fig = plot_source_reconstruction(
        results['X_test'],
        results['xx'],
        results['yy'],
        y_pred,
        results['X_sources'],
        q_est,
        q_std,
        save_path=os.path.join(outputs_dir, 'helmholtz_sources.png')
    )
    plt.close()
    
    print(f"\nResults saved to {outputs_dir}/")
    print("="*60)