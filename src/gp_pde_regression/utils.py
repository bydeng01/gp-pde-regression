"""
Utility functions for experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_grid_2d(bounds, n_points):
    """
    Generate 2D grid for evaluation.
    
    Args:
        bounds: ((xmin, xmax), (ymin, ymax))
        n_points: number of points per dimension
        
    Returns:
        X_grid: (n_points**2, 2) array of grid points
        xx, yy: meshgrid arrays for plotting
    """
    (xmin, xmax), (ymin, ymax) = bounds
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)
    xx, yy = np.meshgrid(x, y)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    return X_grid, xx, yy


def generate_boundary_points(n_points, domain_size=1.0):
    """
    Generate points on boundary of square domain.
    
    Args:
        n_points: number of boundary points
        domain_size: half-width of square domain
        
    Returns:
        X_boundary: (n_points, 2) array
    """
    # Distribute points on 4 edges
    n_per_edge = n_points // 4
    
    points = []
    
    # Bottom edge
    x = np.linspace(-domain_size, domain_size, n_per_edge)
    y = -domain_size * np.ones(n_per_edge)
    points.append(np.column_stack([x, y]))
    
    # Right edge
    x = domain_size * np.ones(n_per_edge)
    y = np.linspace(-domain_size, domain_size, n_per_edge)
    points.append(np.column_stack([x, y]))
    
    # Top edge
    x = np.linspace(domain_size, -domain_size, n_per_edge)
    y = domain_size * np.ones(n_per_edge)
    points.append(np.column_stack([x, y]))
    
    # Left edge
    x = -domain_size * np.ones(n_points - 3*n_per_edge)
    y = np.linspace(domain_size, -domain_size, n_points - 3*n_per_edge)
    points.append(np.column_stack([x, y]))
    
    return np.vstack(points)


def compute_metrics(y_true, y_pred, y_std=None):
    """
    Compute evaluation metrics.
    
    Returns:
        metrics: dict with MAE, max error, relative error, coverage
    """
    abs_error = np.abs(y_true - y_pred)
    
    metrics = {
        'mae': np.mean(abs_error),
        'max_error': np.max(abs_error),
        'rmse': np.sqrt(np.mean(abs_error**2)),
        'relative_error': np.mean(abs_error / (np.abs(y_true) + 1e-10))
    }
    
    if y_std is not None:
        # 95% confidence interval coverage
        within_95 = np.mean(abs_error < 1.96 * y_std)
        metrics['95_coverage'] = within_95
    
    return metrics


def plot_2d_field(ax, xx, yy, values, title='', cmap='viridis', 
                  contour_levels=20, training_points=None):
    """
    Plot 2D scalar field with contours.
    """
    # Reshape values to grid
    ZZ = values.reshape(xx.shape)
    
    # Contour plot
    contour = ax.contour(xx, yy, ZZ, levels=contour_levels, 
                        colors='black', linewidths=0.5, alpha=0.6)
    contourf = ax.contourf(xx, yy, ZZ, levels=contour_levels, cmap=cmap)
    
    # Training points
    if training_points is not None:
        ax.plot(training_points[:, 0], training_points[:, 1], 
               'ko', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    return contourf


def plot_laplace_results(X_grid, xx, yy, y_true, y_pred, y_std, 
                        X_train, save_path=None):
    """
    Create 4-panel plot for Laplace equation results (Figure 1 style).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # True solution
    ax = axes[0, 0]
    cf = plot_2d_field(ax, xx, yy, y_true, 
                       title='Analytical Solution', 
                       training_points=X_train)
    plt.colorbar(cf, ax=ax)
    
    # GP reconstruction
    ax = axes[0, 1]
    cf = plot_2d_field(ax, xx, yy, y_pred, 
                       title='GP Reconstruction', 
                       training_points=X_train)
    plt.colorbar(cf, ax=ax)
    
    # Absolute error
    ax = axes[1, 0]
    abs_error = np.abs(y_true - y_pred)
    cf = plot_2d_field(ax, xx, yy, abs_error, 
                       title='Absolute Error',
                       cmap='Reds',
                       training_points=X_train)
    plt.colorbar(cf, ax=ax)
    
    # 95% confidence interval
    ax = axes[1, 1]
    confidence = 1.96 * y_std
    cf = plot_2d_field(ax, xx, yy, confidence, 
                       title='95% Confidence Interval',
                       cmap='Blues',
                       training_points=X_train)
    plt.colorbar(cf, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_helmholtz_convergence(sensor_counts, errors_pde, errors_se, 
                               save_path=None):
    """
    Plot reconstruction error vs number of sensors (Figure 3 style).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.semilogy(sensor_counts, errors_pde, 'o-', label='PDE-aware kernel', 
               linewidth=2, markersize=8)
    ax.semilogy(sensor_counts, errors_se, 's--', label='Squared exponential', 
               linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Reconstruction error (MAE)', fontsize=12)
    ax.set_title('Helmholtz Equation: Convergence Study', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_likelihood_profile(k0_values, nll_values, k0_true, k0_opt, 
                           save_path=None):
    """
    Plot negative log likelihood vs wavenumber (Figure 3, bottom right).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(k0_values, nll_values, 'b-', linewidth=2, label='PDE kernel')
    ax.axvline(k0_true, color='k', linestyle=':', linewidth=2, 
              label=f'True k₀ = {k0_true:.2f}')
    ax.axvline(k0_opt, color='r', linestyle='--', linewidth=2, 
              label=f'Optimal k₀ = {k0_opt:.2f}')
    
    ax.set_xlabel('Wavenumber k₀', fontsize=12)
    ax.set_ylabel('-log p(y|k₀)', fontsize=12)
    ax.set_title('Marginal Likelihood Profile', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_source_reconstruction(X_grid, xx, yy, y_pred, X_sources, 
                               q_mean, q_std, save_path=None):
    """
    Plot field with source locations and strengths.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Field
    ZZ = y_pred.reshape(xx.shape)
    contourf = ax.contourf(xx, yy, ZZ, levels=20, cmap='RdBu_r')
    ax.contour(xx, yy, ZZ, levels=20, colors='black', 
              linewidths=0.5, alpha=0.4)
    
    # Sources
    for i, (x_src, q_m, q_s) in enumerate(zip(X_sources, q_mean, q_std)):
        ax.plot(x_src[0], x_src[1], 'w*', markersize=20, 
               markeredgecolor='black', markeredgewidth=2)
        ax.text(x_src[0], x_src[1]-0.15, 
               f'q = {q_m:.2f} ± {q_s:.2f}',
               ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(contourf, ax=ax, label='Field value')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Reconstructed Field with Sources')
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig