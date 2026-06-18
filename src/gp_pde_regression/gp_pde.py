"""
Gaussian Process regression for PDE-constrained data.
Based on Albert (2019).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular, cho_solve


class PDEGP:
    """
    Gaussian Process for data satisfying linear PDEs.
    Implements equations (13-14) and (18-19) from the paper.
    """
    
    def __init__(self, kernel, noise_variance=1e-4, 
                 source_locations=None, fundamental_solution=None):
        """
        Args:
            kernel: PDEKernel instance for homogeneous equation
            noise_variance: measurement noise variance σ²ₙ
            source_locations: (n_sources, d) array of source positions
            fundamental_solution: callable G(x, ξ) for source modeling
        """
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.source_locations = source_locations
        self.fundamental_solution = fundamental_solution
        
        # Training data
        self.X_train = None
        self.y_train = None
        
        # Cached matrices
        self.K_y_inv = None
        self.L = None  # Cholesky factor
        self.alpha = None
        
        # Source parameters
        self.H = None
        self.b_mean = None
        self.b_cov = None
    
    def fit(self, X_train, y_train):
        """
        Fit GP to training data.
        
        Args:
            X_train: (n, d) array of training points
            y_train: (n,) array of training values
        """
        self.X_train = X_train
        self.y_train = y_train
        n = len(y_train)
        
        # Compute kernel matrix for training data
        K = self.kernel(X_train, X_train)
        K_y = K + self.noise_variance * np.eye(n)
        
        # Enforce symmetry and dtype for numerical stability
        K_y = np.asarray(0.5 * (K_y + K_y.T), dtype=np.float64)
        
        # Robust Cholesky with adaptive jitter
        jitter = 0.0
        for _ in range(7):
            try:
                self.L = cholesky(K_y + jitter * np.eye(n), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-8 if jitter == 0.0 else jitter * 10
        else:
            # As a last resort, shift by the most negative eigenvalue
            eigvals = np.linalg.eigvalsh(K_y)
            min_eig = eigvals.min()
            if min_eig < 0:
                K_y = K_y + (-min_eig + 1e-12) * np.eye(n)
            K_y = 0.5 * (K_y + K_y.T)
            self.L = cholesky(K_y, lower=True)
        
        # If sources are present, compute source strengths (Eq 18-19)
        if self.source_locations is not None and self.fundamental_solution is not None:
            self.H = self._build_source_matrix(X_train, self.source_locations)
            
            # Solve K_y^{-1} y
            alpha_temp = solve_triangular(self.L, y_train, lower=True)
            K_y_inv_y = solve_triangular(self.L.T, alpha_temp, lower=False)
            
            # Solve K_y^{-1} H^T
            K_y_inv_HT = cho_solve((self.L, True), self.H.T)
            
            # b̄ = (H K_y^{-1} H^T)^{-1} H K_y^{-1} y
            HKH = self.H @ K_y_inv_HT
            HKH = np.asarray(0.5 * (HKH + HKH.T), dtype=np.float64)
            try:
                L_HKH = cholesky(HKH, lower=True)
                temp = solve_triangular(L_HKH, self.H @ K_y_inv_y, lower=True)
                self.b_mean = solve_triangular(L_HKH.T, temp, lower=False)
                
                # cov(q, q) = (H K_y^{-1} H^T)^{-1}
                identity = np.eye(HKH.shape[0])
                temp_cov = solve_triangular(L_HKH, identity, lower=True)
                self.b_cov = solve_triangular(L_HKH.T, temp_cov, lower=False)
            except np.linalg.LinAlgError:
                # Fallback to direct inversion
                self.b_mean = np.linalg.solve(HKH, self.H @ K_y_inv_y)
                self.b_cov = np.linalg.inv(HKH)
        else:
            # Standard GP without sources
            self.alpha = solve_triangular(self.L, y_train, lower=True)
            self.alpha = solve_triangular(self.L.T, self.alpha, lower=False)
    
    def predict(self, X_test, return_std=False, return_cov=False):
        """
        Predict at test points using equations (13-14).
        
        Args:
            X_test: (n_test, d) array of test points
            return_std: whether to return standard deviation
            return_cov: whether to return full covariance
            
        Returns:
            mean: (n_test,) predicted mean
            std: (n_test,) predicted std (if return_std=True)
            cov: (n_test, n_test) predicted covariance (if return_cov=True)
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before predict().")

        K_star = self.kernel(self.X_train, X_test)
        
        if self.source_locations is not None and self.H is not None:
            # Prediction with sources (Eq 13)
            H_star = self._build_source_matrix(X_test, self.source_locations)
            
            # Solve K_y^{-1} (y - H^T b̄)
            residual = self.y_train - self.H.T @ self.b_mean
            temp = solve_triangular(self.L, residual, lower=True)
            K_y_inv_residual = solve_triangular(self.L.T, temp, lower=False)
            
            mean = K_star.T @ K_y_inv_residual + H_star.T @ self.b_mean
            
            if return_std or return_cov:
                K_star_star = self.kernel(X_test, X_test)
                
                # K_*^T K_y^{-1} K_*
                temp = solve_triangular(self.L, K_star, lower=True)
                K_inv_term = temp.T @ temp
                
                # R = H_* - H K_y^{-1} K_*
                K_y_inv_K_star = cho_solve((self.L, True), K_star)
                R = H_star - self.H @ K_y_inv_K_star
                
                # Covariance (Eq 14)
                cov = K_star_star - K_inv_term + R.T @ self.b_cov @ R
                
                if return_cov:
                    return mean, cov
                else:
                    std = np.sqrt(np.maximum(np.diag(cov), 0))
                    return mean, std
        else:
            # Standard GP prediction
            mean = K_star.T @ self.alpha
            
            if return_cov:
                # Full covariance requested: compute K_*_* explicitly
                K_star_star = self.kernel(X_test, X_test)
                temp = solve_triangular(self.L, K_star, lower=True)
                cov = K_star_star - temp.T @ temp
                return mean, cov
            elif return_std:
                # Only standard deviation needed: use diagonal computation
                temp = solve_triangular(self.L, K_star, lower=True)
                prior_var = np.array([self.kernel.kernel_function(x, x) for x in X_test])
                var = prior_var - np.sum(temp**2, axis=0)
                std = np.sqrt(np.maximum(var, 0))
                return mean, std
        
        return mean
    
    def _build_source_matrix(self, X_obs, X_sources):
        """
        Build H matrix using fundamental solutions.
        H_ij = G(x_i, ξ_j) where G is the fundamental solution.
        """
        n_obs = X_obs.shape[0]
        n_sources = X_sources.shape[0]
        H = np.zeros((n_sources, n_obs))
        
        for i in range(n_obs):
            for j in range(n_sources):
                H[j, i] = self.fundamental_solution(X_obs[i], X_sources[j])
        
        return H
    
    def negative_log_likelihood(self):
        """
        Compute negative log marginal likelihood for hyperparameter optimization.
        """
        n = len(self.y_train)
        
        if self.source_locations is not None and self.H is not None:
            # With sources
            residual = self.y_train - self.H.T @ self.b_mean
            temp = solve_triangular(self.L, residual, lower=True)
            data_fit = 0.5 * np.sum(temp**2)
            
            complexity = np.sum(np.log(np.diag(self.L)))
            
            # Add source term. With a vague prior on the m source coefficients
            # the marginal likelihood picks up a 0.5*log|H K_y^-1 H^T| Occam
            # factor and the Gaussian constant runs over n - m dimensions.
            m = self.H.shape[0]
            HKH = self.H @ cho_solve((self.L, True), self.H.T)
            HKH = np.asarray(0.5 * (HKH + HKH.T), dtype=np.float64)
            try:
                L_HKH = cholesky(HKH, lower=True)
                complexity += np.sum(np.log(np.diag(L_HKH)))
            except np.linalg.LinAlgError:
                complexity += 0.5 * np.log(np.linalg.det(HKH + 1e-8*np.eye(m)))

            nll = data_fit + complexity + 0.5 * (n - m) * np.log(2 * np.pi)
        else:
            # Standard GP likelihood
            temp = solve_triangular(self.L, self.y_train, lower=True)
            data_fit = 0.5 * np.sum(temp**2)
            complexity = np.sum(np.log(np.diag(self.L)))
            nll = data_fit + complexity + 0.5 * n * np.log(2 * np.pi)
        
        return nll
    
    def optimize_hyperparameters(self, param_names, bounds, n_restarts=3,
                                 random_state=None):
        """
        Optimize kernel hyperparameters by maximizing the marginal likelihood.

        Args:
            param_names: list of hyperparameter names to optimize
            bounds: list of (min, max) tuples for each parameter
            n_restarts: number of random restarts
            random_state: seed for the restart draws (for reproducibility)

        Returns:
            optimized_params: dict of optimized parameters, or None if every
            restart failed
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before optimize_hyperparameters().")

        rng = np.random.default_rng(random_state)

        def apply_params(param_dict):
            self.kernel.update_hyperparams(**param_dict)
            # Keep the fundamental solution in sync: a parameter such as the
            # wavenumber also lives in the Green's function, so the source
            # matrix H has to follow the kernel.
            if self.fundamental_solution is not None:
                for name, val in param_dict.items():
                    if hasattr(self.fundamental_solution, name):
                        setattr(self.fundamental_solution, name, val)

        def objective(params):
            apply_params(dict(zip(param_names, params)))
            self.fit(self.X_train, self.y_train)
            nll = self.negative_log_likelihood()
            # A bad parameter region can make the kernel ill-conditioned; steer
            # the optimizer away rather than letting NaNs poison the search.
            return nll if np.isfinite(nll) else np.inf

        best_params = None
        best_nll = np.inf

        for _ in range(n_restarts):
            x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
            try:
                result = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(result.fun) and result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        if best_params is None:
            return None

        param_dict = dict(zip(param_names, best_params))
        apply_params(param_dict)
        self.fit(self.X_train, self.y_train)
        return param_dict