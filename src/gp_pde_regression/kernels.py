"""
PDE-aware kernel implementations for Gaussian Process regression.
Based on Albert (2019) - "Gaussian processes for data fulfilling linear differential equations"
"""

import numpy as np
from scipy.special import hankel1, jv, erf
from abc import ABC, abstractmethod


class PDEKernel(ABC):
    """Base class for PDE-aware kernels"""
    
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
    
    @abstractmethod
    def kernel_function(self, x, x_prime):
        """Compute kernel between two points"""
        pass
    
    def __call__(self, X1, X2):
        """
        Compute kernel matrix between X1 and X2
        
        Args:
            X1: (n1, d) array of points
            X2: (n2, d) array of points
            
        Returns:
            K: (n1, n2) kernel matrix
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_function(X1[i], X2[j])
        
        return K
    
    def update_hyperparams(self, **hyperparams):
        """Update hyperparameters"""
        self.hyperparams.update(hyperparams)


class LaplacePolarKernel(PDEKernel):
    """
    Kernel for 2D Laplace equation using polar coordinate separation.
    Equation (26) from paper.
    """
    
    def __init__(self, length_scale=1.0):
        """
        Args:
            length_scale: s parameter, must be > domain radius
        """
        super().__init__(length_scale=length_scale)
    
    def kernel_function(self, x, x_prime):
        """
        Compute kernel k(x, x') with dipole singularity outside domain
        
        k(x,x';s) = (|x̄'|² - x·x̄') / |x - x̄'|²
        where x̄' is mirror point with r̄' = s²/r'
        """
        s = self.hyperparams['length_scale']
        
        # Compute radius and angle for x_prime
        r_prime = np.linalg.norm(x_prime)
        
        if r_prime < 1e-10:
            # Handle singularity at origin
            return 1.0
        
        theta_prime = np.arctan2(x_prime[1], x_prime[0])
        
        # Mirror point
        r_bar_prime = s**2 / r_prime
        x_bar_prime = r_bar_prime * np.array([np.cos(theta_prime), 
                                               np.sin(theta_prime)])
        
        # Kernel formula
        numerator = np.linalg.norm(x_bar_prime)**2 - np.dot(x, x_bar_prime)
        denominator = np.linalg.norm(x - x_bar_prime)**2
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator


class LaplacePolarKernelAlt(PDEKernel):
    """
    Alternative formulation of Laplace kernel (Equation 25)
    """
    
    def __init__(self, length_scale=1.0):
        super().__init__(length_scale=length_scale)
    
    def kernel_function(self, x, x_prime):
        """
        k(x,x';s) = (1 - x·x'/s²) / (1 - 2x·x'/s² + |x|²|x'|²/s⁴)
        """
        s = self.hyperparams['length_scale']
        
        dot_prod = np.dot(x, x_prime)
        norm_x = np.linalg.norm(x)
        norm_x_prime = np.linalg.norm(x_prime)
        
        numerator = 1 - dot_prod / s**2
        denominator = 1 - 2*dot_prod/s**2 + (norm_x * norm_x_prime)**2 / s**4
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator


class LaplaceLogKernel(PDEKernel):
    """
    Logarithmic kernel for 2D Laplace equation (Equation 27)
    Has monopole singularity
    """
    
    def __init__(self, length_scale=1.0):
        super().__init__(length_scale=length_scale)
    
    def kernel_function(self, x, x_prime):
        """
        k(x,x';s) = 1 - ln(|x-x̄'| / |x̄'|)
        """
        s = self.hyperparams['length_scale']
        
        r_prime = np.linalg.norm(x_prime)
        if r_prime < 1e-10:
            return 1.0
        
        theta_prime = np.arctan2(x_prime[1], x_prime[0])
        r_bar_prime = s**2 / r_prime
        x_bar_prime = r_bar_prime * np.array([np.cos(theta_prime), 
                                               np.sin(theta_prime)])
        
        dist = np.linalg.norm(x - x_bar_prime)
        if dist < 1e-10:
            return 1.0
        
        return 1 - np.log(dist / np.linalg.norm(x_bar_prime))


class LaplaceCartesianKernel(PDEKernel):
    """
    Kernel for 2D Laplace equation using Cartesian separation (Equation 30)
    """
    
    def __init__(self, length_scale=1.0, rotation_angle=0.0):
        super().__init__(length_scale=length_scale, rotation_angle=rotation_angle)
    
    def kernel_function(self, x, x_prime):
        """
        k(x,x';s,θ₀) = 0.5 * Re[exp(((x+x')±i(y-y'))²e^{i2θ₀}/s²)]
        """
        s = self.hyperparams['length_scale']
        theta_0 = self.hyperparams['rotation_angle']
        
        x1, y1 = x[0], x[1]
        x2, y2 = x_prime[0], x_prime[1]
        
        exponent = ((x1 + x2) + 1j*(y1 - y2))**2 * np.exp(1j*2*theta_0) / s**2
        result = 0.5 * np.real(np.exp(exponent))
        
        return result


class HelmholtzKernel2D(PDEKernel):
    """
    Stationary kernel for 2D Helmholtz equation based on Bessel/Hankel functions.
    Simplified implementation - full version in Albert 2019 [11].
    """
    
    def __init__(self, wavenumber=1.0, regularization=1e-6):
        super().__init__(wavenumber=wavenumber, regularization=regularization)
    
    def kernel_function(self, x, x_prime):
        """
        Kernel based on Hankel function of first kind
        k(r) ∝ H₀⁽¹⁾(k₀ r) for r = |x - x'|
        """
        k0 = self.hyperparams['wavenumber']
        reg = self.hyperparams['regularization']
        
        r = np.linalg.norm(x - x_prime)
        
        if r < 1e-10:
            # Regularization at singularity
            return reg
        
        # Hankel function of first kind, order 0
        kernel_value = 0.25j * hankel1(0, k0 * r)
        
        # Return real part (imaginary part vanishes for symmetric problems)
        return np.real(kernel_value)


class HelmholtzFundamentalSolution2D:
    """
    Fundamental solution (Green's function) for 2D Helmholtz equation.
    Used for source modeling.
    """
    
    def __init__(self, wavenumber=1.0):
        self.wavenumber = wavenumber
    
    def __call__(self, x, x_source):
        """
        G(x, ξ) = (i/4) H₀⁽¹⁾(k₀|x-ξ|)
        """
        r = np.linalg.norm(x - x_source)
        
        if r < 1e-10:
            return 0.0
        
        G = 0.25j * hankel1(0, self.wavenumber * r)
        return np.real(G)


class HeatKernelNatural(PDEKernel):
    """
    Natural kernel for heat/diffusion equation (Equation 34)
    Stationary in space, non-stationary in time
    """
    
    def __init__(self, diffusivity=1.0):
        super().__init__(diffusivity=diffusivity)
    
    def kernel_function(self, xt, xt_prime):
        """
        k((x,t), (x',t')) = 1/√(4πD(t+t')) exp(-(x-x')²/(4D(t+t')))
        
        Args:
            xt: array [x, t] (can be [x, y, t] for 2D space)
            xt_prime: array [x', t']
        """
        D = self.hyperparams['diffusivity']
        
        # Separate spatial and temporal coordinates
        x = xt[:-1]  # All but last coordinate
        t = xt[-1]   # Last coordinate is time
        x_prime = xt_prime[:-1]
        t_prime = xt_prime[-1]
        
        if t < 0 or t_prime < 0:
            return 0.0  # Only valid for positive times
        
        time_sum = t + t_prime
        if time_sum < 1e-10:
            return 0.0
        
        spatial_diff = x - x_prime
        
        prefactor = 1.0 / np.sqrt(4 * np.pi * D * time_sum)
        exponent = -np.dot(spatial_diff, spatial_diff) / (4 * D * time_sum)
        
        return prefactor * np.exp(exponent)


class HeatKernelSourceFree(PDEKernel):
    """
    Kernel for heat equation in source-free domain (a, b) (Equations 35-36)
    """
    
    def __init__(self, diffusivity=1.0, domain_bounds=(0.0, 1.0)):
        super().__init__(diffusivity=diffusivity, domain_bounds=domain_bounds)
    
    def _erf_term(self, x, t, x_prime, t_prime, s):
        """Helper function g(x,t,x',t';D,s) from Equation 36"""
        D = self.hyperparams['diffusivity']
        
        if t < 1e-10 or t_prime < 1e-10:
            return 0.0
        
        numerator = (s - x)/t + (s - x_prime)/t_prime
        denominator = 2 * np.sqrt(D) * np.sqrt(1/t + 1/t_prime)
        
        return erf(numerator / denominator)
    
    def kernel_function(self, xt, xt_prime):
        """
        k_n((x,t), (x',t')) = k_n(x-x', t+t'; D) [1 - (g(b) - g(a))/2]
        """
        D = self.hyperparams['diffusivity']
        a, b = self.hyperparams['domain_bounds']
        
        # Separate coordinates
        x = xt[0] if len(xt) == 2 else xt[:-1]
        t = xt[-1]
        x_prime = xt_prime[0] if len(xt_prime) == 2 else xt_prime[:-1]
        t_prime = xt_prime[-1]
        
        # Natural kernel
        natural_kernel = HeatKernelNatural(diffusivity=D)
        k_natural = natural_kernel.kernel_function(xt, xt_prime)
        
        # Correction terms
        g_b = self._erf_term(x, t, x_prime, t_prime, b)
        g_a = self._erf_term(x, t, x_prime, t_prime, a)
        
        correction = 1 - (g_b - g_a) / 2
        
        return k_natural * correction


class SquaredExponentialKernel(PDEKernel):
    """
    Standard squared exponential (RBF) kernel for comparison
    """
    
    def __init__(self, length_scale=1.0):
        super().__init__(length_scale=length_scale)
    
    def kernel_function(self, x, x_prime):
        """
        k(x,x') = exp(-|x-x'|²/(2ℓ²))
        """
        length_scale = self.hyperparams['length_scale']
        
        diff = x - x_prime
        distance_sq = np.dot(diff, diff)
        
        return np.exp(-distance_sq / (2 * length_scale**2))