"""
Tests for the GP-PDE regression code.

These check the pieces that the paper actually relies on: the kernels are valid
covariances (symmetric, positive definite), their realizations satisfy the right
PDE, the GP equations (13-14) match a direct computation, and source strengths
(18-19) are recovered correctly. There are also edge-case and reproducibility
checks.

Runs under pytest, or directly with `python tests/test_gp_pde.py` if pytest is
not installed.
"""

import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gp_pde_regression.gp_pde import PDEGP
from gp_pde_regression.kernels import (
    LaplacePolarKernel, LaplacePolarKernelAlt, LaplaceLogKernel,
    LaplaceCartesianKernel, HelmholtzKernel2D, HelmholtzFundamentalSolution2D,
    HeatKernelNatural, HeatKernelSourceFree, SquaredExponentialKernel,
)
from gp_pde_regression.utils import (
    generate_grid_2d, generate_boundary_points, compute_metrics,
)


# --- helpers ----------------------------------------------------------------

def spatial_kernels():
    return [
        SquaredExponentialKernel(length_scale=0.8),
        LaplacePolarKernel(length_scale=2.0),
        LaplacePolarKernelAlt(length_scale=2.0),
        LaplaceLogKernel(length_scale=2.0),
        HelmholtzKernel2D(wavenumber=4.0),
    ]


def random_points(n, seed, lo=-1.0, hi=1.0, d=2):
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, d))


def min_eig(K):
    return np.linalg.eigvalsh(0.5 * (K + K.T)).min()


# --- kernel sanity ----------------------------------------------------------

def test_kernels_symmetric():
    X = random_points(12, 0)
    for k in spatial_kernels():
        K = k(X, X)
        assert np.allclose(K, K.T, atol=1e-10), type(k).__name__


def test_kernels_positive_semidefinite():
    X = random_points(15, 1)
    for k in spatial_kernels():
        assert min_eig(k(X, X)) > -1e-8, type(k).__name__

    # space-time kernels on their own coordinates
    XT = np.column_stack([random_points(15, 2, d=1).ravel(),
                          np.random.default_rng(3).uniform(0.05, 0.6, 15)])
    assert min_eig(HeatKernelNatural(diffusivity=0.2)(XT, XT)) > -1e-8
    assert min_eig(HeatKernelSourceFree(diffusivity=0.2,
                                        domain_bounds=(0.0, 1.0))(XT, XT)) > -1e-8


def test_helmholtz_kernel_is_bessel():
    # Guards against regressing to the singular Green's-function kernel, which
    # is not positive definite. The homogeneous kernel must be J0(k0 r).
    from scipy.special import j0
    k0 = 3.5
    ker = HelmholtzKernel2D(wavenumber=k0)
    X = random_points(8, 4)
    for x in X:
        for xp in X:
            r = np.linalg.norm(x - xp)
            expected = 1.0 + ker.hyperparams['regularization'] if r < 1e-12 else j0(k0 * r)
            assert abs(ker.kernel_function(x, xp) - expected) < 1e-12
    # bounded by ~1 and diagonal normalized to 1
    assert abs(ker.kernel_function(X[0], X[0]) - (1.0 + 1e-6)) < 1e-9
    assert np.all(np.abs(ker(X, X)) <= 1.0 + 1e-6 + 1e-9)


def test_laplace_polar_forms_agree():
    # Eq (25) and Eq (26) are claimed equal in the paper.
    X1, X2 = random_points(10, 5), random_points(10, 6)
    a = LaplacePolarKernel(length_scale=2.0)(X1, X2)
    b = LaplacePolarKernelAlt(length_scale=2.0)(X1, X2)
    assert np.allclose(a, b, atol=1e-12)


def test_laplace_log_kernel_two_forms():
    # k = 1 - 0.5 ln(1 - 2 x.x'/s^2 + |x|^2|x'|^2/s^4) == 1 - ln(|x-xbar'|/|xbar'|)
    s = 2.0
    ker = LaplaceLogKernel(length_scale=s)
    for x, xp in [(np.array([0.1, 0.2]), np.array([0.4, -0.3])),
                  (np.array([-0.5, 0.5]), np.array([0.2, 0.1]))]:
        closed = 1 - 0.5 * np.log(1 - 2 * np.dot(x, xp) / s**2
                                  + (np.dot(x, x) * np.dot(xp, xp)) / s**4)
        assert abs(ker.kernel_function(x, xp) - closed) < 1e-10


def test_laplace_kernel_normalized_at_origin():
    # Paper notes k is normalized to 1 at x = 0.
    ker = LaplacePolarKernel(length_scale=2.0)
    for xp in random_points(5, 7):
        assert abs(ker.kernel_function(np.zeros(2), xp) - 1.0) < 1e-10


def test_cartesian_kernel_valid_at_zero_rotation():
    # Eq (30) is a valid covariance at the default rotation_angle=0: symmetric,
    # PSD and harmonic. (For theta0 != 0 the written formula is not symmetric.)
    ker = LaplaceCartesianKernel(length_scale=2.0, rotation_angle=0.0)
    X = random_points(10, 30, lo=-0.5, hi=0.5)
    K = ker(X, X)
    assert np.allclose(K, K.T, atol=1e-10)
    assert min_eig(K) > -1e-8
    x0 = np.array([0.3, -0.2])
    f = lambda x: ker.kernel_function(x, x0)
    assert abs(laplacian_fd(f, np.array([0.1, 0.15]))) < 1e-3


def test_squared_exponential_values():
    ker = SquaredExponentialKernel(length_scale=1.5)
    assert abs(ker.kernel_function(np.zeros(2), np.zeros(2)) - 1.0) < 1e-12
    x, xp = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    assert abs(ker.kernel_function(x, xp) - np.exp(-0.5 / 1.5**2)) < 1e-12


# --- PDE satisfaction -------------------------------------------------------

def laplacian_fd(f, x, h=1e-3):
    e0 = np.array([h, 0.0])
    e1 = np.array([0.0, h])
    return ((f(x + e0) + f(x - e0) + f(x + e1) + f(x - e1) - 4 * f(x)) / h**2)


def test_laplace_kernel_is_harmonic():
    # k(., x0) satisfies Laplace's equation away from its (exterior) singularity.
    ker = LaplacePolarKernel(length_scale=2.0)
    x0 = np.array([0.6, -0.4])
    f = lambda x: ker.kernel_function(x, x0)
    for x in random_points(6, 8, lo=-0.5, hi=0.5):
        assert abs(laplacian_fd(f, x)) < 1e-3


def test_helmholtz_kernel_satisfies_helmholtz():
    # (Delta + k0^2) J0(k0 r) = 0.
    k0 = 3.0
    ker = HelmholtzKernel2D(wavenumber=k0)
    x0 = np.array([0.2, 0.1])
    f = lambda x: ker.kernel_function(x, x0)
    for x in random_points(6, 9, lo=-0.6, hi=0.6):
        if np.linalg.norm(x - x0) < 0.1:
            continue
        residual = laplacian_fd(f, x) + k0**2 * f(x)
        assert abs(residual) < 1e-2


def test_heat_natural_kernel_satisfies_heat():
    # (d/dt - D d^2/dx^2) k = 0 for the natural heat kernel in (x, t).
    D = 0.25
    ker = HeatKernelNatural(diffusivity=D)
    xt0 = np.array([0.1, 0.3])
    f = lambda x, t: ker.kernel_function(np.array([x, t]), xt0)
    h = 1e-4
    for x, t in [(0.0, 0.4), (0.3, 0.5), (-0.2, 0.25)]:
        dudt = (f(x, t + h) - f(x, t - h)) / (2 * h)
        d2udx2 = (f(x + h, t) - 2 * f(x, t) + f(x - h, t)) / h**2
        assert abs(dudt - D * d2udx2) < 1e-3


def test_posterior_mean_harmonic_for_laplace_kernel():
    # End-to-end: a GP with the Laplace kernel produces a harmonic posterior
    # mean, while the squared-exponential kernel does not.
    X = generate_boundary_points(16, 1.0)
    y = np.exp(X[:, 0]) * np.cos(X[:, 1])          # a harmonic field
    grid, xx, yy = generate_grid_2d(((-0.6, 0.6), (-0.6, 0.6)), 25)

    def mean_abs_laplacian(kernel):
        gp = PDEGP(kernel, noise_variance=1e-8)
        gp.fit(X, y)
        Z = gp.predict(grid).reshape(xx.shape)
        dx = xx[0, 1] - xx[0, 0]
        lap = (Z[1:-1, 2:] + Z[1:-1, :-2] + Z[2:, 1:-1] + Z[:-2, 1:-1]
               - 4 * Z[1:-1, 1:-1]) / dx**2
        return np.mean(np.abs(lap))

    assert mean_abs_laplacian(LaplacePolarKernel(2.0)) < 1e-2
    assert mean_abs_laplacian(SquaredExponentialKernel(0.5)) > 1e-1


def test_laplace_kernel_reconstructs_harmonic_field():
    # The kernel should recover an admissible (harmonic) field almost exactly
    # from boundary data.
    X = generate_boundary_points(24, 1.0)
    truth = lambda P: np.exp(P[:, 0]) * np.cos(P[:, 1])
    gp = PDEGP(LaplacePolarKernel(2.0), noise_variance=1e-10)
    gp.fit(X, truth(X))
    Xt = random_points(80, 11, lo=-0.85, hi=0.85)
    err = np.mean(np.abs(gp.predict(Xt) - truth(Xt)))
    assert err < 1e-2


# --- GP regression equations ------------------------------------------------

def test_gp_matches_direct_formula():
    X, y = random_points(10, 12), np.random.default_rng(13).standard_normal(10)
    Xt = random_points(7, 14)
    s2 = 1e-3
    ker = SquaredExponentialKernel(0.9)
    gp = PDEGP(ker, noise_variance=s2)
    gp.fit(X, y)
    mean, cov = gp.predict(Xt, return_cov=True)

    K = ker(X, X) + s2 * np.eye(10)
    Ks = ker(X, Xt)
    Kss = ker(Xt, Xt)
    Kinv = np.linalg.inv(K)
    mean_ref = Ks.T @ Kinv @ y
    cov_ref = Kss - Ks.T @ Kinv @ Ks
    assert np.allclose(mean, mean_ref, atol=1e-7)
    assert np.allclose(cov, cov_ref, atol=1e-7)


def test_gp_interpolates_training_data():
    X, y = random_points(8, 15), np.random.default_rng(16).standard_normal(8)
    gp = PDEGP(SquaredExponentialKernel(0.7), noise_variance=1e-10)
    gp.fit(X, y)
    mean, std = gp.predict(X, return_std=True)
    assert np.allclose(mean, y, atol=1e-4)
    assert np.all(std < 1e-3)


def test_predict_std_matches_cov_diagonal():
    X, y = random_points(9, 17), np.random.default_rng(18).standard_normal(9)
    gp = PDEGP(SquaredExponentialKernel(0.6), noise_variance=1e-3)
    gp.fit(X, y)
    Xt = random_points(6, 19)
    _, std = gp.predict(Xt, return_std=True)
    _, cov = gp.predict(Xt, return_cov=True)
    assert np.allclose(std, np.sqrt(np.diag(cov)), atol=1e-9)


def test_nll_matches_reference():
    X, y = random_points(10, 20), np.random.default_rng(21).standard_normal(10)
    s2 = 1e-2
    ker = SquaredExponentialKernel(0.8)
    gp = PDEGP(ker, noise_variance=s2)
    gp.fit(X, y)
    K = ker(X, X) + s2 * np.eye(10)
    sign, logdet = np.linalg.slogdet(K)
    ref = 0.5 * y @ np.linalg.solve(K, y) + 0.5 * logdet + 0.5 * 10 * np.log(2 * np.pi)
    assert abs(gp.negative_log_likelihood() - ref) < 1e-6


# --- source modelling (Eqs 18-19) ------------------------------------------

def test_source_strength_recovery():
    k0 = 9.16
    sources = np.array([[0.3, 0.4], [0.7, 0.8]])
    q_true = np.array([0.5, 1.0])
    G = HelmholtzFundamentalSolution2D(wavenumber=k0)

    X = random_points(60, 22, lo=-1.5, hi=1.5)
    y = np.zeros(len(X))
    for i, x in enumerate(X):
        for xs, q in zip(sources, q_true):
            y[i] += q * G(x, xs)

    gp = PDEGP(HelmholtzKernel2D(wavenumber=k0), noise_variance=1e-10,
               source_locations=sources, fundamental_solution=G)
    gp.fit(X, y)
    assert np.allclose(gp.b_mean, q_true, atol=1e-3)
    assert gp.b_cov.shape == (2, 2)
    assert np.all(np.isfinite(gp.b_cov))


def test_source_model_nll_finite():
    k0 = 6.0
    sources = np.array([[0.2, 0.1]])
    G = HelmholtzFundamentalSolution2D(wavenumber=k0)
    X = random_points(20, 23, lo=-1.0, hi=1.0)
    y = np.array([0.7 * G(x, sources[0]) for x in X]) + 1e-3
    gp = PDEGP(HelmholtzKernel2D(wavenumber=k0), noise_variance=1e-4,
               source_locations=sources, fundamental_solution=G)
    gp.fit(X, y)
    assert np.isfinite(gp.negative_log_likelihood())


# --- hyperparameter optimization -------------------------------------------

def test_optimize_requires_fit():
    gp = PDEGP(SquaredExponentialKernel(1.0))
    raised = False
    try:
        gp.optimize_hyperparameters(['length_scale'], [(0.1, 2.0)])
    except RuntimeError:
        raised = True
    assert raised


def test_optimize_reduces_nll_and_recovers_scale():
    rng = np.random.default_rng(24)
    X = rng.uniform(-3, 3, size=(40, 2))
    true_ls = 1.0
    ktrue = SquaredExponentialKernel(true_ls)
    K = ktrue(X, X) + 1e-6 * np.eye(40)
    y = np.linalg.cholesky(K) @ rng.standard_normal(40)   # sample from the GP

    gp = PDEGP(SquaredExponentialKernel(5.0), noise_variance=1e-6)
    gp.fit(X, y)
    nll_before = gp.negative_log_likelihood()
    out = gp.optimize_hyperparameters(['length_scale'], [(0.2, 5.0)],
                                      n_restarts=4, random_state=0)
    assert out is not None
    assert gp.negative_log_likelihood() <= nll_before + 1e-6
    assert 0.4 < out['length_scale'] < 2.5


def test_optimize_syncs_fundamental_solution():
    sources = np.array([[0.3, 0.4], [0.7, 0.8]])
    q_true = np.array([0.5, 1.0])
    G = HelmholtzFundamentalSolution2D(wavenumber=8.0)
    X = random_points(50, 25, lo=-1.5, hi=1.5)
    y = np.zeros(len(X))
    for i, x in enumerate(X):
        for xs, q in zip(sources, q_true):
            y[i] += q * HelmholtzFundamentalSolution2D(9.16)(x, xs)
    y += np.random.default_rng(26).normal(0, 0.01, len(y))

    gp = PDEGP(HelmholtzKernel2D(wavenumber=8.0), noise_variance=0.01**2,
               source_locations=sources, fundamental_solution=G)
    gp.fit(X, y)
    out = gp.optimize_hyperparameters(['wavenumber'], [(7.16, 11.16)],
                                      n_restarts=3, random_state=1)
    assert out is not None
    # kernel and Green's function must agree after optimization
    assert abs(gp.kernel.hyperparams['wavenumber'] - gp.fundamental_solution.wavenumber) < 1e-9


def test_optimize_reproducible():
    rng = np.random.default_rng(27)
    X = rng.uniform(-3, 3, size=(30, 2))
    y = rng.standard_normal(30)
    res = []
    for _ in range(2):
        gp = PDEGP(SquaredExponentialKernel(2.0), noise_variance=1e-4)
        gp.fit(X, y)
        res.append(gp.optimize_hyperparameters(['length_scale'], [(0.2, 5.0)],
                                               n_restarts=3, random_state=42))
    assert abs(res[0]['length_scale'] - res[1]['length_scale']) < 1e-9


# --- utilities and edge cases ----------------------------------------------

def test_boundary_points_distinct_and_on_boundary():
    for n in (4, 8, 16, 7):
        P = generate_boundary_points(n, 1.0)
        assert P.shape == (n, 2)
        # all on the square boundary
        assert np.allclose(np.max(np.abs(P), axis=1), 1.0)
        # all distinct
        assert len({tuple(np.round(p, 9)) for p in P}) == n


def test_grid_generation_shapes():
    grid, xx, yy = generate_grid_2d(((-1, 1), (-2, 2)), 10)
    assert grid.shape == (100, 2)
    assert xx.shape == (10, 10) and yy.shape == (10, 10)


def test_compute_metrics():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 4.0])
    m = compute_metrics(yt, yp)
    assert abs(m['mae'] - 1 / 3) < 1e-12
    assert abs(m['max_error'] - 1.0) < 1e-12
    assert abs(m['rmse'] - np.sqrt(1 / 3)) < 1e-12
    m2 = compute_metrics(yt, yp, y_std=np.array([1.0, 1.0, 1.0]))
    assert '95_coverage' in m2


def test_single_training_point():
    gp = PDEGP(SquaredExponentialKernel(1.0), noise_variance=1e-6)
    gp.fit(np.array([[0.0, 0.0]]), np.array([2.0]))
    mean, std = gp.predict(np.array([[0.0, 0.0], [3.0, 3.0]]), return_std=True)
    assert abs(mean[0] - 2.0) < 1e-2
    assert std[1] > std[0]            # more uncertain far from the data


def test_duplicate_points_do_not_crash():
    X = np.array([[0.1, 0.2], [0.1, 0.2], [0.5, 0.5]])
    y = np.array([1.0, 1.0, 2.0])
    gp = PDEGP(LaplacePolarKernel(2.0), noise_variance=1e-6)
    gp.fit(X, y)
    assert np.all(np.isfinite(gp.predict(X)))


def test_predict_before_fit_raises():
    gp = PDEGP(SquaredExponentialKernel(1.0))
    raised = False
    try:
        gp.predict(np.array([[0.0, 0.0]]))
    except RuntimeError:
        raised = True
    assert raised


# --- standalone runner (no pytest needed) ----------------------------------

if __name__ == '__main__':
    import traceback

    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith('test_') and callable(f))
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
