import numpy as np
import jax.numpy as jnp

from lusee.Fitting import (
    linear_solve_cg,
    linear_solve_dense,
    linear_fisher,
    Param,
    ParamSet,
    ClPrior,
)


class IdentityBlock:
    n_theta = 2

    def theta_to_natural(self, theta):
        return theta

    def natural_to_theta(self, value):
        return jnp.asarray(value)

    def Sinv_from_cl(self, cl):
        return jnp.asarray(cl)


def test_dense_linear_solve_matches_cg_on_small_system():
    M = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]])
    theta_true = jnp.asarray([0.7, -1.2])
    data = M @ theta_true
    N_inv = jnp.asarray([2.0, 0.5, 1.5])
    S_inv = jnp.asarray([0.1, 0.2])

    def A(theta):
        return M @ theta

    dense = linear_solve_dense(A, 2, N_inv, S_inv, data)
    cg = linear_solve_cg(A, 2, N_inv, S_inv, data, tol=1e-12, maxiter=20)

    np.testing.assert_allclose(np.asarray(dense), np.asarray(cg), rtol=1e-10)


def test_linear_fisher_dense_matches_loop_on_small_system():
    M = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]])
    data = jnp.asarray([1.0, -0.5, 2.0])
    N_inv = jnp.asarray([2.0, 0.5, 1.5])
    paramset = ParamSet([
        Param("x", "linear", reparam=IdentityBlock(), prior=ClPrior([0.1, 0.2]))
    ])

    def predict(linear, nonlinear):
        del nonlinear
        return M @ linear["x"]

    dense = linear_fisher(predict, paramset, data, N_inv, {}, method="dense")
    loop = linear_fisher(predict, paramset, data, N_inv, {}, method="loop")

    np.testing.assert_allclose(dense["H"], loop["H"], rtol=1e-10)
    np.testing.assert_allclose(dense["cov"], loop["cov"], rtol=1e-10)
