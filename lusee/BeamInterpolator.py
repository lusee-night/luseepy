import numpy as np
import jax.numpy as jnp


class BeamInterpolator:
	"""RBF interpolation of beam ALM products over a parameter space.

	Cubic polyharmonic spline (r³) with linear polynomial augmentation —
	exact at training points, reproduces linear functions, no shape
	parameter to tune.

	Solves the RBF system at construction, then :meth:`interpolate` is
	pure JAX (JIT/grad-compatible).

	Each element of *beams* must support ``get_healpix_alm(lmax, freq_ndx=,
	return_I_stokes_only=, return_complex_components=False)`` returning an
	array or ``[I, Q, U, V]`` list (see :class:`lusee.Beam`).

	Example::

		interp = BeamInterpolator(beams, [0, 15, 30, 45], lmax=32)
		alm_at_20 = interp.interpolate(jnp.array([20.0]))

	:param beams: Training beams, one per parameter sample.
	:param parameters: Sample locations, shape ``(N,)`` or ``(N, M)``.
	:param lmax: Maximum spherical harmonic degree.
	:param freq_ndx: Frequency indices to precompute (None = all).
	:param return_I_stokes_only: If True, store only Stokes I (default).
	:param smoothing: Ridge regularization on the RBF block; 0 = exact.
	"""

	def __init__(self, beams, parameters, *, lmax, freq_ndx=None,
				 return_I_stokes_only=True, smoothing=0.0):
		beams = list(beams)
		if len(beams) == 0:
			raise ValueError("beams must contain at least one Beam object")

		params = np.asarray(parameters, dtype=float)
		if params.ndim == 1:
			params = params[:, None]
		if params.ndim != 2:
			raise ValueError("parameters must have shape (N_beams, M) or (N_beams,)")
		if params.shape[0] != len(beams):
			raise ValueError("number of parameter vectors must match number of beams")

		self.parameters = jnp.asarray(params)
		n, m = params.shape

		# Extract ALM data from beams
		beam_results = []
		for beam in beams:
			result = beam.get_healpix_alm(
				lmax, freq_ndx=freq_ndx,
				return_I_stokes_only=return_I_stokes_only,
				return_complex_components=False,
			)
			if return_I_stokes_only:
				result = [result]
			beam_results.append(result)

		n_stokes = 1 if return_I_stokes_only else 4
		data_list = [
			np.stack([r[si] for r in beam_results], axis=0)
			for si in range(n_stokes)
		]
		data = jnp.asarray(np.stack(data_list, axis=0))  # (n_stokes, n, ...)
		self._output_shape = (n_stokes,) + data.shape[2:]

		# Solve augmented RBF system: [Φ P; Pᵀ 0] [w; c] = [d; 0]
		diff = self.parameters[:, None, :] - self.parameters[None, :, :]
		Phi = jnp.sum(diff ** 2, axis=2) ** 1.5  # r³

		P = jnp.concatenate([jnp.ones((n, 1)), self.parameters], axis=1)

		A = jnp.zeros((n + m + 1, n + m + 1))
		A = A.at[:n, :n].set(Phi + smoothing * jnp.eye(n))
		A = A.at[:n, n:].set(P)
		A = A.at[n:, :n].set(P.T)

		D = int(np.prod(self._output_shape))
		data_flat = jnp.moveaxis(data, 1, 0).reshape(n, D)
		rhs = jnp.zeros((n + m + 1, D))
		rhs = rhs.at[:n].set(data_flat)

		self._coeffs = jnp.linalg.solve(A, rhs)

	def interpolate(self, parameters):
		"""Evaluate RBF interpolant at query point ``(M,)``.  Pure JAX, JIT/grad safe.

		:param parameters: Query point, shape ``(M,)``.
		:returns: ALM array, shape ``(n_stokes, *alm_shape)``.

		Example::

			loss = lambda q: jnp.sum(interp.interpolate(q) ** 2)
			grad = jax.grad(loss)(jnp.array([20.0]))
		"""
		diff = self.parameters - parameters[None, :]
		phi = jnp.sum(diff ** 2, axis=1) ** 1.5
		poly = jnp.concatenate([jnp.array([1.0]), parameters])
		basis = jnp.concatenate([phi, poly])
		return (basis @ self._coeffs).reshape(self._output_shape)
