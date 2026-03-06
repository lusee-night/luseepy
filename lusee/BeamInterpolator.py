import numpy as np
import jax.numpy as jnp


class BeamInterpolator:
	"""Nadaraya-Watson interpolation of beam ALM products over a parameter space.

	Precomputes ALMs at construction (numpy), then :meth:`interpolate` is
	pure JAX (JIT/grad-compatible).  Stored data shape:
	``(n_stokes, n_beams, *alm_shape)``.
	"""

	def __init__(self, beams, parameters, kernel_width=1.0, *,
				 lmax, freq_ndx=None, return_I_stokes_only=True):
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
		self.n_beams, self.n_params = params.shape

		kw = np.asarray(kernel_width, dtype=float)
		if kw.ndim == 0:
			if float(kw) <= 0:
				raise ValueError("kernel_width must be positive")
			kw = np.full(self.n_params, float(kw), dtype=float)
		elif kw.shape != (self.n_params,):
			raise ValueError("kernel_width must be scalar or have shape (M,)")
		if np.any(kw <= 0):
			raise ValueError("all kernel_width entries must be positive")

		self.kernel_width = jnp.asarray(kw)

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
		self._data = jnp.asarray(np.stack(data_list, axis=0))

	def _weights(self, parameters):
		"""Gaussian softmax weights (log-sum-exp stabilised)."""
		dpar = (self.parameters - parameters[None, :]) / self.kernel_width[None, :]
		dist2 = jnp.sum(dpar * dpar, axis=1)
		logits = -0.5 * dist2
		logits = logits - jnp.max(logits)
		w = jnp.exp(logits)
		return w / jnp.sum(w)

	def interpolate(self, parameters):
		"""Weighted-average ALMs at query point ``(M,)``.  Pure JAX, JIT/grad safe."""
		w = self._weights(parameters)
		return jnp.tensordot(w, self._data, axes=([0], [1]))

