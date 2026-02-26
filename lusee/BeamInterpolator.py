import numpy as np
import jax.numpy as jnp


class BeamInterpolator:
	"""
	Interpolator for beam spherical-harmonic products on irregular parameter samples.

	This class stores a list of :class:`lusee.Beam` objects and an associated set of
	M-dimensional parameter vectors. Harmonic beam products are precomputed on demand
	(and cached), then interpolated using a smooth kernel so that the interpolation
	itself is differentiable with respect to query parameters in JAX.

	:param beams: Training beams.
	:type beams: sequence
	:param parameters: Parameter samples associated one-to-one with ``beams``.
		Shape ``(N_beams, M)`` or ``(N_beams,)`` for 1D parameters.
	:type parameters: array-like
	:param kernel_width: Positive kernel width for interpolation. Either scalar or
		vector of shape ``(M,)``.
	:type kernel_width: float or array-like
	"""

	def __init__(self, beams, parameters, kernel_width=1.0):
		self.beams = list(beams)
		if len(self.beams) == 0:
			raise ValueError("beams must contain at least one Beam object")

		params = np.asarray(parameters, dtype=float)
		if params.ndim == 1:
			params = params[:, None]
		if params.ndim != 2:
			raise ValueError("parameters must have shape (N_beams, M) or (N_beams,)")
		if params.shape[0] != len(self.beams):
			raise ValueError("number of parameter vectors must match number of beams")

		self.parameters_np = params
		self.parameters = jnp.asarray(params)
		self.n_beams, self.n_params = params.shape

		kernel_width_np = np.asarray(kernel_width, dtype=float)
		if kernel_width_np.ndim == 0:
			if float(kernel_width_np) <= 0:
				raise ValueError("kernel_width must be positive")
			kernel_width_np = np.full(self.n_params, float(kernel_width_np), dtype=float)
		elif kernel_width_np.shape != (self.n_params,):
			raise ValueError("kernel_width must be scalar or have shape (M,)")
		if np.any(kernel_width_np <= 0):
			raise ValueError("all kernel_width entries must be positive")

		self.kernel_width_np = kernel_width_np
		self.kernel_width = jnp.asarray(kernel_width_np)

		self._alm_cache = {}
		self._cross_cache = {}

	@staticmethod
	def _normalize_freq_key(freq_ndx):
		if freq_ndx is None:
			return None
		if np.isscalar(freq_ndx):
			return ("scalar", int(freq_ndx))
		return tuple(int(fi) for fi in np.atleast_1d(freq_ndx))

	def _cache_key(self, lmax, freq_ndx, return_I_stokes_only, return_complex_components):
		return (
			int(lmax),
			self._normalize_freq_key(freq_ndx),
			bool(return_I_stokes_only),
			bool(return_complex_components),
		)

	def _query_to_jax(self, parameters):
		q = jnp.asarray(parameters, dtype=self.parameters.dtype)
		if q.ndim != 1:
			raise ValueError("query parameters must be a 1D vector of length M")
		if q.shape[0] != self.n_params:
			raise ValueError(
				f"query vector length must be {self.n_params}, got {q.shape[0]}"
			)
		return q

	def _weights(self, parameters):
		q = self._query_to_jax(parameters)
		dpar = (self.parameters - q[None, :]) / self.kernel_width[None, :]
		dist2 = jnp.sum(dpar * dpar, axis=1)
		logits = -0.5 * dist2
		logits = logits - jnp.max(logits)
		w = jnp.exp(logits)
		return w / jnp.sum(w)

	def _pack_cache(self, beam_results, return_I_stokes_only, return_complex_components):
		if return_I_stokes_only:
			stokes_count = 1
			extractor = lambda result, idx: result
		else:
			stokes_count = 4
			extractor = lambda result, idx: result[idx]

		packed = []
		for stokes_i in range(stokes_count):
			if return_complex_components:
				entries = [extractor(result, stokes_i) for result in beam_results]
				real_stack = jnp.asarray(np.stack([entry[0] for entry in entries], axis=0))
				if entries[0][1] is None:
					imag_stack = None
				else:
					imag_stack = jnp.asarray(np.stack([entry[1] for entry in entries], axis=0))
				packed.append((real_stack, imag_stack))
			else:
				entries = [extractor(result, stokes_i) for result in beam_results]
				packed.append(jnp.asarray(np.stack(entries, axis=0)))
		return packed

	def _get_or_build_cache(
		self,
		lmax,
		freq_ndx,
		other,
		return_I_stokes_only,
		return_complex_components,
	):
		key = self._cache_key(lmax, freq_ndx, return_I_stokes_only, return_complex_components)
		if other is None:
			cache = self._alm_cache
			if key not in cache:
				beam_results = [
					beam.get_healpix_alm(
						lmax,
						freq_ndx=freq_ndx,
						other=None,
						return_I_stokes_only=return_I_stokes_only,
						return_complex_components=return_complex_components,
					)
					for beam in self.beams
				]
				cache[key] = self._pack_cache(
					beam_results,
					return_I_stokes_only=return_I_stokes_only,
					return_complex_components=return_complex_components,
				)
			return cache[key]

		if not isinstance(other, BeamInterpolator):
			raise TypeError("other must be a BeamInterpolator instance or None")

		cross_key = (id(other), key)
		if cross_key not in self._cross_cache:
			if self.n_beams != other.n_beams:
				raise ValueError(
					"self and other must have the same number of beams for cross interpolation"
				)
			if self.n_params != other.n_params:
				raise ValueError(
					"self and other must have the same parameter dimension for cross interpolation"
				)
			if not np.allclose(self.parameters_np, other.parameters_np):
				raise ValueError(
					"self and other parameters must match sample-by-sample for cross interpolation"
				)

			beam_results = [
				beam.get_healpix_alm(
					lmax,
					freq_ndx=freq_ndx,
					other=other_beam,
					return_I_stokes_only=return_I_stokes_only,
					return_complex_components=return_complex_components,
				)
				for beam, other_beam in zip(self.beams, other.beams)
			]
			self._cross_cache[cross_key] = self._pack_cache(
				beam_results,
				return_I_stokes_only=return_I_stokes_only,
				return_complex_components=return_complex_components,
			)
		return self._cross_cache[cross_key]

	@staticmethod
	def _interpolate_packed(packed, weights, return_I_stokes_only, return_complex_components):
		out = []
		for component in packed:
			if return_complex_components:
				real_part = jnp.tensordot(weights, component[0], axes=(0, 0))
				if component[1] is None:
					imag_part = None
				else:
					imag_part = jnp.tensordot(weights, component[1], axes=(0, 0))
				out.append((real_part, imag_part))
			else:
				out.append(jnp.tensordot(weights, component, axes=(0, 0)))

		if return_I_stokes_only:
			return out[0]
		return out

	def clear_cache(self):
		"""Clear all precomputed ALM caches."""
		self._alm_cache.clear()
		self._cross_cache.clear()

	def get_healpix_alm(
		self,
		lmax,
		parameters,
		freq_ndx=None,
		other=None,
		return_I_stokes_only=True,
		return_complex_components=False,
	):
		"""
		Interpolate harmonic beam Stokes products at an arbitrary parameter vector.

		This mirrors :meth:`Beam.get_healpix_alm`, but evaluates an interpolated result
		from cached training beams. Interpolation is smooth and differentiable wrt
		``parameters`` when used with JAX transformations.

		:param lmax: Maximum l value.
		:type lmax: int
		:param parameters: Query parameter vector of shape ``(M,)``.
		:type parameters: array-like
		:param freq_ndx: Optional list of frequency indices, as in ``Beam.get_healpix_alm``.
		:type freq_ndx: int or sequence[int]
		:param other: Optional second interpolator for interpolated cross-power maps.
			For cross interpolation, both interpolators must be built on matching
			parameter samples.
		:type other: BeamInterpolator or None
		:param return_I_stokes_only: If True, return only Stokes-I; if False, return
			``[I, Q, U, V]``.
		:type return_I_stokes_only: bool
		:param return_complex_components: If True, return ``(real_alm, imag_alm)`` per
			Stokes component.
		:type return_complex_components: bool

		:returns: Interpolated ALM output in the same shape convention as
			``Beam.get_healpix_alm``.
		:rtype: jax.Array or tuple(jax.Array, jax.Array) or list
		"""

		weights = self._weights(parameters)
		packed = self._get_or_build_cache(
			lmax=lmax,
			freq_ndx=freq_ndx,
			other=other,
			return_I_stokes_only=return_I_stokes_only,
			return_complex_components=return_complex_components,
		)
		return self._interpolate_packed(
			packed,
			weights,
			return_I_stokes_only=return_I_stokes_only,
			return_complex_components=return_complex_components,
		)
