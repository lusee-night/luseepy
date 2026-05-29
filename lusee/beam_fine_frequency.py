"""
Refine :class:`lusee.Beam.Beam` frequency sampling while keeping the same
``(θ, φ)`` grid.

Linear interpolation along the native frequency axis of ``Etheta``, ``Ephi``,
``gain_conv``, and impedances matches sampling the beam’s
:meth:`~lusee.Beam.Beam.get_Efield_interpolator` at fixed ``(alt, az)`` with
varying frequency (trilinear ``RegularGridInterpolator`` collapses to 1D in
frequency when ``θ, φ`` match grid points).

The cubic-RBF :class:`lusee.BeamInterpolator.BeamInterpolator` in this codebase
currently calls ``get_healpix_alm(..., return_complex_components=False)`` and
does not cover the Croissant cross-polarization path; use linear resampling
here for general simulations (including beam cross-terms).
"""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np


def linear_resample_beam_freq_mhz(beam, fine_freq_mhz):
    """
    Return a deep copy of *beam* with fields resampled onto *fine_freq_mhz* (MHz).

    Interpolates real and imaginary parts of ``Etheta`` / ``Ephi`` along the
    frequency axis, and ``gain_conv``, ``ZRe``, ``ZIm`` (and ``gain`` if present)
    with ``jnp.interp``.
    """
    f0 = np.asarray(beam.freq, dtype=np.float64).reshape(-1)
    ft = np.asarray(fine_freq_mhz, dtype=np.float64).reshape(-1)
    if ft.size < 1:
        raise ValueError("fine_freq_mhz must be non-empty")
    if f0.size < 2:
        raise ValueError("base beam must have at least two frequency samples")

    Et = np.asarray(beam.Etheta)
    Ep = np.asarray(beam.Ephi)
    if Et.shape[0] != f0.size or Ep.shape[0] != f0.size:
        raise ValueError("beam.freq length must match E-field frequency dimension")

    n_th, n_ph = Et.shape[1], Et.shape[2]
    et_out = np.zeros((ft.size, n_th, n_ph), dtype=np.complex128)
    ep_out = np.zeros_like(et_out)
    for it in range(n_th):
        for ip in range(n_ph):
            v = Et[:, it, ip]
            et_out[:, it, ip] = np.interp(ft, f0, v.real) + 1j * np.interp(ft, f0, v.imag)
            w = Ep[:, it, ip]
            ep_out[:, it, ip] = np.interp(ft, f0, w.real) + 1j * np.interp(ft, f0, w.imag)

    out = copy.deepcopy(beam)
    out.Etheta = jnp.asarray(et_out)
    out.Ephi = jnp.asarray(ep_out)
    out.freq = jnp.asarray(ft, dtype=jnp.float64)
    out.Nfreq = int(ft.size)

    f0j = jnp.asarray(f0, dtype=jnp.float64)
    ftj = jnp.asarray(ft, dtype=jnp.float64)
    if getattr(out, "gain_conv", None) is not None:
        out.gain_conv = jnp.interp(ftj, f0j, jnp.asarray(out.gain_conv))
    if getattr(out, "ZRe", None) is not None:
        out.ZRe = jnp.interp(ftj, f0j, jnp.asarray(out.ZRe))
        out.ZIm = jnp.interp(ftj, f0j, jnp.asarray(out.ZIm))
        out.Z = out.ZRe + 1j * out.ZIm
    if getattr(out, "gain", None) is not None:
        out.gain = jnp.interp(ftj, f0j, jnp.asarray(out.gain))
    return out
