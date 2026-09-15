"""Explicit relative power conversions, independent of the packet decoder."""

import numpy as np

from .constants import BITSLICE_REFERENCE, ZOOM_BINS, ZOOM_COMPONENTS


def normalize_zoom(data, *, convention="sdu"):
    """Return a float64 copy of native (..., 4, 64) zoom power.

    Components are AA, BB, ABR, ABI. ``sdu`` divides by 2**37 to match the
    stored bit-31 normal SDU; ``pfb`` divides by 64 for raw PFB power.
    No averaging, notch, or absolute gain correction is applied. The input
    must be native, unnormalized values and is never modified.
    """
    if convention not in ("sdu", "pfb"):
        raise ValueError("zoom convention must be 'sdu' or 'pfb'")
    values = np.asarray(data)
    if values.ndim < 2 or values.shape[-2:] != (ZOOM_COMPONENTS, ZOOM_BINS):
        raise ValueError(
            f"zoom data must have shape (..., {ZOOM_COMPONENTS}, {ZOOM_BINS})"
        )
    if values.dtype.kind not in "iuf":
        raise TypeError("zoom components must be real numeric values")
    divisor = ZOOM_BINS * (2**BITSLICE_REFERENCE if convention == "sdu" else 1)
    return values.astype(np.float64) / divisor


def white_noise_notch_correction(notch):
    """Return the normal-power correction from the metadata notch byte.

    Under the independent white-noise model, multiply normal power by
    N/(N-1), where N = 2**(notch & 7). Modes 0, 2, 4, and 6 are supported.
    Mode 0 or bit 4 (subtraction disabled) gives 1; detector bit 5 does not
    affect this factor. Scalars or row arrays are accepted. This is not a
    general inverse filter and must never be applied to zoom power.
    """
    values = np.asarray(notch)
    if values.dtype.kind not in "iu":
        raise TypeError("notch metadata must contain integer bytes")
    if np.any((values < 0) | (values > 255)):
        raise ValueError("notch metadata must lie in [0, 255]")
    modes = values.astype(np.int16) & 7
    if not np.all(np.isin(modes, (0, 2, 4, 6))):
        raise ValueError("unsupported notch averaging mode")
    retained = np.where(
        (modes == 0) | ((values & 16) != 0),
        1.0,
        1.0 - np.exp2(-modes.astype(np.float64)),
    )
    return 1.0 / retained
