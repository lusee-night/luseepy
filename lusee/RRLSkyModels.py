"""
ULSA sky augmented with radio recombination line (RRL) emission at catalogued
positions. Line rest frequencies use the hydrogenic Rydberg formula; the
spectral line shape is Gaussian in frequency (same width and peak for every
line within the configured band — a placeholder until frequency-dependent
profiles are added).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields, replace
from typing import Literal, Sequence

import fitsio
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants as const
import astropy.units as u

from .SkyModels import FitsSky
from .frequencies import CANONICAL_FREQ_START_MHZ, CANONICAL_FREQ_STOP_MHZ

# RRL line shape in frequency: ``exp(-0.5 ((ν-ν0)/σ)²)`` with
# ``FWHM = 2 √(2 ln 2) σ`` (same σ for every line in :class:`ULSAPlusRRLSky`).
RRL_DEFAULT_LINE_FWHM_KHZ = 25.0
RRL_DEFAULT_LINE_SIGMA_MHZ = float(
    (RRL_DEFAULT_LINE_FWHM_KHZ / 1000.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
)
RRL_DEFAULT_LINE_PEAK_K = 0.5


def _hydrogen_rrl_k_hz() -> float:
    """Hydrogenic Rydberg factor ``c R_∞ (μ/m_e)`` in Hz (multiplies ``1/n_f² − 1/n_i²``)."""
    mu = const.m_e * const.m_p / (const.m_e + const.m_p)
    # Astropy 7+ exposes CODATA R_∞ as ``Ryd``; older releases used ``R_inf``.
    r_inf = getattr(const, "R_inf", const.Ryd)
    r_m = r_inf * mu / const.m_e
    return float((const.c * r_m).to(u.Hz).value)


def hydrogen_rrl_frequency_mhz(n_upper: float, n_lower: float) -> float:
    """
    Rest frequency of a hydrogen radio recombination line for a transition
    from principal quantum number *n_upper* to *n_lower* (with *n_upper* > *n_lower*).

    Uses :math:`\\nu = c R_\\infty (\\mu/m_e)(1/n_{\\rm f}^2 - 1/n_{\\rm i}^2)`
    with reduced mass :math:`\\mu = m_e m_p/(m_e+m_p)`.
    """
    ni = float(n_upper)
    nf = float(n_lower)
    if ni <= nf:
        raise ValueError("n_upper must be greater than n_lower for emission")
    inv = 1.0 / (nf * nf) - 1.0 / (ni * ni)
    return _hydrogen_rrl_k_hz() * inv / 1e6


def _hydrogen_nu_alpha_hz(n: float) -> float:
    """Rest frequency in Hz for hydrogen nα (``n`` → ``n−1``), ``n`` > 1."""
    n = float(n)
    if n <= 1.0:
        raise ValueError("n must be > 1 for nα")
    inv = 1.0 / (n - 1.0) ** 2 - 1.0 / (n * n)
    return _hydrogen_rrl_k_hz() * inv


def hydrogen_rrl_alpha_quantum_numbers_from_frequency_mhz(
    nu_mhz: float,
) -> tuple[int, int]:
    """
    Principal quantum numbers ``(n1, n2)`` for the hydrogen **nα** line
    (``n1`` = upper level, ``n2`` = ``n1 − 1``) whose rest frequency is closest
    to *nu_mhz*.

    This inverts the same Rydberg relation as :func:`hydrogen_rrl_frequency_mhz`
    restricted to Δ*n* = 1.
    """
    nu_hz = float(nu_mhz) * 1e6
    if nu_hz <= 0.0:
        raise ValueError("nu_mhz must be positive")
    k = _hydrogen_rrl_k_hz()
    n_est = max(2, int((2.0 * k / nu_hz) ** (1.0 / 3.0)))
    best_n, best_d = 2, abs(_hydrogen_nu_alpha_hz(2.0) - nu_hz)
    for n in range(max(2, n_est - 4), n_est + 6):
        d = abs(_hydrogen_nu_alpha_hz(float(n)) - nu_hz)
        if d < best_d:
            best_d, best_n = d, n
    return (best_n, best_n - 1)


def hydrogen_rrl_alpha_transitions_in_frequency_band_mhz(
    nu_min_mhz: float | None = None,
    nu_max_mhz: float | None = None,
) -> list[tuple[int, int]]:
    """
    All integer hydrogen **nα** transitions ``(n1, n2) = (n, n−1)`` whose rest
    frequencies lie in ``[nu_min_mhz, nu_max_mhz]`` (inclusive on frequency).

    Defaults match the canonical LuSEE simulator grid (1–50 MHz), i.e. the
    same band as ``freq: start_idx: 0, stop_idx: 50`` in configs such as
    ``sim_choice_realistic.yaml`` (indices 0…49 → 1…50 MHz).

    For hydrogen nα, rest frequency decreases as *n* increases, so the returned
    *n* values form a contiguous range at high principal quantum number
    (thousands of lines in the 1–50 MHz band).
    """
    lo_mhz = CANONICAL_FREQ_START_MHZ if nu_min_mhz is None else float(nu_min_mhz)
    hi_mhz = CANONICAL_FREQ_STOP_MHZ if nu_max_mhz is None else float(nu_max_mhz)
    if lo_mhz > hi_mhz:
        lo_mhz, hi_mhz = hi_mhz, lo_mhz
    nu_min_hz = lo_mhz * 1e6
    nu_max_hz = hi_mhz * 1e6
    if nu_min_hz <= 0.0 or nu_max_hz <= 0.0:
        raise ValueError("band edges must be positive MHz values")

    def smallest_n_nu_leq(target_hz: float) -> int:
        """Smallest *n* ≥ 2 with ν_α(*n*) ≤ *target_hz* (ν decreases with *n*)."""
        lo, hi = 2, 4
        while _hydrogen_nu_alpha_hz(float(hi)) > target_hz:
            hi *= 2
            if hi > 10**9:
                raise ValueError("could not bracket n for upper band edge")
        while lo < hi:
            mid = (lo + hi) // 2
            if _hydrogen_nu_alpha_hz(float(mid)) <= target_hz:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def smallest_n_nu_strictly_below(target_hz: float) -> int:
        """Smallest *n* ≥ 2 with ν_α(*n*) < *target_hz*."""
        lo, hi = 2, 4
        while _hydrogen_nu_alpha_hz(float(hi)) >= target_hz:
            hi *= 2
            if hi > 10**9:
                raise ValueError("could not bracket n for lower band edge")
        while lo < hi:
            mid = (lo + hi) // 2
            if _hydrogen_nu_alpha_hz(float(mid)) < target_hz:
                hi = mid
            else:
                lo = mid + 1
        return lo

    n_lo = smallest_n_nu_leq(nu_max_hz)
    n_first_below_min = smallest_n_nu_strictly_below(nu_min_hz)
    n_hi = n_first_below_min - 1
    if n_hi < 2 or n_lo > n_hi:
        return []
    return [(n, n - 1) for n in range(n_lo, n_hi + 1)]


def _vydula_rydberg_R_hz(nucleus_mass_kg: float, *, z: float = 1.0) -> float:
    """
    Rydberg constant for a nucleus of mass *M* (Vydula+2024 Eq. 2), in Hz.

    :math:`R = R_\\infty \\,(M/(M+m_e))` with :math:`R_\\infty` the infinite-mass
    Rydberg constant. Rest frequency (Eq. 1): :math:`\\nu = c R Z^2 (1/n_1^2 - 1/n_2^2)`
    for transition :math:`n_2 \\rightarrow n_1` (upper :math:`n_2` > lower :math:`n_1`).
    """
    m_e = float(const.m_e.to(u.kg).value)
    m_n = float(nucleus_mass_kg)
    if m_n <= 0.0:
        raise ValueError("nucleus_mass_kg must be positive")
    r_inf = getattr(const, "R_inf", const.Ryd)
    r_m = r_inf * (m_n / (m_n + m_e))
    return float((const.c * r_m * (float(z) ** 2)).to(u.Hz).value)


def _carbon12_nucleus_mass_kg() -> float:
    """Mass of the :math:`^{12}`C nucleus (12 amu) in kg."""
    return float((12.0 * const.u).to(u.kg).value)


def _carbon_rrl_k_hz(*, z: float = 1.0) -> float:
    """Factor :math:`c R_{\\rm C} Z^2` in Hz for Vydula+2024 Eq. (1)–(2), carbon."""
    return _vydula_rydberg_R_hz(_carbon12_nucleus_mass_kg(), z=z)


def carbon_rrl_frequency_mhz(
    n_upper: float,
    n_lower: float,
    *,
    z: float = 1.0,
) -> float:
    """
    Rest frequency (MHz) of a carbon radio recombination line.

    Vydula et al. (2024, AJ 167, 2; `doi:10.3847/1538-3881/ad08ba
    <https://doi.org/10.3847/1538-3881/ad08ba>`_) Eq. (1)–(2) with
    :math:`\\nu_{n_2\\rightarrow n_1} = c R Z^2 (1/n_1^2 - 1/n_2^2)`.
    """
    ni = float(n_upper)
    nf = float(n_lower)
    if ni <= nf:
        raise ValueError("n_upper must be greater than n_lower for emission")
    inv = 1.0 / (nf * nf) - 1.0 / (ni * ni)
    return _carbon_rrl_k_hz(z=z) * inv / 1e6


def _carbon_nu_transition_hz(n_upper: float, delta_n: int) -> float:
    """Rest frequency in Hz for carbon transition ``n_upper`` → ``n_upper − delta_n``."""
    n = float(n_upper)
    dn = int(delta_n)
    if dn < 1:
        raise ValueError("delta_n must be >= 1")
    if n <= float(dn):
        raise ValueError("n_upper must exceed delta_n")
    inv = 1.0 / (n - float(dn)) ** 2 - 1.0 / (n * n)
    return _carbon_rrl_k_hz() * inv


def carbon_rrl_transitions_in_frequency_band_mhz(
    nu_min_mhz: float | None = None,
    nu_max_mhz: float | None = None,
    *,
    delta_n: int = 1,
) -> list[tuple[int, int]]:
    """
    All integer carbon RRL transitions ``(n_upper, n_lower)`` with
    ``n_lower = n_upper − delta_n`` (e.g. ``delta_n=1`` for Cα) whose rest
    frequencies lie in ``[nu_min_mhz, nu_max_mhz]``.

    Frequencies follow Vydula+2024 Eq. (1)–(2). For Cα, *n* decreases with
    increasing frequency across the LuSEE low-frequency band (same ordering as Hα).
    """
    lo_mhz = CANONICAL_FREQ_START_MHZ if nu_min_mhz is None else float(nu_min_mhz)
    hi_mhz = CANONICAL_FREQ_STOP_MHZ if nu_max_mhz is None else float(nu_max_mhz)
    if lo_mhz > hi_mhz:
        lo_mhz, hi_mhz = hi_mhz, lo_mhz
    nu_min_hz = lo_mhz * 1e6
    nu_max_hz = hi_mhz * 1e6
    if nu_min_hz <= 0.0 or nu_max_hz <= 0.0:
        raise ValueError("band edges must be positive MHz values")
    dn = int(delta_n)
    if dn < 1:
        raise ValueError("delta_n must be >= 1")

    def nu_hz(n: int) -> float:
        return _carbon_nu_transition_hz(float(n), dn)

    def smallest_n_nu_leq(target_hz: float) -> int:
        lo_n = dn + 1
        hi_n = lo_n + 2
        while nu_hz(hi_n) > target_hz:
            hi_n *= 2
            if hi_n > 10**9:
                raise ValueError("could not bracket n for upper band edge")
        while lo_n < hi_n:
            mid = (lo_n + hi_n) // 2
            if nu_hz(mid) <= target_hz:
                hi_n = mid
            else:
                lo_n = mid + 1
        return lo_n

    def smallest_n_nu_strictly_below(target_hz: float) -> int:
        lo_n = dn + 1
        hi_n = lo_n + 2
        while nu_hz(hi_n) >= target_hz:
            hi_n *= 2
            if hi_n > 10**9:
                raise ValueError("could not bracket n for lower band edge")
        while lo_n < hi_n:
            mid = (lo_n + hi_n) // 2
            if nu_hz(mid) < target_hz:
                hi_n = mid
            else:
                lo_n = mid + 1
        return lo_n

    n_lo = smallest_n_nu_leq(nu_max_hz)
    n_first_below_min = smallest_n_nu_strictly_below(nu_min_hz)
    n_hi = n_first_below_min - 1
    if n_hi < dn + 1 or n_lo > n_hi:
        return []
    return [(n, n - dn) for n in range(n_lo, n_hi + 1)]


def carbon_rrl_alpha_transitions_in_frequency_band_mhz(
    nu_min_mhz: float | None = None,
    nu_max_mhz: float | None = None,
) -> list[tuple[int, int]]:
    """Carbon Cα transitions ``(n, n−1)`` in the frequency band (``delta_n=1``)."""
    return carbon_rrl_transitions_in_frequency_band_mhz(
        nu_min_mhz, nu_max_mhz, delta_n=1
    )


def _normalize_column_names(names: Sequence[str]) -> dict[str, str]:
    return {n.lower(): n for n in names}


_SPECTRAL_CTYPE_PREFIXES = (
    "VELO",
    "VRAD",
    "VOPT",
    "FREQ",
    "WAVE",
    "AWAV",
    "BETA",
    "STOKES",
)


def spectral_axis_numpy_index_from_header(header) -> int | None:
    """
    Numpy axis index along which a FITS spectral cube varies (for ``hdu.data``).

    ``astropy.io.fits`` image arrays use numpy axis order **slow to fast**:
    ``data.shape == (NAXIS3, NAXIS2, NAXIS1)``. If the spectral axis is FITS
    axis ``k`` (1-based ``CTYPEk``), the corresponding numpy axis is
    ``NAXIS - k``. Many radio cubes have ``VELO`` (or ``FREQ``) on ``CTYPE3``,
    i.e. **numpy axis 0** — not axis 2.

    Returns ``None`` if no spectral ``CTYPE`` is recognised (caller may guess).
    """
    try:
        n_fits = int(header["NAXIS"])
    except (KeyError, TypeError, ValueError):
        return None
    if n_fits < 1:
        return None
    for k in range(1, n_fits + 1):
        ct = str(header.get(f"CTYPE{k}", "") or "").strip().upper()
        if not ct:
            continue
        root = ct.split("-")[0]
        if any(root.startswith(p) for p in _SPECTRAL_CTYPE_PREFIXES):
            return n_fits - k
    return None


def load_rrl_region_positions_gal_deg(
    fits_path: str,
    *,
    max_cube_sources: int = 2000,
    cube_peak_footprint: int = 7,
    cube_collapse: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read galactic longitude and latitude (degrees) for RRL sky regions.

    **Two FITS layouts are supported:**

    1. **Binary table catalog** — rows are sources; only lon/lat (or RA/Dec)
       columns are read. Line or velocity columns are ignored for LuSEE
       simulation (frequencies come from the observation / config grid; spectra
       use the Rydberg + Gaussian model in code).

    2. **(l, b[, v]) spectral cube** — e.g. ``*_lbv.fits`` with ``GLON-CAR``,
       ``GLAT-CAR``, and a spectral axis (``VEL``). The cube is collapsed along
       the spectral axis (default: max), local maxima in the 2D map are taken
       as region centres (bright RRL emission in the stacked line cube), and
       pixel indices are turned into ``(l, b)`` degrees via the WCS.

       Typical HIPASS+ZOA ``*_lbv.fits`` default catalog (same text as the comment
       block at the spectral-cube FITS read site in this module):

       The RRL datacube covering the whole spatial extent of the survey (248 deg in
       longitude, 10 deg in latitude), with +/-335 km/s velocity coverage, spectral
       resolution 20 km/s, and channel width 13.4 km/s. The cube combines the 3 RRLs
       of H166-168alpha. The units are K brightness temperature with an rms noise
       per channel of 2.8 mK.

    Recognized table column pairs (case-insensitive): (GLON, GLAT),
    (G_LON, G_LAT), (LII, BII), (L, B), (GLON_DEG, GLAT_DEG); equatorial
    (RA, DEC), (RAJ2000, DEJ2000), etc., converted to Galactic.
    """
    from astropy.coordinates import SkyCoord

    if not os.path.isfile(fits_path):
        raise FileNotFoundError(fits_path)

    table_hints: list[str] = []
    with fitsio.FITS(fits_path) as fits:
        for i in range(len(fits)):
            hdu = fits[i]
            if hdu.get_exttype() != "BINARY_TBL":
                continue
            data = hdu.read()
            if data is None or data.size == 0:
                continue
            names = _normalize_column_names(data.dtype.names)
            table_hints.append(f"HDU {i}: {list(data.dtype.names)}")
            for a, b in (
                ("glon", "glat"),
                ("g_lon", "g_lat"),
                ("lii", "bii"),
                ("l2", "b2"),
                ("l", "b"),
                ("glon_deg", "glat_deg"),
                ("lon", "lat"),
                ("l_deg", "b_deg"),
                ("l_gal", "b_gal"),
            ):
                if a in names and b in names:
                    l_col, b_col = names[a], names[b]
                    lon = np.asarray(data[l_col], dtype=float)
                    lat = np.asarray(data[b_col], dtype=float)
                    return lon, lat
            for ra_k, de_k in (
                ("ra", "dec"),
                ("raj2000", "dej2000"),
                ("ra_deg", "dec_deg"),
                ("ra2000", "dec2000"),
                ("_raj2000", "_dej2000"),
                ("raj", "dej"),
            ):
                if ra_k in names and de_k in names:
                    ra = np.asarray(data[names[ra_k]], dtype=float)
                    dec = np.asarray(data[names[de_k]], dtype=float)
                    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                    g = c.galactic
                    return np.asarray(g.l.deg), np.asarray(g.b.deg)

    try:
        return _rrl_cube_peak_positions_gal_deg(
            fits_path,
            max_sources=max_cube_sources,
            peak_footprint=cube_peak_footprint,
            collapse=cube_collapse,
        )
    except Exception as cube_err:
        msg = f"No usable RRL source positions in {fits_path!r}."
        if table_hints:
            msg += " Binary table HDUs had columns: " + "; ".join(table_hints)
        msg += f" Spectral-cube fallback failed: {cube_err}"
        raise ValueError(msg) from cube_err


def _rrl_cube_peak_positions_gal_deg(
    fits_path: str,
    *,
    max_sources: int,
    peak_footprint: int,
    collapse: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Peak (l, b) in degrees from a GLON–GLAT–(VEL) FITS cube.

    Primary HDU is opened below with ``astropy.io.fits``. For the default
    HIPASS+ZOA ``lbv`` stack, survey metadata is documented in the comment block
    immediately preceding that open call (and in ``load_rrl_region_positions_gal_deg``).
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from scipy.ndimage import maximum_filter

    # HIPASS+ZOA H166–168α combined-stack spectral cube (e.g. *_lbv.fits):
    # The RRL datacube covering the whole spatial extent of the survey (248 deg in
    # longitude, 10 deg in latitude), with +/-335 km/s velocity coverage, spectral
    # resolution 20 km/s, and channel width 13.4 km/s. The cube combines the 3 RRLs
    # of H166-168alpha. The units are K brightness temperature with an rms noise per
    # channel of 2.8 mK.
    with afits.open(fits_path, memmap=False) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        data = np.asarray(hdu.data, dtype=float)

    if data.ndim == 3:
        spec_ax = spectral_axis_numpy_index_from_header(hdr)
        if spec_ax is None:
            warnings.warn(
                "Could not identify spectral axis from CTYPE* keywords; "
                "assuming numpy axis 0 (typical for VELO on FITS axis 3).",
                UserWarning,
                stacklevel=2,
            )
            spec_ax = 0
        if spec_ax < 0 or spec_ax >= data.ndim:
            raise ValueError(
                f"spectral axis {spec_ax} invalid for data shape {data.shape} / header NAXIS"
            )
        if str(collapse).lower() == "sum":
            plane = np.nansum(data, axis=spec_ax)
        else:
            plane = np.nanmax(data, axis=spec_ax)
    elif data.ndim == 2:
        plane = data
    else:
        raise ValueError(f"expected 2D or 3D FITS data, got shape {data.shape}")

    wcs = WCS(hdr)
    w2 = wcs.celestial
    if w2.naxis != 2:
        raise ValueError("celestial WCS is not 2-dimensional")

    plane = np.where(np.isfinite(plane), plane, np.nan)
    med = float(np.nanmedian(plane))
    mad = float(np.nanmedian(np.abs(plane - med)))
    thresh = med + 5.0 * (mad if mad > 1e-12 else max(1e-6, 0.01 * abs(med)))
    if not np.any(plane > thresh):
        thresh = float(np.nanpercentile(plane, 90.0))

    loc = maximum_filter(plane, size=int(peak_footprint), mode="nearest")
    mask = (plane >= loc) & (plane > thresh) & np.isfinite(plane)
    iy, ix = np.nonzero(mask)
    if iy.size == 0:
        flat = int(np.nanargmax(np.where(np.isfinite(plane), plane, -np.inf)))
        iy, ix = np.unravel_index(flat, plane.shape)
        iy = np.asarray([int(iy)], dtype=int)
        ix = np.asarray([int(ix)], dtype=int)

    vals = plane[iy, ix]
    order = np.argsort(vals)[::-1][: int(max_sources)]
    ix = ix[order]
    iy = iy[order]

    coords = w2.pixel_to_world(ix, iy)
    lon = np.asarray(coords.spherical.lon.to(u.deg).value, dtype=float)
    lat = np.asarray(coords.spherical.lat.to(u.deg).value, dtype=float)
    return lon, lat


def _gaussian_spherical_map(
    nside: int,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    """Sum of axisymmetric Gaussians on the sphere (small-angle in pixel space)."""
    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=float)
    sigma_rad = np.radians(sigma_deg)
    for ell, bee in zip(np.atleast_1d(lon_deg), np.atleast_1d(lat_deg)):
        theta = np.pi / 2.0 - np.radians(bee)
        phi = np.radians(ell) % (2 * np.pi)
        nbr = hp.query_disc(nside, hp.ang2vec(theta, phi), 4 * sigma_rad)
        th2, ph2 = hp.pix2ang(nside, nbr)
        cos_a = np.sin(th2) * np.sin(theta) * np.cos(ph2 - phi) + np.cos(th2) * np.cos(
            theta
        )
        cos_a = np.clip(cos_a, -1.0, 1.0)
        ang = np.arccos(cos_a)
        m[nbr] += np.exp(-0.5 * (ang / sigma_rad) ** 2)
    peak = np.max(m) if m.size else 1.0
    if peak <= 0.0:
        peak = 1.0
    return m / peak


def _interp_mapalm_native_to_target_mhz(
    mapalm_native: np.ndarray,
    freq_native_mhz: np.ndarray,
    freq_target_mhz: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate complex ULSA ``mapalm`` along frequency (axis 0)."""
    fn = np.asarray(freq_native_mhz, dtype=np.float64).reshape(-1)
    ft = np.asarray(freq_target_mhz, dtype=np.float64).reshape(-1)
    x = np.asarray(mapalm_native)
    if x.shape[0] != fn.size:
        raise ValueError("mapalm_native frequency axis must match freq_native_mhz")
    out = np.zeros((ft.size, x.shape[1]), dtype=np.complex128)
    for j in range(x.shape[1]):
        c = x[:, j]
        out[:, j] = np.interp(ft, fn, c.real) + 1j * np.interp(ft, fn, c.imag)
    return out


@jax.tree_util.register_pytree_node_class
class ULSAPlusRRLSky:
    """
    Galactic-frame sky: ULSA cube (:class:`FitsSky`) plus compact Gaussian spots
    at catalog positions, modulated in frequency by a sum of identical Gaussian
    profiles centered at hydrogenic RRL frequencies.

    The RRL FITS catalog is used **only** for source positions (galactic or
    equatorial columns); the simulator frequency axis is set by *sim_freq_mhz*
    when provided (e.g. 1 kHz steps from config), and ULSA maps are linearly
    interpolated onto that axis from the native ULSA frequency sampling.

    :param ulsa_fname: Path to ULSA FITS (same format as :class:`FitsSky`).
    :param rrl_catalog_fname: Path to RRL region catalog FITS (binary table or
        spectral cube; see ``load_rrl_region_positions_gal_deg`` for layout
        and default ``lbv`` cube metadata).
    :param lmax: Harmonic band-limit for Healpix alm synthesis of the RRL spots.
    :param sim_freq_mhz: Optional simulator frequency axis (MHz); when set, ULSA
        alms are interpolated onto this grid and ``self.freq`` matches it.
    :param alpha_transitions: Sequence of (n_upper, n_lower) for each line (default H168α–H166α).
    :param rrl_sigma_mhz: Gaussian σ in MHz for the frequency-domain line shape
        (shared by every line for now); FWHM = ``2√(2 ln 2) σ``. Default matches
        ``RRL_DEFAULT_LINE_FWHM_KHZ``.
    :param rrl_peak_k: Line peak brightness in Kelvin at line centre (multiplier
        on the normalized spatial template). Default ``RRL_DEFAULT_LINE_PEAK_K``.
    :param spot_sigma_deg: On-sky σ of each regional Gaussian (degrees).

    After construction, ``rrl_spot_map`` holds the Healpix map (peak-normalized
    to 1) of the summed Gaussians **before** ``map2alm`` / ``lmax`` band-limiting;
    it is ``None`` on instances restored from :meth:`tree_unflatten`.
    """

    def __init__(
        self,
        ulsa_fname: str,
        rrl_catalog_fname: str,
        lmax: int,
        *,
        sim_freq_mhz: jnp.ndarray | np.ndarray | None = None,
        alpha_transitions: Sequence[tuple[float, float]] | None = None,
        rrl_sigma_mhz: float = RRL_DEFAULT_LINE_SIGMA_MHZ,
        rrl_peak_k: float = RRL_DEFAULT_LINE_PEAK_K,
        spot_sigma_deg: float = 0.25,
    ):
        self._ulsa = FitsSky(ulsa_fname, lmax=lmax)
        self.Nside = int(self._ulsa.Nside)
        self.lmax = int(lmax)
        self.frame = "galactic"

        fn = np.asarray(self._ulsa.freq, dtype=np.float64)
        native_alm = np.asarray(self._ulsa.mapalm)
        if sim_freq_mhz is not None:
            ft = np.asarray(sim_freq_mhz, dtype=np.float64).reshape(-1)
            interp_alm = _interp_mapalm_native_to_target_mhz(native_alm, fn, ft)
            self.mapalm_ulsa = jnp.asarray(interp_alm, dtype=jnp.complex128)
            self.freq = jnp.asarray(ft, dtype=jnp.float64)
        else:
            self.mapalm_ulsa = jnp.asarray(native_alm, dtype=jnp.complex128)
            self.freq = self._ulsa.freq

        lon_deg, lat_deg = load_rrl_region_positions_gal_deg(rrl_catalog_fname)
        spot_map = _gaussian_spherical_map(
            self.Nside, lon_deg, lat_deg, spot_sigma_deg
        )
        self.rrl_spot_map = np.asarray(spot_map, dtype=np.float64)
        alm_sp = hp.map2alm(spot_map, lmax=self.lmax)
        self.alm_spatial = jnp.asarray(alm_sp, dtype=jnp.complex128)

        if alpha_transitions is None:
            alpha_transitions = ((168, 167), (167, 166), (166, 165))
        centers = np.array(
            [hydrogen_rrl_frequency_mhz(nu, nl) for nu, nl in alpha_transitions],
            dtype=float,
        )
        self._line_centers_mhz = jnp.asarray(centers, dtype=jnp.float64)
        self.rrl_sigma_mhz = float(rrl_sigma_mhz)
        self.rrl_peak_k = float(rrl_peak_k)

    def tree_flatten(self):
        children = (self.mapalm_ulsa, self.alm_spatial, self._line_centers_mhz)
        aux_data = (
            self.Nside,
            self.lmax,
            tuple(np.asarray(self.freq).tolist()),
            self.rrl_sigma_mhz,
            self.rrl_peak_k,
            self.frame,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mapalm_ulsa, alm_spatial, line_centers = children
        Nside, lmax, freq_t, sigma, peak, frame = aux_data
        sky = cls.__new__(cls)
        sky.Nside = Nside
        sky.lmax = lmax
        sky.freq = jnp.asarray(freq_t)
        sky.frame = frame
        sky.mapalm_ulsa = mapalm_ulsa
        sky.alm_spatial = alm_spatial
        sky._line_centers_mhz = line_centers
        sky.rrl_sigma_mhz = sigma
        sky.rrl_peak_k = peak
        sky._ulsa = None
        sky.rrl_spot_map = None
        return sky

    def _spectral_weight(self, ndx: jnp.ndarray) -> jnp.ndarray:
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        nu = self.freq[ndx]
        sig = jnp.asarray(self.rrl_sigma_mhz, dtype=nu.dtype)
        peak = jnp.asarray(self.rrl_peak_k, dtype=nu.dtype)
        centers = jnp.asarray(self._line_centers_mhz, dtype=nu.dtype)
        d = nu[:, None] - centers[None, :]
        g = jnp.exp(-0.5 * (d / sig) ** 2)
        return peak * jnp.sum(g, axis=1)

    def get_alm(self, ndx, freq=None):
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        base = self.mapalm_ulsa[ndx]
        w = self._spectral_weight(ndx)
        return base + w[:, None] * self.alm_spatial[None, :]

    def rrl_brightness_map_K(self, freq_index: int) -> np.ndarray:
        """
        Healpix map (Kelvin) of the RRL-only sky at ``self.freq[freq_index]``.

        Returns ``w(ν) T_{\\rm spot}(\\hat{\\mathbf n})`` where ``T_{\\rm spot}`` is
        the real-space reconstruction of the normalized catalog Gaussians (band-limited
        by ``lmax``) and ``w`` is the frequency sum of line-centred Gaussians from
        :meth:`_spectral_weight`. Coordinates match the ULSA cube (Galactic).
        """
        ndx = jnp.atleast_1d(jnp.asarray(int(freq_index)))
        w = self._spectral_weight(ndx)
        w0 = float(np.asarray(w[0], dtype=np.float64))
        alm = np.asarray(self.alm_spatial, dtype=np.complex128)
        m = hp.alm2map(alm, nside=int(self.Nside), lmax=int(self.lmax))
        return (w0 * np.real(m)).astype(np.float64)


def default_rrl_catalog_path(lusee_drive_dir: str | None = None) -> str:
    """Default path to the HIPASS+ZOA RRL region map under the LUSEE drive."""
    root = lusee_drive_dir or os.environ.get("LUSEE_DRIVE_DIR", "")
    if not root:
        raise ValueError("lusee_drive_dir or LUSEE_DRIVE_DIR must be set")
    return os.path.join(
        root,
        "Simulations",
        "SkyModels",
        "RRL_maps",
        "RRL_H166-167-168a_HIPASS+ZOA_lbv.fits",
    )


def default_ulsa_path(lusee_drive_dir: str | None = None) -> str:
    root = lusee_drive_dir or os.environ.get("LUSEE_DRIVE_DIR", "")
    if not root:
        raise ValueError("lusee_drive_dir or LUSEE_DRIVE_DIR must be set")
    return os.path.join(root, "Simulations", "SkyModels", "ULSA_32_ddi_smooth.fits")


# Vydula et al. (2024, AJ 167, 2; doi:10.3847/1538-3881/ad08ba) non-LTE strong-prior fits
# (Table 3): Cα inner-plane and Hα emission cases.
VYDULA2024_NU100_MHZ = 100.0


@dataclass(frozen=True)
class Vydula2024EnvelopeParams:
    """
    Parameters for the smooth RRL envelope, Equation (7) of
    `Vydula et al. 2024 <https://doi.org/10.3847/1538-3881/ad08ba>`_.

    .. math::

        T_{\\mathrm{RRL}}(\\nu_n) \\approx \\tau^{\\mathrm{LTE}}_L
        \\left[ b_n T_e - b_n \\beta_n(\\nu_n)\\, T_R(\\nu_n) \\right]

    with :math:`T_R(\\nu) = f_{\\mathrm{TR}}\\, T_{\\mathrm{cont}}(\\nu)`,
    :math:`T_{\\mathrm{cont}} = T_{100}(\\nu/\\nu_{100})^{\\beta}`, and
    :math:`\\beta_n` from Equation (8) (polynomial in :math:`\\nu/\\nu_{100}`).
    """

    Te_k: float
    tau_lte: float
    T100_k: float
    beta_cont: float
    f_tr: float = 0.2
    bn: float = 1.0
    # coefficients for sum_m beta_nm (nu/nu100)^m, m = 0..M
    beta_n_coeffs: tuple[float, ...] = (0.30, -0.60, 0.56, 0.03)
    nu100_mhz: float = VYDULA2024_NU100_MHZ


# Cα, LST 18 h, non-LTE (Table 3) — cold / carbon-dominated diffuse gas
COLD_GAS_VYDULA2024 = Vydula2024EnvelopeParams(
    Te_k=100.0,
    tau_lte=8.71e-3,
    T100_k=10_000.0,
    beta_cont=-2.4,
    f_tr=0.2,
    bn=1.0,
    beta_n_coeffs=(0.30, -0.60, 0.56, 0.03),
)
# Hα, LST 18 h, non-LTE (Table 3) — hot hydrogen gas
HOT_GAS_VYDULA2024 = Vydula2024EnvelopeParams(
    Te_k=7000.0,
    tau_lte=8.91e-3,
    T100_k=10_000.0,
    beta_cont=-2.4,
    f_tr=0.2,
    bn=1.0,
    beta_n_coeffs=(0.65, -2.29, 4.37, 0.59),
)

GasCaseName = Literal["cold", "hot", "gaussian"]


def vydula2024_envelope_params_from_gas_case(
    gas_case: GasCaseName = "cold",
) -> Vydula2024EnvelopeParams:
    """Return preset :class:`Vydula2024EnvelopeParams` for ``cold`` or ``hot`` gas."""
    key = str(gas_case).lower()
    if key == "cold":
        return COLD_GAS_VYDULA2024
    if key == "hot":
        return HOT_GAS_VYDULA2024
    raise ValueError(
        f"gas_case must be 'cold' or 'hot' for Vydula2024 presets (got {gas_case!r})"
    )


def vydula2024_envelope_params_from_config(
    gas_case: GasCaseName = "cold",
    cfg: dict | None = None,
) -> Vydula2024EnvelopeParams | None:
    """
    Build :class:`Vydula2024EnvelopeParams` from ``gas_case`` and optional YAML-style overrides.

    Returns ``None`` when ``gas_case`` is ``gaussian`` (legacy placeholder envelope).
    """
    key = str(gas_case).lower()
    if key == "gaussian":
        return None
    base = vydula2024_envelope_params_from_gas_case(key)  # type: ignore[arg-type]
    overrides = dict(cfg or {})
    # aliases used in configs
    alias = {
        "Te": "Te_k",
        "tau_L": "tau_lte",
        "T100": "T100_k",
        "beta": "beta_cont",
        "f_TR": "f_tr",
        "bn": "bn",
    }
    kw: dict[str, object] = {}
    valid = {f.name for f in fields(Vydula2024EnvelopeParams)}
    for src, dst in alias.items():
        if src in overrides:
            kw[dst] = overrides.pop(src)
    for name in valid:
        if name in overrides:
            kw[name] = overrides.pop(name)
    if "beta_n_coeffs" in kw:
        kw["beta_n_coeffs"] = tuple(float(x) for x in kw["beta_n_coeffs"])  # type: ignore[arg-type]
    if overrides:
        unknown = ", ".join(sorted(overrides))
        raise ValueError(f"unknown envelope config keys: {unknown}")
    return replace(base, **kw) if kw else base


def beta_n_vydula2024_polynomial(
    nu_mhz: np.ndarray | float,
    beta_n_coeffs: Sequence[float],
    *,
    nu100_mhz: float = VYDULA2024_NU100_MHZ,
) -> np.ndarray:
    """Equation (8): :math:`\\beta_n = \\sum_m \\beta_{n,m} (\\nu/\\nu_{100})^m`."""
    nu = np.asarray(nu_mhz, dtype=np.float64).reshape(-1)
    x = nu / float(nu100_mhz)
    out = np.zeros_like(nu, dtype=np.float64)
    for m, coeff in enumerate(beta_n_coeffs):
        out += float(coeff) * np.power(x, m)
    return out


def T_cont_vydula_mhz(
    nu_mhz: np.ndarray | float,
    *,
    T100_k: float,
    beta_cont: float,
    nu100_mhz: float = VYDULA2024_NU100_MHZ,
) -> np.ndarray:
    """Galactic synchrotron continuum :math:`T_{100}(\\nu/\\nu_{100})^{\\beta}`."""
    nu = np.asarray(nu_mhz, dtype=np.float64).reshape(-1)
    return float(T100_k) * np.power(nu / float(nu100_mhz), float(beta_cont))


def T_R_vydula_mhz(
    nu_mhz: np.ndarray | float,
    params: Vydula2024EnvelopeParams,
) -> np.ndarray:
    """Background radiation temperature :math:`T_R = f_{\\mathrm{TR}} T_{\\mathrm{cont}}`."""
    return float(params.f_tr) * T_cont_vydula_mhz(
        nu_mhz,
        T100_k=params.T100_k,
        beta_cont=params.beta_cont,
        nu100_mhz=params.nu100_mhz,
    )


def rrl_envelope_T_rrl_k_mhz(
    nu_mhz: np.ndarray | float,
    params: Vydula2024EnvelopeParams | None = None,
    *,
    gas_case: GasCaseName = "cold",
) -> np.ndarray:
    """
    Smooth RRL envelope brightness (K) from Vydula+2024 Equation (7).

    Catalog positions still supply the spatial Gaussian template; this function
    gives the frequency-dependent :math:`T_{\\mathrm{RRL}}` multiplier at each
    channel before beam convolution.
    """
    if params is None:
        params = vydula2024_envelope_params_from_gas_case(gas_case)
    nu = np.asarray(nu_mhz, dtype=np.float64).reshape(-1)
    beta_n = beta_n_vydula2024_polynomial(
        nu, params.beta_n_coeffs, nu100_mhz=params.nu100_mhz
    )
    t_r = T_R_vydula_mhz(nu, params)
    return float(params.tau_lte) * (
        float(params.bn) * float(params.Te_k)
        - float(params.bn) * beta_n * t_r
    )


def rrl_smooth_envelope_weight_mhz(
    nu_mhz: np.ndarray | float,
    *,
    nu_ref_mhz: float = 12.5,
    sigma_mhz: float = 3.0,
    amplitude_k: float = 0.5,
    gas_case: GasCaseName = "cold",
    envelope_params: Vydula2024EnvelopeParams | None = None,
) -> np.ndarray:
    """
    Frequency-dependent envelope weight (K) on the spatial spot template.

    Default is Vydula+2024 Equation (7) with ``gas_case`` ``cold`` or ``hot``
    presets (Table 3). Pass ``envelope_params`` to override :math:`T_e`,
    :math:`\\tau_L`, :math:`b_n`, :math:`\\beta_n`, etc. Use ``gas_case='gaussian'``
    for the legacy positive Gaussian placeholder.
    """
    if str(gas_case).lower() == "gaussian":
        nu = np.asarray(nu_mhz, dtype=np.float64).reshape(-1)
        sig = float(sigma_mhz)
        if sig <= 0.0:
            raise ValueError("sigma_mhz must be positive for gaussian envelope")
        return float(amplitude_k) * np.exp(-0.5 * ((nu - float(nu_ref_mhz)) / sig) ** 2)
    return rrl_envelope_T_rrl_k_mhz(
        nu_mhz, envelope_params, gas_case=gas_case
    )


def rydberg_line_spectrum_mhz(
    freq_mhz: np.ndarray,
    transitions: Sequence[tuple[int, int]] | None = None,
    *,
    nu_lo_mhz: float | None = None,
    nu_hi_mhz: float | None = None,
    species: Literal["carbon", "hydrogen"] = "carbon",
    delta_n: int = 1,
    sigma_mhz: float = RRL_DEFAULT_LINE_SIGMA_MHZ,
    peak_k: float = RRL_DEFAULT_LINE_PEAK_K,
) -> np.ndarray:
    """
    Sum of Gaussian line profiles at RRL rest frequencies in the analysis band.

    Default is **carbon** Cα (``delta_n=1``) using Vydula+2024 Eq. (1)–(2)
    (`doi:10.3847/1538-3881/ad08ba <https://doi.org/10.3847/1538-3881/ad08ba>`_).
    When *transitions* is ``None``, every carbon line with
    :math:`n_{\\rm upper} - n_{\\rm lower} = \\delta_n` whose rest frequency lies
    in the band is included (from *freq_mhz* edges or ``nu_lo_mhz``/``nu_hi_mhz``).
    """
    nu = np.asarray(freq_mhz, dtype=np.float64).reshape(-1)
    if nu.size < 1:
        raise ValueError("freq_mhz must have at least one channel")
    lo = float(nu[0]) if nu_lo_mhz is None else float(nu_lo_mhz)
    hi = float(nu[-1]) if nu_hi_mhz is None else float(nu_hi_mhz)
    if lo > hi:
        lo, hi = hi, lo

    if transitions is None:
        if species == "carbon":
            transitions = carbon_rrl_transitions_in_frequency_band_mhz(
                lo, hi, delta_n=int(delta_n)
            )
        elif species == "hydrogen":
            if int(delta_n) != 1:
                raise ValueError("hydrogen band enumeration supports delta_n=1 only")
            transitions = hydrogen_rrl_alpha_transitions_in_frequency_band_mhz(lo, hi)
        else:
            raise ValueError(f"unknown species {species!r}")

    freq_fn = (
        carbon_rrl_frequency_mhz if species == "carbon" else hydrogen_rrl_frequency_mhz
    )
    spec = np.zeros(nu.shape, dtype=np.float64)
    sig = float(sigma_mhz)
    pk = float(peak_k)
    for n_upper, n_lower in transitions:
        nu0 = freq_fn(n_upper, n_lower)
        spec += pk * np.exp(-0.5 * ((nu - nu0) / sig) ** 2)
    return spec


@jax.tree_util.register_pytree_node_class
class ULSAPlusEnvelopeSky:
    """
    ULSA plus a **smooth** RRL envelope on catalog positions (no Rydberg lines in
    ``get_alm``). Used for the beam-resolution Croissant convolution stage of
    :class:`RRLAnalysisPipeline`; lines are added after resampling to the fine grid.
    """

    def __init__(
        self,
        ulsa_fname: str,
        rrl_catalog_fname: str,
        lmax: int,
        *,
        sim_freq_mhz: jnp.ndarray | np.ndarray | None = None,
        spot_sigma_deg: float = 0.25,
        gas_case: GasCaseName = "cold",
        envelope_params: Vydula2024EnvelopeParams | None = None,
        envelope_nu_ref_mhz: float = 12.5,
        envelope_sigma_mhz: float = 3.0,
        envelope_amplitude_k: float = 0.5,
        envelope_weight_fn=None,
    ):
        self._ulsa = FitsSky(ulsa_fname, lmax=lmax)
        self.Nside = int(self._ulsa.Nside)
        self.lmax = int(lmax)
        self.frame = "galactic"
        self.gas_case: GasCaseName = str(gas_case).lower()  # type: ignore[assignment]
        if envelope_params is not None:
            self.envelope_params = envelope_params
        elif self.gas_case == "gaussian":
            self.envelope_params = None
        else:
            self.envelope_params = vydula2024_envelope_params_from_gas_case(
                self.gas_case  # type: ignore[arg-type]
            )
        self.envelope_nu_ref_mhz = float(envelope_nu_ref_mhz)
        self.envelope_sigma_mhz = float(envelope_sigma_mhz)
        self.envelope_amplitude_k = float(envelope_amplitude_k)
        self._envelope_weight_fn = envelope_weight_fn or rrl_smooth_envelope_weight_mhz

        fn = np.asarray(self._ulsa.freq, dtype=np.float64)
        native_alm = np.asarray(self._ulsa.mapalm)
        if sim_freq_mhz is not None:
            ft = np.asarray(sim_freq_mhz, dtype=np.float64).reshape(-1)
            interp_alm = _interp_mapalm_native_to_target_mhz(native_alm, fn, ft)
            self.mapalm_ulsa = jnp.asarray(interp_alm, dtype=jnp.complex128)
            self.freq = jnp.asarray(ft, dtype=jnp.float64)
        else:
            self.mapalm_ulsa = jnp.asarray(native_alm, dtype=jnp.complex128)
            self.freq = self._ulsa.freq

        lon_deg, lat_deg = load_rrl_region_positions_gal_deg(rrl_catalog_fname)
        spot_map = _gaussian_spherical_map(
            self.Nside, lon_deg, lat_deg, spot_sigma_deg
        )
        self.rrl_spot_map = np.asarray(spot_map, dtype=np.float64)
        alm_sp = hp.map2alm(spot_map, lmax=self.lmax)
        self.alm_spatial = jnp.asarray(alm_sp, dtype=jnp.complex128)

    def _envelope_weight(self, ndx: jnp.ndarray) -> jnp.ndarray:
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        nu = np.asarray(self.freq, dtype=np.float64)[np.asarray(ndx, dtype=int)]
        w = self._envelope_weight_fn(
            nu,
            nu_ref_mhz=self.envelope_nu_ref_mhz,
            sigma_mhz=self.envelope_sigma_mhz,
            amplitude_k=self.envelope_amplitude_k,
            gas_case=self.gas_case,
            envelope_params=self.envelope_params,
        )
        return jnp.asarray(w, dtype=jnp.float64)

    def get_alm(self, ndx, freq=None):
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        base = self.mapalm_ulsa[ndx]
        w = self._envelope_weight(ndx)
        return base + w[:, None] * self.alm_spatial[None, :]

    def envelope_brightness_map_K(self, freq_index: int) -> np.ndarray:
        """Healpix map of the smooth envelope term at one frequency channel."""
        ndx = jnp.atleast_1d(jnp.asarray(int(freq_index)))
        w = self._envelope_weight(ndx)
        w0 = float(np.asarray(w[0], dtype=np.float64))
        alm = np.asarray(self.alm_spatial, dtype=np.complex128)
        m = hp.alm2map(alm, nside=int(self.Nside), lmax=int(self.lmax))
        return (w0 * np.real(m)).astype(np.float64)

    def tree_flatten(self):
        children = (self.mapalm_ulsa, self.alm_spatial)
        ep = self.envelope_params
        ep_tuple = None if ep is None else (
            ep.Te_k,
            ep.tau_lte,
            ep.T100_k,
            ep.beta_cont,
            ep.f_tr,
            ep.bn,
            ep.beta_n_coeffs,
            ep.nu100_mhz,
        )
        aux = (
            self.Nside,
            self.lmax,
            tuple(np.asarray(self.freq).tolist()),
            self.gas_case,
            ep_tuple,
            self.envelope_nu_ref_mhz,
            self.envelope_sigma_mhz,
            self.envelope_amplitude_k,
            self.frame,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mapalm_ulsa, alm_spatial = children
        Nside, lmax, freq_t, gas_case, ep_tuple, nu_ref, sig, amp, frame = aux_data
        sky = cls.__new__(cls)
        sky.Nside = Nside
        sky.lmax = lmax
        sky.freq = jnp.asarray(freq_t)
        sky.frame = frame
        sky.mapalm_ulsa = mapalm_ulsa
        sky.alm_spatial = alm_spatial
        sky.gas_case = gas_case
        if ep_tuple is None:
            sky.envelope_params = None
        else:
            sky.envelope_params = Vydula2024EnvelopeParams(
                Te_k=ep_tuple[0],
                tau_lte=ep_tuple[1],
                T100_k=ep_tuple[2],
                beta_cont=ep_tuple[3],
                f_tr=ep_tuple[4],
                bn=ep_tuple[5],
                beta_n_coeffs=ep_tuple[6],
                nu100_mhz=ep_tuple[7],
            )
        sky.envelope_nu_ref_mhz = nu_ref
        sky.envelope_sigma_mhz = sig
        sky.envelope_amplitude_k = amp
        sky._envelope_weight_fn = rrl_smooth_envelope_weight_mhz
        sky._ulsa = None
        sky.rrl_spot_map = None
        return sky


def build_ulsa_rrl_sky(
    lmax: int,
    *,
    ulsa_path: str | None = None,
    rrl_catalog_path: str | None = None,
    lusee_drive_dir: str | None = None,
    **rrl_sky_kwargs,
) -> ULSAPlusRRLSky:
    """Construct :class:`ULSAPlusRRLSky` using default LUSEE drive layout when paths are omitted."""
    drive = lusee_drive_dir or os.environ.get("LUSEE_DRIVE_DIR")
    ulsa = ulsa_path or default_ulsa_path(drive)
    rrl = rrl_catalog_path or default_rrl_catalog_path(drive)
    return ULSAPlusRRLSky(ulsa, rrl, lmax, **rrl_sky_kwargs)
