"""
RRLSimulator: Croissant-based simulator (:class:`CroSimulator`) with the same
beam/observation wiring. Use :class:`lusee.RRLSkyModels.ULSAPlusRRLSky` as
``sky_model`` for ULSA plus catalogued RRL regions and Rydberg line frequencies.
"""

from __future__ import annotations

from .RRLSkyModels import ULSAPlusRRLSky, build_ulsa_rrl_sky

try:
    from .CroSimulator import CroSimulator as _CroSimulator
except (ModuleNotFoundError, ImportError) as e:
    if "croissant" in str(e).lower() or "s2fft" in str(e).lower():
        _CroSimulator = None
    else:
        raise


if _CroSimulator is not None:

    class RRLSimulator(_CroSimulator):
        """
        Same implementation as :class:`CroSimulator`; provided for a clear entry
        point when running ULSA + RRL sky models (:class:`ULSAPlusRRLSky`).

        Frequency grid, time sampling, and antenna location still come from
        ``obs``; beams are prepared identically to :class:`CroSimulator`.
        """

        @classmethod
        def build_ulsa_rrl_sky(
            cls,
            lmax: int,
            *,
            ulsa_path: str | None = None,
            rrl_catalog_path: str | None = None,
            lusee_drive_dir: str | None = None,
            **rrl_sky_kwargs,
        ) -> ULSAPlusRRLSky:
            """
            Convenience: resolve default drive paths and construct :class:`ULSAPlusRRLSky`.

            Parameters in ``rrl_sky_kwargs`` are forwarded to :class:`ULSAPlusRRLSky`
            (e.g. ``alpha_transitions``, ``rrl_sigma_mhz``, ``rrl_peak_k``,
            ``spot_sigma_deg``).
            """
            return build_ulsa_rrl_sky(
                lmax,
                ulsa_path=ulsa_path,
                rrl_catalog_path=rrl_catalog_path,
                lusee_drive_dir=lusee_drive_dir,
                **rrl_sky_kwargs,
            )

else:
    RRLSimulator = None  # type: ignore[misc, assignment]
