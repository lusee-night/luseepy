"""
Staged RRL analysis: ULSA + smooth catalog envelope → beam convolution (beam ν grid)
→ resample to fine Δν → add Rydberg line spectrum.

Positions come from the RRL FITS catalog only; envelope shape and line physics are
set in code (Vydula+2024 Eq. 7 via :func:`rrl_envelope_T_rrl_k_mhz`, YAML ``gas_case``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .frequencies import fine_uniform_frequency_mhz
from .RRLSkyModels import (
    GasCaseName,
    RRL_DEFAULT_LINE_PEAK_K,
    RRL_DEFAULT_LINE_SIGMA_MHZ,
    ULSAPlusEnvelopeSky,
    Vydula2024EnvelopeParams,
    default_rrl_catalog_path,
    default_ulsa_path,
    rydberg_line_spectrum_mhz,
)


@dataclass
class RRLPipelineStages:
    """Outputs from each stage of :meth:`RRLAnalysisPipeline.run`."""

    freq_beam_mhz: np.ndarray
    waterfall_beam: np.ndarray
    freq_fine_mhz: np.ndarray
    waterfall_resampled: np.ndarray
    line_spectrum_fine_k: np.ndarray
    waterfall_final: np.ndarray
    alpha_transitions: tuple[tuple[int, int], ...]
    envelope_sky: ULSAPlusEnvelopeSky


def resample_waterfall_frequency(
    waterfall: np.ndarray,
    freq_in_mhz: np.ndarray,
    freq_out_mhz: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate a simulator waterfall along the frequency axis.

    *waterfall* shape ``(N_times, N_combos, N_freq_in)``.
    """
    w = np.asarray(waterfall, dtype=np.float64)
    fin = np.asarray(freq_in_mhz, dtype=np.float64).reshape(-1)
    fout = np.asarray(freq_out_mhz, dtype=np.float64).reshape(-1)
    if w.ndim != 3:
        raise ValueError(f"expected 3D waterfall, got shape {w.shape}")
    if fin.size != w.shape[2]:
        raise ValueError(
            f"freq_in length {fin.size} != waterfall freq axis {w.shape[2]}"
        )
    out = np.zeros((w.shape[0], w.shape[1], fout.size), dtype=np.float64)
    for ti in range(w.shape[0]):
        for ci in range(w.shape[1]):
            out[ti, ci, :] = np.interp(fout, fin, w[ti, ci, :])
    return out


class RRLAnalysisPipeline:
    """
    Run the staged ULSA + RRL analysis through Croissant beam convolution.

    1. **Sky (beam ν):** ULSA + smooth envelope on catalog positions
       (:class:`ULSAPlusEnvelopeSky`).
    2. **Convolve:** :class:`CroSimulator` at native / resampled beam frequencies.
    3. **Resample:** convolved waterfall to ``fine_step_khz`` (default 0.5 kHz).
    4. **Lines:** add carbon Cα :func:`rydberg_line_spectrum_mhz` on the fine grid
       (Vydula+2024 Eq. 1–2, all lines in band when ``alpha_transitions`` is ``None``).
    """

    def __init__(
        self,
        lmax: int,
        *,
        ulsa_path: str | None = None,
        rrl_catalog_path: str | None = None,
        lusee_drive_dir: str | None = None,
        fine_step_khz: float = 0.5,
        spot_sigma_deg: float = 0.35,
        gas_case: GasCaseName = "cold",
        envelope_params: Vydula2024EnvelopeParams | None = None,
        envelope_nu_ref_mhz: float = 12.5,
        envelope_sigma_mhz: float = 3.0,
        envelope_amplitude_k: float = 0.5,
        envelope_weight_fn=None,
        rrl_sigma_mhz: float = RRL_DEFAULT_LINE_SIGMA_MHZ,
        rrl_peak_k: float = RRL_DEFAULT_LINE_PEAK_K,
        alpha_transitions: Sequence[tuple[int, int]] | None = None,
    ):
        self.lmax = int(lmax)
        drive = lusee_drive_dir or None
        self.ulsa_path = ulsa_path or default_ulsa_path(drive)
        self.rrl_catalog_path = rrl_catalog_path or default_rrl_catalog_path(drive)
        self.fine_step_khz = float(fine_step_khz)
        self.spot_sigma_deg = float(spot_sigma_deg)
        self.gas_case: GasCaseName = str(gas_case).lower()  # type: ignore[assignment]
        self.envelope_params = envelope_params
        self.envelope_nu_ref_mhz = float(envelope_nu_ref_mhz)
        self.envelope_sigma_mhz = float(envelope_sigma_mhz)
        self.envelope_amplitude_k = float(envelope_amplitude_k)
        self.envelope_weight_fn = envelope_weight_fn
        self.rrl_sigma_mhz = float(rrl_sigma_mhz)
        self.rrl_peak_k = float(rrl_peak_k)
        # None → all carbon Cα lines in the run band (Vydula+2024 Eq. 1–2)
        self.alpha_transitions = (
            None if alpha_transitions is None else tuple(alpha_transitions)
        )
        self.stages: RRLPipelineStages | None = None

    def build_envelope_sky(self, freq_mhz: np.ndarray) -> ULSAPlusEnvelopeSky:
        """ULSA + smooth RRL envelope on the given frequency grid (typically ``beam.freq``)."""
        return ULSAPlusEnvelopeSky(
            self.ulsa_path,
            self.rrl_catalog_path,
            self.lmax,
            sim_freq_mhz=np.asarray(freq_mhz, dtype=np.float64),
            spot_sigma_deg=self.spot_sigma_deg,
            gas_case=self.gas_case,
            envelope_params=self.envelope_params,
            envelope_nu_ref_mhz=self.envelope_nu_ref_mhz,
            envelope_sigma_mhz=self.envelope_sigma_mhz,
            envelope_amplitude_k=self.envelope_amplitude_k,
            envelope_weight_fn=self.envelope_weight_fn,
        )

    def fine_frequency_mhz(self, nu_lo_mhz: float, nu_hi_mhz: float) -> np.ndarray:
        """Uniform fine grid spanning the beam band at ``fine_step_khz``."""
        return np.asarray(
            fine_uniform_frequency_mhz(
                float(nu_lo_mhz),
                float(nu_hi_mhz),
                step_khz=self.fine_step_khz,
            ),
            dtype=np.float64,
        )

    def run(
        self,
        obs,
        beam,
        *,
        CroSimulator,
        Tground: float = 0.0,
        combinations: Sequence[tuple[int, int]] | None = None,
        times=None,
    ) -> RRLPipelineStages:
        """
        Execute all pipeline stages and store the result on ``self.stages``.

        *beam* must already be sampled on the desired frequency axis (same as
        will be passed to :class:`CroSimulator`).
        """
        if CroSimulator is None:
            raise RuntimeError("CroSimulator is not available (install croissant and s2fft)")

        freq_beam = np.asarray(beam.freq, dtype=np.float64).reshape(-1)
        if freq_beam.size < 1:
            raise ValueError("beam must have at least one frequency sample")

        sky = self.build_envelope_sky(freq_beam)
        combs = [(0, 0)] if combinations is None else [tuple(int(x) for x in c) for c in combinations]

        sim = CroSimulator(
            obs,
            [beam],
            sky,
            Tground=float(Tground),
            combinations=combs,
            freq=jnp.asarray(freq_beam, dtype=jnp.float64),
            lmax=self.lmax,
            extra_opts={},
        )
        if times is None:
            times = obs.times
        wfall_beam = np.asarray(sim.simulate(times=times), dtype=np.float64)

        nu_lo, nu_hi = float(freq_beam[0]), float(freq_beam[-1])
        freq_fine = self.fine_frequency_mhz(nu_lo, nu_hi)
        wfall_fine = resample_waterfall_frequency(wfall_beam, freq_beam, freq_fine)
        line_spec = rydberg_line_spectrum_mhz(
            freq_fine,
            self.alpha_transitions,
            nu_lo_mhz=nu_lo,
            nu_hi_mhz=nu_hi,
            species="carbon",
            delta_n=1,
            sigma_mhz=self.rrl_sigma_mhz,
            peak_k=self.rrl_peak_k,
        )
        wfall_final = wfall_fine + line_spec.reshape(1, 1, -1)

        self.stages = RRLPipelineStages(
            freq_beam_mhz=freq_beam,
            waterfall_beam=wfall_beam,
            freq_fine_mhz=freq_fine,
            waterfall_resampled=wfall_fine,
            line_spectrum_fine_k=line_spec,
            waterfall_final=wfall_final,
            alpha_transitions=self.alpha_transitions or (),
            envelope_sky=sky,
        )
        return self.stages

    def write_final_fits(self, out_file: str, obs) -> None:
        """Write the fine-resolution final waterfall (same HDU layout as CroSimulator)."""
        import fitsio

        if self.stages is None:
            raise RuntimeError("call run() before write_final_fits()")
        st = self.stages
        fits = fitsio.FITS(out_file, "rw", clobber=True)
        header = {
            "version": 0.2,
            "lunar_day": obs.time_range,
            "lun_lat_deg": obs.lun_lat_deg,
            "lun_long_deg": obs.lun_long_deg,
            "lun_height_m": obs.lun_height_m,
            "deltaT_sec": obs.deltaT_sec,
            "rrl_pipeline": "staged_envelope_conv_lines",
            "fine_step_khz": self.fine_step_khz,
            "envelope_gas_case": self.gas_case,
        }
        fits.write(
            np.asarray(st.waterfall_final, dtype=np.float32),
            header=header,
            extname="data",
        )
        fits.write(np.asarray(st.freq_fine_mhz, dtype=np.float64), extname="freq")
        fits.write(np.array([[0, 0]], dtype=np.int32), extname="combinations")
        fits.close()


def build_rrl_analysis_pipeline(
    lmax: int,
    *,
    lusee_drive_dir: str | None = None,
    **kwargs,
) -> RRLAnalysisPipeline:
    """Construct :class:`RRLAnalysisPipeline` with default drive paths."""
    return RRLAnalysisPipeline(
        lmax,
        lusee_drive_dir=lusee_drive_dir,
        **kwargs,
    )
