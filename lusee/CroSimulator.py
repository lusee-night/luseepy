from functools import partial

from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase
import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os
import jax.numpy as jnp
import croissant as cro
import croissant.jax as crojax
import jax
from lunarsky import LunarTopo

import s2fft

"""
CroSimulator: same inputs as DefaultSimulator (beam, sky, obs, etc.) but uses
the Croissant engine for the actual simulation (MCMF frame, rot_alm_z phases,
crojax.simulator.convolve). Freq, time range, and antenna location come from
the observation object (config). Croissant currently supports single
polarization / single dipole per beam; one beam combination at a time
"""


def healpy_packed_alm_to_croissant_2d(packed_alm, lmax):
    """
    Convert healpy packed alm (1D complex) to croissant (l, m) 2D format.
    Output shape (lmax+1, 2*lmax+1) with m_index = m + lmax (m from -lmax to lmax).
    Healpy stores only m >= 0; negative m are filled via a_l,-m = (-1)^m conj(a_lm).

    :param packed_alm: 1D complex array, healpy convention (e.g. from Beam.get_healpix_alm).
    :param lmax: Maximum l.
    :returns: 2D complex array shape (lmax+1, 2*lmax+1).
    """
    out = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    for ell in range(lmax + 1):
        for m in range(0, ell + 1):
            idx = hp.sphtfunc.Alm.getidx(lmax, ell, m)
            val = packed_alm[idx]
            out[ell, lmax + m] = val
            if m > 0:
                out[ell, lmax - m] = (-1) ** m * np.conj(val)
    return out




class CroSimulator(SimulatorBase):
    """
    Croissant simulator: same inputs as DefaultSimulator (obs, beams, sky_model,
    combinations, freq, lmax) 

    - Freq, time grid, and antenna location are taken from the observation
      object (set from config). [luseepy.observation class]
    - Lunar topo frame is built from obs class (obstime=first time, location=obs.loc).
    - Beam and sky are transformed to MCMF; time evolution uses
      crojax.simulator.rot_alm_z (moon sidereal rotation).
    - Croissant handles single polarization / single dipole per beam; each
      combination is convolved separately to match DefaultSimulator output shape.

    :param obs: Observation (time range, deltaT_sec, lun_lat_deg, lun_long_deg)
    :param beams: Instrument beams [luseepy.beam class]
    :param sky_model: Sky model [luseepy.skymodels class]
    :param combinations: Beam combination indices [(0,0),(1,1),(0,2),(1,3),(1,2)]
    :param lmax: Maximum l
    :param Tground: Ground temperature [K]
    :param freq: Frequencies in MHz (from config / obs)
    :param cross_power: BeamCouplings for cross terms [luseepy.BeamCouplings class]
    :param extra_opts: e.g. cache_transform, dump_beams
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None,
                  lmax = 128, cross_power = None,
                  extra_opts = {}):
        super().__init__(obs, beams, sky_model, Tground, combinations, freq)
        self.lmax = lmax
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self.prepare_beams (beams, combinations)

            
                                
    def simulate(self, times=None):
        """
        Simulate using Croissant.
        freq, time grid, and antenna location from
        observation; beam and sky transformed to MCMF; rot_alm_z phases;
        crojax.simulator.convolve for sky and beam convolution.
        :param times: List of times; if None, use self.obs.times (from config).
        :returns: Waterfall (N_times, N_combos_with_imag, N_freq), same as DefaultSimulator.
        """
        if times is None:
            times = self.obs.times
        ntimes = len(times)
        delta_t = float(self.obs.deltaT_sec)

        if self.sky_model.frame != "galactic":
            raise NotImplementedError(
                f"CroSimulator requires galactic sky frame, got {self.sky_model.frame}"
            )
        self.result = self._simulate_croissant_mcmf(times, ntimes, delta_t)
        return self.result

    def _simulate_croissant_mcmf(self, times, ntimes, delta_t):
        """Croissant pipeline: location/topo from obs class, MCMF frame, rot_alm_z, convolve."""


        # Location and topo from observation class
        topo = LunarTopo(obstime=times[0], location=self.obs.loc)
        sim_L = self.lmax + 1
        # Phases for moon sidereal rotation for all times at once (use N_times, delta_t so JAX gets numeric types)
        phases = crojax.simulator.rot_alm_z(
            self.lmax, N_times=ntimes, delta_t=delta_t, world="moon"
        )

        # Galactic -> MCMF and Topo -> MCMF transforms
        eul_topo, dl_topo = crojax.rotations.generate_euler_dl(
            self.lmax, topo, "mcmf"
        )
        eul_gal, dl_gal = crojax.rotations.generate_euler_dl(
            self.lmax, "galactic", "mcmf"
        )
        topo2mcmf = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_topo,
            dl_array=dl_topo,
        )
        gal2mcmf = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_gal,
            dl_array=dl_gal,
        )

        # Sky: get alm (galactic), convert to 2D, transform to MCMF
        sky = self.sky_model.get_alm(self.freq_ndx_sky, self.freq)
        sky_2d = np.stack([
            healpy_packed_alm_to_croissant_2d(s_, self.lmax) for s_ in sky
        ])
        sky_alm_jax = jnp.array(sky_2d)
        sky_alm_mcmf = jax.vmap(gal2mcmf)(sky_alm_jax)

        # Per combination: beam to MCMF, convolve(beam_mcmf, sky_mcmf, phases), normalize by 4pi, ground
        # Normalization 4*pi matches DefaultSimulator (mean_alm divides by 4*pi).
        # Collect (ntimes, N_freq) per combo then build (N_times, N_outputs, N_freq)
        norm_factor = 4.0 * np.pi
        combo_results = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            beam_2d = np.stack([
                healpy_packed_alm_to_croissant_2d(br_, self.lmax) for br_ in beamreal
            ])
            beam_alm_jax = jnp.array(beam_2d)
            beam_alm_mcmf = jax.vmap(topo2mcmf)(beam_alm_jax)
            vis = crojax.simulator.convolve(
                beam_alm_mcmf, sky_alm_mcmf, phases
            )
            T = np.asarray(vis.real / norm_factor) + self.Tground * groundPowerReal
            combo_results.append((T, None))
            if ci != cj:
                beamimag_2d = np.stack([
                    healpy_packed_alm_to_croissant_2d(bi_, self.lmax) for bi_ in beamimag
                ])
                beamimag_mcmf = jax.vmap(topo2mcmf)(jnp.array(beamimag_2d))
                vis_imag = crojax.simulator.convolve(
                    beamimag_mcmf, sky_alm_mcmf, phases
                )
                Timag = np.asarray(vis_imag.real / norm_factor) + self.Tground * groundPowerImag
                combo_results[-1] = (T, Timag)
        wfall = []
        for ti in range(ntimes):
            res = []
            for T, Timag in combo_results:
                res.append(T[ti])
                if Timag is not None:
                    res.append(Timag[ti])
            wfall.append(res)
        return np.array(wfall)