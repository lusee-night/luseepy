from functools import partial

from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, rot2eul
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
    - If extra_opts["use_observer_frame"] is True (default), sky is rotated to observer at each
      time with the same healpy Rotator as DefaultSimulator and beam is used in topo (observer);
      then convolve is called per (time, frequency). This keeps sky and beam in the same frame
      as Default and avoids last-frequency-bin differences from MCMF/phase conventions.
    - If use_observer_frame is False, the MCMF path is used: beam is transformed topo0→MCMF;
      at each time the sky is rotated to observer (same healpy rotation as Default), then
      observer→MCMF via s2fft, and convolve(beam_mcmf, sky_mcmf_ti) with no phase. This
      matches Default without using rot_alm_z (avoids phase convention mismatch).
    - Output layout is (N_times, N_combos, N_freq).

    :param obs: Observation (time range, deltaT_sec, lun_lat_deg, lun_long_deg)
    :param beams: Instrument beams [luseepy.beam class]
    :param sky_model: Sky model [luseepy.skymodels class]
    :param combinations: Beam combination indices [(0,0),(1,1),(0,2),(1,3),(1,2)]
    :param lmax: Maximum l
    :param Tground: Ground temperature [K]
    :param freq: Frequencies in MHz (from config / obs)
    :param cross_power: BeamCouplings for cross terms [luseepy.BeamCouplings class]
    :param extra_opts: optional dict, e.g. cache_transform, dump_beams (see DefaultSimulator)
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
        if self.extra_opts.get("use_observer_frame", True):
            self.result = self._simulate_observer_frame(times, ntimes)
        else:
            self.result = self._simulate_croissant_mcmf(times, ntimes, delta_t)
        return self.result

    def _simulate_observer_frame(self, times, ntimes):
        """Use same frame as DefaultSimulator: sky rotated to observer at each time, beam in topo.
        Convolve per (time, frequency) so sky and beam are in the same frame before convolution."""
        cache_fn = self.extra_opts.get("cache_transform")
        if cache_fn and os.path.isfile(cache_fn):
            lzl, bzl, lyl, byl = pickle.load(open(cache_fn, "br"))
            if len(lzl) != ntimes:
                raise RuntimeError("Cache file mix-up. Array wrong length!")
        else:
            lzl, bzl = self.obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
            lyl, byl = self.obs.get_l_b_from_alt_az(0.0, 0.0, times)
            if cache_fn:
                pickle.dump((lzl, bzl, lyl, byl), open(cache_fn, "bw"))

        sky_gal = self.sky_model.get_alm(self.freq_ndx_sky, self.freq)
        # Single-time no-rotation phases for croissant convolve
        phases_ones = jnp.ones((1, 2 * self.lmax + 1), dtype=jnp.complex128)
        norm_factor = 4.0 * np.pi
        combo_results = []
        plot_done = False
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            beam_2d = np.stack([
                healpy_packed_alm_to_croissant_2d(br_, self.lmax) for br_ in beamreal
            ])
            beam_alm_jax = jnp.array(beam_2d)
            T = np.zeros((ntimes, self.Nfreq), dtype=np.float64)
            for ti in range(ntimes):
                lz, bz, ly, by = lzl[ti], bzl[ti], lyl[ti], byl[ti]
                zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
                yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
                xhat = np.cross(yhat, zhat)
                R = np.array([xhat, yhat, zhat]).T
                a, b, g = rot2eul(R)
                rot = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
                sky_ti = [rot.rotate_alm(s_) for s_ in sky_gal]
                if ti == 0 and self.extra_opts.get("plot_sky_and_beam") and not plot_done:
                    nside = getattr(self.sky_model, "Nside", 64)
                    self._plot_sky_beam_healpix(
                        sky_ti[0], beamreal[0], nside, self.lmax,
                        outpath="sky_beam_healpix_cro.png", title_prefix="Cro ",
                    )
                    plot_done = True
                sky_2d_ti = np.stack([
                    healpy_packed_alm_to_croissant_2d(s_, self.lmax) for s_ in sky_ti
                ])
                sky_alm_ti = jnp.array(sky_2d_ti)
                for fi in range(self.Nfreq):
                    vis = crojax.simulator.convolve(
                        beam_alm_jax[fi : fi + 1],
                        sky_alm_ti[fi : fi + 1],
                        phases_ones,
                    )
                    v = np.asarray(vis.real).reshape(-1)[0]
                    T[ti, fi] = v / norm_factor + self.Tground * groundPowerReal[fi]
            combo_results.append((T, None))
            if ci != cj:
                beamimag_2d = np.stack([
                    healpy_packed_alm_to_croissant_2d(bi_, self.lmax) for bi_ in beamimag
                ])
                beamimag_jax = jnp.array(beamimag_2d)
                Timag = np.zeros((ntimes, self.Nfreq), dtype=np.float64)
                for ti in range(ntimes):
                    lz, bz, ly, by = lzl[ti], bzl[ti], lyl[ti], byl[ti]
                    zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
                    yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
                    xhat = np.cross(yhat, zhat)
                    R = np.array([xhat, yhat, zhat]).T
                    a, b, g = rot2eul(R)
                    rot = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
                    sky_ti = [rot.rotate_alm(s_) for s_ in sky_gal]
                    sky_2d_ti = np.stack([
                        healpy_packed_alm_to_croissant_2d(s_, self.lmax) for s_ in sky_ti
                    ])
                    sky_alm_ti = jnp.array(sky_2d_ti)
                    for fi in range(self.Nfreq):
                        vis_imag = crojax.simulator.convolve(
                            beamimag_jax[fi : fi + 1],
                            sky_alm_ti[fi : fi + 1],
                            phases_ones,
                        )
                        vi = np.asarray(vis_imag.real).reshape(-1)[0]
                        Timag[ti, fi] = vi / norm_factor + self.Tground * groundPowerImag[fi]
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

    def _simulate_croissant_mcmf(self, times, ntimes, delta_t):
        """Simple MCMF pipeline: sky gal→MCMF once, beam topo→MCMF once, rot_alm_z phases for time
        evolution, then convolve. Does not match Default (phase/frame conventions differ); use
        use_observer_frame=True to match Default."""
        topo = LunarTopo(obstime=times[0], location=self.obs.loc)
        sim_L = self.lmax + 1
        eul_topo, dl_topo = crojax.rotations.generate_euler_dl(
            self.lmax, topo, "mcmf"
        )
        eul_gal, dl_gal = crojax.rotations.generate_euler_dl(
            self.lmax, "galactic", "mcmf"
        )
        gal2mcmf = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_gal,
            dl_array=dl_gal,
        )
        topo2mcmf = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_topo,
            dl_array=dl_topo,
        )
        sky_gal = self.sky_model.get_alm(self.freq_ndx_sky, self.freq)
        sky_2d = np.stack([
            healpy_packed_alm_to_croissant_2d(s_, self.lmax) for s_ in sky_gal
        ])
        sky_mcmf = jax.vmap(gal2mcmf)(jnp.array(sky_2d))
        phases = crojax.simulator.rot_alm_z(
            self.lmax, N_times=ntimes, delta_t=delta_t, world="moon"
        )
        norm_factor = 4.0 * np.pi
        combo_results = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            beam_2d = np.stack([
                healpy_packed_alm_to_croissant_2d(br_, self.lmax) for br_ in beamreal
            ])
            beam_mcmf = jax.vmap(topo2mcmf)(jnp.array(beam_2d))
            vis = crojax.simulator.convolve(beam_mcmf, sky_mcmf, phases)
            T = np.asarray(vis.real) / norm_factor + self.Tground * groundPowerReal
            combo_results.append((T, None))
            if ci != cj:
                beamimag_2d = np.stack([
                    healpy_packed_alm_to_croissant_2d(bi_, self.lmax) for bi_ in beamimag
                ])
                beamimag_mcmf = jax.vmap(topo2mcmf)(jnp.array(beamimag_2d))
                vis_imag = crojax.simulator.convolve(beamimag_mcmf, sky_mcmf, phases)
                Timag = np.asarray(vis_imag.real) / norm_factor + self.Tground * groundPowerImag
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