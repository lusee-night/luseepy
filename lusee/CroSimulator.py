from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, mean_alm, rot2eul
import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os
import croissant as cro
import croissant.jax as crojax
import jax.numpy as jnp

'''
Simulator object that uses Croissant engine.
Uses SimulatorBase (beams in alm from prepare_beams), sky in alm (with rotation
applied per time without changing sky_model), and crojax.simulator.convolve().
'''


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
    Croissant Simulator in luseepy as an alternative to DefaultSimulator.
    

    :param obs: Observation parameters, from lusee.observation class
    :type obs: class
    :param beams: Instrument beams, from lusee.beam class -- for Croissant, this needs to be in alm format
    :type beams: class
    :param sky_model: Simulated model of the sky, from lusee.skymodels
    :type sky_model: class
    :param combinations: Indices for beam combinations/cross correlations to simulate
    :type combinations: list[tuple]
    :param lmax: Maximum l value of beams
    :type lmax: int
    :param taper: Instrument beam taper
    :type taper: float
    :param Tground: Temperature of lunar ground
    :type Tground: float
    :param freq: Frequencies at which instrument observes sky in MHz. If empty, taken from lusee.beam class.
    :type freq: list[float]
    :param cross_power: Beam coupling model for cross-power terms. If empty, uses :class:`lusee.BeamCouplings`.
    :type cross_power: BeamCouplings
    :param beam_smooth: Standard deviation of Gaussian filter for beam smoothing (in pixel units)
    :type beam_smooth: float
    :param extra_opts: Extra options for simulation. Supports "dump_beams" (saves instrument beams to file)
        and "cache_transform" (loads/saves beam transformations from file).
    :type extra_opts: dict
    
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None,
                  lmax = 128, taper = 0.03, cross_power = None, beam_smooth = None,
                  extra_opts = {}):
        super().__init__(obs, beams, sky_model, Tground, combinations, freq)
        self.lmax = lmax
        self.taper = taper
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self.beam_smooth = beam_smooth
        self.prepare_beams (beams, combinations)

            
                                
    def simulate(self, times=None):
        """
        Simulate using SimulatorBase: beams in alm (from prepare_beams), sky in alm
        (from sky_model.get_alm), and crojax.simulator.convolve() for the beam-sky convolution.

        :param times: List of times, defaults to lusee.observation.times if empty
        :type times: list

        :returns: Waterfall style observation data for input times and self.freq
        :rtype: numpy array
        
        """
        if times is None:
            times = self.obs.times
        if self.sky_model.frame == "galactic":
            do_rot = True
            cache_fn = self.extra_opts.get("cache_transform")
            if (cache_fn is not None) and (os.path.isfile(cache_fn)):
                print(f"Loading cached transform from {cache_fn}...")
                lzl, bzl, lyl, byl = pickle.load(open(cache_fn, "br"))
                if len(lzl) != len(times):
                    print("Cache file mix-up. Array wrong length!")
                    raise RuntimeError("Cache file array length mismatch")
                have_transform = True
            else:
                have_transform = False

            if not have_transform:
                print("Getting pole transformations...")
                lzl, bzl = self.obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
                print("Getting horizon transformations...")
                lyl, byl = self.obs.get_l_b_from_alt_az(0.0, 0.0, times)
                if cache_fn is not None:
                    print(f"Saving transforms to {cache_fn}...")
                    pickle.dump((lzl, bzl, lyl, byl), open(cache_fn, "bw"))
        elif self.sky_model.frame == "MCMF":
            do_rot = False
        else:
            raise NotImplementedError(f"frame {self.sky_model.frame} not supported")

        # Phases: identity (shape 1 x (2*lmax+1)) because we rotate sky to observer frame each time
        phases = jnp.ones((1, 2 * self.lmax + 1), dtype=jnp.complex128)

        wfall = []
        Nt = len(times)
        for ti, t in enumerate(times):
            if ti % 100 == 0:
                print(f"{ti/Nt*100}% done ...")
            # Sky in alm from sky_model (unchanged API)
            sky = self.sky_model.get_alm(self.freq_ndx_sky, self.freq)
            # Apply rotation to observer frame if galactic (sky in alm, rotation only here)
            if do_rot:
                lz, bz, ly, by = lzl[ti], bzl[ti], lyl[ti], byl[ti]
                zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
                yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
                xhat = np.cross(yhat, zhat)
                assert np.abs(np.dot(zhat, yhat)) < 1e-10
                R = np.array([xhat, yhat, zhat]).T
                a, b, g = rot2eul(R)
                rot = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
                sky = [rot.rotate_alm(s_) for s_ in sky]
            # Convert sky to croissant 2D alm: (N_freq, lmax+1, 2*lmax+1)
            sky_2d = np.stack([healpy_packed_alm_to_croissant_2d(s_, self.lmax) for s_ in sky])
            sky_alm_jax = jnp.array(sky_2d)

            res = []
            for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
                # Beam in alm (from prepare_beams), convert to croissant 2D
                beam_2d = np.stack([healpy_packed_alm_to_croissant_2d(br_, self.lmax) for br_ in beamreal])
                beam_alm_jax = jnp.array(beam_2d)
                norm = crojax.alm.total_power(beam_alm_jax, self.lmax)
                vis = crojax.simulator.convolve(beam_alm_jax, sky_alm_jax, phases)
                T = np.asarray(vis[0].real / norm) + self.Tground * groundPowerReal
                res.append(T)
                if ci != cj:
                    beamimag_2d = np.stack([healpy_packed_alm_to_croissant_2d(bi_, self.lmax) for bi_ in beamimag])
                    beamimag_jax = jnp.array(beamimag_2d)
                    vis_imag = crojax.simulator.convolve(beamimag_jax, sky_alm_jax, phases)
                    Timag = np.asarray(vis_imag[0].real / norm) + self.Tground * groundPowerImag
                    res.append(Timag)
            wfall.append(res)
        self.result = np.array(wfall)
        return self.result
            
