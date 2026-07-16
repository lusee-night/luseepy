from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .frequencies import FrequencyMap

import numpy as np
import jax.numpy as jnp
import healpy as hp
import fitsio
import sys
import pickle
import os
import json
import hashlib


def default_plot_sky_beam_dir():
    """Default save directory for plot_sky_and_beam (under luseepy/simulation/output/figures)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation", "output", "figures")
    )


def mean_alm(alm1, alm2, lmax):
    """
    Function that calculates the mean of the product of two a_lm arrrays
    
    :param alm1: First a_lm array
    :type alm1: array
    :param alm2: Second a_lm array
    :type alm1: array
    :param lmax: Maximum l value of beams
    :type lmax: int
    
    :returns: Average alm
    :rtype: float
    
    """
    alm1 = jnp.asarray(alm1)
    alm2 = jnp.asarray(alm2)
    prod = alm1*jnp.conj(alm2)
    sm = (jnp.real(prod[:lmax+1]).sum()+2*jnp.real(prod[lmax+1:]).sum())/(4*jnp.pi)
    return sm


def mean_alm_np(alm1, alm2, lmax):
    return np.asarray(mean_alm(alm1, alm2, lmax))

def get_R_gal_to_topo(lz, bz, ly, by):
    """
    Build 3x3 rotation matrix R such that v_topo = R @ v_gal.
    Same construction as DefaultSimulator (zenith zhat, north yhat from (l,b)).
    """
    zhat = jnp.array([jnp.cos(bz) * jnp.cos(lz), jnp.cos(bz) * jnp.sin(lz), jnp.sin(bz)])
    yhat = jnp.array([jnp.cos(by) * jnp.cos(ly), jnp.cos(by) * jnp.sin(ly), jnp.sin(by)])
    xhat = jnp.cross(yhat, zhat)
    return jnp.array([xhat, yhat, zhat]).T


def get_R_gal_to_topo_np(lz, bz, ly, by):
    return np.asarray(get_R_gal_to_topo(lz, bz, ly, by))


def get_topo_z_rotation_angles(obs, times):
    """
    Return z-rotation angles phi[i] (radians) of topo frame at times[i] relative to times[0].
    phi[0] = 0. Uses (l,b)(t) from observation so libration is included.
    """
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    R0 = get_R_gal_to_topo_np(lzl[0], bzl[0], lyl[0], byl[0])
    phis = np.zeros(len(times))
    for i in range(len(times)):
        Ri = get_R_gal_to_topo_np(lzl[i], bzl[i], lyl[i], byl[i])
        R_topo0_to_topo_i = Ri @ R0.T
        phis[i] = np.arctan2(R_topo0_to_topo_i[1, 0], R_topo0_to_topo_i[0, 0])
    return phis


def rot2eul(R):
    """
    Function that converts from rotation matrix to Euler angles
    
    :param R: Rotation matrix
    :type R: array
    
    :returns: Euler angles
    :rtype: numpy array
    
    """
    R = jnp.asarray(R)
    beta = -jnp.arcsin(R[2,0])
    alpha = jnp.arctan2(R[2,1]/jnp.cos(beta),R[2,2]/jnp.cos(beta))
    gamma = jnp.arctan2(R[1,0]/jnp.cos(beta),R[0,0]/jnp.cos(beta))
    return jnp.array((alpha, beta, gamma))


def rot2eul_np(R):
    return np.asarray(rot2eul(R))

def eul2rot(theta) :
    """
    Function that converts from Euler angles to rotation matrix
    
    :param theta: Euler angles
    :type theta: array
    
    :returns: Rotation matrix
    :rtype: numpy array
    
    """
    theta = jnp.asarray(theta)
    R = jnp.array([[jnp.cos(theta[1])*jnp.cos(theta[2]),       jnp.sin(theta[0])*jnp.sin(theta[1])*jnp.cos(theta[2]) - jnp.sin(theta[2])*jnp.cos(theta[0]),      jnp.sin(theta[1])*jnp.cos(theta[0])*jnp.cos(theta[2]) + jnp.sin(theta[0])*jnp.sin(theta[2])],
                  [jnp.sin(theta[2])*jnp.cos(theta[1]),       jnp.sin(theta[0])*jnp.sin(theta[1])*jnp.sin(theta[2]) + jnp.cos(theta[0])*jnp.cos(theta[2]),      jnp.sin(theta[1])*jnp.sin(theta[2])*jnp.cos(theta[0]) - jnp.sin(theta[0])*jnp.cos(theta[2])],
                  [-jnp.sin(theta[1]),                        jnp.sin(theta[0])*jnp.cos(theta[1]),                                                           jnp.cos(theta[0])*jnp.cos(theta[1])]])

    return R


def eul2rot_np(theta) :
    return np.asarray(eul2rot(theta))

    

class SimulatorBase:
    """
    Base Simulator class
    
    :param obs: Observation parameters, from lusee.observation class
    :type obs: class
    :param beams: Instrument beams, from lusee.beam class
    :type beams: class
    :param sky_model: Simulated model of the sky, from lusee.skymodels
    :type sky_model: class
    :param Tground: Temperature of lunar ground in Kelving
    :type Tground: float
    :param combinations: Indices for beam combinations/cross correlations to simulate
    :type combinations: list[tuple]
    :param freq: Frequencies at which instrument observes sky in MHz. If empty, taken from lusee.beam class.
    :type freq: list[float]
    
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None):

        self.obs = obs
        self.sky_model = sky_model
        self.Tground = Tground
        self.combinations = combinations
        self.result = None

        if freq is None:
            self.freq = beams[0].freq
        else:
            self.freq = freq

        ref_freq = np.asarray(beams[0].freq, dtype=float)
        for idx, b in enumerate(beams[1:], start=1):
            other = np.asarray(b.freq, dtype=float)
            if other.shape != ref_freq.shape or not np.allclose(other, ref_freq):
                raise ValueError(
                    f"All beams must share the same native frequency grid; "
                    f"beam[{idx}] (id={getattr(b, 'id', None)!r}) differs from "
                    f"beam[0] (id={getattr(beams[0], 'id', None)!r})."
                )
        try:
            self.freq_map_beam = FrequencyMap.build(self.freq, beams[0].freq)
        except ValueError as exc:
            raise ValueError(f"Beam frequency mismatch: {exc}") from exc
        self.freq_map_sky = None
        self.freq_map_sky = self.sky_freq_map(sky_model)

        self.Nfreq = len(self.freq)

    def sky_freq_map(self, sky_model):
        """FrequencyMap from ``self.freq`` onto ``sky_model``'s native grid.

        Returns None when the model implements ``get_alm_at_freq`` (closed-form
        evaluation needs no map). The map built at construction is reused when
        ``sky_model`` is the constructor sky model; an override sky passed to
        ``simulate(sky=...)`` gets a map built for its own native grid.
        """
        if hasattr(sky_model, "get_alm_at_freq"):
            return None
        if sky_model is self.sky_model and self.freq_map_sky is not None:
            return self.freq_map_sky
        try:
            return FrequencyMap.build(self.freq, getattr(sky_model, "freq", None))
        except ValueError as exc:
            raise ValueError(f"Sky-model frequency mismatch: {exc}") from exc

    def sky_alm_at_freq(self, sky_model, *, xp=np):
        """Sky alm rows evaluated at ``self.freq`` for the given sky model.

        Dispatches on the model itself: closed-form models evaluate exactly via
        ``get_alm_at_freq``; gridded models are interpolated with the map from
        :meth:`sky_freq_map`. ``xp`` (numpy or jax.numpy) selects the array
        namespace; pass ``jnp`` to keep traced values traceable.
        """
        fmap = self.sky_freq_map(sky_model)
        if fmap is None:
            return xp.asarray(sky_model.get_alm_at_freq(self.freq))
        native = sky_model.get_alm(fmap.source_indices)
        return fmap.from_unique(xp.asarray(native))

    @property
    def freq_ndx_beam(self):
        """Compat shim: one native beam-grid index per entry of ``self.freq``.

        Preserves the pre-interpolation contract (target order, duplicates
        kept), which is only defined when every target frequency snaps to a
        native bin; raises ValueError for off-grid targets. Prefer
        ``self.freq_map_beam`` for new code.
        """
        try:
            return self.freq_map_beam.per_target_indices()
        except ValueError as exc:
            raise ValueError(f"freq_ndx_beam: {exc} (see freq_map_beam)") from exc

    @property
    def freq_ndx_sky(self):
        """Compat shim: one native sky-grid index per entry of ``self.freq``.

        Returns ``None`` if the sky model provides ``get_alm_at_freq`` (no
        index mapping exists then); otherwise same contract as
        ``freq_ndx_beam``. Prefer ``self.freq_map_sky`` for new code.
        """
        if self.freq_map_sky is None:
            return None
        try:
            return self.freq_map_sky.per_target_indices()
        except ValueError as exc:
            raise ValueError(f"freq_ndx_sky: {exc} (see freq_map_sky)") from exc

    def simulate(self, times=None):
        """
        Main simulation function. Produces mock observation of sky model from self.sky_model with beams from self.beams

        :param times: List of times, defaults to lusee.observation.times if empty 
        :type times: list
        
        :returns: Waterfall style observation data for input times and self.freq
        :rtype: numpy array

        
        Note:
            This is a base class function and should be implemented in derived classes.
            The result should also be stored in self.result.
        """

        raise NotImplementedError("simulate() not implemented in base class")

    def _cache_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _transform_cache_force_recompute(self):
        return self._cache_bool(
            self.extra_opts.get(
                "force_recompute_cache_transform",
                self.extra_opts.get("recompute_cache_transform", False),
            )
        )

    def _transform_cache_key(self, times):
        time_jd = np.asarray([t.jd for t in times], dtype=float)
        payload = {
            "cache_version": 1,
            "time_jd": time_jd.tolist(),
            "lun_lat_deg": float(self.obs.lun_lat_deg),
            "lun_long_deg": float(self.obs.lun_long_deg),
            "lun_height_m": float(self.obs.lun_height_m),
            "deltaT_sec": float(self.obs.deltaT_sec),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def _transform_cache_file(self, times):
        cache_prefix = self.extra_opts.get("cache_transform")
        if cache_prefix is None:
            return None
        root, ext = os.path.splitext(cache_prefix)
        if ext == "":
            ext = ".pickle"
        return f"{root}__rot_transforms_{self._transform_cache_key(times)}{ext}"

    def _plot_sky_beam_healpix(self, sky_alm, beam_alm, nside, lmax, save_dir="output/figures", save_filename="sky_beam_healpix.png", title_prefix=""):
        """
        Plot sky and beam as healpix mollweide maps (for visual check before convolution).
        Call when extra_opts["plot_sky_and_beam"] is True. When using DefaultSimulator or
        CroSimulator, you can pass save_dir/save_filename via extra_opts["plot_dir"] and
        extra_opts["plot_filename"].
        :param sky_alm: Healpy packed alm (1D complex) for sky at one frequency
        :param beam_alm: Healpy packed alm (1D complex) for beam at one frequency
        :param nside: Healpix Nside for the map
        :param lmax: Maximum l for alm2map
        :param save_dir: Directory to save the figure (default: luseepy/simulation/output/figures)
        :param save_filename: File name for the saved figure (default: sky_beam_healpix.png)
        :param title_prefix: Optional prefix for plot title (e.g. simulator name)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        outpath = os.path.join(save_dir, save_filename)
        os.makedirs(save_dir, exist_ok=True)
        sky_map = hp.alm2map(np.asarray(sky_alm, dtype=np.complex128), nside, lmax=lmax)
        beam_map = hp.alm2map(np.asarray(beam_alm, dtype=np.complex128), nside, lmax=lmax)
        plt.figure(figsize=(12, 5))
        hp.mollview(sky_map, title=(title_prefix + " Sky").strip(), sub=(1, 2, 1))
        hp.mollview(beam_map, title=(title_prefix + " Beam").strip(), sub=(1, 2, 2))
        plt.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  plot_sky_and_beam: saved {outpath}")

    def _plot_sky_beam_maps(self, sky_map, beam_map, outpath="sky_beam_healpix.png", title_prefix=""):
        """
        Plot sky and beam as healpix mollweide maps (precomputed maps, e.g. from s2fft.inverse).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sky_map = np.asarray(sky_map).real
        beam_map = np.asarray(beam_map).real
        plt.figure(figsize=(12, 5))
        hp.mollview(sky_map, title=(title_prefix + " Sky").strip(), sub=(1, 2, 1))
        hp.mollview(beam_map, title=(title_prefix + " Beam").strip(), sub=(1, 2, 2))
        plt.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  plot_sky_and_beam: saved {outpath}")

    def write_fits(self, out_file):
        """
        Function that writes out instrument beam patterns from self.beams to fits file
        
        :param out_file: Name of the output file
        :type out_file: str

        """        
        if self.result is None:
            print ("Nothing to write")
            raise RuntimeError
        fits = fitsio.FITS(out_file,'rw',clobber=True)
        header = {
            "version":      0.1,
            "lunar_day":    self.obs.time_range,
            "lun_lat_deg":  self.obs.lun_lat_deg,
            "lun_long_deg": self.obs.lun_long_deg,
            "lun_height_m": self.obs.lun_height_m,
            "deltaT_sec":   self.obs.deltaT_sec
        }
        fits.write(np.asarray(self.result), header=header, extname='data')
        fits.write(np.asarray(self.freq), extname='freq')
        fits.write(np.array(self.combinations), extname='combinations')
        for i,b in enumerate(self.beams):
            ZRe_target = self.freq_map_beam.from_native(np.asarray(b.ZRe))
            ZIm_target = self.freq_map_beam.from_native(np.asarray(b.ZIm))
            fits.write(ZRe_target, extname=f'ZRe_{i}')
            fits.write(ZIm_target, extname=f'ZIm_{i}')

    def prepare_beams(self, beams, combinations):
        """
        Prepare beams for the simulator (shared by DefaultSimulator and CroSimulator).
        Expects the subclass to have set: lmax, cross_power,
        and optionally extra_opts (e.g. for "dump_beams"), before calling.

        :param beams: Instrument beams, from lusee.beam object
        :type beams: class
        :param combinations: Indices for beam combinations/cross correlations to simulate
        :type combinations: list[tuple]
        """
        
        self.beams = beams
        self.efbeams = []
        bomega = []
        self.combinations = [(int(i),int(j)) for i,j in combinations]
        fmap = self.freq_map_beam

        for i,j in self.combinations:
            bi , bj = beams[i], beams[j]
            print (f"  intializing beam combination {bi.id} x {bj.id} ...")
            gain_i = fmap.from_native(np.asarray(bi.gain_conv))
            gain_j = fmap.from_native(np.asarray(bj.gain_conv))
            norm = np.sqrt(gain_i * gain_j)
            beamreal_native, beamimag_native = bi.get_healpix_alm(
                self.lmax,
                freq_ndx=fmap.source_indices,
                other=bj,
                return_I_stokes_only=True,
                return_complex_components=True,
            )
            beamreal = np.asarray(fmap.from_unique(np.asarray(beamreal_native)))
            beamreal = beamreal * norm[:, None]
            if beamimag_native is not None:
                beamimag = np.asarray(fmap.from_unique(np.asarray(beamimag_native)))
                beamimag = beamimag * norm[:, None]
            else:
                beamimag = None

            if i==j:
                groundPowerReal = np.array([1-np.real(br[0])/np.sqrt(4*np.pi) for br in beamreal])
                beamimag = None
                groundPowerImag = 0.
            else:
                cross_power = self.cross_power.Ex_coupling(bi, bj, fmap)
                print (f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                groundPowerReal = np.array([cp-np.real(br[0])/np.sqrt(4*np.pi) for br,cp in
                                            zip(beamreal,cross_power)])
                groundPowerImag = np.array([0-np.real(bi[0])/np.sqrt(4*np.pi) for bi in beamimag])
            if "dump_beams" in getattr(self, "extra_opts", {}):
                np.save(bi.id+bj.id,beamreal)
            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))
