from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings

import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os


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
    
    prod = alm1*np.conj(alm2)
    sm = (np.real(prod[:lmax+1]).sum()+2*np.real(prod[lmax+1:]).sum())/(4*np.pi)
    return sm

def get_R_gal_to_topo(lz, bz, ly, by):
    """
    Build 3x3 rotation matrix R such that v_topo = R @ v_gal.
    Same construction as DefaultSimulator (zenith zhat, north yhat from (l,b)).
    """
    zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
    yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
    xhat = np.cross(yhat, zhat)
    return np.array([xhat, yhat, zhat]).T


def get_topo_z_rotation_angles(obs, times):
    """
    Return z-rotation angles phi[i] (radians) of topo frame at times[i] relative to times[0].
    phi[0] = 0. Uses (l,b)(t) from observation so libration is included.
    """
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    R0 = get_R_gal_to_topo(lzl[0], bzl[0], lyl[0], byl[0])
    phis = np.zeros(len(times))
    for i in range(len(times)):
        Ri = get_R_gal_to_topo(lzl[i], bzl[i], lyl[i], byl[i])
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
    
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :
    """
    Function that converts from Euler angles to rotation matrix
    
    :param theta: Euler angles
    :type theta: array
    
    :returns: Rotation matrix
    :rtype: numpy array
    
    """
    
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

    

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
            
        freq_ndx_beam = []
        freq_ndx_sky = []
        for f in self.freq:
            try:
                ndx = list(beams[0].freq).index(f)
            except ValueError:
                print ("Error:")
                print (f"Frequency {f} does not exist in beams.")
                sys.exit(1)
            freq_ndx_beam.append(ndx)
            try:
                ndx = list(sky_model.freq).index(f)
            except ValueError:
                print ("Error:")
                print (f"Frequency {f} does not exist in sky model.")
                sys.exit(1)
            freq_ndx_sky.append(ndx)
            
        self.freq_ndx_beam = freq_ndx_beam
        self.freq_ndx_sky = freq_ndx_sky
        self.Nfreq = len(self.freq)

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

    def _plot_sky_beam_healpix(self, sky_alm, beam_alm, nside, lmax, outpath="sky_beam_healpix.png", title_prefix=""):
        """
        Plot sky and beam as healpix mollweide maps (for visual check before convolution).
        Call when extra_opts["plot_sky_and_beam"] is True.
        :param sky_alm: Healpy packed alm (1D complex) for sky at one frequency
        :param beam_alm: Healpy packed alm (1D complex) for beam at one frequency
        :param nside: Healpix Nside for the map
        :param lmax: Maximum l for alm2map
        :param outpath: Output PNG path
        :param title_prefix: Optional prefix for plot title (e.g. simulator name)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
        fits.write(self.result, header=header, extname='data')
        fits.write(self.freq, extname='freq')
        fits.write(np.array(self.combinations), extname='combinations')
        for i,b in enumerate(self.beams):
            fits.write(b.ZRe[self.freq_ndx_beam],extname=f'ZRe_{i}')
            fits.write(b.ZIm[self.freq_ndx_beam],extname=f'ZIm_{i}')

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
        
        for i,j in self.combinations:
            bi , bj = beams[i], beams[j]
            print (f"  intializing beam combination {bi.id} x {bj.id} ...")
            norm = np.sqrt(bi.gain_conv[self.freq_ndx_beam]*bj.gain_conv[self.freq_ndx_beam])
            beamreal, beamimag = bi.get_healpix_alm(
                self.lmax,
                freq_ndx=self.freq_ndx_beam,
                other=bj,
                return_I_stokes_only=True,
                return_complex_components=True,
            )
            beamreal = beamreal * norm[:, None]
            if beamimag is not None:
                beamimag = beamimag * norm[:, None]

            if i==j:
                groundPowerReal = np.array([1-np.real(br[0])/np.sqrt(4*np.pi) for br in beamreal])
                beamimag = None
                groundPowerImag = 0.
            else:
                cross_power = self.cross_power.Ex_coupling(bi,bj,self.freq_ndx_beam)
                print (f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                groundPowerReal = np.array([cp-np.real(br[0])/np.sqrt(4*np.pi) for br,cp in
                                            zip(beamreal,cross_power)])
                groundPowerImag = np.array([0-np.real(bi[0])/np.sqrt(4*np.pi) for bi in beamimag])
            if "dump_beams" in getattr(self, "extra_opts", {}):
                np.save(bi.id+bj.id,beamreal)
            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))