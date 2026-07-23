from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, default_plot_sky_beam_dir, mean_alm_np, rot2eul_np
import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os
import warnings



class TopoNumpySimulator(SimulatorBase):
    """
    Default Simulator in luseepy 
    
    :param obs: Observation parameters, from lusee.observation class
    :type obs: class
    :param beams: Instrument beams, from lusee.beam class
    :type beams: class
    :param sky_model: Simulated model of the sky, from lusee.skymodels
    :type sky_model: class
    :param combinations: Indices for beam combinations/cross correlations to simulate
    :type combinations: list[tuple]
    :param lmax: Maximum l value of beams
    :type lmax: int
    :param Tground: Temperature of lunar ground
    :type Tground: float
    :param freq: Frequencies at which instrument observes sky in MHz. If empty, taken from lusee.beam class.
    :type freq: list[float]
    :param cross_power: Beam coupling model for cross-power terms. If empty, uses :class:`lusee.BeamCouplings`.
    :type cross_power: BeamCouplings
        :param extra_opts: Extra options for simulation. Supports "dump_beams" (saves instrument beams to file),
        "cache_transform" (loads/saves beam transformations from file),
        "force_recompute_cache_transform" (ignores any existing cached transform file),
        and "freq_idx_plot" (int): index of frequency at which to plot sky and beam.
        :type extra_opts: dict
    
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

            
                                
    def simulate (self,times=None):
        """
        Main simulation function. Produces mock observation of sky model from self.sky_model with beams from self.beams

        :param times: List of times, defaults to lusee.observation.times if empty 
        :type times: list
        
        :returns: Waterfall style observation data for input times and self.freq
        :rtype: numpy array

        """
        if times is None:
            times = self.obs.times
        if self.sky_model.frame=="galactic":
            do_rot = True
            cache_fn = self._transform_cache_file(times)
            force_recompute = self._transform_cache_force_recompute()
            if force_recompute and (cache_fn is not None):
                print (f"Ignoring cached transform and recomputing: {cache_fn}")
            if (cache_fn is not None) and (not force_recompute) and (os.path.isfile(cache_fn)):
                print (f"Loading cached transform from {cache_fn}...")
                lzl,bzl,lyl,byl = pickle.load(open(cache_fn,'br'))
                if (len(lzl)!=len(times)):
                    raise RuntimeError("Cache file mix-up. Array wrong length!")
                have_transform = True
            else:
                have_transform = False
                
            if not have_transform:
                print ("Getting pole transformations...")
                lzl,bzl = self.obs.get_l_b_from_alt_az(np.pi/2,0., times)
                print ("Getting horizon transformations...")
                lyl,byl = self.obs.get_l_b_from_alt_az(0.,0., times)  ## astronomical azimuth = 0 = N = our y coordinate
                if cache_fn is not None:
                    print (f"Saving transforms to {cache_fn}...")
                    pickle.dump((lzl,bzl,lyl,byl),open(cache_fn,'bw'))
            
        elif self.sky_model.frame=="MCMF":
            do_rot = False
        else:
            raise NotImplementedError

        sky_target = self.sky_alm_at_freq(self.sky_model)
        sky_base = [np.asarray(s_, dtype=np.complex128) for s_ in sky_target]
        wfall = []
        Nt = len (times)
        for ti, t in enumerate(times):
            if (ti%100==0):
                print (f"{ti/Nt*100}% done ...")
            sky = sky_base
            if do_rot:
                lz,bz,ly,by = lzl[ti],bzl[ti],lyl[ti],byl[ti]
                zhat = np.array([np.cos(bz)*np.cos(lz), np.cos(bz)*np.sin(lz),np.sin(bz)])
                yhat = np.array([np.cos(by)*np.cos(ly), np.cos(by)*np.sin(ly),np.sin(by)])
                xhat = np.cross(yhat,zhat)
                assert(np.abs(np.dot(zhat,yhat))<1e-10)
                R = np.array([xhat,yhat,zhat]).T
                a,b,g = rot2eul_np(R)
                rot = hp.rotator.Rotator(rot=(g,-b,a),deg=False,eulertype='XYZ',inv=False)
                sky = [rot.rotate_alm(s_) for s_ in sky]
            if ti == 0 and self.extra_opts.get("plot_sky_and_beam"):
                nf = len(self.freq)
                freq_idx_plot = int(self.extra_opts.get("freq_idx_plot", 0))
                if nf == 0:
                    raise ValueError("Cannot plot: no frequencies in self.freq")
                if not (0 <= freq_idx_plot < nf):
                    clamped = max(0, min(freq_idx_plot, nf - 1))
                    warnings.warn(
                        f"freq_idx_plot={freq_idx_plot} is out of bounds for "
                        f"len(freq)={nf}; using {clamped}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    freq_idx_plot = clamped
                nside = getattr(self.sky_model, "Nside", 64)
                beamreal0 = self.efbeams[0][2]
                self._plot_sky_beam_healpix(
                    sky[freq_idx_plot], beamreal0[freq_idx_plot], nside, self.lmax,
                    save_dir=self.extra_opts.get("plot_dir", default_plot_sky_beam_dir()),
                    save_filename=self.extra_opts.get("plot_filename", "sky_beam_healpix_default.png"),
                    title_prefix=f"Default at {self.freq[freq_idx_plot]} MHz ",
                )
            res = []
            for ci,cj,beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
                T = np.array([mean_alm_np(br_,sky_,self.lmax) for br_,sky_ in zip(beamreal,sky)])
                T += self.Tground*groundPowerReal
                res.append(T)
                if ci!=cj:
                    Timag = np.array([mean_alm_np(bi_,sky_,self.lmax) for bi_,sky_ in zip(beamimag,sky)])
                    Timag += self.Tground*groundPowerImag
                    res.append(Timag)
            wfall.append(res)
        self.result = np.array(wfall)
        return self.result
            

NumpySimulator = TopoNumpySimulator
