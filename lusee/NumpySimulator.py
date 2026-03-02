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


def _mean_alm_numpy(alm1, alm2, lmax):
    prod = alm1 * np.conj(alm2)
    return (np.real(prod[: lmax + 1]).sum() + 2 * np.real(prod[lmax + 1 :]).sum()) / (4 * np.pi)


class NumpySimulator(SimulatorBase):
    """
    Legacy NumPy default simulator kept as a separate engine.
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
        """
        if times is None:
            times = self.obs.times
        if self.sky_model.frame=="galactic":
            do_rot = True
            cache_fn = self.extra_opts.get("cache_transform")
            if (cache_fn is not None) and (os.path.isfile(cache_fn)):
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

        wfall = []
        Nt = len (times)
        for ti, t in enumerate(times):
            if (ti%100==0):
                print (f"{ti/Nt*100}% done ...")
            # healpy.rotate_alm expects complex128.
            if hasattr(self.sky_model, "get_alm_numpy"):
                sky = np.asarray(
                    self.sky_model.get_alm_numpy(self.freq_ndx_sky, self.freq),
                    dtype=np.complex128,
                )
            else:
                sky = np.asarray(self.sky_model.get_alm(self.freq_ndx_sky, self.freq), dtype=np.complex128)
            if do_rot:
                lz,bz,ly,by = lzl[ti],bzl[ti],lyl[ti],byl[ti]
                zhat = np.array([np.cos(bz)*np.cos(lz), np.cos(bz)*np.sin(lz),np.sin(bz)])
                yhat = np.array([np.cos(by)*np.cos(ly), np.cos(by)*np.sin(ly),np.sin(by)])
                xhat = np.cross(yhat,zhat)
                assert(np.abs(np.dot(zhat,yhat))<1e-10)
                R = np.array([xhat,yhat,zhat]).T
                a,b,g = rot2eul(R)
                rot = hp.rotator.Rotator(rot=(g,-b,a),deg=False,eulertype='XYZ',inv=False)
                sky = [rot.rotate_alm(s_) for s_ in sky]
            res = []
            for ci,cj,beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
                T = np.array([_mean_alm_numpy(br_, sky_, self.lmax) for br_, sky_ in zip(beamreal, sky)])
                T += self.Tground*groundPowerReal
                res.append(T)
                if ci!=cj:
                    Timag = np.array(
                        [_mean_alm_numpy(bi_, sky_, self.lmax) for bi_, sky_ in zip(beamimag, sky)]
                    )
                    Timag += self.Tground*groundPowerImag
                    res.append(Timag)
            wfall.append(res)
        self.result = np.array(wfall)
        return self.result
