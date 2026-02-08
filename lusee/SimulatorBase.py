from scipy.ndimage import gaussian_filter

from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings

import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os


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
        Expects the subclass to have set: lmax, taper, cross_power, beam_smooth,
        and optionally extra_opts (e.g. for "dump_beams"), before calling.

        :param beams: Instrument beams, from lusee.beam object
        :type beams: class
        :param combinations: Indices for beam combinations/cross correlations to simulate
        :type combinations: list[tuple]
        """
        
        self.beams = beams
        self.efbeams = []
        thetas = beams[0].theta
        #gtapr = np.zeros(len(thetas))
        gtapr = (np.arctan((thetas-np.pi/2)/self.taper)/np.pi+0.5)**2
        tapr = 1.0 - gtapr
        bomega = []
        self.combinations = [(int(i),int(j)) for i,j in combinations]
        
        for i,j in self.combinations:
            bi , bj = beams[i], beams[j]
            print (f"  intializing beam combination {bi.id} x {bj.id} ...")
            #f_ground_i, f_ground_j = f_grounds[i], f_grounds[j]
            xP = bi.cross_power(bj)[self.freq_ndx_beam,:,:]
            norm = np.sqrt(bi.gain_conv[self.freq_ndx_beam]*bj.gain_conv[self.freq_ndx_beam])
            beam2 = xP*tapr[None,:,None]*norm[:,None,None]
            if getattr(self, "beam_smooth", None) is not None:
                print ("  smoothing beams with ",self.beam_smooth)
                beam2 = gaussian_filter(beam2,self.beam_smooth)

            # now need to transfrom this to healpy
            # (Note: we cut on freq_ndx above, so yes, range is fine in the line below)
            beamreal =  bi.get_healpix_alm(self.lmax, np.real(beam2), range(self.Nfreq))

            if i==j:
                groundPowerReal = np.array([1-np.real(br[0])/np.sqrt(4*np.pi) for br in beamreal])
                beamimag = None
                groundPowerImag = 0.
            else:
                beamimag = bi.get_healpix_alm(self.lmax, np.imag(beam2), range(self.Nfreq))
                cross_power = self.cross_power.Ex_coupling(bi,bj,self.freq_ndx_beam)
                print (f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                groundPowerReal = np.array([cp-np.real(br[0])/np.sqrt(4*np.pi) for br,cp in
                                            zip(beamreal,cross_power)])
                groundPowerImag = np.array([0-np.real(bi[0])/np.sqrt(4*np.pi) for bi in beamimag])
            if "dump_beams" in getattr(self, "extra_opts", {}):
                np.save(bi.id+bj.id,beamreal)
            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))