
from .observation import LObservation
from .LBeam import LBeam
from .LBeamCouplings import LBeamCouplings
from scipy.ndimage import gaussian_filter
import numpy as np
import healpy as hp
import fitsio
import sys
import pickle
import os

def mean_alm(alm1, alm2, lmax):
    prod = alm1*np.conj(alm2)
    sm = (np.real(prod[:lmax+1]).sum()+2*np.real(prod[lmax+1:]).sum())/(4*np.pi)
    return sm

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R



class Simulator:
    """
    Simulator class
    """

    def __init__ (self, obs, beams, sky_model, 
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], lmax = 128,
                  taper = 0.03, Tground = 200.0, freq = None,
                  cross_power = None, beam_smooth = None,
                  extra_opts = {}):
        self.obs = obs
        self.sky_model = sky_model
        self.lmax = lmax
        self.taper = taper
        self.Tground = Tground
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else LBeamCouplings()
        self.beam_smooth = beam_smooth
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
        self.prepare_beams (beams, combinations)
        self.result = None


    def prepare_beams(self,beams, combinations):
        """
        Beam Preparation

        
        :param combinations: Indices for beams
        :type combinations: tuple

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
            if self.beam_smooth is not None:
                print ("  smoothing beams with ",self.beam_smooth)
                beam2 = gaussian_filter(beam2,self.beam_smooth)

            # now need to transfrom this to healpy
            # (Note: we cut on freq_ndx above, so yes, range is fine in the line below)
            beamreal =  bi.get_healpix(self.lmax, np.real(beam2), range(self.Nfreq))

            if i==j:
                groundPowerReal = np.array([1-np.real(br[0])/np.sqrt(4*np.pi) for br in beamreal])
                beamimag = None
                groundPowerImag = 0.
            else:
                beamimag = bi.get_healpix(self.lmax, np.imag(beam2), range(self.Nfreq))
                cross_power = self.cross_power.Ex_coupling(bi,bj,self.freq_ndx_beam)
                print (f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                groundPowerReal = np.array([cp-np.real(br[0])/np.sqrt(4*np.pi) for br,cp in
                                            zip(beamreal,cross_power)])
                groundPowerImag = np.array([0-np.real(bi[0])/np.sqrt(4*np.pi) for bi in beamimag])
            if "dump_beams" in self.extra_opts:
                np.save(bi.id+bj.id,beamreal)
            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))
            
                                
    def simulate (self,times=None):
        """
        Main simulation loop.
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
                    print ("Cache file mix-up. Array wrong length!")
                    stop()
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
            sky = self.sky_model.get_alm (self.freq_ndx_sky, self.freq)
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
                T = np.array([mean_alm(br_,sky_,self.lmax) for br_,sky_ in zip(beamreal,sky)])
                T += self.Tground*groundPowerReal
                res.append(T)
                if ci!=cj:
                    Timag = np.array([mean_alm(bi_,sky_,self.lmax) for bi_,sky_ in zip(beamimag,sky)])
                    Timag += self.Tground*groundPowerImag
                    res.append(Timag)
            wfall.append(res)
        self.result = np.array(wfall)
        return self.result
            
    def write(self, out_file):
        if self.result is None:
            print ("Nothing to write")
            raise RunTimeError
        fits = fitsio.FITS(out_file,'rw',clobber=True)
        header = {
            "version" : 0.1,
            "lunar_day"  : self.obs.lunar_day,
            "lun_lat_deg"   : self.obs.lun_lat_deg,
            "lun_long_deg"   : self.obs.lun_long_deg,
            "lun_height_m"  : self.obs.lun_height_m,
            "deltaT_sec" : self.obs.deltaT_sec
        }
        fits.write(self.result, header=header, extname='data')
        fits.write(self.freq, extname='freq')
        fits.write(np.array(self.combinations), extname='combinations')
        for i,b in enumerate(self.beams):
            fits.write(b.ZRe[self.freq_ndx_beam],extname=f'ZRe_{i}')
            fits.write(b.ZIm[self.freq_ndx_beam],extname=f'ZIm_{i}')
