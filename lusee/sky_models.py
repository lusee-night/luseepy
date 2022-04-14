import fitsio
import healpy as hp
import numpy as np

class ConstSky:
    def __init__ (self,Nside, lmax,T ):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        Tmap = np.ones(self.Npix)
        self._T = T
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        Tmap[theta>0.75*np.pi] = 0 
        self.mapalm = hp.map2alm(Tmap, lmax=lmax)
        self.frame = "MCMF"

    def T (self,ndx):
        return [self._T]*len(ndx) if type(self._T)==float else self._T[ndx]
    

    def get_alm(self, ndx, freq=None):
        return [self.mapalm*T for T in self.T(ndx)]


class GalCenter (ConstSky):
    def __init__ (self,Nside, lmax, T):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        self._T = T
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        phi[phi>np.pi]-=2*np.pi ## let's have a nice phi around phi=0.
        Tmap = np.exp(-(phi)**2/0.1-(theta-np.pi/2)**2/0.1)
        self.mapalm = hp.map2alm(Tmap, lmax = lmax)
        self.frame = "galactic"


class FitsSky:
    def __init__ (self, fname, lmax):
        header = fitsio.read_header(fname)
        fits = fitsio.FITS(fname,'r')
        maps  = fits[0].read()
        self.maps = maps
        fstart = header['freq_start']
        fend = header['freq_end']
        fstep = header['freq_step']
        self.freq_list = np.arange(fstart, fend+1e-3*fstep, fstep)
        assert (len(self.freq_list) == maps.shape[0])
        self.mapalm = np.array([hp.map2alm(m,lmax = lmax) for m in maps])
        self.frame = "galactic"

    def get_alm (self, ndx, freq):
        assert (np.all(self.freq_list[ndx]==freq))
        return self.mapalm[ndx]



    
