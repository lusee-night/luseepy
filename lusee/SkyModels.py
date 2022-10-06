import fitsio
import healpy as hp
import numpy as np
from .MonoSkyModels import T_C, T_DarkAges, T_DarkAges_Scaled

class ConstSky:
    def __init__ (self,Nside, lmax, T, freq=None, zero_cone = True):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        Tmap = np.ones(self.Npix)
        if type(T) == int:
            T = float(T)
        if type(T) == list:
            T = np.array(T)
        self._T = T
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        if zero_cone:
            # this is strictly speaking not needed, but we want to make sure
            # sky below horizon is ignored
            Tmap[theta>0.75*np.pi] = 0  
        self.mapalm = hp.map2alm(Tmap, lmax=lmax)
        self.frame = "MCMF"
        self.freq=freq

    def T (self,ndx):
        return [self._T]*len(ndx) if type(self._T)==float else self._T[ndx]
    

    def get_alm(self, ndx, freq=None):
        return [self.mapalm*T for T in self.T(ndx)]

class ConstSkyCane1979(ConstSky):
    def __init__(self, Nside, lmax, freq=None):
        self.freq = np.arange(1.0,50.1) if freq is None else freq
        T = T_C(self.freq).value
        ConstSky.__init__(self, Nside, lmax, T, freq)

class DarkAgesMonopole(ConstSky):
    """
    Dark Ages Monopole.
    """
    def __init__(self, Nside, lmax, scaled = True, nu_min = 16.4,
                     nu_rms = 14.0, A = 0.04, freq=None):
        self.freq = np.arange(1.0,50.1) if freq is None else freq
        if scaled:
            T = T_DarkAges_Scaled(self.freq, nu_min, nu_rms, A)
        else:
            T = T_DarkAges(self.freq)
        ConstSky.__init__(self, Nside, lmax, T, freq)


        

class GalCenter (ConstSky):
    """
    The Galaxy Center class.
    """
    def __init__ (self,Nside, lmax, T, freq=None):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        self._T = T
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        phi[phi>np.pi]-=2*np.pi ## let's have a nice phi around phi=0.
        Tmap = np.exp(-(phi)**2/0.1-(theta-np.pi/2)**2/0.1)
        self.mapalm = hp.map2alm(Tmap, lmax = lmax)
        self.frame = "galactic"
        self.freq=freq

    

class FitsSky:
    """
    The 'Fit Sky' class
    """
    def __init__ (self, fname, lmax):
        header      = fitsio.read_header(fname)
        fits        = fitsio.FITS(fname,'r')
        maps        = fits[0].read()
        self.maps   = maps
        fstart      = header['freq_start']
        fend        = header['freq_end']
        fstep       = header['freq_step']
        self.freq   = np.arange(fstart, fend+1e-3*fstep, fstep)

        assert (len(self.freq) == maps.shape[0])

        self.mapalm = np.array([hp.map2alm(m,lmax = lmax) for m in maps])
        self.frame  = "galactic"

    def get_alm (self, ndx, freq):
        """
        Returns a map.

        :param ndx: index
        :type ndx: int
        """
        assert (np.all(self.freq[ndx]==freq))
        return self.mapalm[ndx]



    
