import fitsio
import healpy as hp
import numpy as np
from .MonoSkyModels import T_C, T_DarkAges, T_DarkAges_Scaled

class ConstSky:
    """
    Class that initializes a healpix sky map with a frequency dependent monopole signal given by one of the available Constant Sky models: 
    1) the Cane et al. (1979) radio background model, or 
    2) the Dark Ages monopole model

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param T: Sky temperatures as a function of frequency, defined by model
    :type T: int, float, or list
    :param freq: List of frequencies at which to make sky maps
    :type freq: list
    :param zero_cone: Explicitly zero pixels below horizon
    :type zero_cone: bool
    """
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
        """
        Function that returns sky temperature at specified frequency index or indices

        :param ndx: Frequency index at which to return temperature
        :type ndx: list

        :returns: Sky temperature
        :rtype: list
        """
        return [self._T]*len(ndx) if type(self._T)==float else self._T[ndx]
    

    def get_alm(self, ndx, freq=None):
        """
        Function that calculates a_lm spherical harmonic decomposition for input sky map(s) at specified frequency indices. Uses healpy map2alm.

        :param ndx: Frequency index or list of indices
        :type ndx: list
        :param freq: Frequency list. Not currently implemented.
        :type freq: list

        :returns: A_lm array
        :rtype: array
        """
        return [self.mapalm*T for T in self.T(ndx)]

class ConstSkyCane1979(ConstSky):
    """
    Class that constructs a monopole sky temperature map using the Cane et al. (1979) radio background sky model. Uses ConstSky class to initialize map.

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param freq: List of frequencies at which to make sky maps. If freq=None, defaults to 1-50 MHz with 1 MHz spacing.
    :type freq: list
    """
    def __init__(self, Nside, lmax, freq=None):
        self.freq = np.arange(1.0,50.1) if freq is None else freq
        T = T_C(self.freq).value
        ConstSky.__init__(self, Nside, lmax, T, freq)

class DarkAgesMonopole(ConstSky):
    """
    Class that constructs a monopole sky temperature map using the Dark Ages monopole model. Uses ConstSky class to initialize map. Can optionally generate maps from the monopole model scaled to specified nu_min, nu_rms, and A, or from an explicit list of temperatures, T, as a function of frequency. Scaled model given by lusee.MonoSkyModels.T_DarkAges_Scaled, non-scaled by lusee.MonoSkyModels.T_DarkAges.

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param scaled: Whether to generate maps from frequency scaled model (True), or temperature list model (False) 
    :type scaled: bool
    :param nu_min: Frequency of the minimum of the Dark Ages trough
    :type nu_min: float
    :param nu_rms: Width of the Dark Ages trough
    :type nu_rms: float
    :param A: Monopole signal amplitude scaling factor
    :type A: float
    :param freq: List of frequencies at which to make sky maps.
    :type freq: list
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
    Class that constructs a temperature map using the Galaxy Center model. Uses ConstSky class to initialize map.

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param T: Sky temperatures as a function of frequency, defined by model
    :type T: int, float, or list
    :param freq: List of frequencies at which to make sky maps.
    :type freq: list
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
    Class that reads in a sky map from a FITS file, and reads in the freq list from the FITS header. Computes map A_lms with healpy map2alm, up to specified lmax.

    :param fname: Filename to read in
    :type fname: string
    :param lmax: Maximum l value for maps
    :type lmax: int
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
        Function that calculates a_lm spherical harmonic decomposition for input FITS file sky map(s) at specified frequency indices. Uses healpy map2alm.

        :param ndx: Frequency index at which to return temperature
        :type ndx: list
        :param freq: List of frequencies at which to make sky maps.
        :type freq: list

        :returns: A_lm array
        :rtype: array
        """
        assert (np.all(self.freq[ndx]==freq))
        return self.mapalm[ndx]



    
