import fitsio
import healpy as hp
import numpy as np
import jax
import jax.numpy as jnp

from .MonoSkyModels import T_C, T_DarkAges, T_DarkAges_Scaled
from .frequencies import ALL_FREQUENCIES_MHZ_NP

@jax.tree_util.register_pytree_node_class
class ConstSky:
    """
    Class that initializes a healpix sky map with a frequency dependent monopole signal given by one of the available Constant Sky models: 
    1) the Cane (1979) radio background model, or 
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
        Tmap = jnp.ones(self.Npix)
        if type(T) == int:
            T = float(T)
        if type(T) == list:
            T = jnp.array(T)
        self._T = T
        theta,phi = hp.pix2ang(self.Nside,np.arange(self.Npix))
        theta = jnp.asarray(theta)
        if zero_cone:
            # this is strictly speaking not needed, but we want to make sure
            # sky below horizon is ignored
            Tmap = jnp.where(theta>0.75*jnp.pi, 0.0, Tmap)
        self.mapalm = jnp.asarray(hp.map2alm(np.asarray(Tmap), lmax=lmax))
        self.frame = "MCMF"
        self.freq = None if freq is None else np.asarray(freq, dtype=np.float64)

    def tree_flatten(self):
        children = (self.mapalm, self._T)
        aux_data = (
            self.Nside,
            None if self.freq is None else tuple(np.asarray(self.freq).tolist()),
            self.frame,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Nside, freq, frame = aux_data
        mapalm, T = children
        sky = cls.__new__(cls)
        sky.Nside = Nside
        sky.Npix = Nside**2 * 12
        sky.mapalm = mapalm
        sky._T = T
        sky.frame = frame
        sky.freq = None if freq is None else np.asarray(freq, dtype=np.float64)
        return sky

    def T (self,ndx):
        """
        Function that returns sky temperature at specified frequency index or indices

        :param ndx: Frequency index at which to return temperature
        :type ndx: list

        :returns: Sky temperature
        :rtype: list
        """
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        T = jnp.asarray(self._T)
        return jnp.full(ndx.shape, T) if T.ndim == 0 else T[ndx]
    

    def get_alm(self, ndx, freq=None):
        """
        Function that calculates a_lm spherical harmonic decomposition for input sky map(s) at specified frequency indices. Uses healpy map2alm.

        :param ndx: Frequency index or list of indices
        :type ndx: list
        :param freq: Frequency list. Not currently implemented.
        :type freq: list

        :returns: a_lm array of shape ``(len(ndx), nalm)``
        :rtype: jnp.ndarray
        """
        return self.mapalm[None,:]*self.T(ndx)[:,None]

@jax.tree_util.register_pytree_node_class
class ConstSkyCane1979(ConstSky):
    """
    Class that constructs a monopole sky temperature map using the Cane (1979) radio background sky model. Uses ConstSky class to initialize map.

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param freq: List of frequencies at which to make sky maps. If freq=None, defaults to 1-50 MHz with 1 MHz spacing.
    :type freq: list
    """
    def __init__(self, Nside, lmax, freq=None):
        self.freq = ALL_FREQUENCIES_MHZ_NP if freq is None else np.asarray(freq, dtype=np.float64)
        T = T_C(self.freq).value
        ConstSky.__init__(self, Nside, lmax, T, self.freq)

    def get_alm_at_freq(self, target_freqs):
        """Evaluate alm at arbitrary target frequencies in MHz.

        Uses the closed-form Cane (1979) spectrum directly, so any positive
        frequency is supported without interpolation.
        """
        target = jnp.asarray(target_freqs, dtype=jnp.float64)
        T_at = jnp.asarray(T_C(target).value)
        return self.mapalm[None, :] * T_at[:, None]

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
        self.freq = ALL_FREQUENCIES_MHZ_NP if freq is None else np.asarray(freq, dtype=np.float64)
        self._scaled = scaled
        self._nu_min = nu_min
        self._nu_rms = nu_rms
        self._A = A
        if scaled:
            T = T_DarkAges_Scaled(self.freq, nu_min, nu_rms, A)
        else:
            T = T_DarkAges(self.freq)
        ConstSky.__init__(self, Nside, lmax, T, self.freq)

    def get_alm_at_freq(self, target_freqs):
        """Evaluate alm at arbitrary target frequencies in MHz.

        Uses the closed-form Dark Ages spectrum directly, so any positive
        frequency is supported without interpolation.
        """
        target = jnp.asarray(target_freqs, dtype=jnp.float64)
        if self._scaled:
            T_at = jnp.asarray(T_DarkAges_Scaled(target, self._nu_min, self._nu_rms, self._A))
        else:
            T_at = jnp.asarray(T_DarkAges(target))
        return self.mapalm[None, :] * T_at[:, None]

@jax.tree_util.register_pytree_node_class
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
        theta = jnp.asarray(theta)
        phi = jnp.asarray(phi)
        phi = jnp.where(phi>jnp.pi, phi-2*jnp.pi, phi) ## let's have a nice phi around phi=0.
        Tmap = jnp.exp(-(phi)**2/0.1-(theta-jnp.pi/2)**2/0.1)
        self.mapalm = jnp.asarray(hp.map2alm(np.asarray(Tmap), lmax = lmax))
        self.frame = "galactic"
        self.freq = None if freq is None else np.asarray(freq, dtype=np.float64)


@jax.tree_util.register_pytree_node_class
class HealpixSky:
    """
    Class that contains a sky as a healpix map. Alm representation is precomputed.

    .. note::
       ``mapalm`` is the canonical source of truth for this class (luseepy is
       a harmonic shop). ``maps`` is kept for convenience after ``__init__``
       but is NOT preserved across JAX pytree operations (``tree_flatten`` /
       ``tree_unflatten`` only round-trips ``mapalm``), so ``maps`` may be
       absent on instances reconstructed by JAX transforms.

    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param lmax: Maximum l value for maps
    :type lmax: int
    :param maps: List of healpix maps to use as sky model, one for each frequency in freq list
    :type maps: list of arrays
    :param freq: List of frequencies at which to make sky maps.
    :type freq: list    
    :param frame: Coordinate frame of the sky maps (default: "galactic", also accepts "equatorial" and "ecliptic")
    :type frame: str
    
    """
    def __init__ (self, Nside, lmax, maps, freq=None, frame="galactic"):
        self.Nside = Nside
        self.Npix = Nside**2 * 12
        self.maps = jnp.asarray(maps)
        if freq is None:
            freq = ALL_FREQUENCIES_MHZ_NP[[24]]
        self.freq = np.asarray(freq, dtype=np.float64)
        assert (len(maps)==len(freq))
        self.mapalm = jnp.asarray([hp.map2alm(np.asarray(m),lmax = lmax) for m in self.maps])
        self.frame  = frame

    def tree_flatten(self):
        children = (self.mapalm,)
        aux_data = (
            self.Nside,
            tuple(np.asarray(self.freq).tolist()),
            self.frame,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Nside, freq, frame = aux_data
        (mapalm,) = children
        sky = cls.__new__(cls)
        sky.Nside = Nside
        sky.Npix = Nside**2 * 12
        sky.maps = None
        sky.freq = np.asarray(freq, dtype=np.float64)
        sky.mapalm = mapalm
        sky.frame = frame
        return sky

    def get_alm (self, ndx, freq=None):
        """
        Function that calculates a_lm spherical harmonic decomposition for input FITS file sky map(s) at specified frequency indices. Uses healpy map2alm.

        :param ndx: Frequency index at which to return temperature
        :type ndx: list
        :param freq: List of frequencies at which to make sky maps.
        :type freq: list

        :returns: a_lm array of shape ``(len(ndx), nalm)``
        :rtype: jnp.ndarray
        """
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        return self.mapalm[ndx]

@jax.tree_util.register_pytree_node_class
class FitsSky (HealpixSky):
    """
    Class that reads in a sky map from a FITS file, and reads in the freq list from the FITS header. Computes map A_lms with healpy map2alm, up to specified lmax.

    :param fname: Filename to read in
    :type fname: str
    :param lmax: Maximum l value for maps
    :type lmax: int
    """
    def __init__ (self, fname, lmax):
        header      = fitsio.read_header(fname)
        fits        = fitsio.FITS(fname,'r')
        maps        = fits[0].read()
        fstart      = header['freq_start']
        fend        = header['freq_end']
        fstep       = header['freq_step']
        # the file's own grid: freq_end is its last channel, hence the
        # half-step slop to keep it despite float arange rounding
        freq = np.arange(fstart, fend + 0.5 * fstep, fstep, dtype=np.float64)
        super().__init__(Nside=hp.npix2nside(maps.shape[1]), lmax=lmax, maps=maps, freq=freq, frame="galactic")
        

@jax.tree_util.register_pytree_node_class
class SingleSourceHealpixSky (HealpixSky):
    """
    Class that constructs a temperature map using the Single Source model. Uses HealpixSky class to initialize map.

    :param ra_deg: Right ascension of the single source in degrees
    :type ra_deg: float
    :param dec_deg: Declination of the single source in degrees
    :type dec_deg: float
    :param Nside: Size of Healpix map to create
    :type Nside: int
    :param freq: List of frequencies at which to make sky maps.
    :type freq: list
    """
    def __init__ (self, Nside=128, freq=None, T=1.0, *,
                 ra_deg=None, dec_deg=None, l_deg=None, b_deg=None):
        # convert ra, dec to galactic coordinates and then to pixel number
        if freq is None:
            freq = ALL_FREQUENCIES_MHZ_NP[[24]]
        self.freq = np.asarray(freq, dtype=np.float64)
        T = jnp.atleast_1d(jnp.asarray(T, dtype=float))
        if T.size == 1:
            T = jnp.broadcast_to(T, len(self.freq))

        # Determine frame and convert to (theta, phi) in healpy convention
        has_eq = ra_deg is not None and dec_deg is not None
        has_gal = l_deg is not None and b_deg is not None
        if has_eq == has_gal:
            raise ValueError("provide either (ra_deg, dec_deg) or (l_deg, b_deg), not both")

        if has_eq:
            self.frame = "equatorial"
            theta = jnp.pi / 2 - jnp.radians(dec_deg)
            phi = jnp.radians(ra_deg) % (2 * jnp.pi)
        else:
            self.frame = "galactic"
            theta = jnp.pi / 2 - jnp.radians(b_deg)
            phi = jnp.radians(l_deg) % (2 * jnp.pi)
 
        pix = hp.ang2pix(Nside, float(theta), float(phi))
        Npix = Nside**2 * 12
        map = jnp.zeros(Npix)
        map = map.at[pix].set(1.0)
        map  = map[None,:]*T[:,None]
        super().__init__(Nside, 3*Nside-1, map, freq=freq, frame=self.frame)
        

@jax.tree_util.register_pytree_node_class
class HarmonicPointSourceSky:
    """Point source computed directly in harmonic space — no pixelization or Gibbs ringing.

    Sets a_lm = Y*_lm(θ, φ) for a delta function at the given sky position.
    Supports equatorial (ra, dec) or galactic (l, b) coordinates.

    Example::

        from lusee.frequencies import canonical_frequencies, frequency_indices_from_values
        sky = HarmonicPointSourceSky(
            lmax=64,
            ra_deg=45.0,
            dec_deg=10.0,
            freq=canonical_frequencies(frequency_indices_from_values([10.0])),
        )
        alms = sky.get_alm([0], freq=jnp.array([10.0]))

    :param lmax: Maximum spherical harmonic degree.
    :param freq: Frequency list (the source spectrum is flat; scale via *T*).
    :param T: Source amplitude, scalar or array per frequency.
    :param ra_deg: Right ascension (equatorial). Provide *ra_deg*/*dec_deg* or *l_deg*/*b_deg*.
    :param dec_deg: Declination (equatorial).
    :param l_deg: Galactic longitude.
    :param b_deg: Galactic latitude.
    :param frame: Coordinate frame — inferred from which arguments are given.
    """

    def __init__(self, lmax, freq, T=1.0, *,
                 ra_deg=None, dec_deg=None, l_deg=None, b_deg=None):
        self.lmax = lmax
        self.freq = np.asarray(freq, dtype=np.float64)
        T = jnp.atleast_1d(jnp.asarray(T, dtype=float))
        if T.size == 1:
            T = jnp.broadcast_to(T, len(self.freq))

        # Determine frame and convert to (theta, phi) in healpy convention
        has_eq = ra_deg is not None and dec_deg is not None
        has_gal = l_deg is not None and b_deg is not None
        if has_eq == has_gal:
            raise ValueError("provide either (ra_deg, dec_deg) or (l_deg, b_deg), not both")

        if has_eq:
            self.frame = "equatorial"
            theta = jnp.pi / 2 - jnp.radians(dec_deg)
            phi = jnp.radians(ra_deg) % (2 * jnp.pi)
        else:
            self.frame = "galactic"
            theta = jnp.pi / 2 - jnp.radians(b_deg)
            phi = jnp.radians(l_deg) % (2 * jnp.pi)

        # Build healpy-format alm: a_lm = Y*_lm(θ, φ)
        nalm = hp.Alm.getsize(lmax)
        alm = jnp.zeros(nalm, dtype=complex)
        m,l = jnp.triu_indices(lmax + 1)
        idx = jnp.asarray([hp.Alm.getidx(lmax, int(l_), int(m_)) for m_, l_ in zip(np.asarray(m), np.asarray(l))])
        theta_ = jnp.full_like(l, theta, dtype=jnp.asarray(theta).dtype)
        phi_ = jnp.full_like(l, phi, dtype=jnp.asarray(phi).dtype)
        alm = alm.at[idx].set(jnp.conj(jax.scipy.special.sph_harm_y(l, m, theta_, phi_, n_max=lmax)))

        self._alm = alm
        self._T = T

    def tree_flatten(self):
        children = (self._alm, self._T)
        aux_data = (
            self.lmax,
            tuple(np.asarray(self.freq).tolist()),
            self.frame,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        lmax, freq, frame = aux_data
        alm, T = children
        sky = cls.__new__(cls)
        sky.lmax = lmax
        sky.freq = np.asarray(freq, dtype=np.float64)
        sky.frame = frame
        sky._alm = alm
        sky._T = T
        return sky

    def get_alm(self, ndx, freq=None):
        """Return alm arrays for the requested frequency indices.

        :param ndx: Frequency index or list of indices.
        :param freq: Frequency array (checked against self.freq if provided).
        :returns: a_lm array of shape ``(len(ndx), nalm)``
        :rtype: jnp.ndarray
        """
        ndx = jnp.atleast_1d(jnp.asarray(ndx))
        return self._alm[None,:]*self._T[ndx][:,None]
