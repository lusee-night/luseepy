#
# LuSEE Beam
#

from functools import partial, lru_cache
import os

os.environ["JAX_ENABLE_X64"] = "True"

import fitsio
import numpy as np
import jax
import jax.numpy as jnp
import copy
import matplotlib.pyplot as plt
from jax.tree_util import register_pytree_node_class
from mpl_toolkits.axes_grid1 import make_axes_locatable
import healpy as hp
from scipy.special import sph_harm_y
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from .frequencies import canonicalize_frequencies


@lru_cache(maxsize=None)
def _get_triu_indices(lmax):
    m_idx, l_idx = np.triu_indices(lmax + 1)
    return jnp.asarray(m_idx, dtype=jnp.int32), jnp.asarray(l_idx, dtype=jnp.int32)


@lru_cache(maxsize=None)
def _recurrence_coeffs(lmax):
    """Precompute three-term recurrence coefficients for all (l, m) pairs.

    Returns arrays of shape (lmax+1, lmax+1) indexed as [k, m] where l = m + k.
    """
    ms = np.arange(lmax + 1, dtype=np.float64)
    ks = np.arange(lmax + 1, dtype=np.float64)
    fl = ms[None, :] + ks[:, None]  # l = m + k, shape (lmax+1, lmax+1)
    fm = ms[None, :]

    # Sectoral coefficients: -sqrt((2m+1)/(2m)) for m >= 1
    sect = np.zeros(lmax + 1)
    sect[1:] = -np.sqrt((2.0 * ms[1:] + 1.0) / (2.0 * ms[1:]))

    # Sub-diagonal coefficients: sqrt(2m+3)
    sub_diag = np.sqrt(2.0 * ms + 3.0)

    # Three-term recurrence (k >= 2): P_l^m = alpha * x * P_{l-1}^m - beta * P_{l-2}^m
    denom_a = np.maximum((fl - fm) * (fl + fm), 1.0)
    alpha = np.sqrt(np.maximum((2 * fl + 1) * (2 * fl - 1), 0.0) / denom_a)
    numer_b = np.maximum((2 * fl + 1) * (fl + fm - 1) * (fl - fm - 1), 0.0)
    denom_b = np.maximum((2 * fl - 3) * (fl - fm) * (fl + fm), 1.0)
    beta = np.sqrt(numer_b / denom_b)

    return (
        jnp.asarray(sect),
        jnp.asarray(sub_diag),
        jnp.asarray(alpha),
        jnp.asarray(beta),
    )


def _getLegendre_packed(lmax, theta):
    """Compute fully-normalized associated Legendre functions P_l^m(cos theta)
    for 0 <= m <= l <= lmax, packed in upper-triangular (m, l) order.

    Uses a vectorized three-term recurrence: a single scan over l that processes
    all m columns simultaneously.
    """
    x = jnp.cos(jnp.asarray(theta))
    s = jnp.sin(jnp.asarray(theta))
    sect, sub_diag, alpha, beta = _recurrence_coeffs(lmax)
    ms = jnp.arange(lmax + 1)

    # Step 1: Compute all sectoral values P_m^m via scan
    def sectoral_step(pmm_prev, m):
        pmm = jnp.where(m == 0,
                        1.0 / jnp.sqrt(4.0 * jnp.pi),
                        sect[m] * s * pmm_prev)
        return pmm, pmm

    _, sectoral = jax.lax.scan(sectoral_step, 0.0, ms)

    # Step 2: Scan over k (where l = m + k) for all m simultaneously
    # At each step k, we compute P_{m+k}^m for all m as a vector.
    # k=0: P_m^m (seed), k=1: sub-diagonal, k>=2: three-term recurrence
    def step(carry, k_data):
        p_prev, p_prev2 = carry
        k, alpha_k, beta_k = k_data

        val_k0 = sectoral
        val_k1 = sub_diag * x * sectoral
        val_k2 = alpha_k * x * p_prev - beta_k * p_prev2

        val = jnp.where(k == 0, val_k0, jnp.where(k == 1, val_k1, val_k2))
        return (val, p_prev), val

    ks = jnp.arange(lmax + 1)
    init = (sectoral, jnp.zeros(lmax + 1, dtype=x.dtype))
    _, columns = jax.lax.scan(step, init, (ks, alpha, beta))
    # columns[k, m] = P_{m+k}^m

    m_idx, l_idx = _get_triu_indices(lmax)
    return columns[l_idx - m_idx, m_idx]


def getLegendre(lmax, theta):
    """Compute fully-normalized associated Legendre functions as a 2D array [l, m]."""
    m_idx, l_idx = _get_triu_indices(lmax)
    values = _getLegendre_packed(lmax, theta)
    return jnp.zeros((lmax + 1, lmax + 1), dtype=values.dtype).at[l_idx, m_idx].set(values)


@partial(jax.jit, static_argnums=(3,))
def _grid2healpix_alm_fast_impl(theta, phi, img, lmax):
    dtheta = theta[1] - theta[0]
    dA_theta = jnp.sin(theta) * dtheta
    nphi = phi.shape[0]

    rimg = jnp.fft.rfft(img, axis=1)
    if rimg.shape[1] < lmax + 1:
        rimg = jnp.pad(rimg, ((0, 0), (0, lmax + 1 - rimg.shape[1])))
    else:
        rimg = rimg[:, : lmax + 1]

    m_idx, l_idx = _get_triu_indices(lmax)
    legendre_values = jax.vmap(lambda th: _getLegendre_packed(lmax, th))(theta)
    coeffs = rimg[:, m_idx] * legendre_values * dA_theta[:, None]
    return coeffs.sum(axis=0) * (2 * jnp.pi / nphi)


@partial(jax.jit, static_argnums=(3,))
def _grid2healpix_alm_batch(theta, phi, power_map_batch, lmax):
    return jax.vmap(lambda power_map: _grid2healpix_alm_fast_impl(theta, phi, power_map, lmax))(power_map_batch)


def _to_python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def grid2healpix_alm_reference(theta,phi, img, lmax):
    """
    Function that calculates a_lm spherical harmonic decomposition for input image

    :param theta: Input spherical angle coordinates
    :type theta: numpy array
    :param phi: Input spherical angle coordinates
    :type phi: numpy array
    :param img: Input image (2D)
    :type img: numpy array
    :param lmax: Maximum l value
    :type lmax: int

    :returns: 2D a_lm spherical harmonic array
    :rtype: numpy array
    """
    
    lmax = lmax + 1 ## different conventions
    dphi = phi[1]-phi[0]
    dtheta = theta[1]-theta[0]
    dA_theta = jnp.sin(theta)*dtheta*dphi
    #alm = jnp.zeros((lmax,lmax),complex)
    ell = jnp.arange(lmax)
    theta_list, phi_list = jnp.meshgrid(theta,2*jnp.pi-phi)
    mmax = lmax
    alm = []
    for m in range(lmax):
        for l in range(m,lmax):        
            harm = sph_harm_y (l, m, theta_list, phi_list)
            assert(not jnp.any(jnp.isnan(harm)))
            alm.append((img*harm.T*dA_theta[:,None]).sum())
    alm = jnp.array(alm)
    return alm


def grid2healpix_alm_fast(theta,phi, img, lmax):
    """
    Function that calculates a_lm spherical harmonic decomposition for input image, using fast method

    :param theta: Input spherical angle coordinates
    :type theta: numpy array
    :param phi: Input spherical angle coordinates
    :type phi: numpy array
    :param img: Input image (2D)
    :type img: numpy array
    :param lmax: Maximum l value
    :type lmax: int

    :returns: 2D a_lm spherical harmonic array
    :rtype: numpy array
    """
    
    return _grid2healpix_alm_fast_impl(theta, phi, img, lmax)


def grid2healpix(theta,phi, img, lmax, Nside, fast=True):
    """
    Function that converts from theta-phi orthogonal spherical coordinates to heapix coordinates

    :param theta: Input spherical angle coordinates
    :type theta: numpy array
    :param phi: Input spherical angle coordinates
    :type phi: numpy array
    :param img: Input image (2D)
    :type img: numpy array
    :param lmax: Maximum l value
    :type lmax: int
    :param Nside: Size of output Healpix map
    :type Nside: int
    :param fast: Whether to use fast a_lm method
    :type fast: bool

    :returns: Healpix map of size Nside
    :rtype: numpy array
    """
    
    if fast:
        alm = grid2healpix_alm_fast(theta,phi,img,lmax)
    else:
        alm = grid2healpix_alm_reference(theta,phi,img,lmax)
    return hp.sphtfunc.alm2map(alm,Nside)



def project_to_theta_phi(theta_rad,phi_rad, E):
    """
    Function that projects E_theta and E_phi components of instrument beam from E field in cartesian coordinates, E(x, y, z) 

    :param theta: Input spherical angle coordinates
    :type theta: numpy array
    :param phi: Input spherical angle coordinates
    :type phi: numpy array
    :param E: Electric field
    :type E: numpy array

    :returns: [Etheta, Ephi], Theta and phi components of electric field at [theta, phi] coordinates
    :rtype: numpy array
    """
    
    #create projection matrices
    theta = theta
    phi= phi
    sin = jnp.sin
    cos = jnp.cos
    rad = jnp.array([ sin(theta[:,None])*cos(phi[None,:]), sin(theta[:,None])*sin(phi[None,:]),
                     -cos(theta[:,None])*jnp.ones(self.Nphi)[None,:]])
    tphi =  jnp.array([-sin(phi), +cos(phi)])
    ttheta = jnp.array([ cos(theta[:,None])*cos(phi[None,:]), cos(theta[:,None])*sin(phi[None,:]),
                     +sin(theta[:,None])*jnp.ones(self.Nphi)[None,:]])

    Erad = jnp.einsum('fijk,kij->fij',E,rad)
    Etheta = jnp.einsum('fijk,kij->fij',E,ttheta)
    Ephi = jnp.einsum('fijk,kj->fij',E[:,:,:,:2],tphi)
    Emag2 = (jnp.abs(self.E)**2).sum(axis=3)
    assert(abs(Emag2-jnp.abs(Erad)**2-jnp.abs(Etheta)**2-jnp.abs(Ephi)**2).max()<1e-4)
    #print ((jnp.abs(Erad)/jnp.sqrt(Emag2)).max())
    assert(jnp.all(jnp.abs(Erad)/jnp.sqrt(Emag2)<1e-4))
    return Etheta, Ephi


@register_pytree_node_class
class Beam:
    """
    The main beam class, contains beam data and meta parameters. Only filename of beam to load and ID string are explicitly set in class initialization. All others are normally read in from the beam FITS file, but are included here in documentation for completeness.

    :param fname: Filename of beam to load
    :type fname: str
    :param id: ID string for beam, optional
    :type id: str
    :param version: Beam version
    :type version: int
    :param Etheta: Theta component of electric field
    :type Etheta: numpy array[complex]
    :param Ephi: Phi component of electric field
    :type Ephi: numpy array[complex]
    :param ZRe: Real component of antenna impedance
    :type ZRe: numpy array
    :param ZIm: Imaginary component of antenna impedance
    :type ZIm: numpy array
    :param Z: Complex impedance
    :type Z: numpy array[complex]
    :param gain: Antenna gain
    :type gain: numpy array
    :param f_ground: Ground fraction
    :type f_ground: numpy array
    :param gain_conv: Gain convention
    :type gain_conv: numpy array
    :param freq: Frequency list
    :type freq: numpy array
    :param freq_min: Minimum frequency    
    :type freq_min: float
    :param freq_max: Maximum frequency
    :type freq_max: float
    :param Nfreq: Number of frequencies
    :type Nfreq: int
    :param theta_min: Minimum theta angle
    :type theta_min: float
    :param theta_max: Maximum theta angle
    :type theta_max: float
    :param Ntheta: Number of theta bins
    :type Ntheta: int
    :param phi_min: Minimum phi angle
    :type phi_min: float
    :param phi_max: Maximum phi angle
    :type phi_max: float
    :param Nphi: Number of phi bins
    :type Nphi: int
    :param header: File header
    :type header: dict
    :param theta_deg: Array of theta bins in degrees
    :type theta_deg: numpy array
    :param phi_deg: Array of theta bins in degrees
    :type phi_deg: numpy array
    :param theta: Array of theta bins in radians
    :type theta: numpy array
    :param phi: Array of phi bins in radians
    :type phi: numpy array
    """
    is_jax_pytree_beam = True
    
    def __init__ (self, fname = None, id = None):
        if fname is None:
            fname = base = os.environ['LUSEE_DRIVE_DIR']+"Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
        if not (os.path.isfile (fname) and os.access(fname, os.R_OK)):
            print (f"Cannot open {fname}")
            stop()
        header = dict(fitsio.read_header(fname))
        fits = fitsio.FITS(fname,'r')
        version = _to_python_scalar(header['VERSION'])
        self.id = id
        self.version = version
        self.Etheta = jnp.asarray(fits['Etheta_real'].read()) + 1j*jnp.asarray(fits['Etheta_imag'].read())
        self.Ephi = jnp.asarray(fits['Ephi_real'].read()) + 1j*jnp.asarray(fits['Ephi_imag'].read())
        self.ZRe = jnp.asarray(fits['Z_real'].read())
        self.ZIm = jnp.asarray(fits['Z_imag'].read())
        self.Z = self.ZRe + 1j*self.ZIm
        self.gain = jnp.asarray(fits['gain'].read()) if 'gain' in fits else None
        if version==1:
            self.f_ground = jnp.asarray(fits['f_ground'].read())
        elif version==2:
            self.gain_conv = jnp.asarray(fits['gain_conv'].read())
            self.freq = canonicalize_frequencies(fits['freq'].read(), as_jax=True)
            
        self.freq_min = float(header['FREQ_MIN'])
        self.freq_max = float(header['FREQ_MAX'])
        self.Nfreq = int(header['FREQ_N'])
        self.theta_min = float(header['THETA_MIN'])
        self.theta_max = float(header['THETA_MAX'])
        self.Ntheta = int(header['THETA_N'])
        self.phi_min = float(header['PHI_MIN'])
        self.phi_max = float(header['PHI_MAX'])
        self.Nphi = int(header['PHI_N'])
        self.header = header
        if version==1:
            self.freq = canonicalize_frequencies(
                np.linspace(self.freq_min, self.freq_max, self.Nfreq),
                as_jax=True,
            )
        self.theta_deg = jnp.linspace(self.theta_min, self.theta_max,self.Ntheta)
        self.phi_deg = jnp.linspace(self.phi_min, self.phi_max,self.Nphi)
        self.theta = self.theta_deg/180*jnp.pi
        self.phi = self.phi_deg/180*jnp.pi
        if (self.phi_max != 360) or (self.phi_min != 0):
            print("Code might implicitly assume phi wraparound ... use with care.")
        assert (self.Etheta.shape == (self.Nfreq,self.Ntheta,self.Nphi))

    def tree_flatten(self):
        children = (
            self.Etheta,
            self.Ephi,
            self.ZRe,
            self.ZIm,
            getattr(self, "gain", None),
            getattr(self, "gain_conv", None),
            getattr(self, "f_ground", None),
        )
        aux_data = (
            self.id,
            self.version,
            self.freq_min,
            self.freq_max,
            self.Nfreq,
            self.theta_min,
            self.theta_max,
            self.Ntheta,
            self.phi_min,
            self.phi_max,
            self.Nphi,
            tuple(np.asarray(self.freq).tolist()),  # static metadata, not traced
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            id_,
            version,
            freq_min,
            freq_max,
            Nfreq,
            theta_min,
            theta_max,
            Ntheta,
            phi_min,
            phi_max,
            Nphi,
            freq_tuple,
        ) = aux_data
        (
            Etheta,
            Ephi,
            ZRe,
            ZIm,
            gain,
            gain_conv,
            f_ground,
        ) = children

        beam = cls.__new__(cls)
        beam.id = id_
        beam.version = version
        beam.Etheta = Etheta
        beam.Ephi = Ephi
        beam.ZRe = ZRe
        beam.ZIm = ZIm
        beam.gain = gain
        beam.gain_conv = gain_conv
        beam.f_ground = f_ground
        beam.freq = jnp.asarray(freq_tuple)
        beam.freq_min = freq_min
        beam.freq_max = freq_max
        beam.Nfreq = Nfreq
        beam.theta_min = theta_min
        beam.theta_max = theta_max
        beam.Ntheta = Ntheta
        beam.phi_min = phi_min
        beam.phi_max = phi_max
        beam.Nphi = Nphi
        beam.header = None
        beam.Z = beam.ZRe + 1j * beam.ZIm
        beam.theta_deg = jnp.linspace(theta_min, theta_max, Ntheta)
        beam.phi_deg = jnp.linspace(phi_min, phi_max, Nphi)
        beam.theta = beam.theta_deg / 180 * jnp.pi
        beam.phi = beam.phi_deg / 180 * jnp.pi
        return beam
        
    def rotate(self,deg):
        """
        Function that rotates the beam around the zenith (turntable rotation)

        :param deg: Rotational angle in degrees
        :type deg: float
        
        :returns: Rotated beam as copy of beam object
        :rtype: class
        """
        
        dphi = self.phi_deg[1]-self.phi_deg[0]
        assert (deg%dphi<1e-5)
        if self.phi_max != 360:
            print ("This really only works in you have full phi circle with repetition at the end")
            raise NotImplemented
        
        if deg==0:
            return self.copy_beam()
        rad = deg/180*jnp.pi
        cosrad = jnp.cos(rad)
        sinrad = jnp.sin(rad)
        # some sanity checks
        assert (self.phi_max == 360)
        assert (self.phi_min == 0)
        phi_step = (self.phi_max-self.phi_min)/(self.Nphi-1)
        assert (deg%phi_step == 0)
        m = int(deg // phi_step)
        if (m<0):
            Etheta = jnp.concatenate ((self.Etheta[:,:,m-1:],self.Etheta[:,:,1:m]),axis=2)
            Ephi = jnp.concatenate ((self.Ephi[:,:,m-1:],self.Ephi[:,:,1:m]),axis=2)
        else:
            Etheta = jnp.concatenate ((self.Etheta[:,:,m:],self.Etheta[:,:,1:m+1]),axis=2)
            Ephi = jnp.concatenate ((self.Ephi[:,:,m:],self.Ephi[:,:,1:m+1]),axis=2)

        # No need to rotate in theta  - phi
        #rotmat = jnp.array(([[cosrad, +sinrad, 0],[-sinrad,cosrad,0],[0,0,1]]))
        #E = jnp.einsum('fabj,ij->fabi',E,rotmat)


        return self.copy_beam(Etheta=Etheta, Ephi=Ephi)
     
    def flip_over_yz(self):
        """
        Function that flips beams across yz plane
        
        :returns: Flipped beam as copy of beam object
        :rtype: class
        """
        
        assert (False)
        m = int(90 // self.phi_step)
        n = int(180 // self.phi_step)
        o = int(270 // self.phi_step)
        Ephi = jnp.concatenate ((self.Ephi[:,:,n:0:-1],self.Ephi[:,:,self.Nphi:n-1:-1]),axis=2)
        #E[:,:,:,0]*=-1 ## X flips over
        return self.copy_beam(E=E)

    def power(self):
        """
        Function that calculates the beam power of a single beam
        
        :returns: Beam power
        :rtype: float
        """
        
        P = jnp.abs(self.Etheta**2)+jnp.abs(self.Ephi**2)
        return P

    def power_stokes(self, cross=None):
        """
        Function that calculates the beam power in Stokes components

        :param cross: Optional second beam object. If present, function will compute cross-power between the two beams given by "self" and "cross". If absent, defaults to computing auto-power for "self" beam object.
        :type cross: class
        
        :returns: [I, Q, U, V]
        :rtype: list[float]
        """
        
        if cross is None:
            I = jnp.abs(self.Etheta*self.Etheta)+jnp.abs(self.Ephi*self.Ephi)
            Q = jnp.abs(self.Etheta**2)-jnp.abs(self.Ephi**2)
            T = 2*self.Etheta*jnp.conj(self.Ephi)
            U = jnp.real(T)
            V = -jnp.imag(T)
        else:
            I = self.Etheta*jnp.conj(cross.Etheta) + self.Ephi*jnp.conj(cross.Ephi)
            Q = self.Etheta*jnp.conj(cross.Etheta) - self.Ephi*jnp.conj(cross.Ephi)
            U = self.Etheta*jnp.conj(cross.Ephi)+self.Ephi*jnp.conj(cross.Etheta)
            V = +1j*(self.Etheta*jnp.conj(cross.Ephi)-self.Ephi*jnp.conj(cross.Etheta))
        return [I,Q,U,V]

    
    def cross_power(self, other):
        """
        Function that calculates the cross-power between two beams

        :param other: Second beam object for cross-power
        :type other: class
        
        :returns: Cross power
        :rtype: float
        """
        
        xP = self.Etheta*jnp.conj(other.Etheta) + self.Ephi*jnp.conj(other.Ephi)
        return xP


    def sky_fraction(self, cross = None):
        """
        Function that calculates the fraction of beam power that terminates on the sky (sky fraction), for a single beam or two crossed beams
        
        :param cross: Optional second beam object 
        :type cross: class
        
        :returns: Sky fraction
        :rtype: float
        """
        
        if self.version<2:
            print ("Cannot do this on v1 files.")
        xP=self.power() if cross is None else self.cross_power(cross)
        gain = xP*self.gain_conv[:,None,None]
        dphi = self.phi[1]-self.phi[0]
        dtheta = self.theta[1]-self.theta[0]
        dA_theta = jnp.sin(self.theta)*dtheta*dphi
        return (dA_theta[None, :, None] * gain[:, :, :-1]).sum(axis=(1, 2)) / (4 * jnp.pi)
    
    def ground_fraction(self, cross = None):
        """
        Function that calculates the fraction of beam power that terminates on the ground (ground fraction), for a single beam or two crossed beams

        :param cross: Optional second beam object 
        :type cross: class
        
        :returns: Ground fraction
        :rtype: float
        """
        
        f_ground = 1.0 - self.sky_fraction(cross)
            
        return f_ground
        
    
    def power_hp(self, ellmax, Nside, freq_ndx=None, theta_tapr=None, cross=None, stokes=False):
        """
        Function that calculates the healpix rendering of the beam power

        :param ellmax: Maximum l value
        :type ellmax: int
        :param Nside: Size of output Healpix map
        :type Nside: int
        :param freq_ndx: Optional list of frequency bin indices. Integer indices, not freq values
        :type freq_ndx: list(int)
        :param theta_tapr: Optional tapering profile to apply to beam in theta direction
        :type theta_tapr: numpy array
        :param cross: Optional second beam object for cross-power
        :type cross: class
        :param stokes: Whether to compute Stokes parameters
        :type stokes: bool
        
        :returns: Healpix map containing beam power
        :rtype: numpy array or list(numpy array) if type(freq_ndx) is not int
        """
        
        if not stokes:
            P = self.power() if cross is None else self.power_cross(cross)
            P = [P] # lets' make it a list
        else:
            P = self.power_stokes(cross)

        if theta_tapr is not None:
            P *= theta_tapr[None,:,None]
        take_zero = False
        
        flist = range(self.Nfreq) if freq_ndx is None else jnp.asarray(np.atleast_1d(freq_ndx), dtype=jnp.int32)
        if cross is None:
            result =  [[grid2healpix(self.theta,self.phi[:-1], P_[i,:,:-1], ellmax, Nside) 
                    for i in flist] for P_ in P]
        else:
            result =  [[grid2healpix(self.theta,self.phi[:-1], jnp.real(P_[i,:,:-1]), ellmax, Nside)
                +1j*grid2healpix(self.theta,self.phi[:-1], jnp.imag(P_[i,:,:-1]), ellmax, Nside)
                    for i in flist] for P_ in P]

        if not stokes:
            result = result[0]
            if type(freq_ndx)==int:
                result = result[0]
        else:
            if type(freq_ndx)==int:
                result = [result_[0] for result_ in result]
        return result



    def copy_beam(self,Etheta=None, Ephi=None):
        """
        Function that copies a beam object. Optional field array inputs to, eg. rotate copied beam.

        :param Etheta: Optional new theta component of E-field
        :type Etheta: numpy array[complex]
        :param Ephi: Optional new phi component of E-field
        :type Ephi: numpy array[complex]
        
        :returns: Beam copy
        :rtype: class
        """
        
        ret = copy.deepcopy(self)
        if Etheta is not None:
            ret.Etheta = Etheta
        if Ephi is not None:
            ret.Ephi = Ephi
        return ret

    def taper_and_smooth(self, taper=0.03, beam_smooth=None):
        """
        Apply theta taper and optional Gaussian smoothing directly to beam fields.

        :param taper: Ground/sky taper width in radians. If None, no taper is applied.
        :type taper: float
        :param beam_smooth: Standard deviation for Gaussian smoothing along frequency
            (0th) dimension only. If a sequence is provided, only the first entry is used.
            If None, no smoothing is applied.
        :type beam_smooth: float or sequence

        :returns: Self, with modified Etheta and Ephi fields.
        :rtype: class
        """

        if taper is not None:
            if taper <= 0:
                raise ValueError("taper must be positive when provided")
            gtapr = (jnp.arctan((self.theta - jnp.pi / 2) / taper) / jnp.pi + 0.5) ** 2
            tapr = 1.0 - gtapr
            self.Etheta *= tapr[None, :, None]
            self.Ephi *= tapr[None, :, None]

        if beam_smooth is not None:
            beam_sigma = jnp.atleast_1d(beam_smooth)[0]
            sigma = (beam_sigma, 0.0, 0.0)
            self.Etheta = gaussian_filter(self.Etheta, sigma)
            self.Ephi = gaussian_filter(self.Ephi, sigma)

        return self

    def get_Efield_interpolator(self):
        """
        Return a pair of callables that interpolate the E-field components over the sky
        and frequency, pre-scaled by ``sqrt(gain_conv)``.

        Each callable accepts ``(alt, az, freq)`` in radians / MHz (scalars or
        broadcastable arrays) and returns the complex interpolated field at each
        ``(alt, az, freq)`` triplet.

        The coordinate mapping is: ``theta = pi/2 - alt``, ``phi = az mod 2*pi``.

        :returns: ``(interp_Etheta, interp_Ephi)``, each with signature
            ``f(alt, az, freq) -> complex scalar or ndarray`` whose shape matches
            the broadcast shape of the inputs.
        :rtype: tuple(callable, callable)
        """
        

        scale = jnp.sqrt(self.gain_conv)[:, None, None]          # (Nfreq, 1, 1)
        Et = (self.Etheta * scale).transpose(1, 2, 0)           # (Ntheta, Nphi, Nfreq)
        Ep = (self.Ephi   * scale).transpose(1, 2, 0)

        interp_Et = RegularGridInterpolator(
            (self.theta, self.phi, self.freq), Et,
            method='linear', bounds_error=False, fill_value=None,
        )
        interp_Ep = RegularGridInterpolator(
            (self.theta, self.phi, self.freq), Ep,
            method='linear', bounds_error=False, fill_value=None,
        )

        def _wrapper(interp):
            def call(alt, az, freq):
                alt  = jnp.asarray(alt)
                az   = jnp.asarray(az)
                freq = jnp.asarray(freq)
                scalar = alt.ndim == 0 and az.ndim == 0 and freq.ndim == 0
                shape = jnp.broadcast_shapes(alt.shape, az.shape, freq.shape)
                theta = (jnp.pi / 2 - jnp.broadcast_to(alt,  shape)).ravel()
                phi   = (jnp.broadcast_to(az,   shape).ravel()) % (2 * jnp.pi)
                f     = jnp.broadcast_to(freq, shape).ravel()
                out   = interp(jnp.stack([theta, phi, f], axis=-1))  # (N,)
                if scalar:
                    return complex(out[0])
                return out.reshape(shape)
            return call

        return _wrapper(interp_Et), _wrapper(interp_Ep)


    def plotE(self, freqndx, toplot = None, noabs=False):
        """
        Function that plots 1D cuts of the E-field as a function of theta and phi

        :param freqndx: List of frequency bins to plot. Integer indices, not freq values
        :type freqndx: list(int)
        :param toplot: Optional list of coord arrays to plot, eg. [self.Etheta, self.Ephi]. Defaults to full theta and phi arrays.
        :type toplot: list[jnp.array[complex]]
        :param noabs: Whether to plot absolute value of E-field.
        :type noabs: bool
        
        :returns: None
        """
        
        plt.figure(figsize=(15,10))
        for i in range(2):
            plt.subplot(1,2,i+1)
            ax = plt.gca()
            plt.title (['theta','phi'][i])
            toshow = toplot[i] if toplot is not None else [self.Etheta,self.Ephi][i]
            toshow = jnp.real(toshow[freqndx,:,:,i]) if noabs else jnp.abs(toshow[freqndx,:,:,i])
            im=ax.imshow(toshow,interpolation='nearest',extent=[0,360,180,0],origin='upper')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    def get_healpix_alm(self, lmax, freq_ndx=None, other=None,
                        return_I_stokes_only=True, return_complex_components=False):
        """
        Function that produces healpix harmonic maps of beam Stokes power.

        :param lmax: Maximum l value
        :type lmax: int
        :param freq_ndx: Optional list of frequency bin indices. Integer indices, not freq values
        :type freq_ndx: list(int)
        :param other: Optional second beam object for cross-power Stokes maps
        :type other: class
        :param return_I_stokes_only: If True, return only Stokes-I; if False, return [I,Q,U,V]
        :type return_I_stokes_only: bool
        :param return_complex_components: If True, return (real_alm, imag_alm) for each Stokes map
            where imag_alm is None for real-valued Stokes maps.
        :type return_complex_components: bool

        :returns: If return_I_stokes_only is True, returns Stokes-I alm (or tuple(real, imag)
            when return_complex_components=True). Otherwise returns [I,Q,U,V] in the same format.
        :rtype: numpy array or tuple(numpy array, numpy array) or list
        """

        flist = jnp.arange(self.Nfreq, dtype=jnp.int32) if freq_ndx is None else jnp.asarray(np.atleast_1d(freq_ndx), dtype=jnp.int32)
        stokes_maps = self.power_stokes(cross=other)
        if return_I_stokes_only:
            stokes_maps = [stokes_maps[0]]

        def map_to_alm(power_map):
            return _grid2healpix_alm_batch(self.theta, self.phi[:-1], power_map[flist, :, :-1], lmax)

        result = []
        for stokes_map in stokes_maps:
            if return_complex_components:
                alm_real = map_to_alm(jnp.real(stokes_map))
                alm_imag = map_to_alm(jnp.imag(stokes_map)) if jnp.iscomplexobj(stokes_map) else None
                result.append((alm_real, alm_imag))
            else:
                if jnp.iscomplexobj(stokes_map):
                    alm = map_to_alm(jnp.real(stokes_map)) + 1j * map_to_alm(jnp.imag(stokes_map))
                else:
                    alm = map_to_alm(stokes_map)
                result.append(alm)

        if jnp.isscalar(freq_ndx):
            if return_complex_components:
                result = [
                    (stokes_result[0][0], None if stokes_result[1] is None else stokes_result[1][0])
                    for stokes_result in result
                ]
            else:
                result = [stokes_result[0] for stokes_result in result]

        if return_I_stokes_only:
            return result[0]
        return result
