#
# LuSEE Beam
#
import fitsio
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import healpy as hp
from scipy.special import sph_harm
from pyshtools.legendre import legendre
import os


def getLegendre(lmax, theta):
    """
    Function that calculates Legendre polynomial functions up to specified degree
    
    :param lmax: Maximum degree of Legendre functions
    :type lmax: int
    :param theta: Argument of Legendre function
    :type theta: float
    
    :returns: 2D array of Legendre functions, array indices are l and m
    :rtype: numpy array
    
    """
    
    L=legendre(lmax,np.cos(theta),normalization='ortho')
    L[:,1:]/=np.sqrt(2) # m>0 divide by sqrt(2)
    L[:,1::2]*=-1 # * (-1)**m
    return L


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
    dA_theta = np.sin(theta)*dtheta*dphi
    #alm = np.zeros((lmax,lmax),complex)
    ell = np.arange(lmax)
    theta_list, phi_list = np.meshgrid(theta,2*np.pi-phi)
    mmax = lmax
    alm = []
    for m in range(lmax):
        for l in range(m,lmax):        
            harm = sph_harm (m,l, phi_list, theta_list) #yes idiotic convention
            assert(not np.any(np.isnan(harm)))
            alm.append((img*harm.T*dA_theta[:,None]).sum())
    alm = np.array(alm)
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
    
    # lmax has different definitions
    dtheta = theta[1]-theta[0]
    dA_theta = np.sin(theta)*dtheta
    Nphi = len(phi)
    Ntheta = len(theta)
    #alm = np.zeros((lmax,lmax),complex)
    ell = np.arange(lmax)
    rimg = np.fft.rfft(img,axis=1)
    mmax = rimg.shape[1]
    if mmax<lmax:
        rimg = np.hstack((rimg,np.zeros((Ntheta,lmax-mmax+1),complex)))
    alm = np.zeros(lmax*(lmax+1)//2+lmax+1,complex)
    for th_data, th, dA in zip(rimg,theta,dA_theta):
        L = getLegendre(lmax+1,th)
        contr = np.hstack([th_data[m]*L[m:lmax+1,m]*dA for m in range(lmax+1)])
        alm += contr
    alm*=(2*np.pi/Nphi)
    return alm


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
    sin = np.sin
    cos = np.cos
    rad = np.array([ sin(theta[:,None])*cos(phi[None,:]), sin(theta[:,None])*sin(phi[None,:]),
                     -cos(theta[:,None])*np.ones(self.Nphi)[None,:]])
    tphi =  np.array([-sin(phi), +cos(phi)])
    ttheta = np.array([ cos(theta[:,None])*cos(phi[None,:]), cos(theta[:,None])*sin(phi[None,:]),
                     +sin(theta[:,None])*np.ones(self.Nphi)[None,:]])

    Erad = np.einsum('fijk,kij->fij',E,rad)
    Etheta = np.einsum('fijk,kij->fij',E,ttheta)
    Ephi = np.einsum('fijk,kj->fij',E[:,:,:,:2],tphi)
    Emag2 = (np.abs(self.E)**2).sum(axis=3)
    assert(abs(Emag2-np.abs(Erad)**2-np.abs(Etheta)**2-np.abs(Ephi)**2).max()<1e-4)
    #print ((np.abs(Erad)/np.sqrt(Emag2)).max())
    assert(np.all(np.abs(Erad)/np.sqrt(Emag2)<1e-4))
    return Etheta, Ephi


class Beam:
    """
    The main beam class, contains beam data and meta parameters

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
    
    def __init__ (self, fname = None, id = None):
        if fname is None:
            fname = base = os.environ['LUSEE_DRIVE_DIR']+"Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
        if not (os.path.isfile (fname) and os.access(fname, os.R_OK)):
            print (f"Cannot open {fname}")
            stop()
        header = dict(fitsio.read_header(fname))
        fits = fitsio.FITS(fname,'r')
        version = header['VERSION']
        self.id = id
        self.version = version
        self.Etheta = fits['Etheta_real'].read() + 1j*fits['Etheta_imag'].read()
        self.Ephi = fits['Ephi_real'].read() + 1j*fits['Ephi_imag'].read()
        self.ZRe = fits['Z_real'].read()
        self.ZIm = fits['Z_imag'].read()
        self.Z = self.ZRe + 1j*self.ZIm
        self.gain = fits['gain'].read() if 'gain' in fits else None
        if version==1:
            self.f_ground = fits['f_ground'].read()
        elif version==2:
            self.gain_conv = fits['gain_conv'].read()
            self.freq = fits['freq'].read()
            
        self.freq_min = header['FREQ_MIN']
        self.freq_max = header['FREQ_MAX']
        self.Nfreq = header['FREQ_N']
        self.theta_min = header['THETA_MIN']
        self.theta_max = header['THETA_MAX']
        self.Ntheta = header['THETA_N']
        self.phi_min = header['PHI_MIN']
        self.phi_max = header['PHI_MAX']
        self.Nphi = header['PHI_N']
        self.header = header
        if version==1:
            self.freq = np.linspace(self.freq_min, self.freq_max,self.Nfreq)
        self.theta_deg = np.linspace(self.theta_min, self.theta_max,self.Ntheta)
        self.phi_deg = np.linspace(self.phi_min, self.phi_max,self.Nphi)
        self.theta = self.theta_deg/180*np.pi
        self.phi = self.phi_deg/180*np.pi
        if (self.phi_max != 360) or (self.phi_min != 0):
            print("Code might implicitly assume phi wraparound ... use with care.")
        
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
        rad = deg/180*np.pi
        cosrad = np.cos(rad)
        sinrad = np.sin(rad)
        # some sanity checks
        assert (self.phi_max == 360)
        assert (self.phi_min == 0)
        phi_step = (self.phi_max-self.phi_min)/(self.Nphi-1)
        assert (deg%phi_step == 0)
        m = int(deg // phi_step)
        if (m<0):
            Etheta = np.concatenate ((self.Etheta[:,:,m-1:],self.Etheta[:,:,1:m]),axis=2)
            Ephi = np.concatenate ((self.Ephi[:,:,m-1:],self.Ephi[:,:,1:m]),axis=2)
        else:
            Etheta = np.concatenate ((self.Etheta[:,:,m:],self.Etheta[:,:,1:m+1]),axis=2)
            Ephi = np.concatenate ((self.Ephi[:,:,m:],self.Ephi[:,:,1:m+1]),axis=2)

        # No need to rotate in theta  - phi
        #rotmat = np.array(([[cosrad, +sinrad, 0],[-sinrad,cosrad,0],[0,0,1]]))
        #E = np.einsum('fabj,ij->fabi',E,rotmat)


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
        Ephi = np.concatenate ((self.Ephi[:,:,n:0:-1],self.Ephi[:,:,self.Nphi:n-1:-1]),axis=2)
        #E[:,:,:,0]*=-1 ## X flips over
        return self.copy_beam(E=E)

    def power(self):
        """
        Function that calculates the beam power of a single beam
        
        :returns: Beam power
        :rtype: float
        """
        
        P = np.abs(self.Etheta**2)+np.abs(self.Ephi**2)
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
            I = np.abs(self.Etheta*self.Etheta)+np.abs(self.Ephi*self.Ephi)
            Q = np.abs(self.Etheta**2)-np.abs(self.Ephi**2)
            T = 2*self.Etheta*np.conj(self.Ephi)
            U = np.real(T)
            V = np.imag(T)
        else:
            I = self.Etheta*np.conj(cross.Etheta) + self.Ephi*np.conj(cross.Ephi)
            Q = self.Etheta*np.conj(cross.Etheta) - self.Ephi*np.conj(cross.Ephi)
            U = self.Etheta*np.conj(cross.Ephi)+self.Ephi*np.conj(cross.Etheta)
            V = +1j*self.Etheta*np.conj(cross.Ephi)-self.Ephi*np.conj(cross.Etheta)
        return [I,Q,U,V]

    
    def cross_power(self, other):
        """
        Function that calculates the cross-power between two beams

        :param other: Second beam object for cross-power
        :type other: class
        
        :returns: Cross power
        :rtype: float
        """
        
        xP = self.Etheta*np.conj(other.Etheta) + self.Ephi*np.conj(other.Ephi)
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
        dA_theta = np.sin(self.theta)*dtheta*dphi
        f_sky = np.array([(dA_theta[:,None]*gain[i,:,:-1]).sum()/(4*np.pi) for i in range(self.Nfreq)])
        return f_sky
    
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
        
        flist = range(self.Nfreq) if freq_ndx is None else np.atleast_1d(freq_ndx)
        if cross is None:
            result =  [[grid2healpix(self.theta,self.phi[:-1], P_[i,:,:-1], ellmax, Nside) 
                    for i in flist] for P_ in P]
        else:
            result =  [[grid2healpix(self.theta,self.phi[:-1], np.real(P_[i,:,:-1]), ellmax, Nside) 
                +1j*grid2healpix(self.theta,self.phi[:-1], np.imag(P_[i,:,:-1]), ellmax, Nside)
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

    
    def plotE(self, freqndx, toplot = None, noabs=False):
        """
        Function that plots 1D cuts of the E-field as a function of theta and phi

        :param freqndx: List of frequency bins to plot. Integer indices, not freq values
        :type freqndx: list(int)
        :param toplot: Optional list of coord arrays to plot, eg. [self.Etheta, self.Ephi]. Defaults to full theta and phi arrays.
        :type toplot: list[np.array[complex]]
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
            toshow = np.real(toshow[freqndx,:,:,i]) if noabs else np.abs(toshow[freqndx,:,:,i]) 
            im=ax.imshow(toshow,interpolation='nearest',extent=[0,360,180,0],origin='upper')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    def get_healpix(self,lmax, field, freq_ndx = None):
        """
        Function that produces a healpix map of specified field

        :param lmax: Maximum l value
        :type lmax: int
        :param field: Field to map (eg. Etheta)
        :type field: numpy array[complex]
        :param freq_ndx: Optional list of frequency bin indices. Integer indices, not freq values
        :type freq_ndx: list(int)
        
        :returns: Healpix field map
        :rtype: numpy array
        """
        
        if freq_ndx is None:
            freq_ndx = range(self.Nfreq)
        return  np.array([grid2healpix_alm_fast(self.theta,self.phi[:-1], field[fi,:,:-1],
                                                lmax) for fi in freq_ndx])
