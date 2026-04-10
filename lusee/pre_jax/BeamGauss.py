import numpy as np

from .Beam import Beam
from ..frequencies import ALL_FREQUENCIES_MHZ_NP

def gauss_beam(theta,phi,sigma, theta_c,phi_c=0.):
    """

    Function that creates a map-level gaussian beam in theta and phi of width sigma, centered at theta=theta_c and phi=phi_c.
    E = 0 deg, N = 90 deg (topocentric frame)

    :param theta: theta in radians (from local zenith)
    :type theta: array
    :param phi: azimuth in radians (angle around the local zenith, with E=0, N=90)
    :type phi: array
   
    :param sigma: Beam width (standard deviation)
    :type sigma: float
    :param theta_c: Beam center theta
    :type theta_c: float
    :param phi_c: Beam center phi
    :type phi_c: float
    """
    
    vec  = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    vec_c= [np.sin(theta_c)*np.cos(phi_c), np.sin(theta_c)*np.sin(phi_c), np.cos(theta_c)]
    cos_angle = vec[0]*vec_c[0] + vec[1]*vec_c[1] + vec[2]*vec_c[2]
    return np.exp(- (np.arccos(cos_angle))**2/(2*sigma**2))



class BeamGauss(Beam):
    """
    Class that creates a Gaussian Beam object, centered at the given altitude and azimuth and of width sigma. 

    :param Beam: Beam object
    :type Beam: class
    :param alt_deg: Altitude of the center of the gaussian beam, in degrees 
    :type alt_deg: float
    :param az_deg: Azimuth of the center of the gaussian beam, in degrees, az=0->E, az=90->N
    :type az_deg: float
    :param sigma_deg: Sigma of the gaussian beam at 1MHz, in degrees 
    :type sigma_deg: float
    :param one_over_freq_scaling: Whether to scale beam sigma with 1/f
    :type one_over_freq_scaling: bool
    :param id: ID string for beam, optional
    :type id: str
    """
    def __init__ (self, alt_deg, az_deg=0, sigma_deg=20.0, one_over_freq_scaling=False, id = None):     
        self.version=2.1 #what should this be? 
        # v1 so that self.freq followed the simulator's 1-50 MHz grid
        # >v2 so that self.ground_fraction() can be calculated
        
        self.id = id
        self.freq_min = 1.
        self.freq_max = 50.
        self.Nfreq = 50
        self.theta_min = 0.
        self.theta_max = 90.
        self.Ntheta = 91
        self.phi_min = 0.
        self.phi_max = 360.
        self.Nphi = 361

        self.ZRe=np.zeros(self.Nfreq) #Need this for lusee.Simulator.write()
        self.ZIm=np.zeros(self.Nfreq) #Need this for lusee.Simulator.write()

        
        self.freq = ALL_FREQUENCIES_MHZ_NP
        self.theta_deg = np.linspace(self.theta_min, self.theta_max,self.Ntheta)
        self.phi_deg = np.linspace(self.phi_min, self.phi_max,self.Nphi)
        self.theta = self.theta_deg*np.pi/180.
        self.phi = self.phi_deg*np.pi/180.

        #initialize E
        self.Etheta = np.zeros((self.Nfreq, self.Ntheta, self.Nphi),complex)
        self.Ephi = np.zeros_like(self.Etheta)
        
        # initialize gain_conv and related quantities. 
        # Need to set self.gain_conv so that ground fraction is zero.
        self.gain_conv=np.ones(self.Nfreq)
        dphi=self.phi[1]-self.phi[0]
        dtheta=self.theta[1]-self.theta[0]
        dA_theta=np.sin(self.theta)*dtheta*dphi

        #convert to radians and create meshgrid
        sigma=np.deg2rad(sigma_deg)
        alt_rad=np.deg2rad(alt_deg)
        az_rad=np.deg2rad(az_deg)
        phi_rad = (np.pi/2 - az_rad) 
        theta_rad = np.pi/2 - alt_rad   
        Phi,Theta=np.meshgrid(self.phi,self.theta)

        if one_over_freq_scaling: #slow code, hence separate
            for f,freq in enumerate(self.freq):
                #scale sigma with 1/freq
                sigma_freq=sigma*(10.0/freq) if one_over_freq_scaling else sigma
                
                #create gauss beam centered at altitude=alt and azimuth=az of width sigma_freq
                beam=gauss_beam(Theta,Phi,sigma_freq,theta_rad,phi_rad).astype(complex)
                assert(beam.shape==self.Etheta[f,:,:].shape)
                self.Etheta[f,:,:]=beam

                #set gain_conv such that ground_fraction() is zero
                factor=(dA_theta[:,None]*self.power()[f,:,:-1]).sum()/(4*np.pi)
                self.gain_conv[f]/=factor
        else:
            #create gauss beam centered at altitude=alt and azimuth=az of width sigma
            beam=gauss_beam(Theta,Phi,sigma,theta_rad,phi_rad).astype(complex)
            assert(beam.shape==self.Etheta[0,:,:].shape)
            for f,freq in enumerate(self.freq):
                self.Etheta[f,:,:]=beam

            #set gain_conv such that ground_fraction() is zero
            factor=(dA_theta[:,None]*self.power()[0,:,:-1]).sum()/(4*np.pi)
            self.gain_conv/=factor

        assert(np.all(np.abs(self.ground_fraction())<1e-3)) #confirm ground_fraction==zero


