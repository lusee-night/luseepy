from .Beam import Beam
from .frequencies import ALL_FREQUENCIES_MHZ
import jax
import jax.numpy as jnp

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
    
    vec  = [jnp.sin(theta)*jnp.cos(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(theta)]
    vec_c= [jnp.sin(theta_c)*jnp.cos(phi_c), jnp.sin(theta_c)*jnp.sin(phi_c), jnp.cos(theta_c)]
    cos_angle = vec[0]*vec_c[0] + vec[1]*vec_c[1] + vec[2]*vec_c[2]
    return jnp.exp(- (jnp.arccos(cos_angle))**2/(2*sigma**2))


@jax.tree_util.register_pytree_node_class
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

        self.ZRe=jnp.zeros(self.Nfreq) #Need this for lusee.Simulator.write()
        self.ZIm=jnp.zeros(self.Nfreq) #Need this for lusee.Simulator.write()

        
        self.freq = ALL_FREQUENCIES_MHZ
        self.theta_deg = jnp.linspace(self.theta_min, self.theta_max,self.Ntheta)
        self.phi_deg = jnp.linspace(self.phi_min, self.phi_max,self.Nphi)
        self.theta = self.theta_deg*jnp.pi/180.
        self.phi = self.phi_deg*jnp.pi/180.

        #initialize E
        self.Etheta = jnp.zeros((self.Nfreq, self.Ntheta, self.Nphi),complex)
        self.Ephi = jnp.zeros_like(self.Etheta)
        
        # initialize gain_conv and related quantities. 
        # Need to set self.gain_conv so that ground fraction is zero.
        self.gain_conv=jnp.ones(self.Nfreq)
        dphi=self.phi[1]-self.phi[0]
        dtheta=self.theta[1]-self.theta[0]
        dA_theta=jnp.sin(self.theta)*dtheta*dphi

        #convert to radians and create meshgrid
        sigma=jnp.deg2rad(sigma_deg)
        alt_rad=jnp.deg2rad(alt_deg)
        az_rad=jnp.deg2rad(az_deg)
        phi_rad = (jnp.pi/2 - az_rad)
        theta_rad = jnp.pi/2 - alt_rad
        Phi,Theta=jnp.meshgrid(self.phi,self.theta)

        if one_over_freq_scaling: #slow code, hence separate
            sigma_freq=sigma*(10.0/self.freq)
            
            #create gauss beam centered at altitude=alt and azimuth=az of width sigma_freq
            self.Etheta=jax.vmap(lambda sigma_f: gauss_beam(Theta,Phi,sigma_f,theta_rad,phi_rad).astype(complex))(sigma_freq)
            assert(self.Etheta.shape==(self.Nfreq, self.Ntheta, self.Nphi))

            #set gain_conv such that ground_fraction() is zero
            factor=(dA_theta[None,:,None]*self.power()[:,:,:-1]).sum(axis=(1,2))/(4*jnp.pi)
            self.gain_conv=self.gain_conv/factor
        else:
            #create gauss beam centered at altitude=alt and azimuth=az of width sigma
            beam=gauss_beam(Theta,Phi,sigma,theta_rad,phi_rad).astype(complex)
            assert(beam.shape==self.Etheta[0,:,:].shape)
            self.Etheta=jnp.broadcast_to(beam[None,:,:], self.Etheta.shape)

            #set gain_conv such that ground_fraction() is zero
            factor=(dA_theta[:,None]*self.power()[0,:,:-1]).sum()/(4*jnp.pi)
            self.gain_conv=self.gain_conv/factor

        assert(jnp.all(jnp.abs(self.ground_fraction())<1e-3)) #confirm ground_fraction==zero
