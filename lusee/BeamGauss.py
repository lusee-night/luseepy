from .Beam import Beam
import numpy as np

def gauss_beam(theta,phi,sigma, theta_c,phi_c=0.):
    """
    Creates a map-level gaussian beam in theta, phi of width sigma, centered at theta=theta_c and phi=phi_c
    Uses a naive gaussian function, with wrap around for theta, phi
    """
    
    phiprime=np.min((phi-phi_c,2*np.pi-phi+phi_c),axis=0) #phi wrap around
    norm=1. #beam E^2 is not normalized, E^2*gain_conv is normalized

    return norm*np.exp(- (theta-theta_c)**2/(2*sigma**2)) * np.exp(- (phiprime)**2/(2*(sigma/np.cos(theta))**2))
    

class BeamGauss(Beam):
    """
    Gaussian LBeam object, centered at the given declination (and phi=360-azimuth=0) and of width sigma. 
    """
    def __init__ (self, dec_deg, sigma_deg, phi_deg=90, one_over_freq_scaling=False, id = None):
        """
        dec_deg : declination of the center of the gaussian beam, in degrees
        phi_deg : phi center of the gaussian beam, in degrees, phi=0->E, phi=90->N
        sigma_deg : sigma of the gaussian beam at 1MHz, in degrees
        """
        
        self.version=2.1 #what should this be? 
        # v1 so that self.freq=np.linspace as below
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

        
        self.freq = np.linspace(self.freq_min, self.freq_max,self.Nfreq)
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
        dec=np.deg2rad(dec_deg)
        phi_rad=np.deg2rad(phi_deg)
        self.declination=np.pi/2 - self.theta
        Phi,Declination=np.meshgrid(self.phi,self.declination)

        if one_over_freq_scaling: #slow code, hence separate
            for f,freq in enumerate(self.freq):
                #scale sigma with 1/freq
                sigma_freq=sigma*(10.0/freq) if one_over_freq_scaling else sigma
                
                #create gauss beam centered at declination=dec and phi=0 of width sigma_freq
                beam=gauss_beam(Declination,Phi,sigma_freq,dec,phi_rad).astype(complex)
                assert(beam.shape==self.Etheta[f,:,:].shape)
                self.Etheta[f,:,:]=beam

                #set gain_conv such that ground_fraction() is zero
                factor=(dA_theta[:,None]*self.power()[f,:,:-1]).sum()/(4*np.pi)
                self.gain_conv[f]/=factor
        else:
            #create gauss beam centered at declination=dec and phi=0 of width sigma
            beam=gauss_beam(Declination,Phi,sigma,dec,phi_rad).astype(complex)
            assert(beam.shape==self.Etheta[0,:,:].shape)
            for f,freq in enumerate(self.freq):
                self.Etheta[f,:,:]=beam

            #set gain_conv such that ground_fraction() is zero
            factor=(dA_theta[:,None]*self.power()[0,:,:-1]).sum()/(4*np.pi)
            self.gain_conv/=factor

        assert(np.all(np.abs(self.ground_fraction())<1e-3)) #confirm ground_fraction==zero



