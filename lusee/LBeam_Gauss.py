from .LBeam import LBeam
import numpy as np

def gauss_beam(theta,phi,sigma, theta_c,phi_c=0.):
    """
    Creates a map-level gaussian beam in theta, phi of width sigma, centered at theta=theta_c and phi=phi_c
    Uses a naive gaussian function, with wrap around for theta, phi
    """
    
    phiprime=np.min((phi-phi_c,2*np.pi-phi+phi_c),axis=0) #phi wrap around
    norm=1. #beam E^2 is not normalized, E^2*gain_conv is normalized

    return norm*np.exp(- (theta-theta_c)**2/(2*sigma**2)) * np.exp(- (phiprime)**2/(2*(sigma/np.cos(theta))**2))
    

class LBeam_Gauss(LBeam):
    """
    Gaussian LBeam object, centered at the given declination (and azimuth=0) and of width sigma. 
    """
    def __init__ (self, dec_deg, sigma_deg, azimuth_deg=0):
        """
        dec_deg : declination of the center of the gaussian beam, in degrees
        azimuth_deg : azimuth of the center of the gaussian beam, in degrees
        sigma_deg : sigma of the gaussian beam, in degrees
        """
        
        self.version=2.1 #what should this be? 
        # v1 so that self.freq=np.linspace as below
        # >v2 so that self.ground_fraction() can be calculated
        
        self.freq_min = 1.
        self.freq_max = 50.
        self.Nfreq = 50
        self.theta_min = 0.
        self.theta_max = 90.
        self.Ntheta = 91
        self.phi_min = 0.
        self.phi_max = 360.
        self.Nphi = 361

        
        self.freq = np.linspace(self.freq_min, self.freq_max,self.Nfreq)
        self.theta_deg = np.linspace(self.theta_min, self.theta_max,self.Ntheta)
        self.phi_deg = np.linspace(self.phi_min, self.phi_max,self.Nphi)
        self.theta = self.theta_deg*np.pi/180.
        self.phi = self.phi_deg*np.pi/180.

        self.Etheta = np.zeros((self.Nfreq, self.Ntheta, self.Nphi),complex)
        self.Ephi = np.zeros_like(self.Etheta)
        
        #convert to radians and create meshgrid
        sigma=np.deg2rad(sigma_deg)
        dec=np.deg2rad(dec_deg)
        azimuth=np.deg2rad(azimuth_deg)
        self.declination=np.pi/2 - self.theta
        Phi,Declination=np.meshgrid(self.phi,self.declination)

        #create gauss beam centered at declination=dec and phi=0 of width sigma
        beam=gauss_beam(Declination,Phi,sigma,dec,azimuth).astype(complex)
        assert(beam.shape==self.Etheta[0,:,:].shape)

        #achromatic
        for freq in self.freq:
            self.Etheta[int(freq-1),:,:]=beam
        
        # need to set self.gain_conv so that ground fraction is zero.
        self.gain_conv=np.ones(self.Nfreq)

        dphi=self.phi[1]-self.phi[0]
        dtheta=self.theta[1]-self.theta[0]
        dA_theta=np.sin(self.theta)*dtheta*dphi

        factor=(dA_theta[:,None]*self.power()[0,:,:]).sum()/(4*np.pi) #same factor for all frequencies
        self.gain_conv/=factor

        assert(np.all(np.abs(self.ground_fraction())<1e-3)) #confirm ground_fraction==zero



