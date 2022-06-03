from curses.ascii import ETB
from math import degrees
from .LBeam import LBeam
import numpy as np
import scipy

def gauss_beam(theta,phi,sigma, theta_c):
    """
    Creates a map-level gaussian beam in theta, phi of width sigma, centered at theta=theta_c and phi=0
    Uses a naive gaussian function 
    """
    #ToDo:Verify theta convention: altitude
    
    #the naive Gaussian beam, White(1995): https://adsabs.harvard.edu/full/1995ApJ...443....6W
    #this naive definition makes sense however it does not wrap around correctly for theta or phi. 
    # Appropriate way could be 1. to implement wrap around by hand, 
    #                       or 2. use the alternate idea from Challinor(2000) above.
    return 1/(2*np.pi*sigma**2)* np.exp(- (theta-theta_c)**2/(2*sigma**2)) * np.exp(- phi**2/(2*(sigma*np.cos(theta))**2))
    
    #alternate idea in Challinor(2000), Sec5A: https://arxiv.org/pdf/astro-ph/0008228.pdf
    # allows for easier analytic results, and converges to the naive definition above for sigma<<1
    # also, wrap around is natural.


class LBeam_Gauss(LBeam):
    def __init__ (self, dec_deg, sigma_deg):
        # for the time being, let's hard code everything that we need.
        # We can make all of these things options, if required
        self.freq_min = 1.
        self.freq_max = 50.
        self.Nfreq = 50.
        self.theta_min = 0.
        self.theta_max = 90.
        self.Ntheta = 91
        self.phi_min = 0.
        self.phi_max = 360.
        self.Nphi = 361

        self.Etheta = np.zeros((self.Nfreq, self.Ntheta, self.Nphi),complex)
        self.Ephi = np.zeros_like(self.Etheta)
        
        #convert to radians and create meshgrid
        sigma=np.deg2rad(sigma_deg)
        dec=np.deg2rad(dec_deg)
        Phi,Theta=np.meshgrid(self.phi,self.theta)

        #create gauss beam centered at theta=phi=0
        beam=gauss_beam(Theta,Phi,sigma,dec).astype(complex)
        assert(beam.shape==self.Etheta[0,:,:].shape)

        #achromatic
        for freq in self.Etheta[:,0,0]:
            self.Etheta[freq,:,:]=beam
        #need to check with maps
        
        # need to set self.gain_conv so that ground fraction is zero.
        self.gain_conv=np.zeros(self.Nfreq)
       
        # at the end we want ground fraction to be zero

        assert(np.all(np.abs(self.ground_fraction())<1e-3))



