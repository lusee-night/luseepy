from math import degrees
from .LBeam import LBeam
import numpy as np

def gauss_beam(theta, theta_fwhm):
    """
    Creates a map-level gaussian beam centered at zero

    for a gaussian, fwhm = 2.355*sigma
    """

    return 2.355/(np.sqrt(2*np.pi)*theta_fwhm)*np.exp(-2.355**2 * theta**2 / (2*theta_fwhm**2) )

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
        

        # to check with maps
        Etheta_theta=gauss_beam(self.theta,np.deg2rad(sigma_deg))*gauss_beam(np.deg2rad(dec_deg),np.deg2rad(sigma_deg))
        Etheta_phi=gauss_beam(self.phi,np.deg2rad(sigma_deg))

        self.Etheta=self.Etheta[:,Etheta_theta,Etheta_phi]

        
        # need to set self.gain_conv so that ground fraction is zero.
        self.gain_conv=np.zeros(self.Nfreq)
       
        # at the end we want ground fraction to be zero

        assert(np.all(np.abs(self.ground_fraction())<1e-3))



