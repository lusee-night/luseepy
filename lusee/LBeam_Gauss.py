from .LBeam import LBeam
import numpy as np

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

        # rugved fill in

        #...
        
        # need to set self.gain_conv so that ground fraction is zero.
        
        #...
        
        # at the end we want ground fraction to be zero

        np.assert(np.all(np.abs(self.ground_fraction())<1e-3))



