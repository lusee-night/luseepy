import numpy as np
import os
from scipy.interpolate import interp1d
import  astropy.constants  as const
from .Beam import Beam

class Throughput:
    """
    Class that holds front-end throughput parameters

    :param beam: Beam class object
    :type beam: class
    :param noise_e: Amplifier noise in nV/rtHz
    :type noise_e: float
    :param Cfront: Front-end capacitance in pico-farads
    :type Cfront: float
    :param R4: Front-end resistance in Ohms
    :type R4: float
    """

    def __init__ (self, beam=None, noise_e = 2, Cfront = 35, R4 = 1e6):
        self.noise_e = noise_e
        self.Cfront = 35
        self.R4 = R4
        self._load_spice_sims()
        self.beam = beam if beam is not None else Beam()
        
    def _load_spice_sims(self):
        path = os.path.join(os.environ['LUSEE_DRIVE_DIR'],'Simulations/ElectronicsModel/Model54')
        f,n = np.loadtxt(os.path.join(path,'spectrometer_54-noise.dat')).T
        self.noise = interp1d(f/1e6,n**2*1e-18) ## in V^2/Hz
        f,g,p=np.loadtxt(os.path.join(path,'spectrometer_54_gain-pre.dat')).T
        self._preamp_gain = interp1d(f/1e6,10**(g/20)*np.exp(1j*p/180*np.pi))
        self._gain={}
        for l in "LMH":
            f,g,p=np.loadtxt(os.path.join(path,f'spectrometer_54_gain{l}.dat')).T
            self._gain[l] = interp1d(f/1e6,10**(g/20.)*np.exp(1j*p/180*np.pi))

    def complex_gain(self,freq_MHz, gain_set = 'M'):
        """
        Function that calculates the complex gain of the front-end amplifiers at a specified frequency

        :param freq_MHz: Frequency in MHz
        :type freq_MHz: float
        :param gain_set: Gain setting
        :type gain_set: str

        :returns: Complex gain at input frequency
        :rtype: complex
        """
        return self._gain[gain_set](freq_MHz) # preamp again included in gain set

    def power_gain(self,freq_MHz, gain_set = 'M'):
        """
        Function that calculates the gain of the front-end amplifiers in power at a specified frequency

        :param freq_MHz: Frequency in MHz
        :type freq_MHz: float
        :param gain_set: Gain setting
        :type gain_set: str

        :returns: Gain in power
        :rtype: float
        """
        c = self.complex_gain(freq_MHz, gain_set)
        return np.abs(c**2)
    
    def setCfront(self,Cfront):
        """
        Function that sets the front-end capacitance in the Throughput class

        :param Cfront: Front-end capacitance in pico-farads
        :type Cfront: float

        :returns: None
        :rtype: None
        """
        self.Cfront = Cfront
        #self._calc_conversion_factors()


    def AntennaImpedanceInterp(self,f):
        """
        Function that extrapolates antenna impedance as a function of frequency

        :param f: Array of frequencies at which to calculate impedance
        :type f: array

        :returns: Antenna impedance
        :rtype: array
        """
        out = np.zeros_like(f,complex)
        bfreq = self.beam.freq
        fmin = bfreq[0]
        out[f>=fmin] = interp1d(bfreq, self.beam.Z,fill_value="extrapolate")(f[f>=fmin])
        alpha = (np.log(-np.imag(self.beam.Z[1]))-np.log(-np.imag(self.beam.Z[0]))) / (np.log(bfreq[1])-np.log(bfreq[0]))
        Ai = -np.imag(self.beam.Z[0])*(f[f<fmin]/fmin)**alpha
        out [f<fmin] = -Ai*1j
        return out
        

        
    def Gamma_VD(self,freq):
        """
        Function that calculates gamma at a specified frequency for antenna impedance matching

        :param freq: Frequency in Hz
        :type freq: float

        :returns: Gamma_VD
        :rtype: float
        """
        omega = 2*np.pi*freq*1e6
        Zrec  = 1/(1j*omega*(self.Cfront*1e-12) + 1/self.R4)
        ZAnt = self.AntennaImpedanceInterp(freq)
        Gamma_VD = np.abs(Zrec)/np.abs((ZAnt+Zrec)) ##2 as per t
        return Gamma_VD
    
    def T2Vsq(self,freq):
        """
        Function that calculates 4*k_B*R*Gamma^2 for antenna match

        :param freq: Frequency  [AS: clarify units - MHz or Hz?]
        :type freq: float

        :returns: T2Vsq
        :rtype: float
        """
        kB = const.k_B.value
        c = const.c.value
        ## 1 / i w C , 1e6 = MHz, 1e-12 is pico (farad)
        ZAnt = interp1d(self.beam.freq, self.beam.Z,fill_value="extrapolate")(freq)
        T2Vsq = 4*kB*np.real(ZAnt)*self.Gamma_VD(freq)**2
        return T2Vsq


    def SG2V(self,freq):
        """
        Function that calculates 4*lambda*R*Gamma^2 for antenna match

        :param freq: Frequency
        :type freq: float

        :returns: T2Vsq
        :rtype: float
        """
        kB = const.k_B.value
        c = const.c.value
        ## 1 / i w C , 1e6 = MHz, 1e-12 is pico (farad)
        ZAnt = interp1d(self.beam.freq, self.beam.Z,fill_value="extrapolate")(freq)
        lamb = c/(freq*1e6)
        T2Vsq = 4*lamb*np.real(ZAnt)*self.Gamma_VD(freq)**2
        return T2Vsq

        
