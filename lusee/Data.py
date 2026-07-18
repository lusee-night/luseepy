import  numpy           as np
import  fitsio
from    .Observation import Observation
from .LunarCalendar  import LunarCalendar
from .Throughput import Throughput
from .LabeledArray import label, FRAME_TOPO

class ThroughputBeam:
    """
    Dummy class that contains complex impedance for Throughput class

    :param freq: Frequency
    :type freq: float
    :param Z: Complex impedance
    :type Z: numpy array[complex]
    """
    ## dummy class to carry impedance for throughput
    def __init__(self, freq,Z):
        self.freq = freq
        self.Z = Z

class Data(Observation):
    """
    Class that holds data from an observation

    :param filename: Filename of observation to load
    :type filename: str
    :param throughput: Front-end throughput parameters
    :type throughput: class
    """
    
    def __init__(self, filename, throughput=None):
        header = dict(fitsio.read_header(filename))
        version = header['VERSION']
        lunar_day    = header['LUNAR_DAY']
        lun_lat_deg  = header['LUN_LAT_DEG']
        lun_long_deg = header['LUN_LONG_DEG']
        lun_height_m = header['LUN_HEIGHT_M']
        deltaT_sec   = header['DELTAT_SEC']
        super().__init__(lunar_day, lun_lat_deg, lun_long_deg, lun_height_m, deltaT_sec)
        fits = fitsio.FITS(filename,'r')
        self.data = fits['data'].read()
        # Units of the raw 'data' array.  Simulator FITS output is brightness
        # temperature in Kelvin; an HDF5 ingest path may set this to e.g.
        # "counts" before gain conversion.
        self.data_units = "K"
        self.freq = fits['freq'].read()
        combinations = fits['combinations'].read()
        cc = 0
        comb2ndx = {}
        NBeams = 0
        for i,j in combinations:
            comb2ndx [(i,j)] = cc
            Nbeams = max(NBeams,i,j)
            if i==j:
                cc+=1
            else:
                cc+=2
        self.comb2ndx = comb2ndx
        # sanity checks
        if (i==j):
            assert(cc == self.data.shape[1])
        else:
            assert(cc == self.data.shape[1]+1)
            
        assert (len(self.times) == self.data.shape[0])
        assert (len(self.freq) == self.data.shape[2])
        
        Nbeams +=1
        self.ZRe = []
        self.ZIm = []
        for i in range(Nbeams):
            self.ZRe.append (fits[f'ZRe_{i}'].read())
            self.ZIm.append (fits[f'ZIm_{i}'].read())
        self.Nbeams = NBeams
        self.Nfreq = len(self.freq)
        self.Ntimes = len(self.times)
        self.NComb = len(self.comb2ndx)
        tbeam = ThroughputBeam(self.freq, self.ZRe[0]+1j*self.ZIm[0])
        ## this might need fixing.
        self.T = [Throughput(beam=tbeam) if throughput is None else throughput]*Nbeams
        self.T2Vsq = [T.T2Vsq(self.freq) for T in self.T]

    def __getitem__(self, req):
        # Can do things like
        #   O[0,"01R",:]
        # or 
        #   O[0,(0,1,'R'),:]
        
        day,comb,freq = req
        fact = np.ones(self.Nfreq) # frequency scaling
        if type(comb) == str:
            if comb[0]=="-":
                fact=-1
                comb=comb[1:]
            i = int(comb[0])
            j = int(comb[1])
            if len(comb)>=3:
                what = comb[2:]
            else:
                what = 'R' if (i==j) else 'C'
        else:
            i,j = comb
            if len(comb)==3:
                what = comb[2]
            else:
                what = 'R' if (i==j) else 'C'

       
        vwhat = what[1:]
        what = what[0] 
    
        ndx = self.comb2ndx[(i,j)]
        if what == "R":
            toret = self.data[day,ndx,freq]
        if what == "I":
            assert (i!=j)
            toret= self.data[day,ndx+1,freq]
        if what == "C":
            assert (i!=j)
            toret=self.data[day,ndx,freq]+1j*self.data[day,ndx+1,freq]


        if vwhat == "":
            return label(toret, units=self.data_units, frame=FRAME_TOPO)
        elif vwhat == "V":
                ## ffact can be scalar +1 or -1
                # T2Vsq = 4 kB Re(Z) Gamma^2 is V^2/(Hz K), so multiplying a
                # Kelvin (auto/cross) value by sqrt(T2Vsq_i T2Vsq_j) yields a
                # voltage power spectral density in V^2/Hz, not amplitude.
                T2V = np.sqrt(self.T2Vsq[i]*self.T2Vsq[j])[freq]
                if toret.ndim == 1:
                    return label(toret*T2V, units="V^2/Hz", frame=FRAME_TOPO)
                else:
                    return label(toret*T2V[None,:], units="V^2/Hz", frame=FRAME_TOPO)
        else:
            raise NotImplementedError

        # Should not get here.
        raise NotImplementedError


            
            
    
