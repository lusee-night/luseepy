import  numpy           as np
import  fitsio          
from    .Observation import Observation
from .LunarCalendar  import LunarCalendar
from .Throughput import Throughput

class Data(Observation):
    def __init__(self, filename, throughput=None):
        """
           noise_e is amplifier noise in nV/rtHz
           Cfront is front-end capacticance in pico-farads
        """
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
        self.T = Throughput() if throughput is None else throughput
        self.T2Vsq = self.T.T2Vsq(self.freq)

    def __getitem__(self, req):
        """
         Can do things like
           O[0,"01R",:]
        or 
          O[0,(0,1,'R'),:]
        """
        
        day,comb,freq = req
        fact = +1
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

        if len(what)>1:
            wfact = what[1:]
            if wfact == "V":
                fact *= np.sqrt(self.T2Vsq[i]*self.T2Vsq[j])
            else:
                print ('X',fact,'y')
                raise NotImplemented
            what=what[0]
        ndx = self.comb2ndx[(i,j)]
        if what == "R":
            return fact*self.data[day,ndx,freq]
        if what == "I":
            assert (i!=j)
            return fact*self.data[day,ndx+1,freq]
        if what == "C":
            assert (i!=j)
            return fact*self.data[day,ndx,freq]+1j*self.data[day,ndx+1,freq]

        # Should not get here.
        raise NotImplemented


            
            
    
