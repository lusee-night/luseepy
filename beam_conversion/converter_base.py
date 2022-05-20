import fitsio
import sys
import os
import numpy as np
import glob

class BeamConverter:

    def __init__(self, root, thetamax = 90):
      self.root = root
      self.thetamax = thetamax

    def find_single_file(self, pattern, filt = None, ok_if_not_found = False):
        pat = os.path.join(self.root,pattern)
        flist = glob.glob(pat)
        if filt is not None:
            flist = list(filter(lambda x:filt in x,flist))
        if (len(flist))==0:
            if ok_if_not_found:
                return None
            else:
                print (f"Couldn't find matching file {pat}")
                sys.exit(1)
        if (len(flist)>1):
            print (f"Multiple file candidates:",flist)
            sys.exit(1)
        return flist[0]
            
    
      
    def save_fits(self,outfile):
        ## version history:
        # v1 initial version
        # v2 
        #    fground replaced with a gain_conv field (E^2*gain_conv is gain)
        #    freq does not need to be on a regular grid

        header = {'version':2,
                  'freq_min':self.freq_min,
                  'freq_max':self.freq_max,
                  'freq_N':self.Nfreq,
                  'theta_min':self.theta_min,
                  'theta_max':self.theta_max,
                  'theta_N':self.Ntheta,
                  'phi_min':self.phi_min,
                  'phi_max':self.phi_max,
                  'phi_N':self.Nphi,
                  'source': 'FEKO',
                  'source_root': self.root
        }

        print ('Saving to',outfile,'... ',end="")
        sys.stdout.flush()
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        fits.write(np.real(self.Etheta), extname = 'Etheta_real', header=header)
        fits.write(np.imag(self.Etheta), extname = 'Etheta_imag')
        fits.write(np.real(self.Ephi), extname = 'EPhi_real')
        fits.write(np.imag(self.Ephi), extname = 'EPhi_imag')
        fits.write(self.gainconv, extname = 'gain_conv')
        fits.write(self.freq, extname = 'freq')
        fits.write(self.ZRe, extname = 'Z_real')
        fits.write(self.ZIm, extname = 'Z_imag')
        fits.close()
        print ('Done.')
