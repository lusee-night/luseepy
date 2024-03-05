import fitsio
import sys
import os
import numpy as np
import glob
import argparse

class BeamConverter:

    
    def __init__(self, description, default_output):
      self.parser = argparse.ArgumentParser(description=description)
      self.add_common_options(default_output)
      self.add_options()
      args = self.parser.parse_args()
      self.process_common_options(args)
      self.process_options(args)
      
    def add_common_options (self, default_output):
        self.parser.add_argument('root_name', nargs=1, help='root name')
        self.parser.add_argument('--thetamax', default = 90, type=float, help='do not include data beyond this theta')
        self.parser.add_argument('--freqmin', default = 0, type=float, help='do not include data below this frequency [MHz]')
        self.parser.add_argument('--freqmax', default = 100, type=float, help='do not include data above this frequency [MHz]')
        self.parser.add_argument('--split-impedance',  action="store_true", help='Allow impedance to be split into many files')
        self.parser.add_argument('-g', '--gain', action='store_true', help='add gain field to the output file')
        self.parser.add_argument('-o', '--output_file', type=str, default = default_output, help='output filename')

    def process_common_options(self,args):
        print (args)
        self.root = args.root_name[0]
        self.save_gain = args.gain
        self.thetamax = args.thetamax
        self.freqmin = args.freqmin
        self.freqmax = args.freqmax
        self.output_file = args.output_file
        self.split_impedance = args.split_impedance
                            
      
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
            
    
      
    def save_fits(self,outfile=None):
        ## version history:
        # v1 initial version
        # v2 
        #    fground replaced with a gain_conv field (E^2*gain_conv is gain)
        #    freq does not need to be on a regular grid

        if outfile is None:
            outfile = self.output_file
        
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
        if self.save_gain:
            fits.write(self.gain, extname = 'gain')
        fits.close()
        print ('Done.')
