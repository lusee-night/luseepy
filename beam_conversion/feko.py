#!/usr/bin/env python
import argparse
import numpy as np
import sys
import fitsio
try:
    import lusee
    have_lusee = True
except:
    have_lusee = False
    
class Feko2LBeam:

    def __init__ (self, root, farfield, thetamax = 180):
      self.root = root
      self.farfield = farfield
      self.thetamax = thetamax

    def load(self):
        self.load_beam()

    def load_beam(self):
        fname = self.root+".out"
        data = open(fname).readlines()
        skip = True
        beam_data = []
        freq = -1.0
        farfields = []
        farfield = ""
        print ("Loading frequencies: ", end = "")
        for line in data:
            if skip:
                if ("   THETA    PHI      magn.    phase  " in line) and (farfield == self.farfield):
                    skip = False
                if "FREQ =" in line:
                    cfreq=float(line.split()[-1])/1e6
                    if cfreq!=freq:
                        print (f"{cfreq} ... ", end = "")
                        sys.stdout.flush()
                    freq=cfreq
                if "Far field request with name:" in line:
                    farfield = line[:-1].split()[-1]
                    if farfield not in farfields:
                        farfields.append(farfield)
                    
            else:
                if line == "\n":
                    skip = True
                else:
                    line  = line.split()
                    if len(line)==12:
                        line = [float(x) for x in line[:6]]
                        if line[0]<0:
                            print (line,freq)
                        beam_data.append([freq]+line)
        print()
        print ("Farfields seen:",farfields)
        beam = np.array(beam_data)
        print (f"{beam.shape[0]} rows loaded.")
        plist = []
        for i in range(3):
            plist.append(sorted(list(set(beam[:,i]))))
        freq, theta, phi = plist
        freq_min, freq_max, Nfreq = freq[0], freq[-1], len(freq)
        theta_min, theta_max, Ntheta = theta[0], theta[-1], len(theta)
        phi_min, phi_max, Nphi = phi[0], phi[-1], len(phi)
        dfreq = (freq_max - freq_min)/(Nfreq-1)
        dtheta = (theta_max - theta_min)/(Ntheta-1)
        dphi = (phi_max - phi_min)/(Nphi-1)
        np.testing.assert_almost_equal(freq[1]-freq[0], dfreq)
        np.testing.assert_almost_equal(theta[1]-theta[0], dtheta)
        np.testing.assert_almost_equal(phi[1]-phi[0], dphi)
        print ("Data loaded:")
        print (f"Freq: {freq_min}, {freq_min+dfreq} ... {freq_max} MHz  ({Nfreq} bins)")
        print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        print (f"Phi: {phi_min}, {phi_min+dphi} ... {phi_max} deg ({Nphi} bins)")
        Etheta = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
        Ephi = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
        freqL = ((beam[:,0]-freq_min)/dfreq+1e-6).astype(int)
        thetaL = ((beam[:,1]-theta_min)/dtheta+1e-6).astype(int)
        phiL = ((beam[:,2]-phi_min)/dphi+1e-6).astype(int)
        EthetaL = beam[:,3]*np.exp(1j*2*np.pi/360*beam[:,4])
        EphiL = beam[:,5]*np.exp(1j*2*np.pi/360*beam[:,6])

        Etheta[freqL, thetaL, phiL] = EthetaL
        Ephi[freqL, thetaL, phiL] = EphiL


        newNtheta = int((self.thetamax-theta_min)/dtheta)+1
        newtheta_max = theta_min+dtheta*(newNtheta-1)
        if newNtheta<Ntheta:
            print ("Applying theta cut...")
            Etheta = Etheta [:,:newNtheta, :]
            Ephi = Ephi [:,:newNtheta, :]
            Ntheta = newNtheta
            theta_max = newtheta_max
            print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        
        ## now assert all nans are gone
        assert(not np.any(np.isnan(Etheta)))
        assert(not np.any(np.isnan(Ephi)))
        print ("Beam loading successful.")

        ZRe = np.zeros(Nfreq)+np.nan
        ZIm =np.zeros(Nfreq)+np.nan

        data = np.loadtxt(self.root+"_Z_Re.dat", skiprows=2)
        freqL = ((data[:,0]/1e6-freq_min)/dfreq+1e-6).astype(int)
        ZRe [freqL] = data[:,1]

        data = np.loadtxt(self.root+"_Z_Im.dat", skiprows=2)
        freqL = ((data[:,0]/1e6-freq_min)/dfreq+1e-6).astype(int)
        ZIm [freqL] = data[:,1]

        assert(not np.any(np.isnan(ZRe)))
        assert(not np.any(np.isnan(ZIm)))
        print ("Impedance loading successful.")


        self.Etheta = Etheta
        self.Ephi = Ephi
        self.ZRe = ZRe
        self.ZIm = ZIm
        self.freq_min, self.freq_max, self.Nfreq = freq_min, freq_max, Nfreq
        self.theta_min, self.theta_max, self.Ntheta = theta_min, theta_max, Ntheta
        self.phi_min, self.phi_max, self.Nphi = phi_min, phi_max, Nphi
        
    def save_fits(self,outfile):
        header = {'version':1,
                  'freq_min':self.freq_min,
                  'freq_max':self.freq_max,
                  'freq_N':self.Nfreq,
                  'theta_min':self.theta_min,
                  'theta_max':self.theta_max,
                  'theta_N':self.Ntheta,
                  'phi_min':self.phi_min,
                  'phi_max':self.phi_max,
                  'phi_N':self.Nphi,
                  'ground_fraction':0.5, # placeholder
                  'source': 'FEKO',
                  'source_root': self.root
        }

        print ('Saving to',outfile,'... ',end="")
        sys.stdout.flush()
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        #fits[0].write_keys(header)
        fits.write(np.real(self.Etheta), extname = 'Etheta_real', header=header)
        fits.write(np.imag(self.Etheta), extname = 'Etheta_imag')
        fits.write(np.real(self.Ephi), extname = 'EPhi_real')
        fits.write(np.imag(self.Ephi), extname = 'EPhi_imag')
        fits.write(self.ZRe, extname = 'Z_real')
        fits.write(self.ZIm, extname = 'Z_imag')
        fits.close()
        print ('Done.')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert FEKO Beam to LBEAM.')
    parser.add_argument('root_name', nargs=1, help='root name, ')
    parser.add_argument('--farfield', nargs=1, default = "FarField1", help='farfield to pick')
    parser.add_argument('--thetamax', nargs=1, default = 90, help='do not include data beyond this theta')
    parser.add_argument('-o', '--output_file', nargs=1, default = "feko_converted.fits", help='output filename')
    args = parser.parse_args()
    O = Feko2LBeam(args.root_name[0],farfield = args.farfield, thetamax = args.thetamax)
    return O, args


if __name__=="__main__":
    F2B, args = parse_args()
    F2B.load()
    F2B.save_fits(args.output_file)
    if have_lusee:
        print ("Attempting to reread the file ... ",end="")
        sys.stdout.flush()
        B = lusee.LBeam(args.output_file)
        print ("OK.")
    else:
        print ("No lusee module so no check.")
