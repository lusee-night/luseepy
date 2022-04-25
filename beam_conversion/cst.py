#!/usr/bin/env python
import argparse
import numpy as np
import sys
from converter_base import BeamConverter
import glob
from scipy.interpolate import interp1d

try:
    import lusee
    have_lusee = True
except:
    have_lusee = False
    
class CST2LBeam(BeamConverter):

    def __init__ (self, root, thetamax = 90, maxfreq = None):
        BeamConverter.__init__(self,root,thetamax)
        self.maxfreq = 1e30 if maxfreq is None else maxfreq

    def load(self):
        glob_pattern=self.root+'/ffs/*.ffs'
        flist = glob.glob(glob_pattern)
        if len(flist)==0:
            print (f"Cannot find files in {glob_pattern}")
            assert(False)
            
        beam_data = []
        print ("Loading frequencies: ", end = "")
        for fname in flist:
            freq=fname.split("_")[-2]
            assert("khz" in freq)
            freq=float(freq[:-3])/1e3
            if (freq>self.maxfreq):
                continue
            print (f"{freq} ... ", end = "")
            sys.stdout.flush()
            lines=open(fname).readlines()
            skip = True
            for line in lines:
                if skip:
                    if ("// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):" in line):
                        skip = False
                else:
                    line  = line.split()
                    if len(line)==6:
                        line = [float(x) for x in line[:6]]
                        beam_data.append([freq]+line)
        print('done.')    
        beam = np.array(beam_data)
        print (f"{beam.shape[0]} rows loaded.")
        plist = []
        for i in range(3):
            plist.append(sorted(list(set(beam[:,i]))))
        freq, phi, theta = plist
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
        phiL = ((beam[:,1]-phi_min)/dphi+1e-6).astype(int)
        thetaL = ((beam[:,2]-theta_min)/dtheta+1e-6).astype(int)
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

        fname = glob.glob(self.root+"/*s11_port1.txt")[0]
        data = np.loadtxt(fname, skiprows=2)
        S11_freq = data[:,0]
        S11 = data[:,1]*np.exp(1j*data[:,2]*2*np.pi/360.)
        print ("Assuming 50Ohm for impendance calculation")
        Z = 50*(1+S11)/(1-S11)
        Zint = interp1d(S11_freq,Z)
        Z = np.array([Zint(f) for f in freq])
        ZRe = np.real(Z)
        ZIm = np.imag(Z)

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
        self.ground_fraction = 0.5 # placeholder


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CST Beam to LBEAM.')
    parser.add_argument('root_name', nargs=1, help='root name, ')
    parser.add_argument('--maxfreq', nargs=1, default = 50, help='do not include frequencies beyond this freq [MHz]')
    parser.add_argument('--thetamax', nargs=1, default = 90, help='do not include data beyond this theta')
    parser.add_argument('-o', '--output_file', nargs=1, default = "cst_converted.fits", help='output filename')
    args = parser.parse_args()
    O = CST2LBeam(args.root_name[0], thetamax = args.thetamax, maxfreq=args.maxfreq)
    return O, args


if __name__=="__main__":
    C2B, args = parse_args()
    C2B.load()
    C2B.save_fits(args.output_file)
    if have_lusee:
        print ("Attempting to reread the file ... ",end="")
        sys.stdout.flush()
        B = lusee.LBeam(args.output_file)
        print ("OK.")
    else:
        print ("No lusee module so no check.")
