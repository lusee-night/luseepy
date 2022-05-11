#!/usr/bin/env python
import argparse
import numpy as np
import sys, glob
from converter_base import BeamConverter


try:
    import lusee
    have_lusee = True
except:
    have_lusee = False
    
class HFSS2LBeam(BeamConverter):

    def __init__ (self, root, thetamax = 90):
        BeamConverter.__init__(self,root,thetamax)


    def load(self):
        Edir = self.root+"/ElectricField/"
        Efiles = glob.glob(Edir+'/*.csv')
        Gdir = self.root+"/Directivity/"
        Gfiles = glob.glob(Gdir+'/*.csv')

        freq = []
        freqfname = {}
        for fname in Efiles:
            for field in fname.split('_'):
                if "MHz" in field:
                    cfreq=float(field.replace('MHz',''))
                    freqfname[cfreq]=fname
                    freq.append(cfreq)

        freqgain = {}
        for fname in Gfiles:
            for field in fname.split('_'):
                if "MHz" in field:
                    cfreq=float(field.replace('MHz',''))
                    assert (cfreq in freq)
                    freqgain[cfreq]=fname

        

        freq = sorted(freq)
        Nfreq = len(freq)
        freq_min, freq_max = freq[0], freq[-1]
        print ("Loading frequencies: ", end = "")
        have_size = False
        for i,cfreq in enumerate(freq):
            print (f"{cfreq} ... ", end = "")
            sys.stdout.flush()
            #Efield
            data = open(freqfname[cfreq]).readlines()[1:]
            data = np.array([[float(x) for x in d.split(',')] for d in data])
            phi, theta, ExR, ExI, EyR, EyI, EzR, EzI = data.T
            #gain
            data = open(freqgain[cfreq]).readlines()[1:]
            data = np.array([[float(x) for x in d.split(',')] for d in data])
            gphi, gtheta, gain = data.T
            assert(np.all(gphi==phi))
            assert(np.all(gtheta==theta))
            
            
            
            
            E = np.array([ExR+1j*ExI,  EyR+1j*EyI,  EzR+1j*EzI])
            sin = np.sin
            cos = np.cos
            thetarad = theta/180*np.pi
            phirad = phi/180*np.pi
            trad = np.array([ sin(thetarad)*cos(phirad), sin(thetarad)*sin(phirad),
                             +cos(thetarad)])
            tphi =  np.array([-sin(phirad), +cos(phirad),np.zeros_like(thetarad)])
            ttheta = np.array([ cos(thetarad)*cos(phirad), cos(thetarad)*sin(phirad),
                     -sin(thetarad)])

            Erad = (trad*E).sum(axis=0)
            Ephi = (tphi*E).sum(axis=0)
            Etheta = (ttheta*E).sum(axis=0)
            assert(np.abs((tphi*ttheta).sum(axis=0)).max()<1e-10)
            assert(np.abs((tphi*trad).sum(axis=0)).max()<1e-10)
            assert(np.abs((ttheta*trad).sum(axis=0)).max()<1e-10)
            assert(np.abs(((tphi*tphi).sum(axis=0))-1).max()<1e-10)
            assert(np.abs(((ttheta*ttheta).sum(axis=0))-1).max()<1e-10)
            assert(np.abs(((trad*trad).sum(axis=0))-1).max()<1e-10)

            E = np.sqrt(np.abs(Ephi**2)+np.abs(Erad**2)+np.abs(Etheta**2))
            check = np.where((np.abs(Erad)/E>0.01) & (E>E.max()/10))
            if len(check[0]>0):
                print (" [ Warning, Erad exceeds 1% E at places! ]")
            check = np.where((np.abs(Erad)/E>0.1) & (E>E.max()/10))
            if len(check[0]>0):
                print (" [ Warning, Erad exceeds 10% E at places! Stopping! ]",end="")
                sys.exit(1)
            if not (have_size):
                thetag= sorted(list(set(theta)))
                phig = sorted(list(set(phi)))
                theta_min, theta_max, Ntheta = thetag[0], thetag[-1], len(thetag)
                phi_min, phi_max, Nphi = phig[0], phig[-1], len(phig)
                dtheta = (theta_max - theta_min)/(Ntheta-1)
                dphi = (phi_max - phi_min)/(Nphi-1)
                gEtheta = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
                gEphi = np.zeros((Nfreq,Ntheta,Nphi),complex)+np.nan
                ggain = np.zeros((Nfreq,Ntheta,Nphi),float)+np.nan
                
                thetaL = ((theta-theta_min)/dtheta+1e-6).astype(int)
                phiL = ((phi-phi_min)/dphi+1e-6).astype(int)
                gEtheta[i,thetaL, phiL] = Etheta
                gEphi[i,thetaL, phiL] = Ephi
                ggain[i,thetaL, phiL] = gain
            else:
                gEtheta[i,thetaL, phiL] = Etheta
                gEphi[i,thetaL, phiL] = Ephi
                ggain[i,thetaL, phiL] = ggain
 
                
        Etheta = gEtheta
        Ephi = gEphi
        gain = ggain
        print ("\n")
        print ("Data loaded:")
        print (f"Freq: {freq_min} ... {freq_max} MHz  ({Nfreq} bins)")
        print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        print (f"Phi: {phi_min}, {phi_min+dphi} ... {phi_max} deg ({Nphi} bins)")

        newNtheta = int((self.thetamax-theta_min)/dtheta)+1
        newtheta_max = theta_min+dtheta*(newNtheta-1)
        if newNtheta<Ntheta:
            print ("Applying theta cut...")
            Etheta = Etheta [:, :newNtheta, :]
            Ephi = Ephi [:, :newNtheta, :]
            gain = gain [:, :newNtheta, :]
            Ntheta = newNtheta
            theta_max = newtheta_max
            print (f"Theta: {theta_min}, {theta_min+dtheta} ... {theta_max} deg ({Ntheta} bins)")
        
        ## now assert all nans are gone
        print (np.where(np.isnan(Etheta)))
        assert(not np.any(np.isnan(Etheta)))
        assert(not np.any(np.isnan(Ephi)))
        print ("Beam loading successful.")

        #f_ground = np.array([f_ground[f] for f in freq])
        #print("f_ground = ", f_ground)
        #print ("f_ground parsing successful.")
        print ("Finding gain conversion factors")
        mygain = np.abs(Etheta**2) + np.abs(Ephi**2)
        db2fact = lambda dB: 10**(dB/10)
        ratio = db2fact(gain)/mygain
        gainmax = gain.max()
        gainconv = []
        for i,f in enumerate(freq):
            r = ratio[i,:,:]
            w = np.where(gain[i,:,:]>gainmax-20)
            meanconv = r[w].mean()
            rms = np.sqrt(r[w].var())
            assert (rms/meanconv<1e-3)
            print (f"    {f} MHz    {meanconv:0.3f} ({rms/meanconv*100:0.3f}% err)")
            gainconv.append(meanconv)
        
        ZRe = np.zeros(Nfreq)+np.nan
        ZIm =np.zeros(Nfreq)+np.nan

        data = np.loadtxt(self.root+"_Z_Re.dat", skiprows=2)

        freqL = np.zeros(data.shape[0],int)-1
        for i,f in enumerate(freq):
            freqL[f==data[:,0]/1e6] = i
        ZRe [freqL] = data[:,1]

        data = np.loadtxt(self.root+"_Z_Im.dat", skiprows=2)
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
        self.gainconv = np.array(gainconv)
        self.freq = freq


def parse_args():
    parser = argparse.ArgumentParser(description='Convert HFSS Beam to LBEAM.')
    parser.add_argument('root_name', nargs=1, help='root name, ')
    parser.add_argument('--thetamax', default = 90, type=float, help='do not include data beyond this theta')
    parser.add_argument('-o', '--output_file', default = "hfss_converted.fits", help='output filename')
    args = parser.parse_args()
    O = HFSS2LBeam(args.root_name[0], thetamax = args.thetamax)
    return O, args


if __name__=="__main__":
    H2B, args = parse_args()
    H2B.load()
    H2B.save_fits(args.output_file)
    if have_lusee:
        print ("Attempting to reread the file ... ",end="")
        sys.stdout.flush()
        B = lusee.LBeam(args.output_file)
        print ("OK.")
    else:
        print ("No lusee module so no check.")
